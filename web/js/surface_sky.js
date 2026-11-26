// Observer-local WebGL2 sky. Physical inputs and scattering LUTs arrive from
// the server; this module only reconstructs the radiative transfer display.
// No N-body integration, orbital extrapolation or independent spin clock.
import * as THREE from "three";
import { blackbodyInfo, bvColor, bodySrgb, STAR_FIELD_VERTEX,
  STAR_FIELD_FRAGMENT, STAR_RADIANCE_UNIT, TONE_GLSL,
  photopicLuminance, clamp } from "./color.js";

const EARTH_FLUX = 1361.0;

const SKY_VERTEX = `
precision highp float;
out vec2 vUv;
void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const SKY_FRAGMENT = `
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uStars;
uniform sampler2D uBodies;
uniform sampler2D uTau;
uniform sampler2D uMultiple;
uniform mat3 uCameraBasis;
uniform float uTanHalfFov;
uniform float uAspect;
uniform vec3 uSunDir[3];
uniform vec3 uSunColor[3];
uniform float uSunVisible[3];
uniform int uSunCount;
uniform vec3 uRayleigh;
uniform vec3 uAerosol;
uniform vec3 uAbsorption;
uniform vec3 uGround;
uniform float uRadius;
uniform float uHeight;
uniform float uRayleighH;
uniform float uAerosolH;
uniform float uG;
uniform float uExposure;
uniform float uMuSize;
uniform float uHeightSize;
uniform float uViewSize;
uniform float uSunSize;
uniform bool uHasAtmosphere;
const float PI = 3.14159265358979323846;

vec3 tauToTop(float r, float mu) {
  float h = clamp(r-uRadius, 0.0, uHeight);
  float v = (sqrt(h/uHeight)*(uHeightSize-1.0)+0.5)/uHeightSize;
  float x=sign(mu)*pow(abs(clamp(mu,-1.0,1.0)),1.0/3.0);
  float u = ((x*0.5+0.5)*(uMuSize-1.0)+0.5)/uMuSize;
  return texture(uTau, vec2(u,v)).rgb;
}
vec3 solarT(vec3 q, vec3 s) {
  float b=dot(q,s), c=dot(q,q)-uRadius*uRadius;
  float disc=b*b-c;
  // The observer is 1 cm above the reference sphere. Any positive near
  // intersection is solid ground, regardless of the local Sun elevation.
  if (disc>0.0 && -b-sqrt(disc)>0.001) return vec3(0.0);
  return exp(-min(tauToTop(length(q),dot(normalize(q),s)),vec3(700.0)));
}
vec3 singleScattering(vec3 v, vec3 s) {
  if (v.z<0.0) return vec3(0.0);
  vec3 p0=vec3(0.0,0.0,uRadius+0.01);
  float r=uRadius+0.01;
  float L=-r*v.z+sqrt(max(0.0,pow(uRadius+uHeight,2.0)-r*r*(1.0-v.z*v.z)));
  vec3 tau0=tauToTop(r,v.z);
  float pr=3.0/(16.0*PI)*(1.0+pow(dot(v,s),2.0));
  float g2=uG*uG;
  float pm=(1.0-g2)/(4.0*PI*pow(max(1.0+g2-2.0*uG*dot(v,s),1e-6),1.5));
  vec3 sum=vec3(0.0);
  // Midpoint quadrature over the true curved ray; tau itself is precomputed
  // with float64 Gauss-Legendre quadrature on the server.
  for(int i=0;i<48;i++) {
    float t=(float(i)+0.5)/48.0*L;
    vec3 q=p0+v*t;
    float qr=length(q), h=qr-uRadius;
    vec3 ray=exp(-h/uRayleighH)*uRayleigh;
    vec3 mie=exp(-h/uAerosolH)*uAerosol;
    vec3 tv=exp(-min(max(tau0-tauToTop(qr,dot(q,v)/qr),vec3(0.0)),vec3(700.0)));
    sum += (ray*pr+mie*pm)*tv*solarT(q,s)*(L/48.0);
  }
  return sum;
}
vec3 multipleScattering(float muS,float muV) {
  float u=((clamp(muV,0.0,1.0))*(uViewSize-1.0)+0.5)/uViewSize;
  float v=((clamp(muS,-1.0,1.0)*0.5+0.5)*(uSunSize-1.0)+0.5)/uSunSize;
  return texture(uMultiple,vec2(u,v)).rgb;
}
${TONE_GLSL}
void main() {
  vec2 ndc=vUv*2.0-1.0;
  vec3 dir=normalize(uCameraBasis*vec3(ndc.x*uTanHalfFov*uAspect,ndc.y*uTanHalfFov,-1.0));
  vec3 background=texture(uStars,vUv).rgb;
  vec3 bodies=texture(uBodies,vUv).rgb;
  if (!uHasAtmosphere) {
    vec3 direct=vec3(0.0);
    for(int i=0;i<3;i++) if(i<uSunCount)
      direct+=uSunColor[i]*max(uSunDir[i].z,0.0)*uSunVisible[i];
    vec3 color=dir.z<0.0 ? uGround/PI*direct : background+bodies;
    fragColor=vec4(preserveTone(color,uExposure),1.0);
    return;
  }
  if (dir.z<0.0) {
    vec3 ground=vec3(0.0);
    for(int i=0;i<3;i++) if(i<uSunCount) {
      vec3 s=uSunDir[i];
      ground += uSunColor[i]*max(s.z,0.0)*uSunVisible[i]*solarT(vec3(0,0,uRadius+0.01),s);
    }
    fragColor=vec4(preserveTone(uGround/PI*ground+bodies,uExposure),1.0);
    return;
  }
  vec3 trans=exp(-min(tauToTop(uRadius+0.01,dir.z),vec3(700.0)));
  vec3 radiance=vec3(0.0);
  for(int i=0;i<3;i++) if(i<uSunCount) {
    vec3 s=uSunDir[i];
    radiance += uSunColor[i]*(singleScattering(dir,s)+multipleScattering(s.z,dir.z));
  }
  // Linear HDR composition happens BEFORE display tone mapping. The body
  // target already contains its wavelength-dependent atmospheric extinction.
  fragColor=vec4(preserveTone(background*trans+radiance+bodies,uExposure),1.0);
}
`;

function localVector(v, observer) {
  const e=observer.east_world, n=observer.north_world, u=observer.zenith_world;
  return new THREE.Vector3(v[0]*e[0]+v[1]*e[1]+v[2]*e[2],
                           v[0]*n[0]+v[1]*n[1]+v[2]*n[2],
                           v[0]*u[0]+v[1]*u[1]+v[2]*u[2]);
}

export class SurfaceSky {
  constructor(renderer, canvas, system) {
    this.renderer=renderer; this.canvas=canvas; this.system=system;
    this.active=false; this.data=null; this.atmosphere=null;
    this.camera=new THREE.PerspectiveCamera(70,1,0.1,1e15);
    this.camera.position.set(0,0,0);
    this.worldScene=new THREE.Scene();
    this.starScene=new THREE.Scene();
    this.starScene.background=new THREE.Color(0x000000);
    this.starCamera=this.camera.clone();
    this.target=new THREE.WebGLRenderTarget(1,1,{depthBuffer:false,stencilBuffer:false,
      type:THREE.HalfFloatType});
    this.bodyTarget=new THREE.WebGLRenderTarget(1,1,{depthBuffer:true,stencilBuffer:false,
      type:THREE.HalfFloatType});
    this.screenScene=new THREE.Scene();
    this.screenCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    this.uniforms={
      uStars:{value:this.target.texture},uBodies:{value:this.bodyTarget.texture},
      uTau:{value:null},uMultiple:{value:null},
      uCameraBasis:{value:new THREE.Matrix3()},
      uTanHalfFov:{value:Math.tan(THREE.MathUtils.degToRad(70)/2)},
      uAspect:{value:1},
      uSunDir:{value:[new THREE.Vector3(),new THREE.Vector3(),new THREE.Vector3()]},
      uSunColor:{value:[new THREE.Vector3(),new THREE.Vector3(),new THREE.Vector3()]},
      uSunVisible:{value:[0,0,0]},uSunCount:{value:0},
      uRayleigh:{value:new THREE.Vector3()},uAerosol:{value:new THREE.Vector3()},
      uAbsorption:{value:new THREE.Vector3()},uGround:{value:new THREE.Vector3(0.2,0.2,0.2)},
      uRadius:{value:1},uHeight:{value:1},uRayleighH:{value:1},uAerosolH:{value:1},
      uG:{value:0},uExposure:{value:1},uMuSize:{value:2},uHeightSize:{value:2},
      uViewSize:{value:2},uSunSize:{value:2},uHasAtmosphere:{value:false},
    };
    const material=new THREE.ShaderMaterial({
      glslVersion:THREE.GLSL3,vertexShader:SKY_VERTEX,fragmentShader:SKY_FRAGMENT,
      uniforms:this.uniforms,depthTest:false,depthWrite:false,
    });
    this.screenScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),material));
    this.yaw=0; this.pitch=0;
    // ENU is right-handed (E×N=U). A camera looks along -Z, so its
    // back=-U, right=-E and up=N give (-E)×N=-U: a proper rotation.
    this.homeQuaternion=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3(-1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,-1)));
    this.camera.quaternion.copy(this.homeQuaternion);
    this._bindControls();
    this._makeHud();
  }

  _makeHud() {
    const wrap=this.canvas.parentElement;
    this.hud=document.createElement("div"); this.hud.className="live-sky-hud";
    this.title=document.createElement("div"); this.title.className="live-sky-title";
    this.orientation=document.createElement("div"); this.orientation.className="live-sky-orientation";
    this.hud.append(this.title,this.orientation); wrap.append(this.hud); this.hud.hidden=true;
  }

  _bindControls() {
    let drag=null;
    this.canvas.addEventListener("pointerdown",e=>{
      if(!this.active || e.button!==0) return;
      drag={x:e.clientX,y:e.clientY,yaw:this.yaw,pitch:this.pitch};
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove",e=>{
      if(!this.active || !drag) return;
      this.yaw=drag.yaw-(e.clientX-drag.x)*0.003;
      this.pitch=THREE.MathUtils.clamp(drag.pitch+(e.clientY-drag.y)*0.003,-Math.PI/2+0.002,Math.PI/2-0.002);
    });
    const end=()=>{drag=null;}; this.canvas.addEventListener("pointerup",end);
    this.canvas.addEventListener("pointercancel",end);
    this.canvas.addEventListener("wheel",e=>{
      if(!this.active) return;
      e.preventDefault(); this.camera.fov=THREE.MathUtils.clamp(this.camera.fov+e.deltaY*0.03,20,100);
      this.camera.updateProjectionMatrix();
    },{passive:false});
    window.addEventListener("keydown",e=>{
      if(this.active && e.code==="Home" && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) {
        e.preventDefault(); this.home();
      }
    });
  }

  enter() { this.active=true; this.hud.hidden=false; this.resize(); }
  exit() { this.active=false; this.hud.hidden=true; }
  home() { this.yaw=0; this.pitch=0; }
  toggleStars() { if(this.stars) this.stars.visible=!this.stars.visible; }
  resize() {
    const w=this.canvas.clientWidth,h=this.canvas.clientHeight;
    if(!w||!h)return;
    this.camera.aspect=w/h; this.camera.updateProjectionMatrix();
    this.starCamera.aspect=w/h; this.starCamera.updateProjectionMatrix();
    this.target.setSize(Math.max(1,w),Math.max(1,h));
    this.bodyTarget.setSize(Math.max(1,w),Math.max(1,h));
  }

  _disposeGroup(group) {
    for(const child of [...group.children]) {
      group.remove(child);
      child.geometry?.dispose();
      if(Array.isArray(child.material))child.material.forEach(m=>m.dispose());
      else child.material?.dispose();
    }
  }

  setData(data) {
    this.data=data;
    if(this._bodyName!==data.point.body){
      this._bodyName=data.point.body;
      this.atmosphere=null;
    }
    if(!data.atmosphere)this.atmosphere=null;
    const o=data.observer;
    this.east=new THREE.Vector3(...o.east_world);
    this.north=new THREE.Vector3(...o.north_world);
    this.up=new THREE.Vector3(...o.zenith_world);
    this._disposeGroup(this.worldScene); this._disposeGroup(this.starScene);
    this.starScene.background=new THREE.Color(0x000000);
    const sky=data.sky, pos=new Float32Array(sky.hip.length*3), mag=new Float32Array(sky.hip.length), col=new Float32Array(sky.hip.length*3);
    for(let i=0;i<sky.hip.length;i++) {
      const v=localVector([sky.dir_x[i],sky.dir_y[i],sky.dir_z[i]],o);
      pos.set([v.x*1e6,v.y*1e6,v.z*1e6],i*3); mag[i]=sky.mag[i];
      col.set(bvColor(sky.bv[i]),i*3);
    }
    const geo=new THREE.BufferGeometry();
    geo.setAttribute("position",new THREE.BufferAttribute(pos,3));
    geo.setAttribute("aMag",new THREE.BufferAttribute(mag,1));
    geo.setAttribute("aColor",new THREE.BufferAttribute(col,3));
    const mat=new THREE.ShaderMaterial({
      uniforms:{
        uPixelRatio:{value:this.renderer.getPixelRatio()},
        uRadianceUnit:{value:STAR_RADIANCE_UNIT},
        uExposure:{value:16000},
        uRaw:{value:true},
      },
      vertexShader:STAR_FIELD_VERTEX,
      fragmentShader:STAR_FIELD_FRAGMENT,
      transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,
    });
    this.stars=new THREE.Points(geo,mat);this.starScene.add(this.stars);
    this.starScene.background=new THREE.Color(0x000000);
    const suns=data.bodies.filter(b=>b.kind==="star").slice(0,3);
    this.uniforms.uSunCount.value=suns.length;
    suns.forEach((s,i)=>{
      this.uniforms.uSunDir.value[i].set(...s.direction_enu).normalize();
      const f=(s.illumination_wm2??0)/EARTH_FLUX;
      this.uniforms.uSunColor.value[i].set(...s.spectrum_rgb).multiplyScalar(f);
      this.uniforms.uSunVisible.value[i]=s.occluded_by?.length?0:1;
    });
    const ground=data.atmosphere?.ground_albedo??[0.16,0.16,0.16];
    this.uniforms.uGround.value.set(...ground);
    for(const b of data.bodies) {
      if(!b.above_horizon || b.occluded_by?.length)continue;
      const distance=b.distance_m;
      const radius=Math.max(distance*Math.tan(b.angular_radius_rad),distance*0.00012);
      let rgb,intensity=1;
      if(b.kind==="star") {
        const color=blackbodyInfo(b.teff_k??5772);
        rgb=color.rgb;intensity=color.relativeRadiance;
      } else {
        const light=new THREE.Vector3();
        for(const s of suns) {
          const sv=new THREE.Vector3(...s.direction_enu);
          light.addScaledVector(sv,(s.illumination_wm2??0)/EARTH_FLUX);
        }
        const normal=b.direction_enu??[0,0,1];
        const phase=clamp(0.20+0.80*Math.max(0,light.dot(new THREE.Vector3(...normal))),0.16,1);
        const base=this.system.bodies.find(x=>x.name===b.name);
        const albedo=base?.albedo??0.2;
        rgb=new THREE.Color().setRGB(...bodySrgb(base),THREE.SRGBColorSpace)
          .toArray().map(x=>x*albedo*phase);
      }
      const material=new THREE.ShaderMaterial({
        uniforms:{
          uColor:{value:new THREE.Vector3(...rgb)},
          uTransmittance:{value:new THREE.Vector3(...this.transmittanceSpectrum(b.direction_enu))},
          uRadiance:{value:intensity},
        },
        vertexShader:`varying vec3 vNormal;void main(){vNormal=normal;
          gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
        fragmentShader:`uniform vec3 uColor;uniform vec3 uTransmittance;uniform float uRadiance;
          void main(){gl_FragColor=vec4(uColor*uTransmittance*uRadiance,1.0);}`,
        toneMapped:false,transparent:false,depthWrite:true,
      });
      const mesh=new THREE.Mesh(new THREE.SphereGeometry(radius,48,32),material);
      mesh.position.set(...new THREE.Vector3(...b.direction_enu).multiplyScalar(distance).toArray());
      mesh.userData={bodyName:b.name,physicalAngularRadius:b.angular_radius_rad,
        directionEnu:[...b.direction_enu]};
      this.worldScene.add(mesh);
    }
    this.title.textContent=`${data.point.name} · ${data.point.body}`;
    this.uniforms.uHasAtmosphere.value=Boolean(data.atmosphere && this.atmosphere);
    this.updateExposure();
    if (this._pointName !== data.point.name) {
      this._pointName = data.point.name;
      this.home();
    }
  }

  _tauAt(radius,mu) {
    const a=this.atmosphere,R=this.uniforms.uRadius.value,H=this.uniforms.uHeight.value;
    const h=clamp(radius-R,0,H),m=clamp(mu,-1,1);
    let lo=0,hi=a.mu.length-1;
    while(hi-lo>1){const i=(lo+hi)>>1;if(a.mu[i]<m)lo=i;else hi=i;}
    const mf=(m-a.mu[lo])/(a.mu[hi]-a.mu[lo]);
    const coord=Math.sqrt(h/H)*(a.height_m.length-1);
    const upper=Math.max(1,Math.min(a.height_m.length-1,Math.ceil(coord))),lower=upper-1;
    const hf=(h-a.height_m[lower])/Math.max(a.height_m[upper]-a.height_m[lower],1e-30);
    return [0,1,2].map(c=>{
      const t=a.tau;
      return t[lower][lo][c]*(1-hf)*(1-mf)+t[upper][lo][c]*hf*(1-mf)
        +t[lower][hi][c]*(1-hf)*mf+t[upper][hi][c]*hf*mf;
    });
  }

  transmittanceSpectrum(direction) {
    if(!this.atmosphere || !direction)return [1,1,1];
    return this._tauAt(this.uniforms.uRadius.value+0.01,direction[2])
      .map(t=>Math.exp(-Math.min(t,700)));
  }

  transmittance(direction) {
    const t=this.transmittanceSpectrum(direction);
    return (t[0]+t[1]+t[2])/3;
  }

  _solarTransmittance(q,s) {
    const R=this.uniforms.uRadius.value;
    const b=q[0]*s[0]+q[1]*s[1]+q[2]*s[2];
    const c=q[0]**2+q[1]**2+q[2]**2-R*R,disc=b*b-c;
    if(disc>0 && -b-Math.sqrt(disc)>0.001)return [0,0,0];
    const r=Math.hypot(...q);
    return this._tauAt(r,(q[0]*s[0]+q[1]*s[1]+q[2]*s[2])/r)
      .map(t=>Math.exp(-Math.min(t,700)));
  }

  _singleReference(v,s) {
    const R=this.uniforms.uRadius.value,H=this.uniforms.uHeight.value;
    if(v[2]<0)return [0,0,0];
    const a=this.data.atmosphere,ray0=[0,0,R+0.01],r=R+0.01;
    const L=-r*v[2]+Math.sqrt(Math.max(0,(R+H)**2-r*r*(1-v[2]*v[2])));
    const tau0=this._tauAt(r,v[2]),cos=v[0]*s[0]+v[1]*s[1]+v[2]*s[2];
    const pr=3/(16*Math.PI)*(1+cos*cos),g=a.aerosol_anisotropy;
    const pm=(1-g*g)/(4*Math.PI*Math.max(1+g*g-2*g*cos,1e-6)**1.5);
    const out=[0,0,0];
    for(let i=0;i<32;i++){
      const t=(i+0.5)/32*L,q=v.map((x,j)=>ray0[j]+x*t),qr=Math.hypot(...q),h=qr-R;
      const ray=a.rayleigh_scattering_m_inv.map(x=>Math.exp(-h/a.scale_height_m)*x);
      const mie=a.aerosol_scattering_m_inv.map(x=>Math.exp(-h/a.aerosol_scale_height_m)*x);
      const view=this._tauAt(r,v[2]).map((tau,c)=>Math.max(0,tau0[c]-tau));
      const solar=this._solarTransmittance(q,s);
      for(let c=0;c<3;c++)out[c]+=(ray[c]*pr+mie[c]*pm)*Math.exp(-view[c])*solar[c]*(L/32);
    }
    return out;
  }

  _multipleReference(muS,muV) {
    const a=this.atmosphere,ns=a.sun_cos.length,nv=a.view_cos.length;
    const x=clamp(muV,0,1)*(nv-1),y=(clamp(muS,-1,1)*0.5+0.5)*(ns-1);
    const x0=Math.min(nv-2,Math.floor(x)),y0=Math.min(ns-2,Math.floor(y));
    const fx=x-x0,fy=y-y0;
    return [0,1,2].map(c=>a.multiple[y0][x0][c]*(1-fx)*(1-fy)
      +a.multiple[y0][x0+1][c]*fx*(1-fy)+a.multiple[y0+1][x0][c]*(1-fx)*fy
      +a.multiple[y0+1][x0+1][c]*fx*fy);
  }

  updateExposure() {
    if(!this.atmosphere || !this.data?.atmosphere){
      this.exposure=16000;this.skyLevel=0;this.uniforms.uExposure.value=this.exposure;return;
    }
    const total=[0,0,0],zenith=[0,0,1];
    for(const star of this.data.bodies.filter(b=>b.kind==='star').slice(0,3)){
      const sun=star.direction_enu;
      const one=this._singleReference(zenith,sun),many=this._multipleReference(sun[2],1);
      const f=(star.illumination_wm2??0)/EARTH_FLUX,spectrum=star.spectrum_rgb??[1,1,1];
      for(let c=0;c<3;c++)total[c]+=f*spectrum[c]*(one[c]+many[c]);
    }
    this.skyLevel=Math.max(0,photopicLuminance(total));
    // Display-only visual adaptation. The physical HDR values never change.
    const sensitivity=clamp(0.015/Math.max(this.skyLevel,1e-10),1,2000);
    this.exposure=8*(this.data.atmosphere.exposure??1)*sensitivity;
    this.uniforms.uExposure.value=this.exposure;
  }

  setAtmosphere(model) {
    this.atmosphere=model;
    const nt=model.mu.length,nh=model.height_m.length;
    const tau=new Float32Array(nt*nh*4);
    for(let y=0;y<nh;y++)for(let x=0;x<nt;x++)tau.set([...model.tau[y][x],1],(y*nt+x)*4);
    const ns=model.sun_cos.length,nv=model.view_cos.length;
    const multiple=new Float32Array(nv*ns*4);
    for(let y=0;y<ns;y++)for(let x=0;x<nv;x++)multiple.set([...model.multiple[y][x],1],(y*nv+x)*4);
    for(const key of ["uTau","uMultiple"])this.uniforms[key].value?.dispose();
    // tau rows are altitude, multiple rows are solar zenith.
    const tauTex=new THREE.DataTexture(tau,nt,nh,THREE.RGBAFormat,THREE.FloatType);
    const multiTex=new THREE.DataTexture(multiple,nv,ns,THREE.RGBAFormat,THREE.FloatType);
    for(const t of [tauTex,multiTex]) {
      t.minFilter=THREE.LinearFilter;t.magFilter=THREE.LinearFilter;
      t.wrapS=t.wrapT=THREE.ClampToEdgeWrapping;t.generateMipmaps=false;
      t.flipY=false;t.colorSpace=THREE.NoColorSpace;t.needsUpdate=true;
    }
    this.uniforms.uTau.value=tauTex;this.uniforms.uMultiple.value=multiTex;
    const a=this.data?.atmosphere;
    if(!a)return;
    const body=this.system.bodies.find(b=>b.name===this.data.point.body);
    this.uniforms.uRayleigh.value.set(...a.rayleigh_scattering_m_inv);
    this.uniforms.uAerosol.value.set(...a.aerosol_scattering_m_inv);
    this.uniforms.uAbsorption.value.set(...a.absorption_m_inv);
    this.uniforms.uRadius.value=body.radius_m;this.uniforms.uHeight.value=a.height_m;
    this.uniforms.uRayleighH.value=a.scale_height_m;this.uniforms.uAerosolH.value=a.aerosol_scale_height_m;
    this.uniforms.uG.value=a.aerosol_anisotropy;
    this.uniforms.uMuSize.value=nt;this.uniforms.uHeightSize.value=nh;
    this.uniforms.uViewSize.value=nv;this.uniforms.uSunSize.value=ns;
    this.uniforms.uHasAtmosphere.value=true;
    this.updateExposure();
    for(const child of this.worldScene.children) {
      child.material?.uniforms?.uTransmittance?.value.set(
        ...this.transmittanceSpectrum(child.userData.directionEnu));
    }
  }

  render() {
    if(!this.active || !this.data)return;
    const qYaw=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),this.yaw);
    const qPitch=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),this.pitch);
    this.camera.quaternion.copy(qYaw).multiply(qPitch).multiply(this.homeQuaternion);
    this.camera.updateMatrixWorld();
    this.uniforms.uCameraBasis.value.setFromMatrix4(this.camera.matrixWorld);
    this.uniforms.uTanHalfFov.value=Math.tan(THREE.MathUtils.degToRad(this.camera.fov)/2);
    this.uniforms.uAspect.value=this.camera.aspect;
    this.starCamera.copy(this.camera);this.starCamera.position.set(0,0,0);
    this.starCamera.updateMatrixWorld();
    const r=this.renderer,oldTarget=r.getRenderTarget();
    const oldColor=r.getClearColor(new THREE.Color()),oldAlpha=r.getClearAlpha();
    r.autoClear=true;r.setClearColor(0x000000,0);
    r.setRenderTarget(this.target);r.clear();r.render(this.starScene,this.starCamera);
    r.setRenderTarget(this.bodyTarget);r.clear();r.render(this.worldScene,this.camera);
    // Only this final pass performs exposure and display gamma conversion.
    r.setRenderTarget(oldTarget);r.clear();r.autoClear=false;
    r.render(this.screenScene,this.screenCamera);
    r.setClearColor(oldColor,oldAlpha);
    const direction=new THREE.Vector3(0,0,-1).applyQuaternion(this.camera.quaternion);
    const alt=THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(direction.z,-1,1)));
    const az=(THREE.MathUtils.radToDeg(Math.atan2(direction.x,direction.y))+360)%360;
    this.orientation.textContent=`高度 ${alt.toFixed(1)}° · 方位 ${az.toFixed(1)}°${this.atmosphere?" · 2–4 阶多次散射":""}`;
  }
}
