// Non-stellar draft editor: persistent spin, longitude phase and optical inputs.
import { api } from "./api.js";
import { DEFAULT_BODY_RGB } from "./color.js";

const FIELDS = [
  ["surface_pressure_pa","参考表面气压 (Pa)"],["surface_temperature_k","参考表面温度 (K)"],
  ["height_m","大气顶高度 (km)",0.001],["scale_height_m","分子标高 (km)",0.001],
  ["mean_molecular_weight","平均分子量 (g/mol)"],["aerosol_scale_height_m","气溶胶标高 (km)",0.001],
  ["aerosol_anisotropy","Henyey–Greenstein g"],["infrared_optical_depth","红外光学厚度（灰近似）"],
  ["exposure","显示曝光"],
];
const SPECTRA = [
  ["rayleigh_scattering_m_inv","Rayleigh 散射 (10⁻⁶ m⁻¹)",1e6],
  ["aerosol_scattering_m_inv","气溶胶散射 (10⁻⁶ m⁻¹)",1e6],
  ["absorption_m_inv","吸收 (10⁻⁶ m⁻¹)",1e6],
  ["ground_albedo","参考表面 Lambertian 反照率",1],
];

export function initBodyEditor({system,onCommitted}) {
  const root=document.createElement("section");
  root.id="nonstar-editor";root.className="editor-overlay";root.hidden=true;
  root.innerHTML=`
    <div class="editor-head"><div><div class="modal-eyebrow">NON-STELLAR EDITOR</div>
      <h2 id="ns-title">非恒星</h2><span class="life-badge">质量 · 外观 · 自转 · 大气</span></div>
      <button class="icon-btn" id="ns-close" title="取消，不保存">✕</button></div>
    <p class="modal-note">草稿仅在“保存并退出”时写入。不改变位置和速度；质量修改会影响未来轨道。零经线以自转相位定义。</p>
    <div class="editor-columns"><div class="editor-main">
      <section class="editor-card"><div class="sec-title">物理与外观</div>
        <div class="form-grid">
          <label>质量 (M⊕)<input id="ns-mass" type="number" min="0" step="any"></label>
          <label>反照率<input id="ns-albedo" type="number" min="0" max="1" step="any"></label>
        </div>
        <div class="bb-label">显示颜色（RGB · sRGB · 0–255）</div>
        <div class="bb-grid bb-grid-3">
          <label>R<input id="ns-color-r" type="number" min="0" max="255" step="any"></label>
          <label>G<input id="ns-color-g" type="number" min="0" max="255" step="any"></label>
          <label>B<input id="ns-color-b" type="number" min="0" max="255" step="any"></label>
        </div>
        <div class="note">目录圆点、球体与烟迹使用同一 RGB。反照率控制反射强弱，不是自发光。</div>
      </section>
      <section class="editor-card"><div class="sec-title">自转（北极 = 自转轴方向）</div>
        <div class="form-grid"><label>周期 (h；0 = 不转)<input id="ns-period" type="number" step="any"></label>
        <label>初始相位 (°)<input id="ns-phase" type="number" step="any"></label></div>
        <div class="form-grid"><label>轴 X<input id="ns-axis-x" type="number" step="any"></label>
        <label>轴 Y<input id="ns-axis-y" type="number" step="any"></label></div>
        <label class="field">轴 Z<input id="ns-axis-z" type="number" step="any"></label>
      </section>
      <section class="editor-card"><div class="sec-title">大气 · 球对称指数剖面</div>
        <label class="row-flex"><input type="checkbox" id="ns-has-atm"> 配置并保存大气</label>
        <div class="form-grid" id="ns-basic"></div>
        <label class="field">参考表面类型<select id="ns-surface-model"><option value="solid">固体表面</option><option value="cloud_deck">气体巨行星参考云面</option></select></label>
        <label class="field">体积分数（例如 H2:0.90, He:0.10）<input id="ns-composition" type="text"></label>
        <div id="ns-spectra"></div>
        <div class="note">RGB 为 610/550/465 nm 代表波段。气体名称/比例是组成元数据；实际散射与吸收由下方系数决定，不能仅凭化学名称推断。</div>
      </section>
    </div><aside class="editor-side"><section class="editor-card"><div class="sec-title">光学预览（不保存）</div>
      <div class="kv"><span class="k">天顶光学厚度 R/G/B</span><span class="v" id="ns-tau">—</span></div>
      <div class="kv"><span class="k">高阶散射</span><span class="v">2–4 阶 · 球壳路径</span></div>
      <p class="note">预计算缓存按大气输入摘要生成。高阶散射采用 Monte Carlo 路径积分并对相对方位平均；非光谱级气候求解器。</p>
      <button class="btn ghost" id="ns-preview">验证并预览</button>
    </section></aside></div>
    <div class="modal-foot editor-foot"><span id="ns-status" class="err"></span>
      <button class="btn accent" id="ns-save">保存并退出</button></div>`;
  document.getElementById("viewport-wrap").append(root);
  const $=id=>root.querySelector(`#${id}`);
  for(const [key,label] of FIELDS) {
    const el=document.createElement("label");el.textContent=label;
    const input=document.createElement("input");input.type="number";input.step="any";input.dataset.atm=key;
    el.append(input);$("ns-basic").append(el);
  }
  for(const [key,label] of SPECTRA) {
    const sec=document.createElement("div");sec.className="bb-label";sec.textContent=label;
    const grid=document.createElement("div");grid.className="form-grid";
    ["R","G","B"].forEach((channel,i)=>{
      const l=document.createElement("label");l.textContent=channel;
      const input=document.createElement("input");input.type="number";input.step="any";
      input.dataset.spectrum=key;input.dataset.index=String(i);l.append(input);grid.append(l);
    });
    $("ns-spectra").append(sec,grid);
  }
  let name=null,atmosphere=null,busy=false;
  function syncAtmosphere() {
    const enabled=$("ns-has-atm").checked;
    root.querySelectorAll("[data-atm],[data-spectrum],#ns-composition,#ns-surface-model")
      .forEach(el=>{el.disabled=!enabled;});
  }
  function setAtmosphere(a) {
    atmosphere=a;
    $("ns-has-atm").checked=a!=null;
    if(!a){syncAtmosphere();return;}
    for(const [key,,scale=1] of FIELDS)$( "ns-basic").querySelector(`[data-atm="${key}"]`).value=a[key]*scale;
    $("ns-surface-model").value=a.surface_model??"solid";
    $("ns-composition").value=Object.entries(a.composition??{}).map(([k,v])=>`${k}:${v}`).join(", ");
    for(const [key,,scale=1] of SPECTRA)root.querySelectorAll(`[data-spectrum="${key}"]`)
      .forEach(el=>{el.value=a[key][Number(el.dataset.index)]*scale;});
    syncAtmosphere();
  }
  function payload() {
    const vals=["ns-period","ns-phase","ns-axis-x","ns-axis-y","ns-axis-z"].map(id=>Number($(id).value));
    if(vals.some(x=>!Number.isFinite(x)))throw Error("自转参数必须是有限数值");
    const mass=Number($("ns-mass").value),albedo=Number($("ns-albedo").value);
    const rgb=["r","g","b"].map(c=>Number($(`ns-color-${c}`).value)/255);
    if(!Number.isFinite(mass)||mass<=0||!Number.isFinite(albedo)||albedo<0||albedo>1)
      throw Error("质量须为正，反照率须在 0–1");
    if(rgb.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error("RGB 须在 0–255 内");
    const out={name,mass_earth:mass,albedo,color_rgb:rgb,
      spin_period_h:vals[0],spin_phase_deg:vals[1],spin_axis:vals.slice(2)};
    if(!$("ns-has-atm").checked){out.atmosphere=null;return out;}
    const a={model:"spherical_exponential",surface_model:$("ns-surface-model").value,composition:{}};
    for(const [key,,scale=1] of FIELDS){const v=Number($("ns-basic").querySelector(`[data-atm="${key}"]`).value)/scale;if(!Number.isFinite(v))throw Error("大气参数必须为有限数值");a[key]=v;}
    for(const [key,,scale=1] of SPECTRA){a[key]=[0,1,2].map(i=>Number(root.querySelector(`[data-spectrum="${key}"][data-index="${i}"]`).value)/scale);if(a[key].some(x=>!Number.isFinite(x)))throw Error("光谱系数必须为有限数值");}
    const text=$("ns-composition").value.trim();
    if(text)for(const part of text.split(/[,，]/)){
      const [species,value,...rest]=part.split(":");const fraction=Number(value);
      if(rest.length||!species?.trim()||!Number.isFinite(fraction))throw Error("组成格式应为 H2:0.90, He:0.10");
      a.composition[species.trim()]=fraction;
    }
    out.atmosphere=a;return out;
  }
  async function preview() {
    if(busy)return;
    const p=payload();$("ns-preview").disabled=true;$("ns-status").textContent="";
    try {const r=await api.bodyPreview(p);$("ns-tau").textContent=r.zenith_optical_depth?.map(x=>Number(x).toExponential(3)).join(" / ")??"无大气";}
    catch(e){$("ns-status").textContent=e.message;}finally{$("ns-preview").disabled=false;}
  }
  async function open(bodyName) {
    name=bodyName;busy=false;root.hidden=false;$("ns-status").textContent="";
    try {
      const d=await api.body(name),b=d.body;
      if(name!==bodyName)return;
      $("ns-title").textContent=b.name;
      $("ns-mass").value=b.mass_kg/5.9722e24;
      $("ns-albedo").value=b.albedo;
      (b.extra?.color_rgb??DEFAULT_BODY_RGB).forEach((value,i)=>{
        $(`ns-color-${["r","g","b"][i]}`).value=Number((value*255).toFixed(2));
      });
      $("ns-period").value=(b.rotation_period_s??0)/3600;
      $("ns-phase").value=(b.spin_phase_rad??0)*180/Math.PI;
      ["x","y","z"].forEach((c,i)=>{$(`ns-axis-${c}`).value=b.spin_axis[i];});
      setAtmosphere(b.extra?.atmosphere??null);
    }catch(e){$("ns-status").textContent=e.message;}
  }
  function close(){if(!busy){root.hidden=true;name=null;}}
  $("ns-close").addEventListener("click",close);
  $("ns-has-atm").addEventListener("change",async()=>{
    if($("ns-has-atm").checked&&!atmosphere){
      try{setAtmosphere((await api.atmosphereDefaults()).atmosphere);}catch(e){$("ns-status").textContent=e.message;$("ns-has-atm").checked=false;}
    }
    syncAtmosphere();
  });
  $("ns-preview").addEventListener("click",async()=>{
    try{await preview();}catch(e){$("ns-status").textContent=e.message;}
  });
  $("ns-save").addEventListener("click",async()=>{
    if(busy)return;
    let p;try{p=payload();}catch(e){$("ns-status").textContent=e.message;return;}
    busy=true;$("ns-save").disabled=true;$("ns-status").textContent="正在保存…";
    try{const r=await api.bodyEdit(p);const local=system.bodies.find(b=>b.name===name);if(local)Object.assign(local,r.body);root.hidden=true;name=null;onCommitted?.(r.body);}
    catch(e){$("ns-status").textContent=e.message;}finally{busy=false;$("ns-save").disabled=false;}
  });
  return {open,close,get isOpen(){return !root.hidden;}};
}
