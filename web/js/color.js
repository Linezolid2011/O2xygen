// Shared colour and photometry helpers. Pure JS (no Three.js dependency), so
// the same conversions can be checked with Node and used by both sky modes.
//
// CIE 1931 analytic fits: Wyman, Sloan & Shirley, JCGT 2(2), 2013.
// They approximate the tabulated colour matching functions, not stellar SEDs.

export function clamp(x, lo = 0, hi = 1) {
  return Math.min(hi, Math.max(lo, x));
}

function gaussian(x, mean, left, right) {
  const width = x < mean ? left : right;
  const t = (x - mean) * width;
  return Math.exp(-0.5 * t * t);
}

function cieXYZ(wavelengthNm) {
  const x = 0.362 * gaussian(wavelengthNm, 442.0, 0.0624, 0.0374)
    + 1.056 * gaussian(wavelengthNm, 599.8, 0.0264, 0.0323)
    - 0.065 * gaussian(wavelengthNm, 501.1, 0.0490, 0.0382);
  const y = 0.821 * gaussian(wavelengthNm, 568.8, 0.0213, 0.0247)
    + 0.286 * gaussian(wavelengthNm, 530.9, 0.0613, 0.0322);
  const z = 1.217 * gaussian(wavelengthNm, 437.0, 0.0845, 0.0278)
    + 0.681 * gaussian(wavelengthNm, 459.0, 0.0385, 0.0725);
  return [x, y, z];
}

export function blackbodyInfo(teff) {
  const t = clamp(Number(teff), 500, 100000);
  const h = 6.62607015e-34, c = 299792458.0, k = 1.380649e-23;
  const xyz = [0, 0, 0];
  for (let nm = 380; nm <= 780; nm += 5) {
    const lambda = nm * 1e-9;
    const radiance = 1 / (lambda ** 5 * Math.expm1(h * c / (lambda * k * t)));
    const match = cieXYZ(nm);
    for (let j = 0; j < 3; j++) xyz[j] += radiance * match[j];
  }
  let [r, g, b] = [
    3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2],
    -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2],
    0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0572 * xyz[2],
  ];
  // Display gamut mapping: clamp out-of-gamut negatives and normalize all
  // channels by the SAME factor. The latter preserves chromaticity.
  const max = Math.max(r, g, b, 1e-30);
  const rgb = [r, g, b].map(v => Math.max(0, v) / max);
  return {
    rgb,
    // Bolometric intrinsic radiance ratio, separate from apparent angular
    // radius and observer distance. Do not dim a star disk by 1/d^2.
    relativeRadiance: (t / 5772) ** 4,
    xyz,
  };
}

// Ballesteros (2012), an empirical B-V to effective-temperature proxy.
// B-V is not a unique spectral type or metallicity indicator.
export function bvToTeff(bv) {
  const x = clamp(Number.isFinite(bv) ? bv : 0.656, -0.25, 2.0);
  return 4600 * (1 / (0.92 * x + 1.7) + 1 / (0.92 * x + 0.62));
}

export function bvColor(bv) {
  return blackbodyInfo(bvToTeff(bv)).rgb;
}

export function photopicLuminance(rgb) {
  return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
}

export function magnitudeFlux(magnitude) {
  return 10 ** (-0.4 * magnitude);
}

// A documented V-band zero point, and the Sun's Johnson V magnitude at 1 AU.
// The ratio is used only to put background-star flux in the same normalized
// photometric system as the atmosphere. It is not a claim of absolute human
// eye sensitivity on an arbitrary monitor.
export const VEGA_ILLUMINANCE_LUX = 2.54e-6;
export const SUN_V_MAGNITUDE = -26.74;
export const SUN_EARTH_ILLUMINANCE_LUX = 1.27e5;
// 0 mag / solar V=-26.74 mag. This is about 2e-11, not its reciprocal.
export const STAR_FLUX_UNIT = VEGA_ILLUMINANCE_LUX / SUN_EARTH_ILLUMINANCE_LUX;

// Gaussian point-spread function, 60 arcsec sigma. Pixel footprint remains a
// visual aid; radiance is scaled by this PSF's solid angle, not by an arbitrary
// "brightness = 1.25 - 0.1 * magnitude" ramp.
export const STAR_PSF_SIGMA_RAD = 60 * Math.PI / (180 * 3600);
export const STAR_PSF_SOLID_ANGLE = 2 * Math.PI * STAR_PSF_SIGMA_RAD ** 2;
export const STAR_RADIANCE_UNIT = STAR_FLUX_UNIT / STAR_PSF_SOLID_ANGLE;

// Non-stars store display RGB as normalized sRGB values. Lights and shaders
// convert them to linear sRGB once; the save never stores an MIST-derived star
// colour, since a star's colour remains a function of its current Teff.
export const DEFAULT_BODY_RGB = [0.42, 0.55, 0.72];

export function normalizeBodyRGB(value) {
  if (!Array.isArray(value) || value.length !== 3
      || value.some(v => !Number.isFinite(Number(v)) || Number(v) < 0 || Number(v) > 1)) {
    return [...DEFAULT_BODY_RGB];
  }
  return value.map(v => Number(v));
}

export function linearToSrgb(value) {
  return value.map(v => {
    const x = Math.max(0, v);
    return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
  });
}

export function rgbToHex(value) {
  return "#" + normalizeBodyRGB(value).map(v => Math.round(clamp(v) * 255)
    .toString(16).padStart(2, "0")).join("");
}

export function bodySrgb(body) {
  return body.kind === "star"
    ? linearToSrgb(blackbodyInfo(body.extra?.teff_k ?? 5772).rgb)
    : normalizeBodyRGB(body.extra?.color_rgb);
}

export function bodyHex(body) {
  return rgbToHex(bodySrgb(body));
}

export const STAR_FIELD_VERTEX = `
attribute float aMag;
attribute vec3 aColor;
varying vec3 vColor;
varying float vMag;
varying float vUp;
uniform float uPixelRatio;
void main() {
  vColor = aColor; vMag = aMag; vUp = normalize(position).z;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float size = 1.6 + 0.32 * clamp(5.0 - aMag, 0.0, 8.0);
  gl_PointSize = size * uPixelRatio;
  gl_Position = projectionMatrix * mv;
}`;

export const TONE_GLSL = `
vec3 preserveTone(vec3 c, float exposure) {
  c = max(c, vec3(0.0));
  float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float mapped = 1.0 - exp(-y * exposure);
  c *= mapped / max(y, 1e-20);
  // Gamut compression with a shared factor, not per-channel saturation.
  c /= max(max(c.r, max(c.g, c.b)), 1.0);
  return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}`;

export const STAR_FIELD_FRAGMENT = `
varying vec3 vColor;
varying float vMag;
varying float vUp;
uniform float uRadianceUnit;
uniform float uExposure;
uniform bool uRaw;
${TONE_GLSL}
void main() {
  if (vUp < 0.0) discard;
  float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
  if (r > 1.0) discard;
  float psf = exp(-3.0 * r * r);
  float flux = pow(10.0, -0.4 * vMag);
  vec3 radiance = vColor * flux * uRadianceUnit * psf;
  gl_FragColor = vec4(uRaw ? radiance : preserveTone(radiance, uExposure), 1.0);
}`;
