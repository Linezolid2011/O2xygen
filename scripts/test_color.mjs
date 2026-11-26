import assert from "node:assert/strict";
import {
  blackbodyInfo, bvColor, bvToTeff, magnitudeFlux, bodyHex,
  STAR_FLUX_UNIT, STAR_RADIANCE_UNIT, STAR_PSF_SOLID_ANGLE,
} from "../web/js/color.js";

const warm = blackbodyInfo(4000), hot = blackbodyInfo(10000);
assert.ok(warm.rgb[0] > warm.rgb[2], "4000 K must be warm");
assert.ok(hot.rgb[2] > hot.rgb[0], "10000 K must be blue");
for (const t of [500, 2500, 5772, 10000, 100000]) {
  const c = blackbodyInfo(t);
  assert.ok(c.rgb.every(x => Number.isFinite(x) && x >= 0 && x <= 1));
  assert.ok(c.relativeRadiance > 0);
}
assert.ok(Math.abs(hot.relativeRadiance / (10000 / 5772) ** 4 - 1) < 1e-12);
assert.ok(bvColor(-0.2)[2] > bvColor(-0.2)[0], "negative B-V is blue");
assert.ok(bvColor(1.6)[0] > bvColor(1.6)[2], "large B-V is red");
assert.ok(bvToTeff(0.656) > 4000 && bvToTeff(0.656) < 8000);
assert.ok(Math.abs(magnitudeFlux(1) / magnitudeFlux(6) - 100) < 1e-9);
assert.ok(STAR_FLUX_UNIT > 1e-12 && STAR_FLUX_UNIT < 1e-10);
assert.ok(STAR_PSF_SOLID_ANGLE > 0 && STAR_RADIANCE_UNIT > 0);
assert.equal(bodyHex({ kind: "body" }), "#6b8cb8");
assert.equal(bodyHex({ kind: "body", extra: { color_rgb: [0.8, 0.4, 0.2] } }), "#cc6633");
assert.equal(bodyHex({ kind: "body", extra: { color_rgb: [2, -1, 0.5] } }), "#6b8cb8");
const blueStarHex = bodyHex({ kind: "star", extra: { teff_k: 12000 } });
assert.ok(parseInt(blueStarHex.slice(5, 7), 16) > parseInt(blueStarHex.slice(1, 3), 16));
console.log("body colors", { default: bodyHex({ kind: "body" }), blueStarHex });
console.log("color/photometry tests passed", {
  warm: warm.rgb, hot: hot.rgb,
  blueBV: bvColor(-0.2), redBV: bvColor(1.6),
  fluxRatioOneToSix: magnitudeFlux(1) / magnitudeFlux(6),
  starFluxUnit: STAR_FLUX_UNIT,
});
