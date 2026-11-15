# ORBIT — Reconnaissance Dossier (v0.1, 2026-11-14)

Working hypothesis: "ORBIT / 世界观构造工作台" (Keplerian Simulation v1.0) is a **single-user desktop worldbuilding app for fiction writers** that wraps a hierarchical Keplerian orbit simulator around a pipeline of *standard, public* astrophysics models, and uses an LLM to turn the computed numbers into prose + images.

Critical recon conclusion: **this is a spec-reconstruction problem, not a binary-cracking problem.** Every scientific module is a known public model with published equations/data. The proprietary part is only the integration, UI, prompt library, and parameter choices.

## 1. Feature inventory (from screenshots)

### System / shell
- Local-first desktop app (工作空间 = "白玫瑰"; 已保存到…; 属于你的宇宙 · 设定保存在本地; 打开项目文件夹 / 保存设定 / 打开 / 新建项目 / 新建天体).
- Body catalog tree: 恒星 (star) / 行星 (planet) / 卫星 (moon), hierarchical (moon under planet).
- 3D space view (3D 空间 / 2D 示意 toggle, axes gizmo, orbit lines) + 2D schematic.
- Timeline: 10-year span, speed in 天/秒, play/pause/rewind, UTC clock, 连续调速.

### Star module
- 参数估算 modal: **MIST v1.2 / MESA r7503** non-rotating, solar-scaled tracks.
- Interpolates on initial mass × [Fe/H], then age. Visible nodes: 1.28/1.3 M☉ × [Fe/H] 0/0.25.
- Outputs: spherical radius, L, Teff; cumulative wind loss (初始−当前), current loss rate from track `star_mdot`.
- Explicitly documented limitations (no binaries, no SN/BH, no extrapolation of mass loss).

### Planet/moon structure module
- 参数估算: composition presets (硅酸盐+铁核), **Zeng+2019** high-pressure EOS mass–radius grid (0.01–100 M⊕, core fraction 0–1, interpolated); **PREM**-based model (1–8 M⊕, core fraction 0–0.4); explicitly excludes thick atmosphere/water layers.
- 扁率模型: 慢转近似·均匀流体 / Maclaurin exact / **ToF 4th-order theory of figures** / **CMS 分层自洽迭代** (concentric Maclaurin spheroids).
- Internal structure: polytrope n=1 spherical layered presets, custom density layers, densities normalized to total mass; "按质量、成分与温度估算巨行星" (giant-planet radius estimator).
- Gravity: GM/r² monopole + centrifugal at equator; density from spheroid volume.
- Outputs (Múspellsheimr): 3.15 M⊕, 1.35032 R⊕, 7.068 g/cm³ (note: UI typo "g/cm²"), g_eq 16.927 / g_pole 17.112 / g_eff,eq 16.837 m/s², P_rot 17.1 h, obliquity 25.12°.

### Orbit & irradiation
- Keplerian elements per body around parent (半长轴, 偏心率 0.1924, 周期 7046.37 d, 倾角).
- Insolation = sum of per-star top-of-atmosphere perpendicular beam fluxes at the body's instantaneous position ("不含本星自辐射、遮挡与大气衰减…S⊕=1361 W/m²"). Multi-star systems supported.
- Equilibrium/balance temperature range (~108–131 K quoted for Múspellsheimr; 3–7% S⊕).
- Per-body 温度曲线 (temperature vs days-from-epoch) + 温度表 (sampled table), dual-axis zoom, "跟随当前时刻与温度".

### Atmosphere module
- Composition editor: mole/volume %, gas list (screenshot: CO₂ 91.3% / N₂ 8.7%), presets incl. 氮氧示例 (Earth-like).
- **ExoMol CO₂** line data → Planck-mean absorption, H₂/He-broadened, 100–3400 K, 1e-5–100 bar (self-documented caveat that it's not pure-CO₂ or air broadening).
- 通用气柱幂律 (gray column power-law): reference τ₀, P₀, g₀, pressure index n (default 2), relative absorber strength S; T(P)=T_ref·(P/P_ref)^∇.
- Buttons: 根据大气估算光学厚度 / 估算大气性质.
- **NEW: "Clima" solver** → 温压曲线 (T–P profile plot, log-P axis; observed range 1.1e-5–23.24 bar, 159.6–481.5 K). Almost certainly the Kasting-lineage 1-D radiative–convective model CLIMA (Kopparapu/Ramirez/Wolf et al.).

### One-click pipeline
- 一键计算 / 一键求解进度: ①结构与尺寸 → ②大气与光学厚度 → ③温度表 (13 sampled points across the timeline; cancellable, per-point progress).

### Narrative layer (LLM)
- 介绍 tab = auto-generated markdown stat sheet + AI-generated planet image; rich markdown editor with image embed (≤32 MB), 编辑/预览/并排, 保存介绍 / 放弃修改.
- Generated text includes invented Latin-binomial species ("Cryptonucleus wittenbergii 维滕贝格寄居蟹", "Sclerocladus crystallinus 水晶椿树", "Pyrosphaera pyrus 炽球尾虾") and detailed mineralogy — clearly LLM-generated from computed stats + "一堆存货论文".
- Friend's own words: "提示词那有点多了" → substantial internal prompt library.

## 2. Known-number test vectors (RE gold)
- Múspellsheimr full stat sheet (screenshot 3) — reproduce as regression test.
- Αύρα flux split: Αστραία 3.43977 / Vesper 858.020252 / total 861.460022 W/m²; T-curve 285.61–290.87 K at +13.487 d, 表温 288.238 K.
- Star: 1.2899 M☉ → R 926,278.85 km, L 2.8055 L☉, Teff 6,474.11 K (MIST interp).
- Open question: exact T formula. 861.46 W/m² ↔ ~288–290 K is consistent with a sub-stellar/dayside-average form or a gray-greenhouse correction, NOT simple fast-rotator T⁴=S/4σ (→248 K). Resolve numerically against curve amplitude (ΔT⁴ ∝ ΔS).

## 3. Dependency map (all public)
| Module | Source |
|---|---|
| Stellar evolution | MIST v1.2 grids (Choi et al. 2016; MESA r7503), e.g. via `isochrones`/`MIST_tables` |
| Rocky MR relation | Zeng, Jacobsen & Sasselov 2019 code (lzeng.me), PREM |
| Giants | mass–temperature–composition radius estimate (likely Saturn/Jupiter-like interpolation) |
| Oblateness | Maclaurin; ToF4 (Zharkov & Trubitsyn; Hubbard et al. 2014); CMS (Hubbard 2013) |
| Opacity | ExoMol CO₂ (UCL lists), Planck means, H₂/He broadening |
| Climate | CLIMA 1-D RC model (Kasting 93 → Kopparapu 13 → Wolf+ 22, arXiv:2210.10004) |
| Narrative | LLM API + prompt templates + image model |

## 4. Unknowns to chase (recon checklist)
1. A saved **project/workspace file** (the app persists them locally; even one JSON/YAML answers schema, tech stack hints, and every default parameter). Highest-value artifact obtainable.
2. Tech stack fingerprint: installer/binary? (Electron vs Tauri vs pure local web), or a demo video / shared screen session.
3. Is the two-star system a true hierarchical binary orbit or two independent illuminators?
4. Clima: reimplemented from the paper, or wrapped Fortran? Which bands/gases?
5. LLM provider + prompt structure; image model.
6. Exact temperature formula(s) and albedo/greenhouse defaults.
7. 温度表 sampling strategy (13 points / 10 yr; uniform in time or anomaly?).

## 5. Reconstruction plan (clean-room, open-source)
- **P0 Spec** — module-by-module functional spec from screenshots + papers; freeze test vectors above.
- **P1 Core sim** — body hierarchy, Kepler propagation, time base, irradiation sum, 3D view.
- **P2 Stars** — bundled downsampled MIST grid + trilinear interp + wind bookkeeping.
- **P3 Structure** — Zeng/PREM grids, polytrope presets, layered density, gravity + oblateness.
- **P4 Atmosphere** — composition editor, ExoMol Planck-mean tables, gray column model.
- **P5 Temperature pipeline** — equilibrium T + sampled 温度表, charts.
- **P6 Clima** — port 1-D RC solver.
- **P7 Narrative** — prompt templates + markdown editor + image gen hooks.
- **P8 Validation** — reproduce friend's published numbers; compare against known solar-system bodies.

Stack proposal (vibe-codable, matches UI): Tauri or Electron + React + Three.js/R3F frontend; physics kernels in Rust or Python(PyO3); interpolation grids shipped as compact binary blobs.

---

## 6. Build progress log

### 2026-11-14 — P1 core kernel landed (`D:\test3\o2xygen`, Python 3.13 venv, 清华源)
- Package `openorbit`: `constants` (S⊕=1361 fingerprint), `kepler` (elements→state,
  Newton Kepler solver), `system` (hierarchical catalog, illumination = Σ L⋆/4πd²),
  `temperature` (equilibrium T, regimes fast_rotator/dayside/substellar + gray factor).
- 12 tests pass, incl. recon-by-numbers vectors:
  - **Temperature regime identified**: his Αύρα pair S=861.460022 W/m² ↔ 288.238 K
    implies T⁴·4σ/S = **1.817** — impossible for bare fast-rotator (needs (1−A)≤1).
    Consistent with **dayside-average regime + albedo ≈ 0.09**, or fast-rotator +
    gray greenhouse gh≈0.82. Implemented as per-body parameter pending more data.
  - **R⊕ convention**: his 7.068 g/cm³ matches mean-radius 6371 km better than
    equatorial (7.054 vs 7.038; residual ~0.2% ≈ display rounding of M,R).
  - Gravity: monopole+centrifugal reproduces his 16.927/17.112 pair to <0.5%;
    implied flattening ~0.46% for P_rot=17.1 h.
- `examples/demo_temperature_table.py`: end-to-end 一键求解 stage ③
  (13 samples / 10 yr) producing 距初始时刻/总辐照/温度 table.
- **Next**: P2 star module — download MIST v1.2 grids (to `D:\test3\data`),
  build downsampling + trilinear (Mini, [Fe/H], age) interpolation, validate
  against his 1.2899 M☉ → 926,278.85 km / 2.8055 L☉ / 6474.11 K vector.

### 2026-11-14 (round 2) — P2 scaffolding done, P3 landed; big recon findings

**Philosophy update (per user): his build is a *reference*, not a bit-exact RE
target. Use newest obtainable resource versions; screenshot numbers are
calibration references with tolerance, not acceptance tests. Goal reworded.**

**MIST download saga:**
- waps.cfa.harvard.edu 302→mist.science; per-connection ~5.4 KB/s from this
  network; concurrent connections stall to 0 (per-IP throttle, not per-conn).
  Parallel-range downloads useless here. IPv6 unavailable. Zenodo (isochrones
  test data, incl. full MIST h5) equally slow. MIST online interpolator now
  v2.5-only (his is v1.2) and returns empty `<pre>` for scripted POSTs.
- Decision: v1.2 tarball (107 MB, smallest) single-stream resumable curl
  trickling in background (job pwsh-786fae7924904bf9, ~5.5 h ETA); v2.5 EEPS
  exist on mist.science (tarballs_v2.5/eeps/, ~186 MB each) if we ever want
  them. mist.science also has basic_isos/full_isos txz per version.
- `stars.py`: MIST .track.eep parser + per-metallicity scattered linear
  interpolator over (M_initial, log10 age) + linear [Fe/H] blend + `solve_age`
  inverse (recover age from L,Teff; radius then validates). `star_evolution.py`:
  his 参数估算 bookkeeping (累计损失=初始−当前; mdot from track).
- Star tests written; skipped until data lands (3 skip markers).

**P3 structure module (`mass_radius.py`, `structure.py`) + RECON FINDING:**
- His "PREM: 1–8 M⊕, 核心分数 0–0.4" option = **Zeng, Sasselov & Jacobsen 2016
  (ApJ 819:127) PREM fit: R/R⊕ = (1.07 − 0.21·CMF)·(M/M⊕)^(1/3.7)** —
  the UI ranges match the paper exactly.
- **His Múspellsheimr radius reproduced to 0.008%**: R(3.15 M⊕, CMF=0.38) =
  1.35021 R⊕ vs his displayed 1.35032. The "0.38" visible in his 结构-tab
  screenshot is the core mass fraction; the displayed radius is PREM-model
  output. (His UI's other field "25.12" — likely age Gyr or an orbital
  element; still unidentified.)
- "高压 EOS 表 · Zeng(2019)" proper (0.01–100 M⊕, CMF 0–1) approximated by
  the same CMF-interpolation form with M^1/4 exponent pending real EOS tables.
- Flattening: slow-rotation f=5ω²R³/4GM + exact Maclaurin q(e) inversion
  (clamped at bifurcation q<0.2247). ToF4/CMS raise NotImplementedError.
- Polytrope n=1 analytic profile + his 层/半径范围/相对密度 layered table.
- 23 tests pass, 3 MIST tests skip pending download.

**Next:** star tests activate when download completes; then P4 atmosphere
(ExoMol Planck-mean CO₂ + gray column model) — needs ExoMol data (also a
download, likely slow; may use published Planck-mean tables instead of raw
line lists). Also queue p0.25 tarball afterwards.

### 2026-11-14 (round 3) — MIST data in hand, star module validated, P2 done

- User hand-downloaded the tarball (network too slow for us). Extracted with
  Python tarfile (Windows bsdtar lacks xz) to `D:\test3\data\mist\feh_p0.00/`
  (196 tracks, 0.10-300 M☉, MESA r7503 confirmed in file headers).
- Parser bug found: `# EEPs:` line startswith("EEP") hijacked header
  detection. Fixed to match the star_mass/log_R column row.
- **Interpolation scheme bug found and fixed**: 2-D Delaunay scatter interp
  over (M_init, log age) corrupts R/L badly (Sun at 4.56 Gyr came out
  R=1.094 R☉ instead of 1.023) because triangles span wildly different ages
  between adjacent tracks. Replaced with proper bilinear track-space scheme:
  1-D in log(age) per track, then 1-D across mass. Exact at track nodes.
- **Recon finding — his star's [Fe/H] is bracketed, not zero**: his modal
  lists nodes "1.28/1.3 M☉ × [Fe/H] 0 / 0.25" → his solution sits between the
  two metallicities. Confirmed numerically: at [Fe/H]=0 his (L, Teff, R)
  triplet lands at M≈1.24, age≈1.85 Gyr (R within 1.6%); at M=1.29 the
  solar-metallicity grid runs 2.3% hot in Teff — exactly the direction a
  metal-rich correction fixes. Awaiting the feh_p0.25 tarball from the user
  to close the bracket and validate cross-metallicity blending.
- Full suite: **26 passed**. Star evolution module (estimate + wind
  bookkeeping) live; temperature/flux/structure calibration vectors all green.

### 2026-11-14 (round 5) — WebUI v1: FastAPI + Three.js, headless-Chrome polished

User directive: no Streamlit; build it properly, self-iterate with headless
Chrome. Stack chosen: **FastAPI backend (`server/app.py`) + hand-written
frontend (web/: vanilla ES modules + Three.js 0.187 + uPlot)**, no build step.

- API: /api/system, /api/body (irradiation+period+Teq), /structure (PREM/Zeng),
  /temperature (13-pt table + dense curve), /star/status|ensure|estimate (lazy
  MIST grid in bg thread, 392 tracks), /pipeline (一键计算 3 stages), /add_body.
- UI mirrors his layout: topbar+workspace crumb, left catalog tree (kind colors,
  search, add-body modal), center Three.js (Teff-colored emissive stars + glow,
  hierarchical orbit lines, rings, click-select raycast, floating labels with
  已选择 tag, axis gizmo, 3D/2D toggle, focus/grid buttons), right panel tabs
  基础参数(含MIST参数估算)/结构(CMF滑杆+圈层SVG+分层表, radius matches his
  1.35032 to 0.008%)/大气(成分编辑+预设+气柱幂律)/温度表(uPlot curve with
  current-time dashed marker + follow mode + 13-row table)/介绍, bottom
  timeline (play/pause, 天/秒 speed, UTC clock from his epoch 2026-11-18
  19:00:25Z, span 1y/10y, spacebar shortcut).
- NO procedural starfield — user has a real all-sky renderer (Solar-System-
  relative) to be integrated later. Scene background stays pure dark.
- Bugs found & fixed via screenshot loop: modal `[hidden]` vs CSS display;
  `# EEPs:`-style header mis-detection (earlier); three.module.js 0.187 needs
  three.core.js; catalog render() never invoked; Vesper preset radius ×1000
  (830 R☉ sphere swallowed the camera — the "cream screen"); glow sprite scale;
  scene.update never called at t=0; star self-irradiation d=0 → inf JSON 500
  (now skipped per his 不含本星自辐射 note); async race select-star while
  temperature request in flight (boot crash); gizmo 2D context guard.
- Iteration harness: `scripts/shot.py` (Playwright driving installed Chrome,
  console/pageerror capture, click-wait helper; needs TMP=D:\test3\tmp under
  sandbox) + `scripts/dbg_scene.py`.
- Server: `python server/app.py` → http://127.0.0.1:8501 (running as bg job).
- Screenshot gallery in `openorbit/shots/`.

**Known gaps / next:** real-sky renderer integration (user's tool); P4 proper
ExoMol Planck means; Clima solver; LLM narrative tab (介绍 is a static
placeholder); body editing (orbital elements UI); persistence (project files);
Maclaurin/ToF4/CMS flattening UI; span-aware temperature recompute.

### 2026-11-14 (round 4) — [Fe/H] bracket CLOSED, P2 fully validated

- feh_p0.25 grid extracted (196 tracks). Scan at fixed 1.29 M☉:
  **best fit at [Fe/H]≈+0.10, age≈1.26 Gyr, log-residual 6e-4**
  (R=926,279 km reproduced to ~1‰; Teff/L to <0.5%). His star fully
  triangulated: 1.29 M☉, [Fe/H]~+0.10, ~1.26 Gyr. The four nodes in his
  modal are exactly the bracket corners of his own solution.
- Wind loss convention decoded: MESA `star_mdot` is dM/dt, NEGATIVE on the
  MS (raw ≈ -9.5e-14 at the solution); he quotes |mdot| (1.146e-13). Fixed
  our pipeline to interpolate |mdot|; mass_now matches his 1.28987 to 4e-5.
- **P2 complete: 26/26 tests green** (mass-loss, Sun-sanity, bracket solve).

**Next:** P4 atmosphere module (composition editor data model + ExoMol-class
Planck-mean opacity + 气柱幂律 gray column: τ₀/P₀/g₀/n/S, T(P) power law).
