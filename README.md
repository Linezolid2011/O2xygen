# O₂xygen

世界观构造工作台 —— 面向小说设定的实时恒星系统物理沙盒。
版本见 `docs/version.txt`(唯一真源,ADR-0003)。

与常见的"开普勒推演"类工具不同,O₂xygen 的轨道是**实时 N-body 积分**
算出来的(ADR-0015):恒星会按 MIST 演化轨迹老化掉质量,行星轨道随之
真实膨胀;卫星的瞬时偏心率会因恒星摄动产生数十年量级的 Kozai 型振荡。
每条天体身后的"烟雾"就是它刚刚走过的真实积分路径。

## 功能

- **层级系统**:恒星 / 行星 / 卫星树形目录,双星、行星环、多星辐照求和
  (S⊕ = 1361 W/m²)。
- **实时积分**:rebound IAS15(默认)/ WHFast,检查点存档,时间轴可双向
  拖动(回扫 1000 年后 epoch 瞬时根数逐位恢复)。
- **恒星演化**:MIST v1.2(MESA r7503)轨道空间双线性插值,风损质量
  绝热耦合进积分;恒星在时间轴上真实老去。
- **行星结构**:Zeng+2016 PREM 质量-半径拟合、CMF 推断、Maclaurin 扁率、
  n=1 多方球分层。
- **温度管线**:三种平衡温度体制 + 反照率,温度表沿积分路径采样。
- **WebUI**:FastAPI + Three.js + uPlot,无构建步骤;烟雾轨迹、瞬时根数
  面板、一键计算(结构 → 大气 → 温度表)、可输入的速度/跨度时间轴。

## 运行

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e . rebound fastapi uvicorn
.venv/Scripts/python server/app.py        # http://127.0.0.1:8501
```

测试(38 项,含对公开参照数值的标定):

```bash
.venv/Scripts/python -m pytest
```

## 数据

MIST 网格**不随仓库分发**(约 100 MB/金属度)。放置方式:解压
`MIST_v1.2_feh_<p0.00|p0.25>_afe_p0.0_vvcrit0.4_EEPS.txz` 到
`D:\test3\data\mist\feh_p0.00` 与 `feh_p0.25`(路径可在
`server/app.py::_load_grid` 调整);首次加载后自动生成二进制缓存
(解析约 22 s → 缓存重载约 1.5 s)。数据缺失时恒星演化自动停用,其余功能不受影响。

## 仓库结构

```
src/o2xygen/   物理内核(纯 Python,可独立使用)
server/        FastAPI 服务(Workspace = 活积分器 + 恒星演化耦合)
web/           原生 ES modules 前端(无构建)
tests/         38 项标定与积分验收测试
docs/          CHANGELOG / ADR 档案(decisions.md)/ 重建笔记(RECON.md)
shots/         无头 Chrome 截图存档(UI 迭代留痕)
scripts/       shot.py(截图)/ dbg_scene.py(场景调试)
```

## 许可与出处

干净室重建项目:以好友的闭源应用「ORBIT」的公开输出(截图与显示数值)
为参照系,全部代码与资源独立实现(ADR-0001)。MIST 数据属 MIST 团队,
遵循其发布条款。
