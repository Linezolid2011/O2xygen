# O₂xygen

世界观构造工作台 —— 面向小说设定的实时恒星系统物理沙盒。
版本见 `docs/version.txt`(唯一真源,ADR-0003)。

与常见的"开普勒推演"类工具不同,O₂xygen 的轨道是**实时 N-body 积分**
算出来的(ADR-0015):恒星会按 MIST 演化轨迹老化掉质量,行星轨道随之
真实膨胀;小天体的瞬时偏心率会因恒星摄动产生数十年量级的 Kozai 型
振荡。每条天体身后的"烟雾"就是它刚刚走过的真实积分路径。

**世界即存档(ADR-0016)**:每个世界是一个 `[worldname].oxygen` 文件——
扁平天体(仅恒星/非恒星)+ 笛卡尔初始条件 + 原点天体 + 时间原点。
没有层级,没有"卫星"身份;轨道关系从动力学中涌现,以相对原点天体的
瞬时根数形式被观测。时间以"自原点经过的秒数"计量,显示为
yyyy-mm-dd From origin。

## 功能

- **多世界**:`data/worlds/*.oxygen` 存档,UI 头部下拉切换、另存为,
  首启自动物化内置白玫瑰。
- **实时积分**:rebound IAS15(默认)/ WHFast,检查点存档,时间轴可双向
  拖动(回扫 1000 年后 epoch 瞬时根数逐位恢复)。
- **恒星演化**:MIST v1.2(MESA r7503)轨道空间双线性插值,风损质量
  绝热耦合进积分;恒星在时间轴上真实老去(半径/温度/光度同步演化)。
- **行星结构**:Zeng+2016 PREM 质量-半径拟合、CMF 推断、Maclaurin 扁率、
  n=1 多方球分层。
- **温度管线**:三种平衡温度体制 + 反照率,温度表沿积分路径采样。
- **WebUI**:FastAPI + Three.js + uPlot,无构建步骤;烟雾轨迹、相对
  原点的瞬时根数面板、一键计算(结构 → 大气 → 温度表)、可输入的
  速度/跨度时间轴、以轨道根数为输入的天体放置工具。

## 运行

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e . rebound fastapi uvicorn
.venv/Scripts/python server/app.py        # http://127.0.0.1:8501
```

测试(55 项,含对公开参照数值的标定):

```bash
.venv/Scripts/python -m pytest
```

## 数据

**MIST 恒星网格随仓库分发**(0.1.1 起,用户裁定:下载之苦不转嫁):分卷
压缩包位于 `.data/MIST-compact/`(每卷 ≤64 MB,避开 GitHub 单文件
100 MB 限制;原始包约 107 MB × 2)。首次启动时程序自动拼接分卷并解压到
`.data/MIST/`(纯 Python tarfile,解压产物与解析缓存被 gitignore,
压缩卷保留不删)。世界存档(`*.oxygen`)仍是用户内容,默认存于
`D:\test3\data\worlds`(可在 `server/app.py::WORLDS_DIR` 调整),不进仓库。

## 仓库结构

```
src/o2xygen/   物理内核(纯 Python,可独立使用)
server/        FastAPI 服务(Workspace = 活积分器 + 恒星演化耦合)
web/           原生 ES modules 前端(无构建)
tests/         55 项标定与积分验收测试
docs/          CHANGELOG / ADR 档案(decisions.md)/ 重建笔记(RECON.md)
shots/         无头 Chrome 截图存档(UI 迭代留痕)
scripts/       shot.py(截图)/ dbg_scene.py(场景调试)
```

## 许可与出处

干净室重建项目:以好友的闭源应用「ORBIT」的公开输出(截图与显示数值)
为参照系,全部代码与资源独立实现(ADR-0001)。MIST 数据属 MIST 团队,
遵循其发布条款。
