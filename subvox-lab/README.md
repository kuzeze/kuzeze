# Subvox 实验台：怎么跑、长什么样、第一轮怎么做

Subvox 是 Joe Blau 用一个 prompt 生成的 macOS 原型，用 AirPods 的头部运动数据做「默读识别」。
这个目录放两样东西：

- 本文：装机步骤、三个界面的说明、第一轮实验协议、常见坑。
- `analyze.py`：把 Subvox 的解码流程完整移植到 NumPy，额外加了特征消融和跨佩戴评估。
  只依赖 NumPy，`python3 analyze.py selftest` 可自检。

## 你需要什么

| 项目 | 要求 |
| --- | --- |
| Mac | macOS 14 Sonoma 或更新。代码用了 `@Observable`，老系统编译不过 |
| Xcode | 15 或更新，App Store 免费 |
| 工具 | Homebrew，`brew install xcodegen` |
| AirPods | 支持头部追踪的型号：AirPods Pro 1/2/3、AirPods 3/4、AirPods Max。AirPods 1/2 没有 IMU，不行 |
| 签名 | 一个 Apple ID，免费账号即可；或者用下面的本地签名方式 |
| 权限 | 首次运行会要「运动与健身」；开麦克风时再要「麦克风」和「语音识别」 |

## 安装与构建

```sh
brew install xcodegen
git clone https://github.com/joeblau/subvox.git
cd subvox
xcodegen generate
open Subvox.xcodeproj
```

Xcode 打开后必须改一处：选中 `Subvox` target，进 Signing & Capabilities，把 Team 换成你自己的。
原因是 `project.yml` 里写死了 Joe 的 Team ID `K78G42H4U2`，用他的 team 签名会失败。改完 ⌘R。

不想开 Xcode 界面，也可以改成本地 ad-hoc 签名后用命令行构建：

```sh
sed -i '' 's/DEVELOPMENT_TEAM: K78G42H4U2/DEVELOPMENT_TEAM: ""/; s/CODE_SIGN_STYLE: Automatic/CODE_SIGN_STYLE: Manual/; s/CODE_SIGN_IDENTITY: Apple Development/CODE_SIGN_IDENTITY: "-"/' project.yml
xcodegen generate
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build build
open build/Build/Products/Debug/Subvox.app
```

这条命令行路径我没有 Mac 环境验证过。报签名错误就回到上面的 Xcode 方式选 Team。

可选：麦克风参考转写默认是英文，做中文实验时改成中文。运动解码本身和语言无关，这一步只影响对照用的语音识别。

```sh
sed -i '' 's/en-US/zh-CN/' Subvox/Audio/MicTranscriber.swift
```

连接 AirPods：戴上，蓝牙连到 Mac，在菜单栏声音里把输出设成 AirPods。
工具栏右侧的状态徽章从 Waiting for AirPods 变成 Streaming，并显示采样率，就通了。
一直 Waiting 的话，播放一段音频再试，或者摘下重戴。

## 工作台长什么样

窗口左边是侧栏，三个页面。工具栏右侧有运动状态徽章和 Record Session 按钮。默认打开 Train。

### Sensors

- 左上「AirPods motion」卡：状态、实测采样率、样本数、重力向量。
- 右上「Microphone」卡：输入设备下拉、Start Mic、电平表。提示选 AirPods 做麦克风会切到 HFP 低音质。
- 中间「Live motion (last 8 s)」三条实时曲线：用户加速度、角速度、姿态。
- 底部「Experiment recording」：条件三选一（Normal speech / Whisper / Silent mouthing），prompt 文本框，Record/Stop，Show Last Session。
  录制期间按住空格标记每个发音段，标签就是 prompt 文本框里的内容。

录制输出在 `~/Documents/Subvox/Sessions/<时间戳>-<条件>/`，四个文件：

| 文件 | 内容 |
| --- | --- |
| motion.csv | 每个运动样本，17 列：t、加速度 3、角速度 3、pitch/roll/yaw、重力 3、四元数 4 |
| audio.caf | 麦克风音频，开了麦才有 |
| events.csv | 每个按住空格的段：start、end、kind、label、confidence |
| meta.json | 条件、prompt、音频起始偏移 |

### Train

- 左栏词表：默认 yes / no / next / back / stop，每个词后面是已录条数。底部可以加词，支持中文。右键删除词和它的全部模板。
- 右侧上方一个按住采集的大按钮，鼠标按住或者按住空格都行。提示写的是每个词录 8 到 10 条，头保持不动。
- 「Takes」网格：每条模板一个小瓦片，显示角速度曲线、时长、帧数和删除按钮。
- 「Is there signal?」面板：三个特征复选框 Acceleration / Rotation / Attitude，一个 Evaluate 按钮。
  结果四个数：Accuracy、Chance、Shuffled 95th pct、p-value，然后一句判定和混淆矩阵。至少要 2 个词各 2 条。
- 模板持久化在 `~/Library/Application Support/Subvox/dataset.json`，`analyze.py` 可以直接读。

### Decode

- 分段方式二选一：Hold Space 或 Auto-detect。自动模式先点 Calibrate noise floor 静止 2 秒，再调 Trigger × floor。
- 两个滑块：Max distance 默认 1.5，Min confidence 默认 0.05。被拒绝的词在转写里显示为淡色。
- 「Decoded from motion」大字转写。
- 「Last segment」：每个词的 DTW 距离条、置信度、这一段的角速度曲线。
- 「Microphone transcript (reference)」：麦克风语音识别对照。

### 写死在代码里的参数

| 参数 | 值 |
| --- | --- |
| 采样率 | 系统给多少用多少，AirPods 约 25 Hz |
| 段的下限 | 0.15 秒且至少 4 帧 |
| 段的上限 | 手动 6 秒，自动 3 秒 |
| 预滚 / 尾部延长 | 0.3 秒 / 0.35 秒 |
| 能量 | 角速度模长 + 5 倍加速度模长 |
| 裁剪 | 20 分位做地板，阈值 = 地板 + 0.15 ×（峰值 − 地板），两端各留 2 帧 |
| 特征 | 加速度 3 + 角速度 3 + 相对段首的姿态差 3，各通道去均值，按训练集标准差缩放 |
| DTW | Sakoe-Chiba 带宽 25%，距离除以 (n + m) |
| 评估 | 留一法 1-NN，500 次标签打乱 |

## 第一轮实验，约 30 分钟

目标只有一个：拿到你自己的 p 值。

1. Train 页，默认 5 个英文词，每词 10 条，点 Evaluate，记下四个数。
2. 三个复选框逐个单独勾选再 Evaluate，看信号在哪个通道。
3. 加 5 个中文词，每词 10 条，再 Evaluate。选词原则是音节数和开口度都不同，例如：好、不要、下一个、回去、停。
   避开只差声母或声调的词对，比如开灯和关灯。
4. 跑完整消融表：

   ```sh
   python3 analyze.py dataset "~/Library/Application Support/Subvox/dataset.json" --json day1.json
   ```

5. 跨佩戴测试。到 Sensors 页，条件选 Silent mouthing，prompt 填一个词，点 Record，
   点一下窗口空白处让文本框失焦，按住空格默读 10 次，再回到文本框换下一个词，5 个词录完点 Stop。
   摘下 AirPods 重新戴好，整套再录一个 session。然后：

   ```sh
   python3 analyze.py cross ~/Documents/Subvox/Sessions/<第一次> ~/Documents/Subvox/Sessions/<第二次> --json cross.json
   ```

   这个数字才是 Joe 没有回答的问题：换一次佩戴之后模板还认不认。

把 `day1.json`、`cross.json`，加上 AirPods 型号、macOS 版本、工具栏显示的采样率发回来，就能决定下一步往哪走。

## 常见坑

- 文本框有焦点时空格不触发采集，先点一下窗口空白处。
- Train 和 Decode 不需要麦克风。开麦只是为了对照转写和录音。
- 选 AirPods 做麦克风会掉到 HFP，影响播放音质不影响运动数据。
- 重新佩戴会改变信号，这是已知的。所以第 5 步很重要。
- 窗口小的时候 Train 页会溢出，仓库里有个开放的 PR 在修，先把窗口拉大。
- 拒绝过运动权限的话，去系统设置、隐私与安全性、运动与健身里重新打开。
