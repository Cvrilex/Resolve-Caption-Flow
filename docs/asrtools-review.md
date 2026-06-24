# AsrTools 接口与字幕优化审阅

审阅来源：`wheel/AsrTools`，仓库为 `bozoyan/AsrTools`。

## 在线 ASR 接口

### 必剪 B 接口

- AsrTools 中的 `bk_asr/BcutASR.py` 与当前项目已有 `BcutASR` 能力重复。
- 本次不重复搬运，继续使用当前项目内的必剪 B 接口作为默认在线 ASR。

### 剪映 J 接口

- AsrTools 中的 `bk_asr/JianYingASR.py` 已有对应实现，当前项目也已经有 `JianYingASR`。
- 该接口依赖签名服务。
- 默认第三方签名服务 `https://asrtools-update.bkfeng.top/sign` 实测返回 HTTP 500。
- 当前 WebUI 中保留该接口状态，但只有配置 `JIANYING_SIGN_SERVICE_URL` 后才允许选择。

### 快手 K 接口

- 已搬运到 `mvp_pipeline/tool/online_asr.py`，类名为 `KuaiShouASR`。
- 接口地址：`https://ai.kuaishou.com/api/effects/subtitle_generate`。
- 使用 AsrTools 测试音频实测返回：
  - HTTP 200
  - 业务码 `501`
  - 信息：`效果subtitle_generate禁用`
- 当前 WebUI 展示为已搬运但不可用，不能作为任务启动接口。

## 当前产品 ASR 策略

- WebUI 只展示在线 ASR 下拉菜单。
- 当前默认可用接口为必剪 B 接口。
- 剪映 J 接口需要可用签名服务。
- 快手 K 接口已搬运但接口侧禁用。
- 本地 ASR 功能暂时下线，主流程不再保留 `local` 引擎分支，WebUI 不再展示模型下载、环境准备、本地模型状态或本地 ASR 选项。

## AsrTools 的 SRT 优化流程

AsrTools 的字幕优化流程主要集中在 `main.py` 和 `split_by_llm.py`：

1. 读取 SRT 为结构化 ASRData。
2. 清理纯标点字幕。
3. 将字幕文本拼成长文本。
4. 按字符规模和时间间隔切成多个 chunk。
5. 将每个 chunk 交给 LLM 做语义断句，要求用 `<br>` 标记断句。
6. 用 `difflib.SequenceMatcher` 将 LLM 断句结果重新映射回原始 ASR 字幕片段。
7. 合并匹配到的原始字幕片段，保留原 ASR 时间轴。
8. 对过长合并字幕，再按内部最大时间间隔递归切开。

## 值得借鉴的点

- 长文本先按自然时间间隔切块，再发给 LLM，避免一次性把整条长视频字幕丢给模型。
- LLM 只负责给出语义边界，最终时间轴仍从原始 ASR 片段映射回来。
- 用相似度匹配把 LLM 结果对齐到原始字幕，比直接信任 LLM 输出时间更稳。
- 对过长合并字幕使用原始片段之间的最大时间间隔再次拆分，适合处理 ASR 切点附近的断句。
- LLM 断句结果可以做缓存，便于失败重试和重复调参。

## 不建议直接照搬的点

- 英文字幕会被转小写，医疗缩写和药名可能被破坏。
- 相似度阈值较宽，长医疗字幕需要更严格的内容守恒校验。
- 默认目标长度偏短，医疗课程字幕不宜机械压到过短。
- 全文语义断句仍然可能很慢，必须配合当前项目已有的阶段进度、失败恢复和分批策略。

## 后续融合方向

- 将当前“短字幕修正、长字幕修正、ASR 切点修正”统一为窗口式字幕优化任务。
- 每个窗口交给 LLM 只输出断句边界或修订后的短句，不让 LLM 改写无关内容。
- 断句完成后映射回原始 ASR cue 时间轴，并进行内容守恒校验。
- 对 ASR 分段切点，固定取切点当前句、前一句、后一句作为小窗口，让 LLM 重做语义断句。
- 给每个窗口记录进度、输入哈希、输出文件和失败原因，方便从上次结果继续。
