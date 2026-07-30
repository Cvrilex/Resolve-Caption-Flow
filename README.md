# 达芬奇字幕助手

面向专业长视频的本地字幕校对和成片交付工具：从在线音频 ASR 转写开始，经过术语校对、字幕清理，再按 DaVinci Resolve 模板输出最终成片。

## 项目特点

- **面向长视频字幕校对**：支持在线视频 ASR、术语提取与审核、字幕断句和清理，并保留中间 SRT 以便继续处理。
- **第三方 ASR 服务配额**：当前使用的第三方 ASR 服务在每个网络环境下通常有每日约 **5 小时**的使用配额。这是服务方的配额限制，不是本项目的设计限制；请按服务允许的方式分批处理，或向服务方申请更高配额。
- **大语言模型校对**：可结合参考 PDF、术语表和模型配置进行术语审核与字幕文本校对；医学课程资料是可选示例之一。
- **本地知识库**：可在 WebUI 中维护项目或领域术语知识库，供后续任务复用；当前内置规则以医疗术语为示例，并非唯一适用领域。
- **Resolve 最终交付**：将最终 SRT 导入带有字幕样式的 DaVinci Resolve 模板工程，并调用本机渲染预设输出成片。
- **本地数据边界**：视频、SRT、日志、API Key、模型配置和本地知识库均不上传 GitHub，也不会被打进用户发布包。

## 界面预览

![环境检查页面](docs/images/environment-check.png)

![配置填写页面](docs/images/configuration.png)

![知识库页面](docs/images/knowledge-base.png)

![新建任务页面](docs/images/new-task.png)

![历史任务页面](docs/images/task-history.png)

## 使用手册

### 1. 启动项目

macOS 上双击：

```text
start_web.command
```

首次双击若被 macOS 拦截，请进入“系统设置 → 隐私与安全性”，在“安全性”区域点击“仍要打开”，再使用密码或 Touch ID 验证后打开。

Windows 支持仍在开发中，将在完成实际环境测试后补充。

启动脚本会准备本地 Python 环境并启动 WebUI。也可以手动运行：

```bash
python3 app/web_server.py --host 127.0.0.1 --port 8742
```

然后访问：

```text
http://127.0.0.1:8742/
```

### 2. 安装或检查 FFmpeg

项目需要 `ffmpeg` 和 `ffprobe` 用于视频探测、音频提取和渲染校验。启动时会自动检查；也可在终端执行：

```bash
ffmpeg -version
ffprobe -version
```

若未安装且使用 Homebrew，可执行：

```bash
brew install ffmpeg
```

### 3. 完成首次配置

在“配置填写”页面填写模型名称、API 地址和 API Key，并按需要设置字幕清理选项与“达芬奇输出预设名”。

这些本地配置保存在 `data/work/web_config.json`；其中的 API Key、模型配置和本地知识库均被 Git 忽略，也会被用户发布包排除。可参考不含密钥的 [`resources/web_config.example.json`](resources/web_config.example.json)。

### 4. 准备知识库

在“知识库”页面导入或维护术语资料。知识库只保存在本机，位于 `data/work/knowledge_base/`，不会上传到 GitHub，也不会随用户发布包分发。

### 5. 创建任务

在“新建任务”页面选择视频；如有参考 PDF，也可一并上传以提取和审核术语。选择在线 ASR 接口后，可按需要启用长视频分段处理并设置字幕清理规则。

选择“只生成 SRT”可先检查最终字幕；选择“SRT + 成片”会继续进入 DaVinci Resolve 导入和渲染阶段。第三方 ASR 的每日配额不足时，请按服务允许的方式分批处理或申请更高配额。

### 6. 生成成片前的 DaVinci Resolve 准备

以下准备仅在选择“SRT + 成片”时需要；仅生成 SRT 不需要安装或打开 DaVinci Resolve。

1. 在本机安装并打开 DaVinci Resolve，并在偏好设置中启用 `External scripting: Local`（本地外部脚本权限）。
2. 在 DaVinci Resolve 中制作包含目标字幕样式的模板工程，保存为 `sub.drp`，并放入项目的 `resources/` 目录。程序固定读取 `resources/sub.drp`。
3. 在本机 DaVinci Resolve 中保存需要使用的渲染预设，并在项目界面的“达芬奇输出预设名”中填写该预设名称。最终成片导出会调用该预设。

### 7. 查看最终输出

- 最终 SRT、处理中间文件和报告保存在 `data/work/`。
- 任务日志保存在 `data/logs/`。
- 成片输出保存在 `data/output/`；文件名默认使用原视频文件名主干加“字幕版”。
- 可在“历史任务”页面查看任务状态，并继续使用已有字幕结果。

## 项目目录

```text
app/            WebUI 和本地服务
pipeline/       字幕生产、术语和校对流程
integrations/   在线 ASR 等外部服务适配
caption_core/   可复用领域模型与基础能力
resources/      Resolve 模板、字幕样式说明和示例资源
data/           本地输入、工作文件、输出、日志和知识库
docs/           流程与技术文档
tests/          自动化测试
scripts/        辅助脚本
```

## 开源协议

本项目采用 [MIT License](LICENSE)。
