# 📚 EPUB Bilingual Translator

基于大语言模型（LLM）的 EPUB 双语对照生成工具。将英文（或其他语言）EPUB 电子书转换为「原文段落 + 中文译文段落」逐段对照格式的 EPUB，方便语言学习与对照阅读。

本项目通过 OpenRouter API 接入多种主流大模型（如 Gemini, Claude, GPT-4 等），支持批量翻译、自动重试与样式美化。

## ✨ 主要特性

- **逐段对照**：严格保持原文段落结构，在每段原文后插入对应的中文译文。
- **模型自由**：基于 OpenRouter，可自由切换底层大模型（推荐 Gemini 2.0 Flash 以获得高性价比）。
- **批量处理**：支持分段批量翻译，自动处理速率限制（Rate Limit）。
- **样式美化**：自动注入 CSS 样式，译文采用不同颜色与边框区分，阅读体验更佳。
- **容错机制**：支持网络错误自动重试，提供 `--resume` 选项在遇到错误时跳过继续处理。
- **元数据更新**：自动修改书籍标题（添加“中英对照”后缀）与语言标识。

## 📦 安装依赖

确保已安装 Python 3.6+，然后安装所需库：

```bash
pip install ebooklib beautifulsoup4 requests
```

## 🚀 快速开始

### 1. 获取 API Key

本项目使用 OpenRouter API。请访问 [OpenRouter](https://openrouter.ai/) 获取 API Key。

你可以选择以下方式之一配置 Key：

- **环境变量**（推荐）：
  ```bash
  export OPENROUTER_API_KEY="your_api_key_here"
  ```
- **命令行参数**：
  在使用命令时通过 `--api-key` 指定。

### 2. 基本用法

转换单个 EPUB 文件（输出文件默认名为 `原文件名_bilingual.epub`）：

```bash
python epub_bilingual.py input.epub
```

### 3. 高级用法

指定输出文件、模型及翻译参数：

```bash
python epub_bilingual.py input.epub output_bilingual.epub \
    --model google/gemini-2.0-flash-001 \
    --batch 5 \
    --delay 1.0 \
    --from-lang English
```

## ⚙️ 命令行选项

| 参数          | 说明                               | 默认值                        |
| :------------ | :--------------------------------- | :---------------------------- |
| `input`       | 输入 EPUB 文件路径                 | **必填**                      |
| `output`      | 输出 EPUB 文件路径                 | `输入文件名_bilingual.epub`   |
| `--api-key`   | OpenRouter API Key                 | 环境变量 `OPENROUTER_API_KEY` |
| `--model`     | OpenRouter 模型 ID                 | `google/gemini-2.0-flash-001` |
| `--batch`     | 每次请求翻译的段落数               | `5`                           |
| `--delay`     | 每批请求之间的间隔秒数             | `1.0`                         |
| `--from-lang` | 原文语言描述（用于提示词）         | `English`                     |
| `--resume`    | 遇到错误时跳过该段落，继续处理后续 | `False`                       |

## 🤖 推荐模型

通过 OpenRouter 可以使用多种模型，以下是代码中推荐的常用模型：

| 模型 ID                             | 特点             | 适用场景                   |
| :---------------------------------- | :--------------- | :------------------------- |
| `google/gemini-2.0-flash-001`       | 快速、低价       | **默认推荐**，适合长篇书籍 |
| `anthropic/claude-sonnet-4-5`       | 高质量、理解力强 | 对翻译质量要求极高         |
| `openai/gpt-4o-mini`                | 均衡             | 性能与成本平衡             |
| `deepseek/deepseek-chat`            | 中文友好         | 适合中文语境优化           |
| `meta-llama/llama-3.3-70b-instruct` | 开源             | 性价比选择                 |

## 🎨 输出效果

生成的 EPUB 会在每个原文段落后插入译文段落，并应用以下样式：

- **译文颜色**：深蓝色 (`#1a5276`)
- **字体**：衬线体 (Noto Serif SC / 宋体)
- **标识**：左侧带有浅蓝色边框，便于视觉区分
- **元数据**：书籍标题自动追加 `（中英对照）`

**示例 HTML 结构：**

```html
<p lang="en">Original English text...</p>
<p lang="zh-Hans" class="zh-translation">对应的中文译文...</p>
```

## ⚠️ 注意事项

1. **翻译一致性**：由于是分批翻译，极少数情况下可能出现术语前后不一致，建议翻译完成后通读检查。
2. **复杂排版**：脚本主要处理文本段落（`p`, `h1-h6`, `li` 等标签）。复杂的表格、脚注或图片内的文字可能无法被提取翻译。
3. **网络环境**：确保运行环境可以访问 `openrouter.ai`，否则会遇到连接错误。

## 🛠️ 工作原理

1. **解析**：使用 `ebooklib` 读取 EPUB，通过 `BeautifulSoup` 提取所有章节的文本块。
2. **翻译**：将文本块分批发送至 OpenRouter API，利用 System Prompt 约束模型输出严格对应的段落。
3. **注入**：将返回的译文创建为新的 HTML 标签，插入到原文标签之后，并添加专用 CSS 类。
4. **保存**：更新书籍元数据，写入新的 EPUB 文件。

## 📄 许可证

本项目代码仅供学习与个人使用。请遵守相关版权法规，仅对拥有合法版权的书籍进行转换。

---

_如有问题或建议，欢迎提交 Issue。_
