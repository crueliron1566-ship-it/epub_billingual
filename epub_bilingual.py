#!/usr/bin/env python3
"""
epub_bilingual.py
将 EPUB 或 PDF（文字版 / 扫描版）转为「原文段落 + 中文译文段落」逐段对照 EPUB。
使用 OpenRouter API，可自由切换底层大模型。

用法：
    python epub_bilingual.py input.epub [output.epub] [选项]
    python epub_bilingual.py input.pdf  [output.epub] [选项]

PDF 处理逻辑：
    1. 先用 pdfminer 尝试提取文字
    2. 若每页平均字符 < 50，判定为扫描版，自动切换 OCR 模式
    3. OCR 模式需要本机安装 tesseract + 对应语言包，以及 pdf2image / poppler

选项：
    --api-key KEY         OpenRouter API Key（或环境变量 OPENROUTER_API_KEY）
    --model   MODEL       模型 ID，默认 google/gemini-2.0-flash-001
    --batch   N           每批翻译段落数，默认 5
    --delay   SECONDS     批次间隔，默认 1.0
    --from-lang LANG      原文语言，默认 "English"
    --resume              出错时跳过继续
    --min-chars N         文字PDF：段落最少字符数（默认 40）
    --ocr-lang  LANG      OCR 语言代码，默认 eng（多语言如 eng+fra）
    --ocr-dpi   DPI       OCR 渲染分辨率，默认 300
    --force-ocr           强制使用 OCR，跳过文字提取

常用 OpenRouter 模型：
    google/gemini-2.0-flash-001   （快速低价，推荐）
    anthropic/claude-sonnet-4-5   （高质量）
    openai/gpt-4o-mini            （均衡）
    deepseek/deepseek-chat        （中文友好）

安装依赖：
    pip install ebooklib beautifulsoup4 requests pdfminer.six
    # OCR 模式额外需要：
    pip install pdf2image pytesseract
    # 并安装 tesseract-ocr 及语言包：
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   macOS:   brew install tesseract tesseract-lang
    #   Ubuntu:  apt install tesseract-ocr tesseract-ocr-eng
"""

import argparse
import os
import re
import sys
import time
import copy
import json
import uuid
from pathlib import Path
from typing import Optional

# ── 核心依赖 ──────────────────────────────────

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    sys.exit("请先安装 ebooklib：pip install ebooklib")

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:
    sys.exit("请先安装 beautifulsoup4：pip install beautifulsoup4")

try:
    import requests
except ImportError:
    sys.exit("请先安装 requests：pip install requests")


# ── 可选依赖（按需导入）──────────────────────

def _require_pdfminer():
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        return extract_pages, LTTextContainer
    except ImportError:
        sys.exit("处理 PDF 需要：pip install pdfminer.six")


def _require_ocr():
    missing = []
    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append("pdf2image")
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")
    if missing:
        sys.exit(
            f"OCR 模式需要额外安装：pip install {' '.join(missing)}\n"
            "同时请确保系统已安装 tesseract-ocr 及对应语言包。\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS:   brew install tesseract tesseract-lang\n"
            "  Ubuntu:  apt install tesseract-ocr tesseract-ocr-eng"
        )
    from pdf2image import convert_from_path
    import pytesseract
    return convert_from_path, pytesseract


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL      = "google/gemini-2.0-flash-001"
SCANNED_THRESHOLD  = 50   # 每页平均字符数低于此值 → 判定为扫描版


# ══════════════════════════════════════════════
#  OpenRouter 调用（含网络重试）
# ══════════════════════════════════════════════

_RETRYABLE = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def call_openrouter(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
    timeout: int = 120,
    net_retries: int = 4,
    net_retry_delay: float = 8.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/epub-bilingual",
        "X-Title": "EPUB Bilingual Translator",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    last_err: Exception = RuntimeError("未知错误")
    for attempt in range(1, net_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_API_URL, headers=headers,
                json=payload, timeout=timeout,
            )
        except _RETRYABLE as e:
            last_err = e
            wait = net_retry_delay * attempt
            print(f"\n      🔄 网络错误（{type(e).__name__}），{wait:.0f}s 后重试"
                  f"（{attempt}/{net_retries}）...", end=" ", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = net_retry_delay * attempt * 2
            print(f"\n      🔄 触发限速(429)，{wait:.0f}s 后重试"
                  f"（{attempt}/{net_retries}）...", end=" ", flush=True)
            time.sleep(wait)
            last_err = RuntimeError(f"限速 429：{resp.text[:200]}")
            continue

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter 请求失败 [{resp.status_code}]：{resp.text[:300]}"
            )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter 返回错误：{data['error']}")

        choice       = data.get("choices", [{}])[0]
        finish       = choice.get("finish_reason", "unknown")
        content      = choice.get("message", {}).get("content")

        if content is None:
            raise RuntimeError(
                f"模型返回 content=null（finish_reason={finish}）。"
                "可能原因：内容过滤/输出截断/模型不支持 system prompt。"
                f"\n完整响应：{json.dumps(data, ensure_ascii=False)[:400]}"
            )
        return content.strip()

    raise RuntimeError(f"网络重试 {net_retries} 次后仍失败：{last_err}")


# ══════════════════════════════════════════════
#  翻译核心
# ══════════════════════════════════════════════

TRANSLATE_SYSTEM = """你是一位专业文学翻译，擅长将{from_lang}文本译为流畅、准确的简体中文。
翻译要求：
1. 忠实原文，语言自然，符合中文表达习惯。
2. 保留段落结构，不合并、不拆分段落。
3. 不添加任何解释或注释。
4. 直接输出译文，段落间用单个空行分隔，与输入段落一一对应。
5. 输入有几段，输出就有几段，段落数必须严格对应。"""


def _translate_single(api_key: str, text: str, from_lang: str, model: str) -> str:
    """翻译单个段落，批量失败时的最终回退。"""
    try:
        return call_openrouter(
            api_key=api_key, model=model,
            system=TRANSLATE_SYSTEM.format(from_lang=from_lang),
            user=f"请将以下段落译为简体中文，只输出译文：\n\n{text}",
        )
    except Exception:
        return ""


def translate_paragraphs(
    api_key: str,
    paragraphs: list[str],
    from_lang: str = "English",
    model: str = DEFAULT_MODEL,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> list[str]:
    """批量翻译；失败重试；全失败则降级逐段翻译。"""
    if not paragraphs:
        return []

    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))
    prompt = (
        f"请将以下 {len(paragraphs)} 个段落逐段译为简体中文。\n"
        "保持编号 [N] 前缀，每段译文独占一段，段落间用空行分隔。\n\n"
        f"{numbered}"
    )

    raw: Optional[str] = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = call_openrouter(
                api_key=api_key, model=model,
                system=TRANSLATE_SYSTEM.format(from_lang=from_lang),
                user=prompt,
            )
            break
        except RuntimeError as e:
            last_err = e
            if attempt < retries:
                print(f"\n      ⚠ 第{attempt}次失败，{retry_delay}s 后重试：{e}", flush=True)
                time.sleep(retry_delay)

    if raw is None:
        print(f"\n      ↓ 批量翻译失败（{last_err}），降级为逐段翻译 ...", flush=True)
        results = []
        for i, para in enumerate(paragraphs):
            print(f"        段落 {i+1}/{len(paragraphs)} ...", end=" ", flush=True)
            zh = _translate_single(api_key, para, from_lang, model)
            results.append(zh)
            print("✓" if zh else "✗（已留空）")
            if i < len(paragraphs) - 1:
                time.sleep(1.0)
        return results

    # 解析 [N] 编号输出
    result: list[str] = [""] * len(paragraphs)
    current_idx: Optional[int] = None
    current_lines: list[str] = []

    def _flush():
        if current_idx is not None and 0 < current_idx <= len(paragraphs):
            result[current_idx - 1] = " ".join(current_lines).strip()

    for line in raw.splitlines():
        m = re.match(r"^\[(\d+)\]\s*(.*)", line)
        if m:
            _flush()
            current_idx = int(m.group(1))
            current_lines = [m.group(2)] if m.group(2) else []
        elif current_idx is not None:
            current_lines.append(line)
    _flush()

    # 回退：解析失败则按空行切割
    if all(r == "" for r in result):
        parts = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
        for i, p in enumerate(parts[: len(result)]):
            result[i] = p

    return result


# ══════════════════════════════════════════════
#  通用翻译批次调度（EPUB / PDF 共用）
# ══════════════════════════════════════════════

def _run_translation_batches(
    paragraphs: list[str],
    api_key: str,
    model: str,
    batch_size: int,
    delay: float,
    from_lang: str,
    resume: bool,
) -> list[str]:
    """对段落列表分批翻译，返回等长中文列表。"""
    all_zh: list[str] = []
    n = len(paragraphs)
    for i in range(0, n, batch_size):
        batch = paragraphs[i: i + batch_size]
        end = min(i + len(batch), n)
        print(f"    翻译第 {i+1}–{end} 段 ...", end=" ", flush=True)
        try:
            zh_batch = translate_paragraphs(api_key, batch, from_lang, model)
            all_zh.extend(zh_batch)
            print("✓")
        except Exception as e:
            print(f"✗ 错误：{e}")
            if resume:
                all_zh.extend([""] * len(batch))
            else:
                raise
        if i + batch_size < n:
            time.sleep(delay)
    return all_zh


# ══════════════════════════════════════════════
#  HTML / EPUB 工具
# ══════════════════════════════════════════════

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "td", "th", "caption", "figcaption", "dt", "dd",
}

BILINGUAL_CSS = """\
/* 双语对照样式 */
.zh-translation {
    color: #1a5276;
    font-family: "Noto Serif SC", "Source Han Serif CN", "SimSun", serif;
    margin-top: 0.3em;
    margin-bottom: 0.8em;
    border-left: 3px solid #aed6f1;
    padding-left: 0.6em;
    font-size: 0.97em;
    line-height: 1.75;
}
"""


def _extract_text_blocks(soup: BeautifulSoup) -> list[Tag]:
    return [
        tag for tag in soup.find_all(BLOCK_TAGS)
        if len(tag.get_text(" ", strip=True)) > 1
    ]


def _inject_translations(
    blocks: list[Tag], translations: list[str], orig_lang: str = "en"
) -> None:
    for block, zh_text in zip(blocks, translations):
        if not zh_text:
            continue
        zh_tag = copy.copy(block)
        for child in list(zh_tag.children):
            child.extract()
        zh_tag.string = zh_text
        zh_tag["lang"] = "zh-Hans"
        zh_tag["class"] = (zh_tag.get("class", []) or []) + ["zh-translation"]
        block.insert_after(zh_tag)
        block["lang"] = orig_lang


def _add_bilingual_style(soup: BeautifulSoup) -> None:
    head = soup.find("head")
    if head:
        tag = soup.new_tag("style")
        tag.string = BILINGUAL_CSS
        head.append(tag)


def _build_epub_from_chapters(
    chapters: list[dict],
    book_title: str,
    translations: list[list[str]],
) -> epub.EpubBook:
    """
    chapters: [{"title": str, "paragraphs": [str]}]
    translations: 与 chapters 对应，每章是等长中文列表
    """
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(f"{book_title}（中英对照）")
    book.set_language("zh-Hans")

    css_item = epub.EpubItem(
        uid="style", file_name="Styles/style.css",
        media_type="text/css", content=BILINGUAL_CSS.encode("utf-8"),
    )
    book.add_item(css_item)

    spine = ["nav"]
    toc   = []

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for ch_idx, (chapter, zh_list) in enumerate(zip(chapters, translations)):
        paras_html = []
        for orig, zh in zip(chapter["paragraphs"], zh_list):
            paras_html.append(f'<p lang="en">{esc(orig)}</p>')
            if zh:
                paras_html.append(
                    f'<p class="zh-translation" lang="zh-Hans">{esc(zh)}</p>'
                )

        html_content = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<!DOCTYPE html>'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-Hans">'
            '<head>'
            f'<title>{esc(chapter["title"])}</title>'
            '<link rel="stylesheet" type="text/css" href="../Styles/style.css"/>'
            '</head><body>'
            f'<h2>{esc(chapter["title"])}</h2>'
            + "\n".join(paras_html)
            + '</body></html>'
        )

        fn   = f"Text/chapter_{ch_idx+1:04d}.xhtml"
        item = epub.EpubHtml(title=chapter["title"], file_name=fn, lang="zh-Hans")
        item.content = html_content.encode("utf-8")
        item.add_item(css_item)
        book.add_item(item)
        spine.append(item)
        toc.append(epub.Link(fn, chapter["title"], f"ch{ch_idx+1}"))

    book.toc   = toc
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    return book


# ══════════════════════════════════════════════
#  EPUB 流程
# ══════════════════════════════════════════════

def process_epub(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 5,
    delay: float = 1.0,
    from_lang: str = "English",
    resume: bool = False,
    **_,
) -> None:
    print(f"📖 正在读取 EPUB：{input_path}")
    print(f"🤖 使用模型：{model}")
    book = epub.read_epub(input_path)

    docs = [
        item for item in book.get_items()
        if item.get_type() == ebooklib.ITEM_DOCUMENT
    ]
    print(f"📄 共找到 {len(docs)} 个章节文档")

    total_p = translated_p = 0
    for idx, doc in enumerate(docs, 1):
        raw_html = doc.get_content().decode("utf-8", errors="replace")
        soup     = BeautifulSoup(raw_html, "html.parser")
        blocks   = _extract_text_blocks(soup)
        if not blocks:
            continue

        texts    = [b.get_text(" ", strip=True) for b in blocks]
        total_p += len(texts)
        print(f"\n  [{idx}/{len(docs)}] {doc.get_name()}  —  {len(texts)} 段")

        all_zh = _run_translation_batches(
            texts, api_key, model, batch_size, delay, from_lang, resume
        )
        translated_p += sum(1 for z in all_zh if z)

        _inject_translations(blocks, all_zh)
        _add_bilingual_style(soup)
        doc.set_content(str(soup).encode("utf-8"))

    title_list = book.get_metadata("DC", "title")
    orig_title = title_list[0][0] if title_list else Path(input_path).stem
    book.set_unique_metadata("DC", "title", f"{orig_title}（中英对照）")
    book.set_unique_metadata("DC", "language", "zh-Hans")

    print(f"\n💾 正在写出：{output_path}")
    epub.write_epub(output_path, book)
    print(f"\n✅ 完成！共翻译 {translated_p}/{total_p} 段。\n   输出文件：{output_path}")


# ══════════════════════════════════════════════
#  PDF 文字提取
# ══════════════════════════════════════════════

def _extract_text_pdf(pdf_path: str, min_chars: int) -> tuple[list[dict], int]:
    """
    用 pdfminer 提取文字型 PDF。
    返回 (chapters, total_chars_per_page_avg)。
    chapters = [{"title": str, "paragraphs": [str]}]
    """
    extract_pages, LTTextContainer = _require_pdfminer()

    chapters: list[dict] = []
    total_chars = 0
    total_pages = 0

    for page_num, page_layout in enumerate(extract_pages(pdf_path), 1):
        total_pages += 1
        lines: list[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if text:
                    lines.append(text)

        page_text = " ".join(lines)
        total_chars += len(page_text)

        # 合并碎行为段落
        paragraphs: list[str] = []
        buffer = ""
        for line in lines:
            clean = line.replace("\n", " ").strip()
            if not clean:
                continue
            buffer = (buffer + " " + clean).strip() if buffer else clean
            if len(buffer) >= min_chars and re.search(r'[.!?。！？]["»]?\s*$', buffer):
                paragraphs.append(buffer)
                buffer = ""
        if buffer:
            paragraphs.append(buffer)

        if paragraphs:
            chapters.append({"title": f"第 {page_num} 页", "paragraphs": paragraphs})

    avg = total_chars // max(total_pages, 1)
    return chapters, avg


# ══════════════════════════════════════════════
#  PDF OCR 提取
# ══════════════════════════════════════════════

def _extract_ocr_pdf(
    pdf_path: str,
    ocr_lang: str = "eng",
    ocr_dpi: int = 300,
    min_chars: int = 40,
) -> list[dict]:
    """
    将扫描版 PDF 转为图像，用 tesseract OCR 提取文字。
    返回 chapters = [{"title": str, "paragraphs": [str]}]
    """
    convert_from_path, pytesseract = _require_ocr()

    print(f"   🔍 OCR 模式（语言={ocr_lang}，DPI={ocr_dpi}）")
    print(f"   正在将 PDF 转换为图像，这可能需要几分钟 ...", flush=True)

    try:
        images = convert_from_path(pdf_path, dpi=ocr_dpi)
    except Exception as e:
        sys.exit(
            f"PDF 转图像失败：{e}\n"
            "请确认已安装 poppler：\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  macOS:   brew install poppler\n"
            "  Ubuntu:  apt install poppler-utils"
        )

    print(f"   共 {len(images)} 页，开始 OCR 识别 ...")
    chapters: list[dict] = []

    for page_num, image in enumerate(images, 1):
        print(f"   OCR 第 {page_num}/{len(images)} 页 ...", end=" ", flush=True)
        try:
            raw_text = pytesseract.image_to_string(image, lang=ocr_lang)
        except Exception as e:
            print(f"✗（{e}）")
            continue

        # 按空行分段，过滤过短噪声行
        paragraphs: list[str] = []
        buffer = ""
        for line in raw_text.splitlines():
            clean = line.strip()
            if not clean:
                if buffer and len(buffer) >= min_chars:
                    paragraphs.append(buffer)
                buffer = ""
            else:
                buffer = (buffer + " " + clean).strip() if buffer else clean
        if buffer and len(buffer) >= min_chars:
            paragraphs.append(buffer)

        print(f"✓ 提取 {len(paragraphs)} 段")
        if paragraphs:
            chapters.append({"title": f"第 {page_num} 页", "paragraphs": paragraphs})

    return chapters


# ══════════════════════════════════════════════
#  PDF 主流程（自动检测 + 分发）
# ══════════════════════════════════════════════

def process_pdf(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 5,
    delay: float = 1.0,
    from_lang: str = "English",
    resume: bool = False,
    min_chars: int = 40,
    ocr_lang: str = "eng",
    ocr_dpi: int = 300,
    force_ocr: bool = False,
    **_,
) -> None:
    print(f"📄 正在分析 PDF：{input_path}")
    print(f"🤖 使用模型：{model}")

    if force_ocr:
        print("   ⚡ 已指定 --force-ocr，直接进入 OCR 模式")
        chapters = _extract_ocr_pdf(input_path, ocr_lang, ocr_dpi, min_chars)
    else:
        # 先尝试文字提取
        print("   🔎 尝试提取文字型 PDF ...", end=" ", flush=True)
        chapters, avg_chars = _extract_text_pdf(input_path, min_chars)
        print(f"每页平均 {avg_chars} 字符")

        if avg_chars < SCANNED_THRESHOLD:
            print(f"   ⚠  字符数过少（< {SCANNED_THRESHOLD}），判定为扫描版，切换 OCR 模式")
            chapters = _extract_ocr_pdf(input_path, ocr_lang, ocr_dpi, min_chars)
        else:
            print(f"   ✓ 文字型 PDF，提取到 {len(chapters)} 页")

    total_p = sum(len(c["paragraphs"]) for c in chapters)
    if total_p == 0:
        sys.exit(
            "⚠  未提取到任何文本段落。\n"
            "   • 若为扫描版，请确认 tesseract 及语言包已正确安装。\n"
            "   • 可用 --ocr-lang 指定正确语言（如 --ocr-lang deu 表示德语）。"
        )

    print(f"\n   共 {len(chapters)} 页，{total_p} 个段落，开始翻译 ...\n")

    all_translations: list[list[str]] = []
    translated_p = 0

    for ch_idx, chapter in enumerate(chapters):
        paragraphs = chapter["paragraphs"]
        print(f"  [{ch_idx+1}/{len(chapters)}] {chapter['title']}  —  {len(paragraphs)} 段")
        ch_zh = _run_translation_batches(
            paragraphs, api_key, model, batch_size, delay, from_lang, resume
        )
        all_translations.append(ch_zh)
        translated_p += sum(1 for z in ch_zh if z)

    print(f"\n📚 正在构建 EPUB ...")
    book_title = Path(input_path).stem
    book = _build_epub_from_chapters(chapters, book_title, all_translations)

    out = Path(output_path)
    if out.suffix.lower() != ".epub":
        out = out.with_suffix(".epub")

    print(f"💾 正在写出：{out}")
    epub.write_epub(str(out), book)
    print(f"\n✅ 完成！共翻译 {translated_p}/{total_p} 段。\n   输出文件：{out}")


# ══════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将 EPUB / PDF（文字版或扫描版）转换为逐段中文对照 EPUB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="输入文件（.epub 或 .pdf）")
    parser.add_argument("output", nargs="?",
                        help="输出路径（默认：原文件名 + _bilingual.epub）")
    parser.add_argument("--api-key",   default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API Key")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"模型 ID（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--batch",     type=int,   default=5,
                        help="每批段落数（默认 5）")
    parser.add_argument("--delay",     type=float, default=1.0,
                        help="批次间隔秒数（默认 1.0）")
    parser.add_argument("--from-lang", default="English",
                        help="原文语言（默认 English）")
    parser.add_argument("--resume",    action="store_true",
                        help="出错时跳过继续")
    parser.add_argument("--min-chars", type=int,   default=40,
                        help="段落最少字符数（默认 40）")
    parser.add_argument("--ocr-lang",  default="eng",
                        help="OCR 语言代码（默认 eng；多语言如 eng+deu）")
    parser.add_argument("--ocr-dpi",   type=int,   default=300,
                        help="OCR 渲染 DPI（默认 300；扫描质量差时可调高至 400）")
    parser.add_argument("--force-ocr", action="store_true",
                        help="强制使用 OCR，跳过文字提取尝试")

    args = parser.parse_args()

    if not args.api_key:
        sys.exit(
            "❌ 未找到 API Key。\n"
            "   请通过 --api-key 传入，或设置环境变量 OPENROUTER_API_KEY。"
        )

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"❌ 找不到输入文件：{input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in (".epub", ".pdf"):
        sys.exit(f"❌ 不支持的文件类型：{suffix}（仅支持 .epub / .pdf）")

    output_path = args.output or str(
        input_path.with_stem(input_path.stem + "_bilingual").with_suffix(".epub")
    )

    kwargs = dict(
        output_path = output_path,
        api_key     = args.api_key,
        model       = args.model,
        batch_size  = args.batch,
        delay       = args.delay,
        from_lang   = args.from_lang,
        resume      = args.resume,
        min_chars   = args.min_chars,
        ocr_lang    = args.ocr_lang,
        ocr_dpi     = args.ocr_dpi,
        force_ocr   = args.force_ocr,
    )

    if suffix == ".epub":
        process_epub(input_path=str(input_path), **kwargs)
    else:
        process_pdf(input_path=str(input_path), **kwargs)


if __name__ == "__main__":
    main()
