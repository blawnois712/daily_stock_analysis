# -*- coding: utf-8 -*-
"""
===================================
格式化工具模塊
===================================

提供各種內容格式化工具函式，用於將通用格式轉換為平台特定格式。
"""

import re
from typing import Callable, List, Optional

import markdown2

TRUNCATION_SUFFIX = "\n...(本段內容過長已截斷)"
PAGE_MARKER_PREFIX = "\n📄"  # 單一換行，不留空白行
PAGE_MARKER_SAFE_BYTES = 16 # "\n📄 9999/9999"
PAGE_MARKER_SAFE_LEN = 13   # "\n📄 9999/9999"
MIN_MAX_WORDS = 10
MIN_MAX_BYTES = 40
FENCED_CODE_BLOCK_RE = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
FENCED_CODE_BLOCK_PLACEHOLDER = "@@DSA_FENCED_CODE_BLOCK_{}@@"
HIDDEN_MARKDOWN_METADATA_RE = re.compile(
    r"^\[dsa-[^\]]+\]:\s+#\s+\([^)\n]*\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Unicode code point ranges for special characters.
_SPECIAL_CHAR_RANGE = (0x10000, 0xFFFFF)
_SPECIAL_CHAR_REGEX = re.compile(r'[\U00010000-\U000FFFFF]')


def _page_marker(i: int, total: int) -> str:
    return f"{PAGE_MARKER_PREFIX} {i+1}/{total}"


def _is_special_char(c: str) -> bool:
    """判斷字元是否為特殊字元

    Args:
        c: 字元

    Returns:
        True 如果字元為特殊字元，False 否則
    """
    if len(c) != 1:
        return False
    cp = ord(c)
    return _SPECIAL_CHAR_RANGE[0] <= cp <= _SPECIAL_CHAR_RANGE[1]


def _count_special_chars(s: str) -> int:
    """
    計算字元串中的特殊字元數量

    Args:
        s: 字元串
    """
    # reg find all (0x10000, 0xFFFFF)
    match = _SPECIAL_CHAR_REGEX.findall(s)
    return len(match)


def _effective_len(s: str, special_char_len: int = 2) -> int:
    """
    計算字元串的有效長度

    Args:
        s: 字元串
        special_char_len: 每個特殊字元的長度，預設為 2

    Returns:
        s 的有效長度
    """
    n = len(s)
    n += _count_special_chars(s) * (special_char_len - 1)
    return n


def _slice_at_effective_len(s: str, effective_len: int, special_char_len: int = 2) -> tuple[str, str]:
    """
    按有效長度分割字元串

    Args:
        s: 字元串
        effective_len: 有效長度
        special_char_len: 每個特殊字元的長度，預設為 2

    Returns:
        分割後的前、後部分字元串
    """
    if _effective_len(s, special_char_len) <= effective_len:
        return s, ""

    s_ = s[:effective_len]
    n_special_chars = _count_special_chars(s_)
    residual_lens = n_special_chars * (special_char_len - 1) + len(s_) - effective_len
    while residual_lens > 0:
        residual_lens -= special_char_len if _is_special_char(s_[-1]) else 1
        s_ = s_[:-1]
    return s_, s[len(s_):]


def markdown_to_html_document(markdown_text: str) -> str:
    """
    Convert Markdown to a complete HTML document (for email, md2img, etc.).

    Uses markdown2 with table and code block support, wraps with inline CSS
    for compact, readable layout. Reused by notification email and md2img.

    Args:
        markdown_text: Raw Markdown content.

    Returns:
        Full HTML document string with DOCTYPE, head, and body.
    """
    html_content = markdown2.markdown(
        markdown_text,
        extras=["tables", "fenced-code-blocks", "break-on-newline", "cuddled-lists"],
    )

    css_style = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                line-height: 1.5;
                color: #24292e;
                font-size: 14px;
                padding: 15px;
                max-width: 900px;
                margin: 0 auto;
            }
            h1 {
                font-size: 20px;
                border-bottom: 1px solid #eaecef;
                padding-bottom: 0.3em;
                margin-top: 1.2em;
                margin-bottom: 0.8em;
                color: #0366d6;
            }
            h2 {
                font-size: 18px;
                border-bottom: 1px solid #eaecef;
                padding-bottom: 0.3em;
                margin-top: 1.0em;
                margin-bottom: 0.6em;
            }
            h3 {
                font-size: 16px;
                margin-top: 0.8em;
                margin-bottom: 0.4em;
            }
            p {
                margin-top: 0;
                margin-bottom: 8px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 12px 0;
                display: block;
                overflow-x: auto;
                font-size: 13px;
            }
            th, td {
                border: 1px solid #dfe2e5;
                padding: 6px 10px;
                text-align: left;
            }
            th {
                background-color: #f6f8fa;
                font-weight: 600;
            }
            tr:nth-child(2n) {
                background-color: #f8f8f8;
            }
            tr:hover {
                background-color: #f1f8ff;
            }
            blockquote {
                color: #6a737d;
                border-left: 0.25em solid #dfe2e5;
                padding: 0 1em;
                margin: 0 0 10px 0;
            }
            code {
                padding: 0.2em 0.4em;
                margin: 0;
                font-size: 85%;
                background-color: rgba(27,31,35,0.05);
                border-radius: 3px;
                font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            }
            pre {
                padding: 12px;
                overflow: auto;
                line-height: 1.45;
                background-color: #f6f8fa;
                border-radius: 3px;
                margin-bottom: 10px;
            }
            hr {
                height: 0.25em;
                padding: 0;
                margin: 16px 0;
                background-color: #e1e4e8;
                border: 0;
            }
            ul, ol {
                padding-left: 20px;
                margin-bottom: 10px;
            }
            li {
                margin: 2px 0;
            }
        """

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                {css_style}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """


def markdown_to_plain_text(markdown_text: str) -> str:
    """
    將 Markdown 轉換為純文字

    移除 Markdown 格式標記，保留可讀性
    """
    text = markdown_text

    # 移除 Markdown reference definitions（例如隱藏元數據標記）
    text = strip_markdown_reference_definitions(text)

    # 移除標題標記 # ## ###
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 移除加粗 **text** -> text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # 移除斜體 *text* -> text
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # 移除引用 > text -> text
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # 移除列表標記 - item -> item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)

    # 移除分隔線 ---
    text = re.sub(r'^---+$', '────────', text, flags=re.MULTILINE)

    # 移除表格語法 |---|---|
    text = re.sub(r'\|[-:]+\|[-:|\s]+\|', '', text)
    text = re.sub(r'^\|(.+)\|$', r'\1', text, flags=re.MULTILINE)

    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def strip_hidden_markdown_metadata(markdown_text: str) -> str:
    """Strip hidden internal Markdown metadata lines while preserving normal refs."""

    return HIDDEN_MARKDOWN_METADATA_RE.sub("", markdown_text)


def strip_markdown_reference_definitions(markdown_text: str) -> str:
    """Strip Markdown reference-definition lines while preserving the rest."""

    return re.sub(r'^\[[^\]]+\]:\s+.*$', '', markdown_text, flags=re.MULTILINE)


def _bytes(s: str) -> int:
    return len(s.encode('utf-8'))


def utf8_len(s: str) -> int:
    """Return the number of UTF-8 bytes used by ``s``."""

    return len(s.encode("utf-8"))


def utf16_len(s: str) -> int:
    """Return the number of UTF-16 code units used by ``s``.

    Telegram's 4096-character message limit is effectively counted in UTF-16
    units, so astral-plane characters such as emoji consume two units.
    """

    return len(s.encode("utf-16-le")) // 2


def _custom_unit_to_index(text: str, budget: int, len_fn: Callable[[str], int]) -> int:
    """Map a custom-unit budget to the largest safe Python string index."""

    if len_fn(text) <= budget:
        return len(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _has_unclosed_inline_code(text: str) -> bool:
    """Return whether ``text`` ends inside a single-backtick inline code span."""

    escaped = False
    count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and not escaped:
            escaped = True
            i += 1
            continue
        if ch == "`" and not escaped:
            # Triple backticks are handled as fenced code by the caller; skip
            # them here so fence delimiters do not look like inline spans.
            if text[i:i + 3] == "```":
                i += 3
                escaped = False
                continue
            count += 1
        escaped = False
        i += 1
    return count % 2 == 1


def _last_unclosed_markdown_link_start(text: str) -> int:
    """Return the start index of an inline Markdown link split in progress."""

    last_open_paren = text.rfind("](")
    last_close_paren = text.rfind(")")
    if last_open_paren > last_close_paren:
        label_start = text.rfind("[", 0, last_open_paren)
        return label_start if label_start >= 0 else last_open_paren

    last_open_bracket = text.rfind("[")
    last_close_bracket = text.rfind("]")
    if last_open_bracket > last_close_bracket:
        return last_open_bracket

    return -1


def chunk_markdown_preserving_blocks(
    content: str,
    max_units: int,
    *,
    len_fn: Optional[Callable[[str], int]] = None,
    add_page_marker: bool = False,
) -> List[str]:
    """Split Markdown while preserving common formatting boundaries.

    The splitter is intentionally conservative and does not alter report
    semantics.  If a split lands inside a fenced code block, the current chunk is
    closed and the next chunk reopens the same fence language.  It also avoids
    splitting inside inline code spans and Markdown links, and supports custom
    length functions such as :func:`utf16_len`.
    """

    measure = len_fn or len
    if max_units < MIN_MAX_WORDS:
        raise ValueError(f"max_units={max_units} < {MIN_MAX_WORDS}, 可能陷入無限遞迴。")
    if measure(content) <= max_units:
        return [content]

    marker_reserve = measure(_page_marker(9998, 9998)) if add_page_marker else 0
    indicator_reserve = measure("\n\n(9999/9999)")
    fence_close = "\n```"
    chunks: List[str] = []
    remaining = content
    carry_lang: Optional[str] = None

    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        headroom = max_units - marker_reserve - indicator_reserve - measure(prefix) - measure(fence_close)
        if headroom < MIN_MAX_WORDS:
            headroom = max(MIN_MAX_WORDS, max_units - marker_reserve - indicator_reserve - measure(prefix))
        if headroom <= 0:
            raise ValueError("max_units is too small for markdown-preserving chunking")

        if measure(prefix) + measure(remaining) <= max_units - marker_reserve - indicator_reserve:
            chunks.append(prefix + remaining)
            break

        cp_limit = (
            _custom_unit_to_index(remaining, headroom, measure)
            if measure is not len else min(headroom, len(remaining))
        )
        region = remaining[:cp_limit]
        split_at = region.rfind("\n\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind("\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            split_at = cp_limit

        candidate = remaining[:split_at]
        unsafe_start = len(candidate)
        if _has_unclosed_inline_code(candidate):
            last_tick = candidate.rfind("`")
            if last_tick >= 0:
                unsafe_start = min(unsafe_start, last_tick)

        link_start = _last_unclosed_markdown_link_start(candidate)
        if link_start >= 0:
            unsafe_start = min(unsafe_start, link_start)

        if unsafe_start < len(candidate):
            safe_split = max(candidate.rfind(" ", 0, unsafe_start), candidate.rfind("\n", 0, unsafe_start))
            if safe_split > 0:
                split_at = safe_split

        chunk_body = remaining[:split_at].rstrip()
        in_code = carry_lang is not None
        lang = carry_lang or ""
        for line in chunk_body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    in_code = False
                    lang = ""
                else:
                    in_code = True
                    tag = stripped[3:].strip()
                    lang = tag.split()[0] if tag else ""

        next_remaining = remaining[split_at:]
        if next_remaining.startswith("\n"):
            next_remaining = next_remaining[1:]
        elif not in_code and next_remaining.startswith(" "):
            line_start = remaining.rfind("\n", 0, split_at) + 1
            if remaining[line_start:split_at].strip(" \t"):
                next_remaining = next_remaining[1:]
        remaining = next_remaining
        full_chunk = prefix + chunk_body

        if in_code:
            full_chunk += fence_close
            carry_lang = lang
        else:
            carry_lang = None

        chunks.append(full_chunk)

    if len(chunks) > 1:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            suffix = f"\n\n({i + 1}/{total})"
            if add_page_marker:
                suffix += _page_marker(i, total)
            chunks[i] = chunk + suffix
    elif add_page_marker:
        chunks[0] = chunks[0] + _page_marker(0, 1)
    return chunks


def _is_markdown_table_separator(row: str) -> bool:
    return bool(re.match(r'^\s*\|?\s*[:-]+\s*(\|\s*[:-]+\s*)+\|?\s*$', row))


def _parse_markdown_table_row(row: str) -> List[str]:
    stripped = row.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split('|')]


def _strip_inline_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
    text = re.sub(r'^\*(.+)\*$', r'\1', text)
    text = text.replace("**", "")
    return text.strip()


def _format_two_column_table_row(header: List[str], row: List[str]) -> str:
    key = _strip_inline_markdown(row[0]) if row else ""
    value = _strip_inline_markdown(row[1]) if len(row) > 1 else ""
    value_header = _strip_inline_markdown(header[1]) if len(header) > 1 else ""

    if not value:
        return key
    if value.upper() == "N/A" and value_header in {"類型", "类型", "Type"}:
        return key
    return f"{key}：{value}"


def _flush_table_as_key_value_rows(buffer: List[str], output: List[str], *, bullet: str) -> None:
    if not buffer:
        return

    rows = []
    for raw in buffer:
        if _is_markdown_table_separator(raw):
            continue
        parsed = _parse_markdown_table_row(raw)
        if parsed:
            rows.append(parsed)

    if not rows:
        return

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        if len(header) == 2 and len(row) >= 2:
            output.append(f"{bullet} {_format_two_column_table_row(header, row)}")
            continue

        pairs = []
        for idx, cell in enumerate(row):
            key = _strip_inline_markdown(header[idx]) if idx < len(header) else f"欄{idx + 1}"
            pairs.append(f"{key}：{_strip_inline_markdown(cell)}")
        output.append(f"{bullet} {' | '.join(pairs)}")


def _protect_fenced_code_blocks(content: str) -> tuple[str, List[str]]:
    blocks: List[str] = []

    def _replace(match: re.Match) -> str:
        blocks.append(match.group(0))
        return FENCED_CODE_BLOCK_PLACEHOLDER.format(len(blocks) - 1)

    return FENCED_CODE_BLOCK_RE.sub(_replace, content), blocks


def _restore_fenced_code_blocks(content: str, blocks: List[str]) -> str:
    restored = content
    for idx, block in enumerate(blocks):
        restored = restored.replace(FENCED_CODE_BLOCK_PLACEHOLDER.format(idx), block)
    return restored


def _transform_outside_fenced_code_blocks(content: str, transform: Callable[[str], str]) -> str:
    protected, blocks = _protect_fenced_code_blocks(content)
    return _restore_fenced_code_blocks(transform(protected), blocks)


def _markdown_tables_to_key_value_rows_unprotected(content: str, *, bullet: str) -> str:
    """Convert pipe tables to compact key-value rows for chat clients."""

    lines: List[str] = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        if table_buffer:
            _flush_table_as_key_value_rows(table_buffer, lines, bullet=bullet)
            table_buffer = []

        lines.append(line)

    if table_buffer:
        _flush_table_as_key_value_rows(table_buffer, lines, bullet=bullet)

    return "\n".join(lines).strip()


def markdown_tables_to_key_value_rows(content: str, *, bullet: str = "•") -> str:
    """Convert pipe tables to compact key-value rows outside fenced code blocks."""

    return _transform_outside_fenced_code_blocks(
        content,
        lambda text: _markdown_tables_to_key_value_rows_unprotected(text, bullet=bullet),
    ).strip()


def _chunk_by_max_bytes(content: str, max_bytes: int) -> List[str]:
    if _bytes(content) <= max_bytes:
        return [content]
    if max_bytes < MIN_MAX_BYTES:
        raise ValueError(f"max_bytes={max_bytes} < {MIN_MAX_BYTES}, 可能陷入無限遞迴。")

    sections: List[str] = []
    suffix = TRUNCATION_SUFFIX
    effective_max_bytes = max_bytes - _bytes(suffix)
    if effective_max_bytes <= 0:
        effective_max_bytes = max_bytes
        suffix = ""

    while True:
        chunk, content = slice_at_max_bytes(content, effective_max_bytes)
        if content.strip() != "":
            sections.append(chunk + suffix)
        else:
            # 最後一段了，直接添加並離開循環
            sections.append(chunk)
            break
    return sections


def chunk_content_by_max_bytes(content: str, max_bytes: int, add_page_marker: bool = False) -> List[str]:
    """
    按位元組數智能分割訊息內容

    Args:
        content: 完整訊息內容
        max_bytes: 單條訊息最大位元組數
        add_page_marker: 是否添加分頁標記

    Returns:
        分割後的區塊列表
    """
    def _chunk(content: str, max_bytes: int) -> List[str]:
        # 優先按分隔線/標題分割，保證分頁自然
        if max_bytes < MIN_MAX_BYTES:
            raise ValueError(f"max_bytes={max_bytes} < {MIN_MAX_BYTES}, 可能陷入無限遞迴。")

        if _bytes(content) <= max_bytes:
            return [content]

        sections, separator = _chunk_by_separators(content)
        if separator == "" and len(sections) == 1:
            # 無法智能分割，則強制按字數分割
            return _chunk_by_max_bytes(content, max_bytes)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_bytes = 0
        separator_bytes = _bytes(separator) if separator else 0
        effective_max_bytes = max_bytes - separator_bytes

        for section in sections:
            section += separator
            section_bytes = _bytes(section)

            # 如果單個 section 就超長，需要強制截斷
            if section_bytes > effective_max_bytes:
                # 先保存當前積累的內容
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_bytes = 0

                # 強制按位元組截斷，避免整段被截斷丟失
                section_chunks = _chunk(
                    section[:-separator_bytes], effective_max_bytes
                )
                section_chunks[-1] = section_chunks[-1] + separator
                chunks.extend(section_chunks)
                continue

            # 檢查加入後是否超長
            if current_bytes + section_bytes > effective_max_bytes:
                # 保存當前塊，開始新塊
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [section]
                current_bytes = section_bytes
            else:
                current_chunk.append(section)
                current_bytes += section_bytes

        # 添加最後一塊
        if current_chunk:
            chunks.append("".join(current_chunk))

        # 移除最後一個塊的分割符
        if (chunks and
            len(chunks[-1]) > separator_bytes and
            chunks[-1][-separator_bytes:] == separator
        ):
            chunks[-1] = chunks[-1][:-separator_bytes]

        return chunks

    if add_page_marker:
        max_bytes = max_bytes - PAGE_MARKER_SAFE_BYTES

    chunks = _chunk(content, max_bytes)
    if add_page_marker:
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk.rstrip("\n") + _page_marker(i, total_chunks)
    return chunks


def slice_at_max_bytes(text: str, max_bytes: int) -> tuple[str, str]:
    """
    按位元組數截斷字元串，確保不會在多位元組字元中間截斷

    Args:
        text: 要截斷的字元串
        max_bytes: 最大位元組數

    Returns:
        (截斷後的字元串, 剩余未截斷內容)
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, ""

    # 從最大位元組數開始向前查找，找到完整的 UTF-8 字元邊界
    truncated = encoded[:max_bytes]
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]

    truncated = truncated.decode('utf-8', errors='ignore')
    return truncated, text[len(truncated):]


def _format_feishu_markdown_unprotected(content: str) -> str:
    lines = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()

        # 處理表格行
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        # 刷新表格緩衝區
        if table_buffer:
            _flush_table_as_key_value_rows(table_buffer, lines, bullet="•")
            table_buffer = []

        # 轉換標題（# ## ### 等）
        if re.match(r'^#{1,6}\s+', line):
            title = re.sub(r'^#{1,6}\s+', '', line).strip()
            line = f"**{title}**" if title else ""
        # 轉換引用塊
        elif line.startswith('> '):
            quote = line[2:].strip()
            line = quote
        # 轉換分隔線
        elif line.strip() == '---':
            line = '────────'
        # 轉換列表項
        elif line.startswith('- '):
            line = f"• {line[2:].strip()}"

        lines.append(line)

    # 處理末尾的表格
    if table_buffer:
        _flush_table_as_key_value_rows(table_buffer, lines, bullet="•")

    return "\n".join(lines).strip()


def format_feishu_markdown(content: str) -> str:
    """
    將通用 Markdown 轉換為飛書 lark_md 更友好的格式

    轉換規則：
    - 飛書不支援 Markdown 標題（# / ## / ###），用加粗代替
    - 引用塊使用前綴替代
    - 分隔線統一為細線
    - 表格轉換為條目列表

    Args:
        content: 原始 Markdown 內容

    Returns:
        轉換後的飛書 Markdown 格式內容

    Example:
        >>> markdown = "# 標題\\n> 引用\\n| 列1 | 列2 |"
        >>> formatted = format_feishu_markdown(markdown)
        >>> print(formatted)
        **標題**
        💬 引用
        • 列1：值1 | 列2：值2
    """
    def _flush_table_rows(buffer: List[str], output: List[str]) -> None:
        """將表格緩衝區中的行轉換為飛書格式"""
        if not buffer:
            return

        def _parse_row(row: str) -> List[str]:
            """解析表格行，提取單元格"""
            return _parse_markdown_table_row(row)

        rows = []
        for raw in buffer:
            # 跳過分隔行（如 |---|---|）
            if re.match(r'^\s*\|?\s*[:-]+\s*(\|\s*[:-]+\s*)+\|?\s*$', raw):
                continue
            parsed = _parse_row(raw)
            if parsed:
                rows.append(parsed)

        if not rows:
            return

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        for row in data_rows:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            pairs = []
            for idx, cell in enumerate(row):
                key = header[idx] if idx < len(header) else f"欄{idx + 1}"
                pairs.append(f"{key}：{cell}")
            output.append(f"• {' | '.join(pairs)}")

    lines = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()

        # 處理表格行
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        # 刷新表格緩衝區
        if table_buffer:
            _flush_table_rows(table_buffer, lines)
            table_buffer = []

        # 轉換標題（# ## ### 等）
        if re.match(r'^#{1,6}\s+', line):
            title = re.sub(r'^#{1,6}\s+', '', line).strip()
            line = f"**{title}**" if title else ""
        # 轉換引用塊
        elif line.startswith('> '):
            quote = line[2:].strip()
            line = f"💬 {quote}" if quote else ""
        # 轉換分隔線
        elif line.strip() == '---':
            line = '────────'
        # 轉換列表項
        elif line.startswith('- '):
            line = f"• {line[2:].strip()}"

        lines.append(line)

    # 處理末尾的表格
    if table_buffer:
        _flush_table_rows(table_buffer, lines)

    return "\n".join(lines).strip()


def _format_telegram_markdown_unprotected(content: str) -> str:
    """Convert common report Markdown to Telegram legacy Markdown."""

    result = _markdown_tables_to_key_value_rows_unprotected(content, bullet="-")
    result = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', result, flags=re.MULTILINE)
    result = re.sub(r'\*\*(.+?)\*\*', r'*\1*', result)
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    result = _escape_telegram_non_link_markdown_chars(result)
    return result.strip()


def _escape_telegram_non_link_markdown_chars(content: str) -> str:
    """Escape Telegram Markdown link metacharacters outside valid links."""

    links: list[str] = []

    def _save_link(match: re.Match) -> str:
        links.append(match.group(0))
        return f"@@DSA_TELEGRAM_LINK_{len(links) - 1}@@"

    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _save_link, content)
    for char in ("[", "]", "(", ")"):
        result = result.replace(char, f"\\{char}")

    for index, link in enumerate(links):
        result = result.replace(f"@@DSA_TELEGRAM_LINK_{index}@@", link)
    return result


def format_telegram_markdown(content: str) -> str:
    """Convert common report Markdown to Telegram legacy Markdown."""

    return _transform_outside_fenced_code_blocks(
        content,
        _format_telegram_markdown_unprotected,
    ).strip()


def format_wechat_markdown(content: str) -> str:
    """Keep WeChat Markdown style while making pipe tables mobile-readable."""

    result = markdown_tables_to_key_value_rows(content, bullet="•")
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    return result.strip()


def _format_slack_mrkdwn_unprotected(content: str) -> str:
    """Convert common report Markdown to Slack mrkdwn."""

    result = _markdown_tables_to_key_value_rows_unprotected(content, bullet="•")
    result = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<\2|\1>', result)
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', result)
    result = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', result, flags=re.MULTILINE)
    result = re.sub(r'\*\*(.+?)\*\*', r'*\1*', result)
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    return result.strip()


def format_slack_mrkdwn(content: str) -> str:
    """Convert common report Markdown to Slack mrkdwn."""

    return _transform_outside_fenced_code_blocks(
        content,
        _format_slack_mrkdwn_unprotected,
    ).strip()


def _chunk_by_separators(content: str) -> tuple[list[str], str]:
    """
    通過分隔線等特殊字元將訊息內容分割為多個區塊

    Args:
        content: 完整訊息內容

    Returns:
        sections: 分割後的區塊列表
        separator: 區塊之間的分隔符號，None 表示無法分割
    """
    # 智能分割：優先按 "---" 分隔（股票之間的分隔線）
    # 其次嘗試各級標題分割
    if "\n---\n" in content:
        sections = content.split("\n---\n")
        separator = "\n---\n"
    elif "\n# " in content:
        # 按 # 分割 (相容一級標題)
        parts = content.split("\n# ")
        sections = [parts[0]] + [f"# {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n## " in content:
        # 按 ## 分割 (相容二級標題)
        parts = content.split("\n## ")
        sections = [parts[0]] + [f"## {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n### " in content:
        # 按 ### 分割
        parts = content.split("\n### ")
        sections = [parts[0]] + [f"### {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n**" in content:
        # 按 ** 加粗標題分割 (相容 AI 未輸出標准 Markdown 標題的情況)
        parts = content.split("\n**")
        sections = [parts[0]] + [f"**{p}" for p in parts[1:]]
        separator = "\n"
    elif "\n" in content:
        # 按 \n 分割
        sections = content.split("\n")
        separator = "\n"
    else:
        return [content], ""
    return sections, separator


def _chunk_by_max_words(content: str, max_words: int, special_char_len: int = 2) -> list[str]:
    """
    按字數分割訊息內容

    Args:
        content: 完整訊息內容
        max_words: 單條訊息最大字數
        special_char_len: 每個特殊字元的長度，預設為 2

    Returns:
        分割後的區塊列表
    """
    if _effective_len(content, special_char_len) <= max_words:
        return [content]
    if max_words < MIN_MAX_WORDS:
        raise ValueError(
            f"max_words={max_words} < {MIN_MAX_WORDS}, 可能陷入無限遞迴。"
        )

    sections = []
    suffix = TRUNCATION_SUFFIX
    effective_max_words = max_words - len(suffix)  # 預留後綴，避免邊界超限
    if effective_max_words <= 0:
        effective_max_words = max_words
        suffix = ""

    while True:
        chunk, content = _slice_at_effective_len(content, effective_max_words, special_char_len)
        if content.strip() != "":
            sections.append(chunk + suffix)
        else:
            # 最後一段了，直接添加並離開循環
            sections.append(chunk)
            break
    return sections


def chunk_content_by_max_words(
    content: str,
    max_words: int,
    special_char_len: int = 2,
    add_page_marker: bool = False
    ) -> list[str]:
    """
    按字數智能分割訊息內容

    Args:
        content: 完整訊息內容
        max_words: 單條訊息最大字數
        special_char_len: 每個特殊字元的長度，預設為 2
        add_page_marker: 是否添加分頁標記

    Returns:
        分割後的區塊列表
    """
    def _chunk(content: str, max_words: int, special_char_len: int = 2) -> list[str]:
        if max_words < MIN_MAX_WORDS:
            # Safe guard，避免無限遞迴
            # 理論上，max_words在每次遞迴中可以減小到無限小，但實際中不太可能發生，
            # 除非每次_chunk_by_separators都能成功返回分隔符號，且max_words初始值太小。
            raise ValueError(f"max_words={max_words} < {MIN_MAX_WORDS}, 可能陷入無限遞迴。")

        if _effective_len(content, special_char_len) <= max_words:
            return [content]

        sections, separator = _chunk_by_separators(content)
        if separator == "" and len(sections) == 1:
            # 無法智能分割，則強制按字數分割
            return _chunk_by_max_words(content, max_words, special_char_len)

        chunks = []
        current_chunk = []
        current_word_len = 0
        separator_len = len(separator) if separator else 0
        effective_max_words = max_words - separator_len # 預留分割符長度，避免邊界超限

        for section in sections:
            section += separator
            section_word_len = _effective_len(section, special_char_len)

            # 如果單個 section 就超長，需要強制截斷
            if section_word_len > max_words:
                # 先保存當前積累的內容
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_word_len = 0

                # 強制截斷這個超長 section
                section_chunks = _chunk(
                    section[:-separator_len], effective_max_words, special_char_len
                    )
                section_chunks[-1] = section_chunks[-1] + separator
                chunks.extend(section_chunks)
                continue

            # 檢查加入後是否超長
            if current_word_len + section_word_len > max_words:
                # 保存當前塊，開始新塊
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [section]
                current_word_len = section_word_len
            else:
                current_chunk.append(section)
                current_word_len += section_word_len

        # 添加最後一塊
        if current_chunk:
            chunks.append("".join(current_chunk))

        # 移除最後一個塊的分割符
        if (chunks and
            len(chunks[-1]) > separator_len and
            chunks[-1][-separator_len:] == separator
        ):
            chunks[-1] = chunks[-1][:-separator_len]
        return chunks


    if add_page_marker:
        max_words = max_words - PAGE_MARKER_SAFE_LEN

    chunks = _chunk(content, max_words, special_char_len)
    if add_page_marker:
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk.rstrip("\n") + _page_marker(i, total_chunks)
    return chunks

