from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


DEFAULT_MIN_SIMILARITY = 0.52
DEFAULT_RELATIVE_SCORE = 0.90
DEFAULT_MIN_LEXICAL_OVERLAP = 0.30
WEAK_SCORE_BOUNDARY = 0.65
MAX_CHUNKS_PER_SOURCE = 2

CATEGORIES = ("政策制度", "专业介绍", "新闻通知", "通用知识")

_INTENT_CATEGORY = {
    "policy": "政策制度",
    "major": "专业介绍",
    "news": "新闻通知",
}

_CATEGORY_KEYWORDS = {
    "专业介绍": ("专业", "学院", "培养", "课程", "学科", "招生", "大创", "学籍"),
    "新闻通知": ("新闻", "通知", "公告", "活动", "讲座", "开放", "安排", "动态"),
    "政策制度": (
        "规章",
        "办法",
        "条例",
        "规定",
        "违纪",
        "处分",
        "制度",
        "纪律",
        "政策",
        "手册",
        "管理",
    ),
}

_DATE_PATTERNS = (
    re.compile(r"(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?"),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
)
_STUB_CONTENT_PATTERN = re.compile(r"(详见附件|见附件|附件下载|点击下载|详见原文)")

_QUERY_NOISE = (
    "请帮我",
    "帮我",
    "请问",
    "麻烦",
    "查询",
    "告诉我",
    "介绍一下",
    "是什么",
    "有哪些",
    "需要哪些",
    "怎么办",
    "怎么做",
    "如何",
    "哪些",
    "什么",
    "是否",
    "可以",
    "能否",
    "今天",
    "具体",
    "几点",
    "一下",
    "相关",
    "情况",
    "问题",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_url(value: Any) -> str:
    """Normalize a source URL while keeping the original transport scheme."""
    url = str(value or "").strip()
    if not url:
        return ""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return url

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return url

    try:
        port = parsed.port
    except ValueError:
        return url
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, parsed.query, ""))


def url_identity(value: Any) -> str:
    """Ignore http/https differences when the same official page was crawled twice."""
    url = canonical_url(value)
    if not url:
        return ""
    parsed = urlsplit(url)
    if parsed.scheme in ("http", "https"):
        return urlunsplit(("", parsed.netloc, parsed.path, parsed.query, ""))
    return url


def infer_category(title: Any, content: Any) -> str:
    text = f"{normalize_text(title)} {normalize_text(content)[:150]}"
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "通用知识"


def extract_document_date(title: Any, content: Any) -> str:
    text = f"{normalize_text(title)} {normalize_text(content)[:800]}"
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        year, month, day = (int(part) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def is_stub_source(title: Any, content: Any) -> bool:
    normalized = normalize_text(content)
    if not normalized:
        return True
    if len(normalized) <= 160 and _STUB_CONTENT_PATTERN.search(normalized):
        return True
    return len(normalized) < 24


def preferred_category(intent: str) -> str:
    return _INTENT_CATEGORY.get(intent, "")


def _text_fingerprint(text: Any) -> str:
    normalized = re.sub(r"[\W_]+", "", normalize_text(text).lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _query_terms(query: str) -> set[str]:
    text = normalize_text(query).lower()
    for phrase in sorted(_QUERY_NOISE, key=len, reverse=True):
        text = text.replace(phrase, " ")

    terms: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    terms.update(re.findall(r"[a-z0-9]{2,}", text))
    return terms


def lexical_overlap(query: str, text: str) -> float:
    query_terms = _query_terms(query)
    if not query_terms:
        return 1.0
    text_terms = _query_terms(text)
    return len(query_terms & text_terms) / len(query_terms)


def make_source(
    raw: dict[str, Any],
    *,
    citation_id: str = "",
    source_type: str = "knowledge",
) -> dict[str, Any]:
    title = normalize_text(raw.get("title")) or "校园知识库资料"
    content = normalize_text(raw.get("content"))
    source_url = str(raw.get("source_url") or "").strip()
    score = float(raw.get("score") or 0.0)
    content_status = "metadata_only" if is_stub_source(title, content) else "full_text"
    return {
        "id": citation_id,
        "title": title,
        "content": content,
        "score": round(score, 6),
        "source_url": source_url,
        "canonical_url": canonical_url(source_url),
        "category": infer_category(title, content),
        "date": extract_document_date(title, content),
        "source_type": source_type,
        "content_status": content_status,
    }


def select_relevant_sources(
    raw_sources: Iterable[dict[str, Any]],
    query: str,
    *,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    relative_score: float = DEFAULT_RELATIVE_SCORE,
    min_lexical_overlap: float = DEFAULT_MIN_LEXICAL_OVERLAP,
    preferred: str = "",
) -> list[dict[str, Any]]:
    """
    Filter, deduplicate and rank retrieved chunks.

    Exact duplicate chunks are collapsed even when their source URLs differ only
    by http/https. At most two chunks are kept for one canonical page.
    """
    if top_k <= 0:
        return []

    candidates = [make_source(raw) for raw in raw_sources if raw]
    if not candidates:
        return []

    top_score = max(source["score"] for source in candidates)
    adaptive_floor = max(float(min_similarity), top_score * float(relative_score))
    filtered: list[dict[str, Any]] = []

    for source in candidates:
        if source["score"] < adaptive_floor:
            continue
        if source["score"] < WEAK_SCORE_BOUNDARY:
            overlap = lexical_overlap(
                query,
                f'{source["title"]} {source["content"][:1200]}',
            )
            if overlap < min_lexical_overlap:
                continue
        filtered.append(source)

    # Prefer an https source when two crawls contain the exact same chunk.
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for source in filtered:
        fingerprint = _text_fingerprint(source["content"])
        existing = by_fingerprint.get(fingerprint)
        if existing is None:
            by_fingerprint[fingerprint] = source
            continue
        existing_https = existing["canonical_url"].startswith("https://")
        current_https = source["canonical_url"].startswith("https://")
        if current_https and not existing_https:
            by_fingerprint[fingerprint] = source

    ranked = list(by_fingerprint.values())
    ranked.sort(
        key=lambda source: (
            source["score"] + (0.04 if preferred and source["category"] == preferred else 0.0),
            source["score"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    chunks_per_source: dict[str, int] = {}
    for source in ranked:
        identity = url_identity(source["source_url"]) or source["title"]
        used = chunks_per_source.get(identity, 0)
        if used >= MAX_CHUNKS_PER_SOURCE:
            continue
        chunks_per_source[identity] = used + 1
        selected.append(source)
        if len(selected) >= top_k:
            break

    for index, source in enumerate(selected, start=1):
        source["id"] = f"S{index}"
    return selected


def citation_validation(answer: str, sources: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {str(source.get("id")) for source in sources if source.get("id")}
    references = re.findall(r"\[(S\d+)\]", answer or "", flags=re.IGNORECASE)
    normalized_references = [reference.upper() for reference in references]
    cited_ids = list(dict.fromkeys(normalized_references))
    invalid_ids = [citation for citation in cited_ids if citation not in valid_ids]

    status = "not_required"
    if valid_ids:
        if not cited_ids:
            status = "missing"
        elif invalid_ids:
            status = "invalid"
        else:
            status = "valid"
    return {
        "status": status,
        "cited_ids": [citation for citation in cited_ids if citation in valid_ids],
        "invalid_ids": invalid_ids,
    }


def extractive_fallback(sources: Iterable[dict[str, Any]], limit: int = 900) -> str:
    source_list = list(sources)
    if not source_list:
        return (
            "知识库中暂未找到足够可靠的依据，因此我不能给出确定性结论。"
            "建议换一种问法，或向学校相关业务部门确认。"
        )

    full_text_sources = [
        source for source in source_list if source.get("content_status") != "metadata_only"
    ]
    if not full_text_sources:
        source = source_list[0]
        return (
            f'检索到《{source.get("title", "校园资料")}》的政策入口，但知识库只收录了'
            f'“{source.get("content", "")[:120]}”，没有附件正文，因此无法确认更具体的条件。'
            f'请打开来源原文核对。[{source["id"]}]'
        )

    source = max(
        full_text_sources,
        key=lambda item: (float(item.get("score") or 0.0), len(item.get("content", ""))),
    )
    content = source.get("content", "").strip()
    excerpt = content[:limit] + ("..." if len(content) > limit else "")
    return (
        "自动回答未能正确关联引用。为保证信息可靠，先提供最相关资料原文：\n\n"
        f"{excerpt}\n\n[{source['id']}]"
    )
