from source_utils import (
    citation_validation,
    extract_document_date,
    extractive_fallback,
    select_relevant_sources,
    url_identity,
)


def test_http_and_https_urls_share_identity():
    assert url_identity("http://example.edu.cn/info/1") == url_identity(
        "https://example.edu.cn/info/1/"
    )


def test_extract_document_date_only_returns_explicit_dates():
    assert (
        extract_document_date("通知", "发布日期：2026年9月3日")
        == "2026-09-03"
    )
    assert extract_document_date("校园通知", "本通知长期有效") == ""


def test_select_sources_deduplicates_scheme_and_filters_relative_noise():
    raw = [
        {
            "title": "综合奖学金评定办法",
            "content": "一等奖学金申请条件包括无处分且绩点达到要求。",
            "score": 0.7573,
            "source_url": "https://example.edu.cn/scholarship",
        },
        {
            "title": "综合奖学金评定办法",
            "content": "一等奖学金申请条件包括无处分且绩点达到要求。",
            "score": 0.7573,
            "source_url": "http://example.edu.cn/scholarship",
        },
        {
            "title": "规章制度",
            "content": "博士研究生参加学术会议的资助办法。",
            "score": 0.6790,
            "source_url": "https://example.edu.cn/conference",
        },
    ]

    sources = select_relevant_sources(raw, "一等奖学金申请条件", top_k=5)

    assert len(sources) == 1
    assert sources[0]["id"] == "S1"
    assert sources[0]["source_url"].startswith("https://")
    assert sources[0]["category"] == "政策制度"


def test_select_sources_rejects_weak_lexically_unrelated_match():
    raw = [
        {
            "title": "研究生新生入学须知",
            "content": "银行卡由学校统一寄送，毕业证明书可以申请补办。",
            "score": 0.6233,
            "source_url": "https://example.edu.cn/notice",
        }
    ]

    assert select_relevant_sources(raw, "校园卡丢失怎么办", top_k=5) == []


def test_citation_validation_reports_missing_and_invalid_ids():
    sources = [{"id": "S1"}]

    assert citation_validation("结论。", sources)["status"] == "missing"
    assert citation_validation("结论。[S2]", sources)["status"] == "invalid"
    assert citation_validation("结论。[S1]", sources)["status"] == "valid"


def test_extractive_fallback_always_has_a_source_when_available():
    answer = extractive_fallback(
        [{"id": "S1", "content": "这是最相关的校内资料原文。"}]
    )

    assert "这是最相关的校内资料原文。" in answer
    assert answer.endswith("[S1]")


def test_metadata_only_source_does_not_claim_attachment_details():
    answer = extractive_fallback(
        [
            {
                "id": "S1",
                "title": "综合奖学金评定办法",
                "content": "《综合奖学金评定办法》详见附件",
                "content_status": "metadata_only",
                "score": 0.8,
            }
        ]
    )

    assert "没有附件正文" in answer
    assert "详见附件" in answer
    assert answer.endswith("[S1]")
