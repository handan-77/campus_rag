import sys
import types
from types import SimpleNamespace

from source_utils import make_source


fake_retriever = types.ModuleType("retriever")
fake_retriever.search_documents = lambda *args, **kwargs: []
sys.modules.setdefault("retriever", fake_retriever)

import agent_graph as ag


class FakeModel:
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "判断问题属于哪个类别" in prompt:
            return SimpleNamespace(content="policy")
        return SimpleNamespace(content=self.answer)


class RaisingModel:
    def invoke(self, prompt):
        raise ConnectionError("model unavailable")


def sample_source():
    return make_source(
        {
            "title": "综合奖学金评定办法",
            "content": "一等奖学金要求无处分，并达到学校规定的绩点要求。",
            "score": 0.78,
            "source_url": "https://example.edu.cn/scholarship",
        },
        citation_id="S1",
    )


def test_agent_uses_one_retrieval_and_returns_linked_source(monkeypatch):
    model = FakeModel("一等奖学金要求无处分并达到绩点要求。[S1]")
    calls = []

    def fake_search(*args, **kwargs):
        calls.append((args, kwargs))
        return [sample_source()]

    monkeypatch.setattr(ag, "_get_model", lambda: model)
    monkeypatch.setattr(ag, "search_documents", fake_search)

    result = ag.agent.invoke(
        {
            "question": "一等奖学金申请条件是什么？",
            "chat_history": [],
            "thinking_steps": [],
            "retrieval_top_k": 8,
            "profile": {},
        }
    )

    assert len(calls) == 1
    assert calls[0][1]["top_k"] == 8
    assert result["sources"][0]["id"] == "S1"
    assert result["citation_status"] == "valid"
    assert result["missing_profile_fields"] == []


def test_agent_does_not_invent_an_answer_when_no_source_is_available(monkeypatch):
    model = FakeModel("这个答案不应该被生成。[S1]")
    monkeypatch.setattr(ag, "_get_model", lambda: model)
    monkeypatch.setattr(ag, "search_documents", lambda *args, **kwargs: [])

    result = ag.agent.invoke(
        {
            "question": "图书馆今天开放到几点？",
            "chat_history": [],
            "thinking_steps": [],
            "retrieval_top_k": 5,
            "profile": {},
        }
    )

    assert result["sources"] == []
    assert "暂未找到足够可靠" in result["answer"]
    assert len(model.prompts) == 1  # Intent classification only.


def test_eligibility_tool_requires_real_profile_values(monkeypatch):
    model = FakeModel("奖学金申请要求以学校文件为准。[S1]")
    monkeypatch.setattr(ag, "_get_model", lambda: model)
    monkeypatch.setattr(ag, "search_documents", lambda *args, **kwargs: [sample_source()])

    result = ag.agent.invoke(
        {
            "question": "我绩点3.8，能申请一等奖学金吗？",
            "chat_history": [],
            "thinking_steps": [],
            "retrieval_top_k": 5,
            "profile": {},
        }
    )

    assert result["missing_profile_fields"] == ["绩点", "处分次数"]
    assert "请在个人中心补充" in result["answer"]


def test_completed_profile_appends_a_tool_source():
    result = ag.run_skills(
        {
            "question": "我能申请一等奖学金吗？",
            "intent": "policy",
            "sources": [sample_source()],
            "thinking_steps": [],
            "profile": {"gpa": 3.8, "punishment_count": 0, "grade": "大二"},
        }
    )

    assert result["tool_result"]
    assert result["sources"][-1]["source_type"] == "skill"
    assert result["sources"][-1]["id"] == "S2"


def test_model_outage_falls_back_to_retrieved_source_text(monkeypatch):
    monkeypatch.setattr(ag, "_get_model", lambda: RaisingModel())
    monkeypatch.setattr(ag, "search_documents", lambda *args, **kwargs: [sample_source()])

    result = ag.agent.invoke(
        {
            "question": "一等奖学金申请条件是什么？",
            "chat_history": [],
            "thinking_steps": [],
            "retrieval_top_k": 5,
            "profile": {},
        }
    )

    assert "语言模型服务暂时不可用" in result["answer"]
    assert "一等奖学金要求无处分" in result["answer"]
    assert result["citation_status"] == "valid"
