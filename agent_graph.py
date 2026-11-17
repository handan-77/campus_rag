import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import END, StateGraph

from retriever import search_documents
from skills_tools import query_scholarship
from source_utils import citation_validation, extractive_fallback, make_source


_CST = timezone(timedelta(hours=8))
_CAMPUS_HINTS = (
    "学校",
    "学院",
    "校园",
    "图书馆",
    "奖学金",
    "处分",
    "学籍",
    "课程",
    "专业",
    "考试",
    "宿舍",
    "校区",
    "教务",
    "大创",
)


def _get_api_key():
    """按优先级找 API Key：.env 环境变量 -> Streamlit secrets。"""
    load_dotenv()
    key = os.getenv("ZHIPU_API_KEY")
    if key:
        return key
    try:
        import streamlit as _st

        return _st.secrets.get("ZHIPU_API_KEY")
    except Exception:
        return None


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError("未配置 ZHIPU_API_KEY，无法调用语言模型。")
        _MODEL = ChatZhipuAI(model="glm-4-flash", api_key=api_key, temperature=0.1)
    return _MODEL


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    sources: list[dict[str, Any]]
    answer: str
    evidence_status: str
    citation_status: str
    retrieval_top_k: int
    chat_history: list[dict]
    thinking_steps: list[str]
    profile: dict[str, Any]
    tool_result: str
    missing_profile_fields: list[str]


def _history_text(history: list[dict], limit: int = 6) -> str:
    if not history:
        return ""
    lines = [
        f'{message.get("role", "unknown")}: {message.get("content", "")}'
        for message in history[-limit:]
    ]
    return "对话历史：\n" + "\n".join(lines) + "\n"


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _asks_for_scholarship_eligibility(question: str) -> bool:
    if "奖学金" not in question:
        return False
    return any(
        phrase in question
        for phrase in (
            "我能",
            "能申请",
            "是否符合",
            "符合",
            "资格",
            "够不够",
            "能拿",
            "可以申请",
            "能否申请",
            "能否获得",
        )
    )


def _rule_based_intent(question: str) -> str:
    if any(word in question for word in ("新闻", "通知", "公告", "动态")):
        return "news"
    if any(word in question for word in ("专业", "课程", "培养", "招生", "学院介绍")):
        return "major"
    if any(
        word in question
        for word in ("规章", "办法", "条例", "规定", "处分", "奖学金", "学籍", "政策")
    ):
        return "policy"
    if any(hint in question for hint in _CAMPUS_HINTS):
        return "service"
    if re.fullmatch(r"\s*(你好|您好|嗨|hi|hello|谢谢|再见)[！!。.\s]*", question, re.IGNORECASE):
        return "chat"
    return "service"


def classify_intent(state: AgentState):
    history = state.get("chat_history", [])
    thinking = list(state.get("thinking_steps", []))
    thinking.append("分析问题意图")

    prompt = f"""
{_history_text(history)}
用户最新问题：{state["question"]}

判断问题属于哪个类别，只输出一个词：
policy / major / news / service / chat
"""
    try:
        response = _get_model().invoke(prompt)
        raw = str(response.content).strip().lower()
        match = re.search(r"\b(policy|major|news|service|chat)\b", raw)
        intent = match.group(1) if match else "chat"
        thinking.append("使用模型完成意图分类")
    except Exception as exc:
        intent = _rule_based_intent(state["question"])
        thinking.append(f"模型分类不可用，使用本地规则（{type(exc).__name__}）")

    # A broad classifier can mistake concrete campus questions for small talk.
    if intent == "chat" and any(hint in state["question"] for hint in _CAMPUS_HINTS):
        intent = "service"

    thinking.append(f"意图：{intent}")
    return {"intent": intent, "thinking_steps": thinking}


def retrieve_sources(state: AgentState):
    thinking = list(state.get("thinking_steps", []))
    top_k = int(state.get("retrieval_top_k") or 5)
    thinking.append(f"检索校园知识库（{top_k} 条以内）")

    sources = search_documents(
        state["question"],
        top_k=top_k,
        intent=state.get("intent", ""),
    )
    top_score = max((source.get("score", 0.0) for source in sources), default=0.0)
    evidence_status = "none" if not sources else ("strong" if top_score >= 0.65 else "weak")
    thinking.append(f"保留 {len(sources)} 条可追溯来源")
    return {
        "sources": sources,
        "evidence_status": evidence_status,
        "thinking_steps": thinking,
    }


def run_skills(state: AgentState):
    """Only run profile-dependent tools when the user has supplied real values."""
    thinking = list(state.get("thinking_steps", []))
    sources = list(state.get("sources", []))
    question = state["question"]
    profile = state.get("profile") or {}
    updates: dict[str, Any] = {}

    if not _asks_for_scholarship_eligibility(question):
        return {
            "sources": sources,
            "tool_result": "",
            "missing_profile_fields": [],
            "thinking_steps": thinking,
        }

    gpa = _to_float(profile.get("gpa"))
    punishment_count = _to_int(profile.get("punishment_count"))
    missing = []
    if gpa is None:
        missing.append("绩点")
    if punishment_count is None:
        missing.append("处分次数")

    if missing:
        thinking.append("奖学金判断缺少个人资料：" + "、".join(missing))
        updates["missing_profile_fields"] = missing
        updates["tool_result"] = ""
    else:
        result = query_scholarship.invoke(
            {
                "grade": str(profile.get("grade") or "未提供"),
                "gpa": gpa,
                "punishment_count": punishment_count,
            }
        )
        tool_source = make_source(
            {
                "title": "奖学金资格判断工具",
                "content": result,
                "score": 1.0,
                "source_url": "",
            },
            citation_id=f"S{len(sources) + 1}",
            source_type="skill",
        )
        tool_source["category"] = "业务工具"
        tool_source["date"] = ""
        sources.append(tool_source)
        updates["missing_profile_fields"] = []
        updates["tool_result"] = result
        thinking.append("已使用当前用户资料判断奖学金资格")

    updates["sources"] = sources
    updates["thinking_steps"] = thinking
    return updates


def route_after_classify(state: AgentState):
    return "direct_answer" if state.get("intent") == "chat" else "retrieve_sources"


def direct_answer(state: AgentState):
    thinking = list(state.get("thinking_steps", []))
    thinking.append("直接回答，不检索校园知识库")
    prompt = f"""
你是一个友好、简洁的校园助手。请结合对话历史回答用户，但不要虚构学校制度、
办理流程、日期或办事结果。

{_history_text(state.get("chat_history", []))}
用户最新问题：{state["question"]}
"""
    try:
        response = _get_model().invoke(prompt)
        answer = str(response.content).strip()
        thinking.append("回答已生成")
    except Exception as exc:
        answer = "当前无法连接语言模型服务。你可以改为查询校园制度、办事流程或校内通知，我会优先展示知识库中的可追溯资料。"
        thinking.append(f"模型回答不可用（{type(exc).__name__}）")
    return {
        "answer": answer,
        "sources": [],
        "evidence_status": "direct",
        "citation_status": "not_required",
        "thinking_steps": thinking,
    }


def generate_answer(state: AgentState):
    thinking = list(state.get("thinking_steps", []))
    sources = list(state.get("sources", []))
    missing_fields = list(state.get("missing_profile_fields", []))

    if not sources:
        thinking.append("没有足够可靠的来源，返回证据不足提示")
        answer = (
            "知识库中暂未找到足够可靠、可追溯的信息，因此我不能给出确定性结论。"
            "建议补充更具体的时间、校区或业务名称，或向学校相关业务部门确认。"
        )
        if missing_fields:
            answer += (
                "\n\n另外，如需判断你的奖学金资格，请在个人中心补充："
                + "、".join(missing_fields)
                + "。"
            )
        return {
            "answer": answer,
            "sources": [],
            "evidence_status": "none",
            "citation_status": "not_required",
            "thinking_steps": thinking,
        }

    thinking.append("基于可追溯来源生成回答")
    context = "\n\n".join(
        (
            f'[{source["id"]}] {source["title"]}\n'
            f'{source["content"]}\n'
            f'来源类型：{source.get("source_type", "knowledge")} '
            f'日期：{source.get("date") or "未提供"} '
            f'正文状态：{source.get("content_status", "full_text")}'
        )
        for source in sources
    )
    valid_ids = ", ".join(source["id"] for source in sources)
    missing_instruction = ""
    if missing_fields:
        missing_instruction = (
            "\n用户个人资料缺少："
            + "、".join(missing_fields)
            + "。不要替用户假设这些值；如问题涉及资格判断，请明确请用户补充。"
        )

    today = datetime.now(_CST).strftime("%Y-%m-%d")
    prompt = f"""
今天是 {today}。
{_history_text(state.get("chat_history", []))}
用户最新问题：{state["question"]}

参考资料：
{context}

回答规则：
1. 只能依据参考资料回答，不得补充资料外的事实、日期或办理承诺。
2. 每个包含事实结论的句子末尾必须标注来源编号，例如 [S1]。
3. 只能引用这些编号：{valid_ids}。
4. 如果资料不足以回答，明确说“现有资料不足”，不要猜测。
5. 如果资料之间冲突，指出冲突并建议用户核对原始来源。
6. 不要替用户假设个人成绩、处分、年级等信息。
7. 正文状态为 metadata_only 的来源通常只包含“详见附件”等入口信息，
   不能据此编造附件中的条件；如果所有来源都是 metadata_only，请明确提示用户打开原文核对。
{missing_instruction}
"""
    try:
        response = _get_model().invoke(prompt)
        answer = str(response.content).strip()
        validation = citation_validation(answer, sources)
    except Exception as exc:
        answer = "语言模型服务暂时不可用。以下是最相关资料原文：\n\n" + extractive_fallback(sources)
        validation = citation_validation(answer, sources)
        thinking.append(f"模型回答不可用，返回资料原文摘录（{type(exc).__name__}）")

    if validation["status"] == "missing":
        answer = extractive_fallback(sources)
        thinking.append("模型回答缺少引用，已改为资料原文摘录")
    elif validation["status"] == "invalid":
        answer = extractive_fallback(sources)
        thinking.append("模型引用了不存在的来源，已改为资料原文摘录")
    else:
        thinking.append("回答引用校验通过")

    if missing_fields:
        answer += (
            "\n\n如需进一步判断你的奖学金资格，请在个人中心补充："
            + "、".join(missing_fields)
            + "。"
        )

    final_validation = citation_validation(answer, sources)
    return {
        "answer": answer,
        "sources": sources,
        "evidence_status": state.get("evidence_status", "strong"),
        "citation_status": final_validation["status"],
        "thinking_steps": thinking,
    }


builder = StateGraph(AgentState)
builder.add_node("classify", classify_intent)
builder.add_node("retrieve_sources", retrieve_sources)
builder.add_node("run_skills", run_skills)
builder.add_node("generate_answer", generate_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify")
builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "retrieve_sources": "retrieve_sources",
        "direct_answer": "direct_answer",
    },
)
builder.add_edge("retrieve_sources", "run_skills")
builder.add_edge("run_skills", "generate_answer")
builder.add_edge("generate_answer", END)
builder.add_edge("direct_answer", END)

agent = builder.compile()


if __name__ == "__main__":
    print("终端测试模式")
    chat_history = []
    while True:
        question = input("你: ")
        if question.lower() in ("exit", "退出"):
            break
        result = agent.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "thinking_steps": [],
                "retrieval_top_k": 5,
                "profile": {},
            }
        )
        print("\n处理过程:")
        for step in result.get("thinking_steps", []):
            print("  ", step)
        print("\n助手:", result["answer"])
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result["answer"]})
