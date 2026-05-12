"""
llm_client.py — LLM 调用客户端 + 响应解析工具

职责：
1. query()                        调用 DeepSeek API，返回模型回复
2. extract_string()               从回复中解析 <表达式> 和 explanation
3. convert_fields_to_lowercase()  将表达式中的字段名统一转小写
4. contains_future_function()     检测表达式是否含未来函数（负窗口）
5. contains_too_many_fields()     检测表达式是否使用了过多字段

所有函数均无状态，可在多线程中安全调用。
"""

import re
import os
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def query(question: str) -> tuple[str, str]:
    """
    调用 DeepSeek API。

    Args:
        question: 发送给模型的提示词

    Returns:
        (response_text, response_text): 两个值相同，保持与原接口兼容
                                        （原 Qwen 版本会返回 think + answer 两段）
    """
    client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user",   "content": question},
        ],
        temperature=0.6,
        max_tokens=3000,
    )
    resp = response.choices[0].message.content
    print("LLM 回复:", resp)
    return resp, resp


def extract_string(text: str, brackets: str) -> str:
    """
    从 LLM 回复中提取因子表达式或解释。

    Args:
        text:     LLM 回复文本
        brackets: 'alpha'  → 提取 <表达式>
                  'reason' → 提取 explanation: 之后的内容

    Returns:
        提取到的字符串，失败时返回空列表 []
    """
    if brackets == "alpha":
        pattern = r"<([^>]+)>"
        matches = re.findall(pattern, text)
        return matches[0] if matches else []
    elif brackets == "reason":
        match = re.search(r"explanation:(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else []
    else:
        print("brackets 参数错误，应为 'alpha' 或 'reason'")
        return []


def convert_fields_to_lowercase(text: str, fields: list) -> str:
    """
    将表达式中出现的字段名统一转为小写。
    （防止 LLM 输出大写的 Close / VOLUME 等）
    """
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, fields)) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: m.group(0).lower(), text)


def contains_future_function(expression: str) -> bool:
    """
    检测表达式中是否存在负时间窗口（未来函数）。

    例：Ref(close, -5) → True（非法，会用到未来数据）
    """
    pattern = r",\s*-\d+"
    return bool(re.search(pattern, expression))


def contains_too_many_fields(
    expression: str,
    ops_keys: list,
    max_fields: int = 10,
) -> bool:
    """
    统计表达式中使用的算子数量，超过 max_fields 则返回 True。

    Args:
        expression: 因子表达式字符串
        ops_keys:   所有合法算子名称列表（Operators 的方法名）
        max_fields: 算子数量上限（默认 10）
    """
    expr_lower = expression.lower()
    used = set()
    for op in ops_keys:
        if re.search(r"\b" + re.escape(op.lower()) + r"\b", expr_lower):
            used.add(op)
    return len(used) > max_fields
