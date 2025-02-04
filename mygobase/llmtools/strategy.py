from enum import Enum

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel

from mygobase import BASE_URL, CHAT_MODEL

PROMPT = """
你是動畫台詞劇本分析專家。
请根据当前的問題，生成对当前回答問題的合适策略。

分析步骤：

1. 首先根据当前的問題型態，选择你接下来打算采取的思考检索模式：

FAST: 直接检索模式
- 需要快速确立基本观点
- 想要做出直接明确的回应
- 想要对于某个特定问题进行讨论

ASSOCIATE: 关联检索模式
- 需要对单一事物进行多角度分析或深入解释
- 现有讨论没有进展，需要从单一事物发散思考
- 探索论点的连带影响和延伸问题

返回格式：
{
    "mode": "int",  // 检索模式：FAST/ASSOCIATE
}
"""


class StrategyEnum(Enum):
    FAST = 0
    ASSOCIATE = 1


class Strategy(BaseModel):
    strategy: StrategyEnum


class StrategyPicker:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
        self.temperature = 0.3

    def pick(self, text: str) -> Strategy:
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": PROMPT,
                },
                {
                    "role": "user",
                    "content": f"問題：\n\n{text}",
                },
            ],
            model=CHAT_MODEL,
            stream=False,
            max_tokens=-1,
            temperature=self.temperature,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchema(
                    name="Strategy", schema=Strategy.model_json_schema()
                ),
                type="json_schema",
            ),
        )

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return Strategy.model_validate_json(content)
