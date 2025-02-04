from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel

from mygobase import BASE_URL, CHAT_MODEL

ENTITY_EXTRACTION_PROMPT = """
我想要构建知识图谱，请从给定的文本中提取核心实体，遵循以下指南：
- 只选择最核心、出现频率最高的2到4个实体
- 请严格判断，如果找到的实体不能反映所有评论整体反映的主题，或者评论是明显无意义的重复评论，无需提取实体，entities请返回空列表
- 将简写的实体名补充为完整形式
- 提取的实体需要和话题或评论整体反映的主题紧密相关。请提取具体的、有实际意义的实体，避免空泛的概念，实体应该是独立的名词形式
- 实体字串去掉前后空格
- 中文回答实体名稱

输出格式：
   以 JSON 格式输出结果，结构如下：
   ```json
   {
     "entities": ["实体1", "实体2"...],
   }
   ```

请分析给定的文本，并直接输出json，无需额外的解释说明。
"""

RELATION_EXTRACTION_PROMPT = """
我想要构建知识图谱，请根据给定的实体列表，从文本中提取这些实体之间的关系，遵循以下指南：
- 只需判断给定的实体列表中这些实体的关系，source和target都应该在给定的实体中
- 每个关系应该反映的是评论的某种观点，包含这组实体之间完整的上下文、逻辑联系和背景信息，避免提取孤立的、缺乏意义的关系
- 评论内容是围绕话题展开的讨论，所选的评论有相似的主题，所以请选择这组评论中最为典型、能够反映整体的实体关系
- 一组实体间只能有一个关系。请严格判断，不是所有的实体都需要有联系，如果没有找到有效关系，relations请返回空列表
- 实体和关系字串去掉前後空格
- 中文回答实体與关系名稱

输出格式：
   以 JSON 格式输出结果，结构如下：
   ```json
   {
     "relations": [
       {
         "source": "实体1",
         "target": "实体2",
         "relation": "关系描述"
       },
       // 更多关系...
     ]
   }
   ```

请分析给定的文本，并直接输出json，无需额外的解释说明。
"""


class Relation(BaseModel):
    source: str
    target: str
    relation: str


class Relations(BaseModel):
    relations: List[Relation]


type Entity = str


class Entities(BaseModel):
    entities: List[Entity]


class EntityRelationExtractor:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
        self.temperature = 0.3

    def extract_entities(self, text: str) -> Entities:
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": ENTITY_EXTRACTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"请从以下文本中提取与话题相关的核心实体：\n\n{text}",
                },
            ],
            model=CHAT_MODEL,
            stream=False,
            max_tokens=-1,
            temperature=self.temperature,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchema(
                    name="Entities", schema=Entities.model_json_schema()
                ),
                type="json_schema",
            ),
        )

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return Entities.model_validate_json(content)

    def extract_relations(self, text: str, entities: List[str]) -> Relations:
        entities_str = ", ".join(entities)
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": RELATION_EXTRACTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"已知实体列表：{entities_str}\n\n请从以下文本中提取这些实体之间的关系：\n\n{text}",
                },
            ],  # type: ignore
            model=CHAT_MODEL,
            stream=False,
            max_tokens=-1,
            temperature=self.temperature,
            response_format=ResponseFormatJSONSchema(
                json_schema=JSONSchema(
                    name="Relations", schema=Relations.model_json_schema()
                ),
                type="json_schema",
            ),
        )

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return Relations.model_validate_json(content)
