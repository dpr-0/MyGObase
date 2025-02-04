from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel
from tqdm import tqdm

from mygobase import BASE_URL, CHAT_MODEL
from mygobase.graph import Retriever
from mygobase.llmtools.strategy import StrategyEnum, StrategyPicker
from mygobase.llmtools.summarizer import Summarizer

summarizer = Summarizer()


class Response(BaseModel):
    answer: str
    explanation: str


json_schema = ResponseFormatJSONSchema(
    json_schema=JSONSchema(
        name="response", description="", schema=Response.model_json_schema()
    ),
    type="json_schema",
)


class RAG:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
        self.retriever = Retriever()
        self.strategy_picker = StrategyPicker()
        self.summarizer = Summarizer()

    def query(self, query: str):
        pbar = tqdm(total=3)
        strategy = self.strategy_picker.pick(query).strategy
        pbar.update(1)
        match strategy:
            case StrategyEnum.FAST:
                contexts = self.retriever.fast_retrieve(query)
            case StrategyEnum.ASSOCIATE:
                contexts = self.retriever.associate_retrieve(query)
            case _:
                raise NotImplementedError

        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                        基于以下已知信息，请回答问题：

                        已知劇情：
                        {contexts}

                        请详细回答以下问题，如果無法從已知信息中推斷可能答案就說不知道，不要假裝知道：
                        {query}

                        回答格式：
                        {{
                           "answer":<簡短insight_answer>,
                           "explanation": <详细insight_explanation>
                        }}
                        """.format(
                        contexts=contexts.generate_context(),
                        query=query,
                    ),
                }
            ],  # type: ignore
            model=CHAT_MODEL,
            stream=False,
            max_tokens=-1,
            temperature=0.3,
            response_format=json_schema,
        )  # type: ignore
        content: str = chat_completion.choices[0].message.content  # type: ignore
        pbar.update(1)
        return Response.model_validate_json(content)


if __name__ == "__main__":
    # query = "為何爽世同意愛音組新樂團？"
    query = "為何祥子找燈加入樂團？"
    print(f"問：{query}")
    rag = RAG()
    answer = rag.query(query)
    print("答：")
    print(answer.answer)
    print(answer.explanation)
