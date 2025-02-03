import sqlite3

import sqlite_vec
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel
from sqlite_vec import serialize_float32
from tqdm import tqdm

from mygobase.embedding import EmbeddingExtractor
from mygobase.graph import MyGOKnowledgeGraph
from mygobase.ner import EntityRelationExtractor
from mygobase.summarizer import Report, Summarizer

embedding_extractor = EmbeddingExtractor()
conn = sqlite3.connect("db/mygo.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)


def search_related_entities(entity: str):
    embedding = embedding_extractor.extract(entity)
    with conn:
        res = conn.execute(
            """
            SELECT
                entity
            FROM 
                entity_embedding
            WHERE 
                embedding match ? AND k = 1
            """,
            (serialize_float32(embedding),),
        ).fetchone()[0]
    return res


def search_related_scenes(entity: str):
    embedding = embedding_extractor.extract(entity)
    with conn:
        res = conn.execute(
            """
            SELECT
                scene
            FROM 
                scene_embedding
            WHERE 
                embedding match ? AND k = 3
            """,
            (serialize_float32(embedding),),
        ).fetchall()
    return list([r[0] for r in res])


def search_commnuity_report(entity: str) -> Report | None:
    with conn:
        res = conn.execute(
            """
            SELECT
                community_report.report
            FROM 
                entity_embedding_rowids
            INNER JOIN 
                community_report 
                ON 
                entity_embedding_rowids.community_report_id = community_report.id
            WHERE 
                entity_embedding_rowids.id = ?
            """,
            (entity,),
        ).fetchone()[0]
    report = Report.model_validate_json(res)
    if report.rating > 0:
        return report
    return None


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
        self.MODEL = "gemma-2-9b-it"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def query(self, query: str):
        pbar = tqdm(total=5)
        er_extractor = EntityRelationExtractor()
        extracted_entities = [
            search_related_entities(e.name)
            for e in er_extractor.extract(query).entities
        ]
        pbar.update(1)
        # print(f"#Entities:{len(extracted_entities)}")

        #
        extracted_scenes = []
        for e in extracted_entities:
            for s in search_related_scenes(e):
                extracted_scenes.append(summarizer.summarize(query, s))
        pbar.update(2)
        # print(f"#Scenes:{len(extracted_scenes)}")

        #
        extracted_ners = []
        mygo_graph = MyGOKnowledgeGraph.from_db()
        for e in extracted_entities:
            extracted_ners.append(mygo_graph.query(e))
        pbar.update(3)
        # print(f"#NER:{len(extracted_ners)}")

        #
        commnuity_reports = []
        for e in extracted_entities:
            if (report := search_commnuity_report(e)) is not None:
                commnuity_reports.append(report.model_dump_json())
        # print(f"#Reports:{len(commnuity_reports)}")
        context = f"""
            相關場景：
            {summarizer.summarize(query, "\n".join(extracted_scenes))}
            相關人事物關係：
            {summarizer.summarize(query, "\n".join(extracted_ners))}
            相關背景知識:
            {summarizer.summarize(query, "\n".join(commnuity_reports))}
        """
        pbar.update(4)
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                        基于以下已知信息，请回答问题：

                        已知信息：
                        {context}

                        请详细回答以下问题，如果無法從已知信息中推斷可能答案就說不知道，不要假裝知道：
                        {query}

                        回答格式：
                        {{
                           "answer":<簡短insight_answer>,
                           "explanation": <详细insight_explanation>
                        }}
                        """.format(context=context, query=query),
                }
            ],  # type: ignore
            model=self.MODEL,
            stream=False,
            max_tokens=-1,
            temperature=0.6,
            response_format=json_schema,
        )  # type: ignore
        content: str = chat_completion.choices[0].message.content  # type: ignore
        pbar.update(5)
        return Response.model_validate_json(content)


if __name__ == "__main__":
    query = "為何爽世同意愛音的邀請而加入MyGO樂團？"
    print(f"問：{query}")
    rag = RAG()
    answer = rag.query(query)
    print("答：")
    print(answer.answer)
    print(answer.explanation)
