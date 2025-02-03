import sqlite3
from typing import List

import polars as pl
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore


class Entity(BaseModel):
    name: str
    type: str


class Relation(BaseModel):
    subject: str
    relation: str
    object: str


class NER(BaseModel):
    entities: List[Entity]
    relations: List[Relation]


json_schema = ResponseFormatJSONSchema(
    json_schema=JSONSchema(name="ner", description="", schema=NER.model_json_schema()),
    type="json_schema",
)


class EntityRelationExtractor:
    def __init__(self) -> None:
        self.MODEL = "gemma-2-9b-it"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def extract(self, text: str):
        chat_completion: ChatCompletion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                        請從以下動畫台詞腳本中 提取實體 (entities) 與關係 (relations)，並以結構化格式輸出。

                        📌 任務要求：
                        
                        實體類型 (Entities)

                        角色 (Character)：出現在對話中的人物名稱
                        物品 (Object)：台詞中提及的特定物品 (如飲料、樂器等)
                        行為 (Action)：角色執行的動作 (如「點單」、「試彈」等)
                        概念 (Concept)：抽象概念 (如「社交網絡」)
                        
                        關係類型 (Relations)

                        角色之間的關係 (Character Relationships) (如「朋友」、「隊友」等)
                        角色與物品的關係 (Character-Object Relations) (如「點了什麼飲料」)
                        角色與行為的關係 (Character-Action Relations) (如「正在找樂隊成員」)
                        角色與概念的關係 (Character-Concept Relations) (如「討論某個概念」)
                        
                        輸出格式：JSON
                        範例Output:
                        
                        {
                            "entities": [
                                {"type": "Character", "name": "愛音"},
                                {"type": "Character", "name": "爽世"},
                                {"type": "Character", "name": "立希"},
                                {"type": "Character", "name": "樂奈"},
                                {"type": "Object", "name": "格雷伯爵茶"},
                                {"type": "Concept", "name": "社交網絡"},
                                {"type": "Action", "name": "找樂隊成員"}
                            ],
                            "relations": [
                                {"subject": "爽世", "relation": "點了", "object": "熱格雷伯爵茶"},
                                {"subject": "愛音", "relation": "尋找", "object": "樂隊成員"},
                                {"subject": "愛音", "relation": "討論", "object": "社交網絡"},
                                {"subject": "立希", "relation": "送上", "object": "兩杯熱格雷伯爵茶"}
                            ]
                        }
                """,
                },
                {
                    "role": "user",
                    "content": f"動畫台詞腳本:\n{text}\n\n請按照上述要求提取實體與關係，並以 JSON 格式輸出結果。",
                },
            ],  # type: ignore
            model=self.MODEL,
            stream=False,
            max_tokens=-1,
            temperature=0.3,
            response_format=json_schema,
        )  # type: ignore

        content: str = chat_completion.choices[0].message.content  # type: ignore
        return NER.model_validate_json(content)


def load_df():
    with sqlite3.connect("db/mygo.db") as conn:
        df = pl.read_database(
            """
                SELECT
                    id,
                    scene,
                    frame_number,
                    role,
                    subtitle
                FROM
                    storyboards
                WHERE 
                    scene NOT IN (SELECT scene FROM ner)
                ORDER BY
                    scene,
                    frame_number
                """,
            connection=conn,
        )
        return df


def load_scenes(df: pl.DataFrame) -> List[tuple[int, str]]:
    scene_docs = []
    for (scene_id,), scene_db in df.group_by("scene", maintain_order=True):
        script = ""
        for role, subtitle in scene_db.select(["role", "subtitle"]).iter_rows():
            script += f"{role}:{subtitle}\n"
        scene_docs.append((scene_id, script))
    return scene_docs  # type: ignore


def load_ner(path: str = "db/mygo.db") -> List[NER]:
    with sqlite3.connect(path) as conn:
        res = conn.execute("""
            SELECT
                ner
            FROM ner
        """).fetchall()
    ners = []
    for (ner_json,) in res:
        ners.append(NER.model_validate_json(ner_json))
    return ners


def insert_ner(scene_id: int, ner: NER):
    try:
        with sqlite3.connect("db/mygo.db") as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ner (scene, ner)
                VALUES (?, ?)
                """,
                (scene_id, ner.model_dump_json()),
            )
    except sqlite3.OperationalError as e:
        print(e)


if __name__ == "__main__":
    scene_docs = load_scenes(load_df())
    print("load scenes completed")
    extractor = EntityRelationExtractor()
    for scene_id, script in tqdm(scene_docs):
        try:
            ner = extractor.extract(script)
        except Exception as e:
            print(e)
        else:
            insert_ner(scene_id, ner)
