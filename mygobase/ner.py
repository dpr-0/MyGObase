import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

import polars as pl
import sqlite_vec
from pydantic import BaseModel, ConfigDict
from sqlite_vec import serialize_float32
from tqdm import tqdm

from mygobase import DB_PATH
from mygobase.llmtools.embedding import EmbeddingExtractor
from mygobase.llmtools.er import Entity, EntityRelationExtractor, Relations


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
                ORDER BY
                    scene,
                    frame_number
                """,
            connection=conn,
        )
        return df


def load_scenes(df: pl.DataFrame) -> List[tuple[int, str]]:
    scene_docs: List[tuple[int, str]] = []
    for (scene_id,), scene_db in df.group_by("scene", maintain_order=True):
        script = ""
        for role, subtitle in scene_db.select(["role", "subtitle"]).iter_rows():
            script += f"{role.strip()}:{subtitle.strip()}\n"
        scene_docs.append((scene_id, script))  # type: ignore
    print("load scenes completed")
    return scene_docs


class Content(BaseModel):
    title: str
    content: str

    model_config = ConfigDict(frozen=True)


class Contents(BaseModel):
    contents: List[Content]


replace_table = {
    "高松燈": "燈",
    "小灯": "燈",
    "小燈": "燈",
    "小 lamp": "燈",
    "lamp": "燈",
    "灯": "燈",
    "Lamp": "燈",
    "小 祥": "祥子",
    "豐川祥子": "祥子",
    "小祥": "祥子",
    "初华": "初華",
    "小愛": "愛音",
    "千早": "愛音",
    "爱音": "愛音",
    "乐奈": "樂奈",
    "千早愛音": "愛音",
    "Tomorin": "燈",
    "Rikk": "立希",
    "Rikki": "立希",
    "椎名立希": "立希",
    "长崎": "爽世",
    "Soyorin": "爽世",
    "小睦": "睦",
    "Cryochic": "CRYCHIC",
    "Crychic": "CRYCHIC",
    "notebook": "筆記本",
    "other": "其他",
    "lamp同学的家里": "燈同学的家里",
    "be斯手": "贝斯手",
    "BES": "贝斯",
    "祐天寺若麥": "祐天寺",
    "small 灯": "燈",
}


def replace(entity: Entity) -> Entity:
    entity = entity.strip()
    if entity in replace_table:
        return replace_table[entity]
    return entity


if __name__ == "__main__":
    scenes = load_scenes(load_df())

    er_extractor = EntityRelationExtractor()
    all_entities: Dict[Entity, List[Content]] = defaultdict(list)
    all_ners: Dict[int, Tuple[List[Entity], Relations]] = {}
    for scene_id, script in tqdm(scenes, desc="extracting entity and relation"):
        try:
            entities = er_extractor.extract_entities(script).entities
            relations = er_extractor.extract_relations(script, entities)

            entities = list(map(replace, entities))
            for relation in relations.relations:
                relation.source = replace(relation.source)
                relation.target = replace(relation.target)

            content = Content(title=", ".join(entities), content=script)
            for entity in entities:
                all_entities[entity].append(content)
            all_ners[scene_id] = (entities, relations)
        except Exception as e:
            print(e)

    embeddings: Dict[Entity, List[float]] = {}
    embedd_extractor = EmbeddingExtractor()
    for entity in all_entities.keys():
        try:
            embeddings[entity] = embedd_extractor.extract(entity)
        except Exception as e:
            print(e)

    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("DELETE FROM entity_embedding")
        conn.execute("DELETE FROM entity")
        conn.execute("DELETE FROM ner")
        conn.commit()

        for entity, contents in tqdm(
            all_entities.items(), desc="inserting entity embeddings"
        ):
            contents = Contents(contents=contents)

            conn.execute(
                """
                INSERT INTO entity_embedding (entity, embedding)
                VALUES (?, ?)
            """,
                (
                    entity,
                    serialize_float32(embeddings[entity]),
                ),
            )
            conn.execute(
                """
                INSERT INTO entity (entity, content)
                VALUES (?, ?)
            """,
                (
                    entity,
                    contents.model_dump_json(),
                ),
            )

        for scene_id, (_, relations) in tqdm(
            all_ners.items(), desc="inserting relations"
        ):
            conn.execute(
                """
                INSERT INTO ner (scene, ner)
                VALUES (?, ?)
                """,
                (scene_id, relations.model_dump_json()),
            )
