import sqlite3
from typing import List

import sqlite_vec
from openai import OpenAI
from sqlite_vec import serialize_float32

from mygobase.ner import NER


class EmbeddingExtractor:
    def __init__(self) -> None:
        self.MODEL = "text-embedding-nomic-embed-text-v1.5@f32"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

    def extract(self, text: str):
        resp = self.client.embeddings.create(
            input=text, model=self.MODEL, encoding_format="float"
        )
        embedding = resp.data[0].embedding

        # print(len(embedding))
        return embedding


def load_ner() -> List[NER]:
    with sqlite3.connect("db/mygo.db") as conn:
        res = conn.execute("""
            SELECT
                ner
            FROM ner
        """).fetchall()
    ners = []
    for (ner_json,) in res:
        ners.append(NER.model_validate_json(ner_json))
    return ners


if __name__ == "__main__":
    embedd_extractor = EmbeddingExtractor()
    ners = load_ner()
    embeddings = {}
    for ner in ners:
        for e in ner.entities:
            if e.name in embeddings:
                continue
            embedding = embedd_extractor.extract(e.name)
            embeddings[e.name] = embedding

    with sqlite3.connect("db/mygo.db") as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        for entity, embedd in embeddings.items():
            try:
                conn.execute(
                    """
                    INSERT INTO entity_embedding (entity, embedding)
                    VALUES (?, ?)
                """,
                    (entity, serialize_float32(embedd)),
                )
            except sqlite3.OperationalError as e:
                if (
                    e.args[0]
                    != "UNIQUE constraint failed on entity_embedding primary key"
                ):
                    print(e)
