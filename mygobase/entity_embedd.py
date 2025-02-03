import sqlite3

import sqlite_vec
from sqlite_vec import serialize_float32

from mygobase.embedding import EmbeddingExtractor
from mygobase.ner import load_ner

if __name__ == "__main__":
    ners = load_ner()
    embedd_extractor = EmbeddingExtractor()
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

        for entity, embedding in embeddings.items():
            try:
                conn.execute(
                    """
                    INSERT INTO entity_embedding (entity, embedding)
                    VALUES (?, ?)
                """,
                    (entity, serialize_float32(embedding)),
                )
            except sqlite3.OperationalError as e:
                if (
                    e.args[0]
                    != "UNIQUE constraint failed on entity_embedding primary key"
                ):
                    print(e)
