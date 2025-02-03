import sqlite3

import polars as pl
import sqlite_vec
from sqlite_vec import serialize_float32
from tqdm import tqdm

from mygobase.embedding import EmbeddingExtractor
from mygobase.ner import load_scenes


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


if __name__ == "__main__":
    scene_docs = load_scenes(load_df())
    embedd_extractor = EmbeddingExtractor()
    print(len(scene_docs))
    embeddings = {}
    for scene_id, script in tqdm(scene_docs):
        embedding = embedd_extractor.extract(script)
        embeddings[script] = embedding
    print(len(embeddings))
    with sqlite3.connect("db/mygo.db") as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        for script, embedding in embeddings.items():
            try:
                conn.execute(
                    """
                        INSERT INTO scene_embedding (scene, embedding)
                        VALUES (?, ?)
                    """,
                    (script, serialize_float32(embedding)),
                )
            except sqlite3.OperationalError as e:
                if (
                    e.args[0]
                    != "UNIQUE constraint failed on entity_embedding primary key"
                ):
                    print(e)
