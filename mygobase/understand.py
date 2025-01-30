import sqlite3
from dataclasses import dataclass
from typing import Iterator

from mygobase.model import multimodal_understanding


@dataclass
class Storyboard:
    episode: int
    frame_number: int
    subtitle: str
    picture: bytes


def namedtuple_factory(cursor, row):
    return Storyboard(*row)


def db_data() -> Iterator[Storyboard]:
    try:
        with sqlite3.connect("db/mygo.db") as conn:
            conn.row_factory = namedtuple_factory
            cursor = conn.cursor()
            res = cursor.execute("""
                            SELECT 
                                episode,
                                frame_number, 
                                subtitle,
                                picture
                            FROM 
                                storyboards 
                            ORDER BY 
                                episode, 
                                frame_number
                            LIMIT 10
                        """)
            for i in res:
                yield i
    except sqlite3.OperationalError as e:
        print("Failed to open database:", e)


if __name__ == "__main__":
    for sb in db_data():
        question = f"台詞:{sb.subtitle}"
        answer = multimodal_understanding(sb.picture, question)
        print(sb.episode, sb.frame_number, answer)
