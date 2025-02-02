import pickle
import sqlite3
from typing import Iterator

from tqdm import tqdm  # type: ignore

from mygobase.datatypes import Storyboard
from mygobase.model import multimodal_understanding


def namedtuple_factory(cursor, row):
    return Storyboard(*row)


def db_data() -> Iterator[Storyboard]:
    try:
        with sqlite3.connect("db/mygo.db") as conn:
            conn.row_factory = namedtuple_factory
            cursor = conn.cursor()
            res = cursor.execute("""
                            SELECT 
                                id,
                                episode,
                                frame_number, 
                                subtitle,
                                picture
                            FROM 
                                storyboards 
                            ORDER BY 
                                episode, 
                                frame_number
                        """)
            for i in res:
                yield i
    except sqlite3.OperationalError as e:
        print("Failed to open database:", e)


def janus1b_understanding():
    result = {}
    for sb in tqdm(db_data(), total=3946):
        prompt = """
        Main character names and appearance descriptions:

        * Tomori: She has short, purple hair that is slightly messy and covers part of her forehead. Her eyes are large and pink, with visible red veins.
        * Anon: She  has long, straight pink hair that falls over her shoulders. She has large, expressive blue eyes with visible reflections.
        * Soyo: She has long, light brown hair that cascades over her shoulders. She is wearing a dark green school uniform with a white collar and a large bow on the front. Her eyes are a bright blue.
        * Riki: She has long, straight brown hair that falls over her shoulders. Her eyes are large and expressive, with a light purple color.
        * Sakiko: She  has long, wavy hair that is light blue in color. She is wearing a white blouse with long sleeves and a black bow at the collar. She has a gray plaid skirt that reaches just above her knees. She is also wearing white socks and black shoes.
        * Mutsumi: She has long, light-yellow-colored hair that appears to be silver or gray. 
        * Rena: She has short, light-colored hair that appears to be silver or gray. She has a gentle expression on her face, with a slight smile and a hint of blush on her cheeks. Her eyes are large and expressive, with one eye being blue and the other being yellow
        Who is speaker? Just ansmer name
        """
        question = prompt
        answer = multimodal_understanding(sb.picture, question)
        result[sb.id] = answer
    with open("result.pickle", "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    janus1b_understanding()
