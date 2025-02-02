import sqlite3

import sqlite_vec

statements = [
    """
    CREATE TABLE IF NOT EXISTS storyboards (
                id INTEGER PRIMARY KEY, 
                episode INT NOT NULL, 
                frame_number INT NOT NULL, 
                subtitle TEXT NOT NULL,
                picture BLOB NOT NULL
    );
""",
    """
    CREATE INDEX IF NOT EXISTS idx_episode 
    ON storyboards (episode);
""",
    """
    CREATE INDEX IF NOT EXISTS idx_episode_frame_number 
    ON storyboards (episode, frame_number);
""",
    """
    ALTER TABLE storyboards ADD COLUMN role TEXT
""",
    """
    ALTER TABLE storyboards ADD COLUMN scene INT
""",
    """
    CREATE TABLE IF NOT EXISTS ner (
                id INTEGER PRIMARY KEY, 
                scene INT NOT NULL, 
                ner JSON NOT NULL,
                FOREIGN KEY(scene) REFERENCES storyboards(scene)
    );
""",
    """
    CREATE VIRTUAL TABLE entity_embedding USING vec0(
        entity TEXT PRIMARY KEY,
        embedding FLOAT[768]
    );
""",
]
with sqlite3.connect("db/mygo.db") as conn:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cursor = conn.cursor()
    for statement in statements:
        try:
            cursor.execute(statement)
            conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
        else:
            print(f"statement ok: {statement}")
