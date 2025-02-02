import sqlite3

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
]
with sqlite3.connect("db/mygo.db") as conn:
    cursor = conn.cursor()
    for statement in statements:
        try:
            cursor.execute(statement)
            conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
        else:
            print(f"statement ok: {statement}")
