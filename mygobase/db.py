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
]
try:
    with sqlite3.connect("db/mygo.db") as conn:
        cursor = conn.cursor()
        for statement in statements:
            cursor.execute(statement)
        conn.commit()
except sqlite3.OperationalError as e:
    print("Failed to open database:", e)
