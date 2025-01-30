import sqlite3

statements = [
    """
    CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY, 
                episode INT NOT NULL, 
                frame_number INT NOT NULL, 
                subtitle TEXT NOT NULL,
                picture BLOB NOT NULL
    );
""",
    """
    CREATE INDEX IF NOT EXISTS idx_episode 
    ON episodes (episode);
""",
    """
    CREATE INDEX IF NOT EXISTS idx_episode_frame_number 
    ON episodes (episode, frame_number);
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

# try:
#     with sqlite3.connect("db/mygo.db") as conn:
#         cursor = conn.cursor()
#         ep = 1
#         fn = 85
#         pic = (65).to_bytes()
#         sub = "Hello world"
#         cursor.execute(
#             "INSERT INTO episodes (episode, frame_number, subtitle, picture) VALUES (?,?,?,?)",
#             (ep, fn, sub, pic),
#         )
# except sqlite3.OperationalError as e:
#     print("Failed to open database:", e)
