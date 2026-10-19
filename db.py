# db.py
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "emotions.db"
)

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name  TEXT,
            timestamp   DATETIME,
            fps         REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER,
            track_id      INTEGER,
            emotion       TEXT,
            category      TEXT,
            dwell_seconds REAL,
            revisit       INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER,
            track_id      INTEGER,
            embedding     TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()

def insert_embedding(session_id, track_id, embedding):
    import json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO face_embeddings (session_id, track_id, embedding) VALUES (?, ?, ?)",
        (session_id, track_id, json.dumps(embedding))
    )
    conn.commit()
    conn.close()

def get_all_embeddings():
    import json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, track_id, embedding FROM face_embeddings")
    rows = [(r[0], r[1], json.loads(r[2])) for r in cursor.fetchall()]
    conn.close()
    return rows

def create_session(video_name, fps):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (video_name, timestamp, fps) VALUES (?, ?, ?)",
        (video_name, datetime.now().isoformat(), fps)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def insert_detection(session_id, track_id, emotion, category, dwell_seconds, revisit=False):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO detections 
           (session_id, track_id, emotion, category, dwell_seconds, revisit) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, track_id, emotion, category, dwell_seconds, int(revisit))
    )
    conn.commit()
    conn.close()

def get_all_detections():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, category, dwell_seconds, revisit FROM detections")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_category_dwell():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT category, AVG(dwell_seconds)
        FROM detections
        WHERE dwell_seconds >= 0
        GROUP BY category
    """)
    rows = dict(cursor.fetchall())
    conn.close()
    return rows