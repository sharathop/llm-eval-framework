import sqlite3
import json
from datetime import datetime

DB_PATH = "data/evaluations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT,
            question TEXT,
            llm_response TEXT,
            final_verdict TEXT,
            cosine_score REAL,
            bert_score REAL,
            nli_label TEXT,
            nli_score REAL,
            fluency_verdict TEXT,
            full_result TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_evaluation(context, question, llm_response, result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations (
            context, question, llm_response,
            final_verdict, cosine_score, bert_score,
            nli_label, nli_score, fluency_verdict,
            full_result, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        context,
        question,
        llm_response,
        result["final_verdict"],
        result["cosine"]["score"],
        result["bert_score"]["score"],
        result["nli"]["label"],
        result["nli"]["score"],
        result["fluency"]["verdict"],
        json.dumps(result),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def get_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows