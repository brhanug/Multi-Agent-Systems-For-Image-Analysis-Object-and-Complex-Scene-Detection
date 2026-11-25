import sqlite3
import json

DB_PATH = "/home/brhanu/.local/share/label-studio/label_studio.sqlite3"
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.id, t.data, tc.result 
        FROM task t
        JOIN task_completion tc ON t.id = tc.task_id
        WHERE t.id <= 300
        ORDER BY t.id ASC
    """)
    rows = cursor.fetchall()
    print(f"Number of retrieved tasks: {len(rows)}")
    
    empty_completions = 0
    non_empty_completions = 0
    for task_id, data_str, result_str in rows:
        result = json.loads(result_str)
        if len(result) == 0:
            empty_completions += 1
        else:
            non_empty_completions += 1
            if non_empty_completions <= 5:
                print(f"Task ID {task_id}: non-empty result count={len(result)}")
                
    print(f"Empty completions: {empty_completions}")
    print(f"Non-empty completions: {non_empty_completions}")
    
    conn.close()
except Exception as e:
    print(f"Error checking database details: {e}")
