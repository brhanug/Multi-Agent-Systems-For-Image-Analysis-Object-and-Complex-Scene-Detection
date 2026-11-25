import sqlite3

DB_PATH = "/home/brhanu/.local/share/label-studio/label_studio.sqlite3"
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM task_completion")
    completions = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM task")
    tasks = cursor.fetchone()[0]
    print(f"Total tasks in database: {tasks}")
    print(f"Total task completions in database: {completions}")
    
    # Query tasks to see what IDs are there
    cursor.execute("SELECT MIN(task_id), MAX(task_id) FROM task_completion")
    min_id, max_id = cursor.fetchone()
    print(f"Completions task_id range: {min_id} to {max_id}")
    
    conn.close()
except Exception as e:
    print(f"Error checking database: {e}")
