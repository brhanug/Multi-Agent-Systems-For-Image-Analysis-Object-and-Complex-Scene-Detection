import datetime, sys, os

LOG_FILE = os.path.expanduser("~/thesis_project/logs/progress_log.md")

entry = " ".join(sys.argv[1:]) or "No message given."
timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
with open(LOG_FILE, "a") as f:
    f.write(f"{timestamp} — {entry}\n")

print("✅ Log updated:", entry)
