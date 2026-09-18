import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("dashboard/index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'id="secDemo"' in line:
        print(f"=== Found at Line {i+1} ===")
        print("".join(lines[i:i+60]))
        break
