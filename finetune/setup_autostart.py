"""
finetune/setup_autostart.py
Registers auto_train.py with Windows Task Scheduler.
Run once as Administrator: python finetune\setup_autostart.py
"""

import subprocess
import sys
from pathlib import Path

THIS_DIR    = Path(__file__).parent.absolute()
python_exe  = sys.executable
script_path = THIS_DIR / "auto_train.py"
task_name   = "JARVIS_V3_AutoTrain"

print(f"Python:  {python_exe}")
print(f"Script:  {script_path}")
print(f"Task:    {task_name}")

result = subprocess.run([
    "schtasks", "/create",
    "/tn", task_name,
    "/tr", f'"{python_exe}" "{script_path}"',
    "/sc", "onlogon",
    "/rl", "highest",
    "/f",
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n[OK] '{task_name}' registered!")
    print("     Starts automatically on every Windows login.")
    print(f"\nTo remove: schtasks /delete /tn {task_name} /f")
else:
    print(f"\n[ERROR] {result.stderr}")
    print("Run as Administrator.")