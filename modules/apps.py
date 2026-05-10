"""
modules/apps.py
Handles launching applications on Windows.
"""

import re
import subprocess
import os


# ── App registry — add YOUR apps here ─────────────────────────────────────────
# Format: "what you say": "what to run"
APP_REGISTRY = {
    # Browsers
    "chrome":           "chrome",
    "google chrome":    "chrome",
    "firefox":          "firefox",
    "edge":             "msedge",
    "browser":          "chrome",

    # Dev tools
    "vs code":          "code",
    "vscode":           "code",
    "visual studio code": "code",
    "terminal":         "wt",           # Windows Terminal
    "powershell":       "powershell",
    "cmd":              "cmd",
    "git bash":         "git-bash",
    "jupyter":          "jupyter notebook",
    "pycharm":          "pycharm64",

    # Office / productivity
    "word":             "winword",
    "excel":            "excel",
    "powerpoint":       "powerpnt",
    "notepad":          "notepad",
    "notepad++":        "notepad++",
    "calculator":       "calc",
    "paint":            "mspaint",

    # Communication
    "discord":          "discord",
    "slack":            "slack",
    "teams":            "teams",
    "zoom":             "zoom",
    "whatsapp":         "whatsapp",
    "telegram":         "telegram",
    "outlook":          "outlook",

    # Media & entertainment
    "spotify":          "spotify",
    "vlc":              "vlc",
    "steam":            "steam",
    "epic games":       "epicgameslauncher",
    "obs":              "obs64",
    "premiere":         "premiere",
    "photoshop":        "photoshop",

    # System
    "task manager":     "taskmgr",
    "settings":         "ms-settings:",
    "control panel":    "control",
    "file explorer":    "explorer",
    "explorer":         "explorer",
    "files":            "explorer",
    "bluetooth":        "ms-settings:bluetooth",
    "wifi":             "ms-settings:network-wifi",
    "display settings": "ms-settings:display",

    # Add your own apps:
    # "my app": "myapp.exe",
}


def launch_app(name: str) -> str:
    """Launch an app by name. Fuzzy matches the registry."""
    name_lower = name.lower().strip()

    # Exact match
    if name_lower in APP_REGISTRY:
        cmd = APP_REGISTRY[name_lower]
        return _run(cmd, name)

    # Partial match
    for key, cmd in APP_REGISTRY.items():
        if key in name_lower or name_lower in key:
            return _run(cmd, key)

    # Try running directly (in case it's an installed app)
    return _run(name_lower, name)


def _run(cmd: str, display_name: str) -> str:
    """Actually run the command."""
    try:
        if cmd.startswith("ms-"):
            os.startfile(cmd)
        else:
            subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        return f"Launching {display_name} now."
    except Exception as e:
        return f"I couldn't launch {display_name}. Error: {e}"


def parse_and_handle(text: str) -> str | None:
    """
    Returns response string if this module handles it, else None.
    """
    text_lower = text.lower().strip()

    # Match: "open/launch/start/run X"
    m = re.search(
        r"\b(open|launch|start|run)\b\s+(.+?)(?:\s+app|\s+application|\s+program)?$",
        text_lower
    )
    if m:
        target = m.group(2).strip()
        # Don't steal file/folder intents
        if any(w in target for w in ["file", "folder", "document", "my "]):
            return None
        return launch_app(target)

    return None
