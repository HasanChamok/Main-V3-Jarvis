"""
modules/apps.py
Launches applications by voice command on Windows.
"""

import re
import subprocess
import os
import shutil


APP_REGISTRY = {
    # Browsers
    "chrome":               "chrome",
    "google chrome":        "chrome",
    "firefox":              "firefox",
    "edge":                 "msedge",
    "microsoft edge":       "msedge",
    "browser":              "chrome",

    # Dev tools
    "vs code":              "code",
    "vscode":               "code",
    "visual studio code":   "code",
    "visual studio":        "code",
    "terminal":             "wt",
    "windows terminal":     "wt",
    "powershell":           "powershell",
    "cmd":                  "cmd",
    "command prompt":       "cmd",
    "git bash":             "git-bash",
    "jupyter":              "jupyter notebook",
    "pycharm":              "pycharm64",

    # Office
    "word":                 "winword",
    "microsoft word":       "winword",
    "excel":                "excel",
    "microsoft excel":      "excel",
    "powerpoint":           "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "notepad":              "notepad",
    "notepad++":            "notepad++",
    "calculator":           "calc",
    "paint":                "mspaint",

    # Communication
    "discord":              "discord",
    "slack":                "slack",
    "teams":                "teams",
    "microsoft teams":      "teams",
    "zoom":                 "zoom",
    "whatsapp":             "whatsapp",
    "telegram":             "telegram",
    "outlook":              "outlook",

    # Media
    "spotify":              "spotify",
    "vlc":                  "vlc",
    "steam":                "steam",
    "epic games":           "epicgameslauncher",
    "obs":                  "obs64",
    "youtube":              "https://youtube.com",

    # System
    "task manager":         "taskmgr",
    "settings":             "ms-settings:",
    "control panel":        "control",
    "control":              "control",
    "file explorer":        "explorer",
    "explorer":             "explorer",
    "files":                "explorer",
    "bluetooth":            "ms-settings:bluetooth",
    "wifi":                 "ms-settings:network-wifi",
    "display":              "ms-settings:display",
    "display settings":     "ms-settings:display",
    "sound settings":       "ms-settings:sound",
    "camera":               "microsoft.windows.camera:",
}


def launch_app(name: str) -> str:
    """
    Try to launch an app by name.
    Order of attempts:
    1. Exact match in registry
    2. Partial match in registry
    3. Try running the name directly as a command
    4. Try opening as a URL if it looks like one
    """
    name_lower = name.lower().strip()
    # Remove filler words
    name_lower = re.sub(r"\b(the|app|application|program|software)\b", "", name_lower).strip()

    # 1 — Exact match
    if name_lower in APP_REGISTRY:
        return _run(APP_REGISTRY[name_lower], name_lower)

    # 2 — Partial match — check if any registry key is contained in the request
    best_match = None
    best_len   = 0
    for key, cmd in APP_REGISTRY.items():
        if key in name_lower and len(key) > best_len:
            best_match = (key, cmd)
            best_len   = len(key)

    if best_match:
        return _run(best_match[1], best_match[0])

    # 3 — Try running directly as executable
    if shutil.which(name_lower):
        return _run(name_lower, name_lower)

    # 4 — Try as URL
    if "." in name_lower and " " not in name_lower:
        url = name_lower if name_lower.startswith("http") else f"https://{name_lower}"
        return _run(url, name_lower)

    return f"I don't know how to open {name}. You can add it to the app registry in modules/apps.py."


def _run(cmd: str, display_name: str) -> str:
    """Execute the command."""
    try:
        if cmd.startswith("ms-") or cmd.startswith("microsoft."):
            # Windows URI scheme — opens system apps
            os.startfile(cmd)
        elif cmd.startswith("http"):
            # URL — open in default browser
            import webbrowser
            webbrowser.open(cmd)
        else:
            subprocess.Popen(
                cmd,
                shell  = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
        return f"Opening {display_name}."
    except Exception as e:
        return f"Couldn't open {display_name}. {e}"


def parse_and_handle(text: str) -> str | None:
    text_lower = text.lower().strip()

    m = re.search(
        r"\b(open|launch|start|run|show me|bring up)\b\s+(.+?)(?:\s+(?:app|application|program|software|browser))?$",
        text_lower
    )
    if m:
        target = m.group(2).strip()
        # Don't steal file/folder intents — those go to files module in V4
        if any(w in target for w in ["my file", "my folder", "my document"]):
            return None
        return launch_app(target)

    return None