"""
modules/files.py
Handles everything related to opening files and folders.
"""

import os
import re
import subprocess
from pathlib import Path


# Common folder shortcuts — add your own below
FOLDER_SHORTCUTS = {
    "desktop":      Path.home() / "Desktop",
    "downloads":    Path.home() / "Downloads",
    "documents":    Path.home() / "Documents",
    "pictures":     Path.home() / "Pictures",
    "music":        Path.home() / "Music",
    "videos":       Path.home() / "Videos",
    "home":         Path.home(),
    # Add your custom folders here:
    # "projects":   Path("C:/Users/YourName/Projects"),
    # "work":       Path("C:/Users/YourName/Work"),
}

# File type associations (Windows)
FILE_OPENERS = {
    ".pdf":  "start",
    ".docx": "start",
    ".xlsx": "start",
    ".txt":  "notepad",
    ".py":   "code",        # opens in VS Code
    ".mp4":  "start",
    ".mp3":  "start",
    ".jpg":  "start",
    ".png":  "start",
}

# Trigger words that mean "open something"
OPEN_TRIGGERS = r"\b(open|go to|navigate to|show me|launch|take me to|browse)\b"

# Trigger words that mean "list contents"
LIST_TRIGGERS = r"\b(list|show|what'?s? in|contents of|what's inside)\b"

# Words that signal a file (not a folder)
FILE_SIGNALS = r"\b(file|pdf|document|doc|video|image|photo|song|mp3|mp4|txt|excel|spreadsheet)\b"


def open_folder(name: str) -> str:
    """Open a folder by shortcut name or full path."""
    name_lower = name.lower().strip()

    # Check shortcuts first
    if name_lower in FOLDER_SHORTCUTS:
        path = FOLDER_SHORTCUTS[name_lower]
        if path.exists():
            os.startfile(str(path))
            return f"Opening your {name_lower} folder."
        return f"I couldn't find your {name_lower} folder."

    # Try as a direct path
    path = Path(name).expanduser()
    if path.exists() and path.is_dir():
        os.startfile(str(path))
        return f"Opening {path.name}."

    # Fuzzy search in home directory
    matches = list(Path.home().rglob(f"*{name}*"))
    dirs = [m for m in matches if m.is_dir()]
    if dirs:
        os.startfile(str(dirs[0]))
        return f"Found and opening {dirs[0].name}."

    return f"I couldn't find a folder called '{name}'."


def open_file(name: str) -> str:
    """Open a file by name or path. Searches if not found directly."""
    path = Path(name).expanduser()

    # Direct path
    if path.exists() and path.is_file():
        _open_with_association(path)
        return f"Opening {path.name}."

    # Fuzzy search
    matches = list(Path.home().rglob(f"*{name}*"))
    files = [m for m in matches if m.is_file()]
    if files:
        _open_with_association(files[0])
        return f"Found and opening {files[0].name}."

    return f"I couldn't find a file called '{name}'."


def list_folder(name: str) -> str:
    """List contents of a folder."""
    name_lower = name.lower().strip()

    if name_lower in FOLDER_SHORTCUTS:
        path = FOLDER_SHORTCUTS[name_lower]
    else:
        path = Path(name).expanduser()

    if not path.exists():
        return f"I can't find the {name} folder."

    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    folders = [i.name for i in items if i.is_dir()][:10]
    files   = [i.name for i in items if i.is_file()][:10]

    parts = []
    if folders:
        parts.append(f"Folders: {', '.join(folders)}")
    if files:
        parts.append(f"Files: {', '.join(files)}")

    total = len(list(path.iterdir()))
    if total > 20:
        parts.append(f"...and {total - 20} more items.")

    return f"In your {name} folder — " + ". ".join(parts) + "."


def _open_with_association(path: Path):
    """Open a file using Windows default or a specific app."""
    ext = path.suffix.lower()
    opener = FILE_OPENERS.get(ext, "start")
    if opener == "start":
        os.startfile(str(path))
    else:
        subprocess.Popen([opener, str(path)], shell=True)


def _extract_target(text: str, trigger_pattern: str) -> str | None:
    """
    Strips the trigger verb (open/show/list etc.) and filler words
    from the command and returns the target name.
    e.g. "open my downloads folder" → "downloads"
    """
    # Remove the trigger word
    text = re.sub(trigger_pattern, "", text, flags=re.IGNORECASE).strip()
    # Remove filler words
    text = re.sub(r"\b(my|the|a|an|please|up|some|me)\b", "", text, flags=re.IGNORECASE).strip()
    # Remove trailing/leading noise words
    text = re.sub(r"\b(folder|directory|file|document)\b", "", text, flags=re.IGNORECASE).strip()
    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def parse_and_handle(text: str) -> str | None:
    """
    Called by the brain. Returns a response string if this module
    can handle the intent, or None to let the LLM handle it.
    """
    text_lower = text.lower().strip()

    # ── List folder contents ───────────────────────────────────────────────────
    if re.search(LIST_TRIGGERS, text_lower):
        # Check if any known shortcut is mentioned
        for name in FOLDER_SHORTCUTS:
            if name in text_lower:
                return list_folder(name)
        # Extract target generically
        target = _extract_target(text_lower, LIST_TRIGGERS)
        if target:
            return list_folder(target)

    # ── Open something ─────────────────────────────────────────────────────────
    if re.search(OPEN_TRIGGERS, text_lower):

        # Check if any known folder shortcut is mentioned directly
        for name in FOLDER_SHORTCUTS:
            if name in text_lower:
                # Is it a file signal? If so, search for a file with that name
                if re.search(FILE_SIGNALS, text_lower):
                    target = _extract_target(text_lower, OPEN_TRIGGERS)
                    if target:
                        return open_file(target)
                return open_folder(name)

        # Extract generic target
        target = _extract_target(text_lower, OPEN_TRIGGERS)
        if not target:
            return None

        # Decide: file or folder?
        if re.search(FILE_SIGNALS, text_lower):
            return open_file(target)
        else:
            # Try folder first, then file
            result = open_folder(target)
            if "couldn't find" in result:
                result = open_file(target)
            return result

    return None  # not handled here