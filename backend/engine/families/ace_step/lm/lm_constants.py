"""ACE-Step 5Hz LM constrained-decoding constants (aligned with upstream)."""
from __future__ import annotations

VALID_LANGUAGES = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
]

KEYSCALE_NOTES = ["A", "B", "C", "D", "E", "F", "G"]
KEYSCALE_ACCIDENTALS = ["", "#", "b", "♯", "♭"]
KEYSCALE_MODES = ["major", "minor"]

VALID_KEYSCALES: set[str] = set()
for _note in KEYSCALE_NOTES:
    for _acc in KEYSCALE_ACCIDENTALS:
        for _mode in KEYSCALE_MODES:
            VALID_KEYSCALES.add(f"{_note}{_acc} {_mode}")

BPM_MIN = 30
BPM_MAX = 300
DURATION_MIN = 10
DURATION_MAX = 600
VALID_TIME_SIGNATURES = [2, 3, 4, 6]

MAX_AUDIO_CODE = 63999
