"""The class list lives in one place so ids can't drift between scripts."""

CLASS_NAMES = ["cone", "barrier", "stop_sign"]

# what actually gets printed on the boxes
DISPLAY_NAMES = {
    "cone": "Cone",
    "barrier": "Barrier",
    "stop_sign": "Stop Sign",
}

# BGR, picked so the three classes are easy to tell apart on street photos
CLASS_COLORS = {
    "cone": (0, 140, 255),      # orange
    "barrier": (255, 200, 0),   # cyan-ish
    "stop_sign": (60, 20, 220), # red
}


def display_name(class_name: str) -> str:
    """Pretty name for a class, falls back to the raw name if it's unknown."""
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())
