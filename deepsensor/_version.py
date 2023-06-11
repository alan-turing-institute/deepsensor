import os


def string():
    try:
        with open(os.path.dirname(__file__) + "/VERSION", "r", encoding="utf-8") as fh:
            version = fh.read().strip()
            if version:
                return version
    except:
        pass
    return "unknown (git checkout)"