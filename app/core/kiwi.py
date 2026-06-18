from kiwipiepy import Kiwi

_instance: Kiwi | None = None


def get() -> Kiwi:
    global _instance
    if _instance is None:
        _instance = Kiwi()
    return _instance
