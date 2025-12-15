try:
    from .zotero import ZoteroDB
    __all__ = ["ZoteroDB"]
except ImportError:
    __all__ = []
