# app/utils/mongodb.py
import os
from pymongo import MongoClient
from functools import lru_cache

@lru_cache()
def get_db():
    """Return a tuple (client, db) or raise if not configured."""
    uri = os.environ.get("MONGODB_URI")
    name = os.environ.get("MONGODB_NAME")
    if not uri or not name:
        raise RuntimeError("MongoDB URI or database name not configured")
    client = MongoClient(uri)
    db = client[name]
    return client, db