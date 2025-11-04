from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

app = FastAPI(title="Stub Vectorizer")


class EmbedRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    embeddings = [
        [0.0] * EMBED_DIM
        for _ in req.texts
    ]
    return {"embeddings": embeddings}

