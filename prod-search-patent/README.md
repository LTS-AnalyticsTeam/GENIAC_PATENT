# prod-search-patent

End-to-end patent search pipeline designed for production parity with the Graph-RAG architecture.  
The stack is fully containerised with Docker Compose and covers the following services:

- **api** – FastAPI ingestion/search service with job orchestration
- **frontend** – React/Vite UI for XML upload, progress monitoring, and results
- **worker** – Background pipeline executor (Redis-backed)
- **redis** – Message broker for ingestion jobs
- **elasticsearch** – Primary vector store for Stage 1 index
- **neo4j** – Graph store (APOC enabled) powering Stage 2 graph enrichment & Graph-RAG
- **cosmos-proxy** – Optional façade over Azure Cosmos DB (stub implementation)
- **vectorizer** – Local HTTP embedding/summary service (stub; Azure OpenAI compatible schema)

## Key pipeline flow (①〜⑧)

1. **XML parsing** – Extract title, summary, claim1, and classification (IPC/FI/F-term).
2. **Cosmos lookup** – IPC prefix OR search (startswith semantics). IPC absence short-circuits with 4xx.
3. **Stage 1 ingest** – Trim results to 40,000 (publication date desc -> `_ts` desc fallback) and reuse/embed vectors via `text-embedding-3-large`.
4. **Query generation** – Generate hybrid vector/text queries (LLM powered, optional override).
5. **Vector search** – Run Stage 1 `knn` search (`k=1000`, `num_candidates` from env).
6. **Stage 2 ingest** – Reset and enrich Neo4j graph.
7. **Graph-RAG** – Combine Neo4j paths and return Top 10.
8. **Result delivery** – `/result/{job_id}` API responds with scored Top 10 + evidence.

## Repository layout

```
prod-search-patent/
├── docker-compose.yml
├── .env.example
├── README.md
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── parsing_service.py
│   │   ├── cosmos_client.py
│   │   ├── trimming.py
│   │   ├── embedding_service.py
│   │   ├── elasticsearch_stage1.py
│   │   ├── stage2_indexer.py
│   │   ├── graph_rag.py
│   │   ├── job_manager.py
│   │   └── pipeline_runner.py
│   └── tests/
│       ├── __init__.py
│       └── test_pipeline.py
└── services/
    ├── api/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── app/
    │       ├── __init__.py
    │       ├── main.py
    │       ├── models.py
    │       ├── deps.py
    │       ├── routes.py
    │       └── settings.py
    ├── frontend/
    │   ├── Dockerfile
    │   ├── package.json
    │   └── src/
    ├── worker/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── worker.py
    ├── vectorizer/
    │   ├── Dockerfile
    │   └── app.py
    └── cosmos-proxy/
        ├── Dockerfile
        └── app.py
```

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

アクセス:
- Frontend: <http://localhost:3000>
- API: <http://localhost:8080>
- Elasticsearch: <http://localhost:9200>
- Neo4j Browser: <http://localhost:7474>
- Redis: `redis-cli -h localhost -p 6379 ping`

### Health checks

- API: <http://localhost:8080/healthz>
- Vectorizer (stub): <http://localhost:9000/health>
- Cosmos proxy (stub): <http://localhost:7070/health>

### フロントエンドでできること

- テキストファイル（.txt 内に XML を格納）をアップロードして `/ingest` を呼び出す
- `/status/{job_id}` を 3 秒おきにポーリングし、ステージ単位の進捗を表示
- ジョブ完了後、自動的に `/result/{job_id}` から Graph-RAG Top10 を取得し、カード表示

## Tests

Run pipeline unit tests locally:

```bash
cd prod-search-patent
python3 -m venv .venv && source .venv/bin/activate
pip install -r services/api/requirements.txt -r src/tests/requirements.txt
pytest src/tests
```

## Licensing

MIT (placeholder). Update as appropriate.
