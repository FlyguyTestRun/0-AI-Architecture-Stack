# Runbook

Operational procedures. Keep this current: an out of date runbook is worse than none.

---

## Starting up

### Local, offline

```bash
make setup
make demo
```

No services required. Every layer uses its fallback.

### Local, full stack

```bash
make up            # Qdrant, Ollama, Phoenix
make pull-model    # gemma3:4b, roughly 3 GB
make doctor        # confirm the stack picked the services up
```

`make doctor` is the source of truth. If it reports `offline` or `in-memory` when you
expected otherwise, the service is not reachable. Do not guess, read the table.

---

## Diagnosing

### `zerostack doctor` shows the offline LLM provider

Ollama is not reachable at `ZEROSTACK_LLM_BASE_URL`.

```bash
docker compose ps ollama
curl -s localhost:11434/api/tags
docker compose logs ollama --tail 50
```

If the container is healthy but the model is missing, `make pull-model`.

### Answers say "No context was retrieved"

The vector collection is empty for this process.

```bash
zerostack doctor          # check indexed_chunks
zerostack ingest          # index data/corpus
```

If `indexed_chunks` is zero right after ingesting, the store is probably in process
rather than embedded, so the index died with the previous process. Check the
`vector_store_mode` column: `embedded` or `server` persist, `in-memory` does not.

### `vector_store_mode` is `in-memory` when you expected `embedded`

Another process holds the lock on `data/qdrant`. This is expected when the Streamlit
app and the CLI run at the same time.

Either stop the other process, or start the Qdrant container, which removes the
limitation entirely:

```bash
docker compose up -d qdrant
```

### Analytics reports `"engine": "sqlite"` instead of `"duckdb"`

DuckDB could not load its sqlite scanner extension, usually because the machine has no
outbound network access. The fallback computes the same aggregate in SQLite and
returns the same shape. No action needed unless you specifically want DuckDB.

### Retrieval returns the wrong documents

1. Confirm what is actually indexed: `zerostack doctor`.
2. Check the embedding backend. `hashing` matches shared vocabulary, not meaning, so
   a question phrased with synonyms will miss. Install the `embeddings` extra.
3. Inspect the scores in the trace: `tail -5 data/traces.jsonl`.
4. If the top score is high but the chunk is wrong, the chunk size is likely too large
   for the document structure. Lower `ZEROSTACK_RAG_CHUNK_SIZE`.

### The API returns 500

```bash
zerostack doctor
tail -50 data/traces.jsonl        # spans carry the error field
```

Every span records its error, so the failing layer is identifiable without reproducing
the request.

---

## Changing the embedding model

Embedding dimensions are baked into the vector collection. Changing the model without
reindexing produces silent garbage, because the search runs against vectors from a
different space.

```bash
# 1. update both values together
ZEROSTACK_RAG_EMBEDDING_MODEL=<new model>
ZEROSTACK_RAG_EMBEDDING_DIMENSIONS=<its dimension count>

# 2. drop the collection
rm -rf data/qdrant                                    # embedded
# or, for a server:
curl -X DELETE localhost:6333/collections/zerostack

# 3. reindex
zerostack ingest ./your-documents

# 4. verify
zerostack doctor
```

Never skip step 2.

---

## Deploying

Work through `modes/DEPLOY.md`. The point that matters most:

**Set every backend selector explicitly in production.** Leaving `auto` means an
outage in Qdrant or Ollama degrades silently to a fallback and keeps answering, with
worse answers, and nobody is paged.

```bash
ZEROSTACK_LLM_PROVIDER=ollama
ZEROSTACK_RAG_BACKEND=qdrant
ZEROSTACK_ORCHESTRATOR_KIND=langgraph
```

With these set, an unreachable service raises instead of degrading.

### Pre-flight

```bash
make check                    # must be green
zerostack doctor              # must show intended backends, no fallbacks
```

- Secrets present in the target environment, absent from the repository.
- Vector collection populated, dimensions matched to the model.
- Rollback tested.

### Rollback

State lives in `data/zerostack.db` and the vector collection. Neither is destroyed by
redeploying the application, so a rollback is a revert of the application version.
Back up the database file before any migration that alters the schema.

---

## Routine maintenance

| Task | Frequency | Command |
|------|-----------|---------|
| Rotate the trace log | Weekly, or when large | `mv data/traces.jsonl data/traces.$(date +%F).jsonl` |
| Back up application state | Daily in production | copy `data/zerostack.db` |
| Reindex the corpus | When source documents change | `zerostack ingest` |
| Review latency | Weekly | `zerostack analytics` |
| Update dependencies | Monthly | `pip install -e ".[dev]" --upgrade` then `make check` |
