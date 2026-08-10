# ragkit

Reusable RAG (retrieval-augmented generation) building blocks for Go,
originally extracted from the
[`rag-ttc`](https://github.com/the-tree-center/rag-ttc) research codebase.

The package tree is contracts-first: `rag` holds the domain types and narrow
component interfaces; everything else is a replaceable implementation.

```text
ragkit/
  rag/                 Document/Chunk/Representation/Hit/Evidence types,
                       Chunker/Embedder/Generator/Searcher/Reranker interfaces,
                       validation, deterministic hit ordering
  rag/chunking         fixed, markdown, and heading-aware chunkers
  rag/representations  raw/breadcrumbs/small-to-big + generated (batched) kinds,
                       PromptSet (prompts are cache identity)
  rag/embedding        embedding fan-out, caching, budget caps, hash fixture
  rag/lexical          in-memory BM25; rag/lexical/bleve persistent index
  rag/vector           exact cosine; rag/vector/sqliteexact persistent index
  rag/indexbundle      immutable content-addressed bundles (atomic publication)
  rag/retrieval        collapse, weighted RRF fusion, hydration, filters
  rag/reranking        term-overlap + cached reranker decorator
  rag/provider/geppetto validated Geppetto embedding/reranking adapters
  rag/generation       generation caching/observation + flow adapters
  rag/answering        retrieval strategies (bm25/vector/rrf/rrf-reranked/
                       multi-query/hyde), context policy, grounded-answer contract
  rag/evaluation       precision/recall@k, MRR, nDCG, target-level validation
  rag/dataset          corpus/evaluation loading with digest validation
  digest, text, vector, execution, flow   domain-neutral infrastructure
```

Invariants preserved from upstream:

- Chunk text is verbatim source: `ValidateChunk` asserts the chunk equals the
  exact byte slice of its document.
- Representations are retrieval material; chunks are evidence. Search returns
  representation hits, `Collapse` maps them to chunks, `Hydrate` produces
  evidence.
- `HitRanksBefore` defines a complete ordering (score, document ID, chunk ID,
  representation ID) for deterministic retrieval results.

Changes relative to rag-ttc:

- Module path `github.com/go-go-golems/ragkit`; logcopter areas renamed.
- Representation prompts moved from package constants to an injectable
	`representations.PromptSet`; callers construct and own their domain prompts
	explicitly.
- The grounded-answer contract kind is injectable via
  `answering.Service.ContractKind` (default `ttc-grounded-answer-v1`).
- Dataset tests use self-contained fixtures instead of the TTC corpora.
- A boundary test forbids geppetto/pinocchio/glazed/cobra/bubbletea in the
  ragkit core. The explicit `rag/provider/...` adapter tree is the only place
  model-framework dependencies may enter; CLI frameworks remain forbidden.
- `rag/provider/geppetto` wraps already-configured providers. Applications keep
  deployment settings, task prefixes, candidate rendering, cache policy, and
  artifact identity; ragkit validates and projects provider responses.

Extraction is a cache epoch: `execution` cache keys are semantically
compatible with upstream, but no attempt is made to share cache directories
with rag-ttc installations.

## Opinionated Geppetto adapters

```go
import raggeppetto "github.com/go-go-golems/ragkit/rag/provider/geppetto"

embedder, err := raggeppetto.NewEmbedder(embeddingProvider)
reranker, err := raggeppetto.NewReranker(rerankProvider)
```

The constructors accept providers configured by the application. They add no
endpoints, credentials, model defaults, task prefixes, document formatting, or
cache policy. This keeps the dependency direction strict:

```text
application → ragkit/rag/provider/geppetto → Geppetto
```

## Development

```bash
make ci-check
```

## License

[MIT](LICENSE)
