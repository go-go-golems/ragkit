# Geppetto provider adapters

This package is ragkit's opinionated bridge to Geppetto's model-service
contracts. It does not construct transports or choose deployment defaults.

```text
application configuration
        |
        v
Geppetto embeddings.Provider / rerank.Provider
        |
        v
rag/provider/geppetto
        |
        v
rag.Embedder / rag.Reranker
```

## Ownership and identity contract

| Value | Authority | Durable identity rule |
| --- | --- | --- |
| Provider endpoint, credentials, transport policy | Application + Geppetto | Never copied into a ragkit cache or artifact identity. |
| Provider and model name | Geppetto provider | Must be non-empty; request and response model identity must match exactly. |
| Embedding dimensions | Geppetto provider | Validated for every vector; applications include model and dimensions in bundle identity. |
| Document/query task prefixes | Application | Must remain in the application's vector-channel or adapter identity. |
| Candidate document rendering | Application | Any title/breadcrumb transformation requires an application cache/adapter epoch. |
| Provider result IDs and indices | Geppetto provider | Must map completely and uniquely back to caller chunk IDs. |
| Provider usage and cost | Geppetto provider | Projected without changing unknown (`nil`) versus explicit zero semantics, including on errors. |
| Result limit | ragkit caller | Geppetto is asked to score every candidate; the adapter applies the requested final limit only after complete validation. |
| Adapter implementation | ragkit release | Consumers pin the ragkit module version; no hidden adapter mode or local default exists. |

## Constructors

```go
embedder, err := geppetto.NewEmbedder(configuredEmbeddingProvider)
reranker, err := geppetto.NewReranker(configuredRerankProvider)
```

The embedding adapter rejects model mismatches, empty batches, response-count
mismatches, wrong dimensions, and non-finite values. The reranking adapter
rejects missing/duplicate candidate IDs, incomplete or foreign results,
provider/model drift, inconsistent indices/ranks, non-deterministic ordering,
non-finite scores, invalid usage, and invalid cost.

Applications still own prefixes, candidate formatting, durable cache keys,
provider construction, and product policy. Keeping those values outside these
constructors prevents a dependency cleanup from silently changing scientific
or artifact identity.
