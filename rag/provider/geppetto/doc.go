// Package geppetto adapts configured Geppetto model-service providers to
// ragkit's retrieval-domain interfaces.
//
// Geppetto owns provider transport, security, model identity, usage, and cost.
// This package owns only the validated projection from embeddings.Provider to
// rag.Embedder and from rerank.Provider to rag.Reranker. Applications retain
// deployment settings, task prefixes, candidate rendering, cache policy, and
// artifact identity.
package geppetto
