// Package flow is the one typed step/pipeline layer over pkg/execution: it
// unifies bounded parallelism, fail-closed admission, retry with error
// classification, content-addressed per-item caching, batching-with-repair,
// and per-item failure policy behind a single Step type.
//
// flow owns mechanics; anything resembling workflow belongs to the calling
// program. There is no DAG scheduler, no persisted control state, and no
// distributed anything (RAG-TTC-FLOW-001 DR-1). Durability is memoization:
// the Store interface carries the execution cache contract, the default
// implementation is the existing content-addressed *execution.FileCache, and
// swapping the durable mechanism means providing another Store — steps and
// pipelines never change. Resume = replay: a killed process costs one
// in-flight item per worker.
//
// The package is deliberately generic. It knows nothing about RAG, models,
// or providers; domain adapters (generation, embeddings) live next to their
// domain types and produce Steps.
package flow
