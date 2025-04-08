# Agreements from Development Sessions

This document keeps track of design agreements made during development sessions to ensure consistent implementation across the codebase.

## Document Similarity API Design

### RerankClient Trait Methods

The agreed design for the RerankClient trait is:

```rust
pub trait RerankClient: std::fmt::Debug + Send + Sync {
    /// Finds documents similar to a query and returns a stream of document similarities.
    fn find_similar_documents(&self, query: &str, documents: Vec<Document>) -> impl Stream<Item = DocumentSimilarity>;
    
    /// Finds the single most similar document to a query.
    fn most_similar_document(&self, query: &str, documents: Vec<Document>) -> AsyncDocumentSimilarity;
    
    /// Create a rerank request with more options
    fn rerank_request(&self) -> RerankRequestBuilder;
}
```

Where:
- `Stream<Item = DocumentSimilarity>` directly returns a stream of DocumentSimilarity items
- `AsyncDocumentSimilarity` is a concrete type that implements Future and resolves to a `DocumentSimilarity`
- Both return types use the Hidden Box/Pin pattern to encapsulate async complexity

### DocumentSimilarity Structure

The `DocumentSimilarity` struct represents a single document with its similarity information:

```rust
#[derive(Debug, Clone)]
pub struct DocumentSimilarity {
    /// Position in the ranking (0 = most similar)
    pub rank: usize,
    /// Similarity score from 0.0 to 1.0, higher is more similar
    pub similarity: f64,
    /// The document content
    pub document: Document,
}
```

## Async Implementation Patterns

We follow the "Hidden Box/Pin" pattern as described in CLAUDE.md:

- ❌ NEVER use `async_trait` or `async fn` in traits
- ❌ NEVER return `Box<dyn Future>` or `Pin<Box<dyn Future>>` from client interfaces
- ✅ Return concrete domain-specific types that can be directly awaited by the user
- ✅ Hide async complexity behind `channel` and `task` `spawn`
- ✅ Client code simply awaits the concrete type (e.g., `let doc = client.most_similar_document(query, docs).await?;`)

## API Design Principles

- Use domain-appropriate singular names for individual items (e.g., `DocumentSimilarity`, not `RankedResult`)
- Use `Stream<T>` to represent multiple items rather than plural type names
- Use semantic names that accurately represent what an operation does (e.g., "similarity" versus "match")
- Method names should clearly reflect their purpose (e.g., `find_similar_documents`, `most_similar_document`)