use crate::models::embeddings::{EmbeddingsInput, EmbeddingsResponse};
use crate::client::SearchRequest;
use crate::client::SearchResult;
use tokio::sync::oneshot;

/// Domain-specific future type for embeddings that can be awaited
pub struct EmbeddingTask {
    receiver: oneshot::Receiver<Result<EmbeddingsResponse, Box<dyn std::error::Error + Send + Sync>>>
}

impl EmbeddingTask {
    pub fn new(receiver: oneshot::Receiver<Result<EmbeddingsResponse, Box<dyn std::error::Error + Send + Sync>>>) -> Self {
        Self { receiver }
    }
}

// Implement Future trait for EmbeddingTask for clean .await usage
impl std::future::Future for EmbeddingTask {
    type Output = Result<EmbeddingsResponse, Box<dyn std::error::Error + Send + Sync>>;
    
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.receiver).poll(cx)
            .map(|result| result.unwrap_or_else(|_| Err(Box::new(crate::errors::VoyageError::Other("Embedding task canceled".to_string())))))
    }
}

/// Domain-specific future type for search results that can be awaited
pub struct SearchTask {
    receiver: oneshot::Receiver<Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>>>
}

impl SearchTask {
    pub fn new(receiver: oneshot::Receiver<Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>>>) -> Self {
        Self { receiver }
    }
}

// Implement Future trait for SearchTask for clean .await usage
impl std::future::Future for SearchTask {
    type Output = Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>>;
    
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.receiver).poll(cx)
            .map(|result| result.unwrap_or_else(|_| Err(Box::new(crate::errors::VoyageError::Other("Search task canceled".to_string())))))
    }
}

pub trait VoyageAiClientExt {
    /// Create embeddings for input text and return a future that resolves to the embedding
    fn embed<T>(&self, input: T) -> EmbeddingTask
    where
        T: Into<EmbeddingsInput> + Send + 'static;

    /// Create a rerank request builder for more options
    fn rerank_request(&self) -> crate::client::rerank_client::RerankRequestBuilder;
    
    /// Find documents similar to a query and return a stream of similarity results
    fn find_similar_documents(&self, query: &str, documents: Vec<String>) -> tokio_stream::wrappers::ReceiverStream<crate::client::rerank_client::DocumentSimilarity>;
    
    /// Find the most similar document to a query
    fn most_similar_document(&self, query: &str, documents: Vec<String>) -> crate::client::rerank_client::AsyncDocumentSimilarity;

    /// Search using the provided request
    fn search(&self, request: SearchRequest) -> SearchTask;
}
