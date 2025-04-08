use std::sync::Arc;
use crate::{
    client::{
        embeddings_client::Client as EmbeddingsClient,
        rerank_client::{DefaultRerankClient, RerankClient},
        search_client::SearchClient,
        RateLimiter
    },
    config::VoyageConfig,
    models::{
        embeddings::EmbeddingsRequest
    },
};

pub struct VoyageAiClientConfig {
    pub config: VoyageConfig,
    pub embeddings_client: Arc<EmbeddingsClient>,
    pub rerank_client: Arc<DefaultRerankClient>,
    pub search_client: Arc<SearchClient>,
}

pub struct VoyageAiClient {
    pub config: VoyageAiClientConfig,
}

impl VoyageAiClient {
    pub fn new() -> Self {
        let config = VoyageConfig::default();
        Self::new_with_config(config)
    }
    
    pub fn with_key(api_key: impl Into<String>) -> Self {
        let config = VoyageConfig::new(api_key.into());
        Self::new_with_config(config)
    }
    
    pub fn new_with_config(config: VoyageConfig) -> Self {
        let rate_limiter = Arc::new(RateLimiter::new());
        let embeddings_client = EmbeddingsClient::new(config.clone());
        let rerank_client = DefaultRerankClient::new(config.clone(), rate_limiter.clone());
        
        // Create the search client with the unwrapped clients
        let search_client = Arc::new(SearchClient::new(embeddings_client.clone(), rerank_client.clone()));
        
        // Now wrap the base clients in Arc for our config
        let embeddings_client = Arc::new(embeddings_client);
        let rerank_client = Arc::new(rerank_client);
        
        let client_config = VoyageAiClientConfig {
            config,
            embeddings_client,
            rerank_client,
            search_client,
        };
        
        Self {
            config: client_config,
        }
    }

    pub fn embeddings_client(&self) -> &Arc<EmbeddingsClient> {
        &self.config.embeddings_client
    }

    /// Create a rerank request builder for more options
    pub fn rerank_request(&self) -> crate::client::rerank_client::RerankRequestBuilder {
        self.config.rerank_client.rerank_request()
    }
    
    /// Finds documents similar to a query and returns a stream of similarity results.
    pub fn find_similar_documents(&self, query: &str, documents: Vec<String>) -> tokio_stream::wrappers::ReceiverStream<crate::client::rerank_client::DocumentSimilarity> {
        self.config.rerank_client.find_similar_documents(query, documents)
    }
    
    /// Finds the single most similar document to a query.
    pub fn most_similar_document(&self, query: &str, documents: Vec<String>) -> crate::client::rerank_client::AsyncDocumentSimilarity {
        self.config.rerank_client.most_similar_document(query, documents)
    }
    
    // Implement embeddings method for backward compatibility
    pub fn embeddings(&self, request: EmbeddingsRequest) -> crate::traits::voyage::EmbeddingTask {
        // Clone everything needed for the async task
        let embeddings_client = self.config.embeddings_client.clone();
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        tokio::task::spawn(async move {
            let result = embeddings_client.create_embedding(&request).await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
            let _ = tx.send(result);
        });
        
        crate::traits::voyage::EmbeddingTask::new(rx)
    }
    
    // Implement search method for backward compatibility
    pub fn search(&self, request: crate::client::SearchRequest) -> crate::traits::voyage::SearchTask {
        // Clone everything needed for the async task
        let search_client = self.config.search_client.clone();
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        tokio::task::spawn(async move {
            let result = search_client.search(&request).await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
            let _ = tx.send(result);
        });
        
        crate::traits::voyage::SearchTask::new(rx)
    }
}
