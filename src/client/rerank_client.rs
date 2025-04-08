use log::{debug, info, warn};
use reqwest::Client;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;

use crate::client::RateLimiter;
use crate::config::VoyageConfig;
use crate::errors::VoyageError;
use crate::models::rerank::{RerankRequest, RerankResponse};

/// Base URL for the Voyage AI API.
const BASE_URL: &str = "https://api.voyageai.com/v1";

/// Builder for rerank requests with additional configuration options
#[derive(Debug, Clone)]
pub struct RerankRequestBuilder {
    query: Option<String>,
    documents: Vec<String>,
    model: crate::models::rerank::RerankModel,
    top_k: Option<usize>,
}

impl RerankRequestBuilder {
    /// Create a new request builder
    pub fn new() -> Self {
        Self {
            query: None,
            documents: Vec::new(),
            model: Default::default(),
            top_k: None,
        }
    }
    
    /// Set the query to compare documents against
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }
    
    /// Add a document to be compared
    pub fn add_document(mut self, document: impl Into<String>) -> Self {
        self.documents.push(document.into());
        self
    }
    
    /// Add multiple documents to be compared
    pub fn add_documents(mut self, documents: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.documents.extend(documents.into_iter().map(|d| d.into()));
        self
    }
    
    /// Set the model to use for reranking
    pub fn model(mut self, model: crate::models::rerank::RerankModel) -> Self {
        self.model = model;
        self
    }
    
    /// Limit the number of results to return
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Build the RerankRequest
    pub fn build(self) -> Result<RerankRequest, crate::models::rerank::ValidationError> {
        let query = self.query.ok_or_else(|| {
            crate::models::rerank::ValidationError::EmptyDocuments
        })?;
        
        RerankRequest::new(
            query,
            self.documents,
            self.model,
            self.top_k,
        )
    }
}

/// A single document with its similarity score to a query
#[derive(Debug, Clone)]
pub struct DocumentSimilarity {
    /// Position in the ranking (0 = most similar)
    pub rank: usize,
    /// Similarity score from 0.0 to 1.0, higher is more similar
    pub similarity: f64,
    /// The document content
    pub document: String,
}

/// A future that resolves to a single document similarity
pub struct AsyncDocumentSimilarity {
    receiver: oneshot::Receiver<Result<DocumentSimilarity, VoyageError>>,
}

impl AsyncDocumentSimilarity {
    fn new(receiver: oneshot::Receiver<Result<DocumentSimilarity, VoyageError>>) -> Self {
        Self { receiver }
    }
}

impl Future for AsyncDocumentSimilarity {
    type Output = Result<DocumentSimilarity, VoyageError>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(VoyageError::Other("Similarity task canceled".to_string()))),
            Poll::Pending => Poll::Pending,
        }
    }
}


/// Client trait for finding similar documents based on semantic similarity.
pub trait RerankClient: std::fmt::Debug + Send + Sync {
    /// Finds documents similar to a query and returns a stream of document similarities.
    fn find_similar_documents(&self, query: &str, documents: Vec<String>) -> ReceiverStream<DocumentSimilarity>;
    
    /// Finds the single most similar document to a query.
    fn most_similar_document(&self, query: &str, documents: Vec<String>) -> AsyncDocumentSimilarity;
    
    /// Create a rerank request with more options
    fn rerank_request(&self) -> RerankRequestBuilder;
}

/// Default implementation of RerankClient
#[derive(Clone, Debug)]
pub struct DefaultRerankClient {
    client: Client,
    config: VoyageConfig,
    rate_limiter: Arc<RateLimiter>,
}

impl DefaultRerankClient {
    /// Creates a new `DefaultRerankClient` instance.
    pub fn new(config: VoyageConfig, rate_limiter: Arc<RateLimiter>) -> Self {
        debug!("Creating new DefaultRerankClient");
        Self {
            client: Client::new(),
            config,
            rate_limiter,
        }
    }

    fn estimate_tokens(&self, request: &RerankRequest) -> u32 {
        fn tokenize(text: &str) -> usize {
            text.split(|c: char| c.is_whitespace() || !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .count()
        }

        let query_tokens = tokenize(&request.query);
        let doc_tokens: usize = request.documents.iter().map(|doc| tokenize(doc)).sum();

        let total_tokens = query_tokens + doc_tokens;
        debug!("Estimated token count: {}", total_tokens);
        total_tokens as u32
    }
    
    /// Create a RerankRequest from a query and documents
    fn create_request(&self, query: &str, documents: Vec<String>) -> RerankRequest {
        RerankRequest::new(
            query.to_string(),
            documents,
            Default::default(), // Use default model
            None,
        ).unwrap_or_else(|_| panic!("Failed to create rerank request"))
    }
    
    /// Internal implementation of the rerank operation
    async fn perform_rerank(&self, request: RerankRequest) -> Result<RerankResponse, VoyageError> {
        let url = format!("{}/rerank", BASE_URL);
        let api_key = self.config.api_key().to_string();
        let estimated_tokens = self.estimate_tokens(&request);
        
        debug!("Reranking documents with URL: {}", url);
        debug!("Estimated tokens for request: {}", estimated_tokens);

        let wait_time = self.rate_limiter
            .check_reranking_limit(estimated_tokens)
            .await;

        if wait_time.as_secs() > 0 {
            info!(
                "Rate limit reached. Waiting for {} seconds",
                wait_time.as_secs()
            );
            sleep(wait_time).await;
        }

        let response = self.client
            .post(&url)
            .bearer_auth(api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        match status {
            reqwest::StatusCode::OK => {
                debug!("Rerank request successful");
                debug!("Raw API response: {}", text);
                let rerank_response: RerankResponse = serde_json::from_str(&text).map_err(|e| {
                    warn!("Failed to parse rerank response: {:?}", e);
                    warn!("Raw response: {}", text);
                    VoyageError::JsonError(e.to_string())
                })?;

                if rerank_response.data.is_empty() {
                    warn!("Rerank response contains no results");
                } else {
                    debug!(
                        "Rerank response contains {} results",
                        rerank_response.data.len()
                    );
                }

                self.rate_limiter
                    .update_reranking_usage(rerank_response.usage.total_tokens)
                    .await;

                Ok(rerank_response)
            }
            reqwest::StatusCode::UNAUTHORIZED => {
                warn!("Unauthorized request: {}", text);
                Err(VoyageError::Unauthorized)
            }
            _ => {
                warn!("Rerank request failed with status: {}", status);
                warn!("Error response body: {}", text);
                Err(VoyageError::ApiError(status, text))
            }
        }
    }
}

impl RerankClient for DefaultRerankClient {
    fn find_similar_documents(&self, query: &str, documents: Vec<String>) -> ReceiverStream<DocumentSimilarity> {
        let (tx, rx) = mpsc::channel(16);
        let client = self.clone();
        let input_docs = documents.clone();
        let request = self.create_request(query, documents);
        
        tokio::spawn(async move {
            match client.perform_rerank(request).await {
                Ok(response) => {
                    for (rank, result) in response.data.into_iter().enumerate() {
                        // Directly use the original document at the matching index
                        let document = DocumentSimilarity {
                            rank,
                            similarity: result.relevance_score,
                            document: input_docs[result.index].clone(),
                        };
                        
                        if tx.send(document).await.is_err() {
                            break; // receiver dropped
                        }
                    }
                }
                Err(e) => {
                    log::error!("Error performing rerank: {:?}", e);
                    // Channel is closed, receiver will get an end of stream
                }
            }
        });
        
        ReceiverStream::new(rx)
    }
    
    fn most_similar_document(&self, query: &str, documents: Vec<String>) -> AsyncDocumentSimilarity {
        let client = self.clone();
        let input_docs = documents.clone();
        let request = self.create_request(query, documents);
        let (tx, rx) = oneshot::channel();
        
        tokio::spawn(async move {
            let result = match client.perform_rerank(request).await {
                Ok(response) => {
                    if let Some(best_match) = response.data.into_iter().next() {
                        // Use the original document text directly by index
                        Ok(DocumentSimilarity {
                            rank: 0,
                            similarity: best_match.relevance_score,
                            document: input_docs[best_match.index].clone(),
                        })
                    } else {
                        Err(VoyageError::Other("No matching documents found".to_string()))
                    }
                }
                Err(e) => Err(e),
            };
            
            let _ = tx.send(result);
        });
        
        AsyncDocumentSimilarity::new(rx)
    }
    
    fn rerank_request(&self) -> RerankRequestBuilder {
        RerankRequestBuilder::new()
    }
}
