use crate::errors::VoyageError;
use crate::models::embeddings::{EmbeddingModel, EmbeddingsInput, EmbeddingsRequest};
use crate::VoyageAiClient;
use crate::client::rerank_client::RerankClient;
use tokio::sync::oneshot;
use tokio::task;

/// Domain-specific future type for single text embedding that can be awaited
pub struct TextEmbedding {
    receiver: oneshot::Receiver<Result<Vec<f32>, VoyageError>>,
}

impl TextEmbedding {
    fn new(receiver: oneshot::Receiver<Result<Vec<f32>, VoyageError>>) -> Self {
        Self { receiver }
    }
}

// Implement Future trait for TextEmbedding for clean .await usage
impl std::future::Future for TextEmbedding {
    type Output = Result<Vec<f32>, VoyageError>;
    
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.receiver).poll(cx)
            .map(|result| result.unwrap_or_else(|_| Err(VoyageError::Other("Embedding task canceled".to_string()))))
    }
}

/// Domain-specific future type for batch text embeddings that can be awaited
pub struct BatchEmbedding {
    receiver: oneshot::Receiver<Result<Vec<Vec<f32>>, VoyageError>>,
}

impl BatchEmbedding {
    fn new(receiver: oneshot::Receiver<Result<Vec<Vec<f32>>, VoyageError>>) -> Self {
        Self { receiver }
    }
}

// Implement Future trait for BatchEmbedding for clean .await usage
impl std::future::Future for BatchEmbedding {
    type Output = Result<Vec<Vec<f32>>, VoyageError>;
    
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.receiver).poll(cx)
            .map(|result| result.unwrap_or_else(|_| Err(VoyageError::Other("Batch embedding task canceled".to_string()))))
    }
}

/// A stream of document similarities
pub type DocumentSimilarityStream = tokio_stream::wrappers::ReceiverStream<crate::client::rerank_client::DocumentSimilarity>;

/// A stream of text embeddings
pub type TextEmbeddingStream = tokio_stream::wrappers::ReceiverStream<Vec<f32>>;

/// Interface for embedding text into vectors
pub trait Embedder: Send + Sync + 'static {
    /// Get embedding for a single text (returns a future)
    fn embed(&self, text: &str) -> TextEmbedding;

    /// Get embeddings for multiple texts (returns a future with all embeddings)
    fn embed_batch(&self, texts: &[String]) -> BatchEmbedding;
    
    /// Get embeddings for multiple texts as a stream (optional method)
    fn embed_stream(&self, texts: Vec<String>) -> TextEmbeddingStream;
    // Default implementation is removed - each implementor must provide their own implementation
}

/// Interface for reranking documents
pub trait Reranker: Send + Sync + 'static {
    /// Rerank documents based on a query and return a stream of document similarities
    fn rerank(&self, query: &str, documents: Vec<String>) -> DocumentSimilarityStream;
}

impl Embedder for VoyageAiClient {
    fn embed(&self, text: &str) -> TextEmbedding {
        // Clone everything needed for the async task
        let text = text.to_string();
        // Create a cloned instance of the client for the task
        let embeddings_client = self.embeddings_client().clone();
        
        let (tx, rx) = oneshot::channel();
        
        task::spawn(async move {
            let result = async {
                let request = EmbeddingsRequest {
                    input: EmbeddingsInput::Single(text),
                    model: EmbeddingModel::Voyage3Large,
                    input_type: None,
                    truncation: None,
                    encoding_format: None,
                };

                let embeddings = embeddings_client.create_embedding(&request).await?;
                Ok(embeddings.data[0].embedding.clone())
            }.await;
            
            let _ = tx.send(result);
        });
        
        TextEmbedding::new(rx)
    }

    fn embed_batch(&self, texts: &[String]) -> BatchEmbedding {
        // Clone everything needed for the async task
        let texts = texts.to_vec();
        // Create a cloned instance of the client for the task
        let embeddings_client = self.embeddings_client().clone();
        
        let (tx, rx) = oneshot::channel();
        
        task::spawn(async move {
            let result = async {
                let request = EmbeddingsRequest {
                    input: EmbeddingsInput::Multiple(texts),
                    model: EmbeddingModel::Voyage3Large,
                    input_type: None,
                    truncation: None,
                    encoding_format: None,
                };

                let embeddings = embeddings_client.create_embedding(&request).await?;
                Ok(embeddings.data.into_iter().map(|d| d.embedding).collect())
            }.await;
            
            let _ = tx.send(result);
        });
        
        BatchEmbedding::new(rx)
    }
    
    fn embed_stream(&self, texts: Vec<String>) -> TextEmbeddingStream {
        // Implementation that creates a stream
        let embeddings_client = self.embeddings_client().clone();
        let (tx, rx) = tokio::sync::mpsc::channel(texts.len());
        
        tokio::spawn(async move {
            let request = EmbeddingsRequest {
                input: EmbeddingsInput::Multiple(texts),
                model: EmbeddingModel::Voyage3Large,
                input_type: None,
                truncation: None,
                encoding_format: None,
            };
            
            match embeddings_client.create_embedding(&request).await {
                Ok(response) => {
                    for embedding_data in response.data {
                        if tx.send(embedding_data.embedding).await.is_err() {
                            break; // receiver dropped
                        }
                    }
                },
                Err(e) => {
                    log::error!("Error in embed_stream: {:?}", e);
                    // Channel will be closed, receiver will get end of stream
                }
            }
        });
        
        tokio_stream::wrappers::ReceiverStream::new(rx)
    }
}

impl Reranker for VoyageAiClient {
    fn rerank(&self, query: &str, documents: Vec<String>) -> DocumentSimilarityStream {
        // This is direct passthrough to the find_similar_documents API
        self.config.rerank_client.find_similar_documents(query, documents)
    }
}
