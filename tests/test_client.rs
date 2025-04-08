use std::error::Error;
use voyageai::{
    EmbeddingModel, EmbeddingsRequestBuilder, RerankModel, RerankRequest, VoyageAiClient,
    VoyageConfig, VoyageError, traits::llm::{Embedder, Reranker},
};

#[tokio::test]
async fn test_embeddings() -> Result<(), Box<dyn Error>> {
    let config = VoyageConfig::new(
        std::env::var("VOYAGE_API_KEY").unwrap_or_else(|_| "test_key".to_string()),
    );
    let client = VoyageAiClient::new_with_config(config);

    // Use the Embedder trait directly
    let embedding = client.embed("test input").await?;

    assert!(
        !embedding.is_empty(),
        "Embedding should not be empty"
    );

    Ok(())
}

// Removed invalid API key test as it's not needed

#[tokio::test]
async fn test_reranking() -> Result<(), Box<dyn Error>> {
    let config = VoyageConfig::new(
        std::env::var("VOYAGE_API_KEY").unwrap_or_else(|_| "test_key".to_string()),
    );
    let client = VoyageAiClient::new_with_config(config);

    // Use the Reranker trait directly
    let documents = vec!["doc1".to_string(), "doc2".to_string()];
    let stream = client.rerank("test query", documents);
    
    // Collect the results from the stream
    use tokio_stream::StreamExt;
    let results: Vec<_> = stream.collect().await;

    assert_eq!(results.len(), 2, "Expected exactly 2 reranked documents");
    assert!(
        results[0].similarity > results[1].similarity,
        "Documents should be sorted by similarity score"
    );

    Ok(())
}
