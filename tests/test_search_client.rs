use voyageai::{VoyageAiClient, VoyageConfig, traits::llm::{Embedder, Reranker}};
use tokio_stream::StreamExt;

#[tokio::test]
async fn test_search_client_integration() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
    let config = VoyageConfig::new(api_key);
    let client = VoyageAiClient::new_with_config(config);

    // Test direct embedding
    let text = "Test document";
    let embedding = client.embed(text).await?;
    assert!(!embedding.is_empty());

    // Test batch embedding
    let texts = vec!["test document 1".to_string(), "test document 2".to_string()];
    let embeddings = client.embed_batch(&texts).await?;
    assert_eq!(embeddings.len(), 2);
    assert!(!embeddings[0].is_empty());
    assert!(!embeddings[1].is_empty());

    // Test reranking
    let query = "What is Rust?";
    let documents = vec![
        "Rust is a systems programming language".to_string(),
        "Python is interpreted".to_string(),
    ];

    // Use Reranker trait directly
    let similarity_stream = client.rerank(query, documents);
    let results = similarity_stream.collect::<Vec<_>>().await;
    
    assert!(
        !results.is_empty(),
        "Rerank response should not be empty"
    );
    Ok(())
}
