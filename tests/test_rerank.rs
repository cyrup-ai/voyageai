use voyageai::{RerankModel, RerankRequest, VoyageAiClient, traits::llm::{Reranker, Embedder}};
use tokio_stream::StreamExt;

#[tokio::test]
async fn test_rerank() -> Result<(), Box<dyn std::error::Error>> {
    let config = voyageai::VoyageConfig::new(
        std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set")
    );
    let client = VoyageAiClient::new_with_config(config);

    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital of France.".to_string(),
        "London is the capital of the United Kingdom.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
    ];

    // Use the Reranker trait directly
    let stream = client.rerank(query, documents);
    
    // Collect results 
    let response: Vec<_> = stream.collect().await;
    
    // Take the first 2 results if there are enough
    let response = if response.len() >= 2 {
        response.into_iter().take(2).collect()
    } else {
        response
    };

    assert!(
        !response.is_empty(),
        "Rerank operation returned empty results"
    );
    assert_eq!(
        response.len(),
        2,
        "Expected 2 results due to take(2) parameter"
    );
    assert!(
        response[0].similarity >= response[1].similarity,
        "Results should be sorted by similarity score"
    );
    assert!(
        response[0].document.contains("Paris"),
        "Most relevant document should be the first one (about Paris)"
    );
    Ok(())
}

#[tokio::test]
async fn test_rerank_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
    let _api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
    let _client = VoyageAiClient::new();

    let result = RerankRequest::new("".to_string(), vec![], RerankModel::Rerank2, Some(2));

    assert!(result.is_err());

    Ok(())
}
