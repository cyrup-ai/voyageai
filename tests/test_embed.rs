use dotenvy::dotenv;
use voyageai::{
    builder::embeddings::EmbeddingsRequestBuilder,
    models::embeddings::{EmbeddingModel, EmbeddingsInput},
    InputType, VoyageAiClient, VoyageConfig,
    traits::llm::Embedder,
};

#[tokio::test]
async fn test_embedding() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables and check if we should run the test
    dotenv().ok();
    let api_key = match std::env::var("VOYAGE_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping test: VOYAGE_API_KEY not set");
            return Ok(());
        }
    };

    let config = VoyageConfig::new(api_key);
    let client = VoyageAiClient::new_with_config(config);

    let inputs = [
        "Soul music emerged in the 1950s.",
        "Rock and roll revolutionized popular music.",
        "Soul and rock both influenced modern music.",
    ];

    let request = EmbeddingsRequestBuilder::new()
        .input(EmbeddingsInput::Multiple(
            inputs.iter().map(|&s| s.to_string()).collect(),
        ))
        .model(EmbeddingModel::Voyage3Large)
        .input_type(InputType::Document)
        .build()
        .expect("Failed to build embeddings request");

    // Option 1: Using the Embedder trait directly
    let texts_vec: Vec<String> = inputs.iter().map(|&s| s.to_string()).collect();
    let embeddings = client.embed_batch(&texts_vec).await
        .expect("Failed to create embeddings");

    assert_eq!(
        embeddings.len(),
        inputs.len(),
        "Number of embeddings should match number of inputs"
    );
    embeddings.iter().for_each(|embedding| {
        assert!(
            !embedding.is_empty(),
            "Embedding should not be empty"
        );
    });
    Ok(())
}

#[tokio::test]
async fn test_embedding_single_input() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables and check if we should run the test
    dotenv().ok();
    let api_key = match std::env::var("VOYAGE_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping test: VOYAGE_API_KEY not set");
            return Ok(());
        }
    };

    let config = VoyageConfig::new(api_key);
    let client = VoyageAiClient::new_with_config(config);

    let input = "Soul rock music combines elements of both genres.";

    let request = EmbeddingsRequestBuilder::new()
        .input(EmbeddingsInput::Single(input.to_string()))
        .model(EmbeddingModel::Voyage3Large)
        .input_type(InputType::Document)
        .build()
        .expect("Failed to build embeddings request");

    // Use the Embedder trait to get a single embedding
    let embedding = client.embed(input).await
        .expect("Failed to create embedding");

    assert!(
        !embedding.is_empty(),
        "Embedding should not be empty"
    );
    Ok(())
}
