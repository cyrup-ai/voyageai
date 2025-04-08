#[cfg(test)]
mod tests {
    use voyageai::{
        builder::embeddings::EmbeddingsRequestBuilder,
        cosine_similarity,
        models::embeddings::{EmbeddingModel, EmbeddingsInput},
        InputType, VoyageAiClient, VoyageConfig,
        traits::llm::Embedder,
    };

    #[tokio::test]
    async fn test_embeddings_similarity() {
        // Load environment variables and check if we should run the test
        dotenvy::dotenv().ok();
        let api_key = match std::env::var("VOYAGE_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: VOYAGE_API_KEY not set");
                return;
            }
        };

        let config = VoyageConfig::new(api_key);
        let client = VoyageAiClient::new_with_config(config);

        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn canine leaps above an idle hound",
            "The sky is blue",
        ];

        // Option 1: Use embed_batch directly
        let texts_vec: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        let embeddings = client.embed_batch(&texts_vec).await.expect("Failed to create embeddings");
        
        assert_eq!(embeddings.len(), 3, "Expected 3 embeddings");

        let embedding1 = &embeddings[0];
        let embedding2 = &embeddings[1];
        let embedding3 = &embeddings[2];

        let similarity_1_2 = cosine_similarity(embedding1, embedding2);
        let similarity_1_3 = cosine_similarity(embedding1, embedding3);
        let similarity_2_3 = cosine_similarity(embedding2, embedding3);

        println!("Similarity between 1 and 2: {}", similarity_1_2);
        println!("Similarity between 1 and 3: {}", similarity_1_3);
        println!("Similarity between 2 and 3: {}", similarity_2_3);

        // Test relative ordering of similarities only since actual values will vary
        assert!(
            similarity_1_2 > similarity_1_3,
            "Similarity between similar sentences should be higher"
        );
        assert!(
            similarity_1_2 > similarity_2_3,
            "Similarity between similar sentences should be higher"
        );
    }
}
