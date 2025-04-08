use dotenvy::dotenv;
use env_logger::Builder;
use log::{debug, info, LevelFilter};
use std::io::Write;
use voyageai::{
    builder::embeddings::EmbeddingsRequestBuilder,
    client::SearchRequest,
    models::{
        embeddings::{EmbeddingModel, EmbeddingsInput},
        search::{SearchModel, SearchQuery, SearchType},
    },
    InputType, RerankModel, RerankRequest, VoyageBuilder,
    traits::llm::{Embedder, Reranker},
};
use tokio_stream::StreamExt;

#[tokio::test]
async fn test_voyage_ai_client() -> Result<(), Box<dyn std::error::Error>> {
    // Set up logging first for better debugging
    let mut builder = Builder::from_default_env();
    builder
        .format(|buf, record| writeln!(buf, "{} - {}", record.level(), record.args()))
        .filter(None, LevelFilter::Debug)
        .init();

    // Load environment variables and check if we should run the test
    dotenv().ok();
    let api_key = match std::env::var("VOYAGE_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping test: VOYAGE_API_KEY not set");
            return Ok(());
        }
    };

    let client = VoyageBuilder::new()
        .with_api_key(api_key)
        .build()
        .expect("Failed to build client");

    // Test embeddings
    let texts = vec![
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany",
    ];

    let embeddings_request = EmbeddingsRequestBuilder::default()
        .input(EmbeddingsInput::Multiple(
            texts.iter().map(ToString::to_string).collect(),
        ))
        .model(EmbeddingModel::Voyage3Large)
        .input_type(InputType::Document)
        .build()
        .expect("Failed to build embeddings request");

    info!("Submitting embeddings request");

    let embeddings_response = client
        .embeddings(embeddings_request)
        .await
        .expect("Failed to get embeddings");

    info!("Received embeddings response successfully");

    for (i, embedding_data) in embeddings_response.data.iter().enumerate() {
        assert_eq!(embedding_data.object, "embedding");
        assert_eq!(embedding_data.index, i);
        assert!(!embedding_data.embedding.is_empty());
        println!("Embedding {} length: {}", i, embedding_data.embedding.len());
    }

    // Test rerank using Reranker trait
    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital of France.".to_string(),
        "London is the capital of the United Kingdom.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
    ];

    info!("Creating rerank stream");
    
    // Use the Reranker trait directly
    let similarity_stream = client.rerank(query, documents);
    
    // Collect all results
    let rerank_results = similarity_stream.collect::<Vec<_>>().await;
    
    info!("Rerank response received successfully");
    debug!("Raw rerank response: {:?}", rerank_results);

    if rerank_results.is_empty() {
        panic!("Rerank results are empty");
    } else {
        info!("Rerank results received successfully");

        // Verify that the top result has the highest similarity score
        let top_result = &rerank_results[0];
        info!("Top result document: {}", top_result.document);
        assert!(
            top_result.similarity >= rerank_results[1].similarity,
            "Top result should have the highest similarity score"
        );

        // Print all rerank results for debugging
        info!("All rerank results:");
        for (i, result) in rerank_results.iter().enumerate() {
            info!("Result {}: Score: {}", i, result.similarity);
        }
    }

    // Test search
    let search_query = "capital of France".to_string();
    let search_request = SearchRequest {
        query: SearchQuery {
            query: search_query.clone(),
            model: SearchModel::default(),
            max_results: None,
            num_results: Some(10),
            include_metadata: None,
        },
        documents: Some(texts.iter().map(|&s| s.to_string()).collect()),
        embeddings: Some(
            embeddings_response
                .data
                .iter()
                .map(|d| d.embedding.clone())
                .collect(),
        ),
        model: SearchModel::default(),
        top_k: None,
        search_type: SearchType::Similarity,
    };

    let search_response = client
        .search(search_request)
        .await
        .expect("Failed to perform search");

    info!("Search results:");
    for result in search_response {
        info!("Score: {}, Index: {}", result.score, result.index);
    }
    Ok(())
}
