use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::sync::Arc;
use voyageai::{
    EmbeddingModel, VoyageAiClient, VoyageConfig,
    traits::llm::Embedder,
    client::embeddings_client::Client as EmbeddingsClient,
    client::rerank_client::DefaultRerankClient,
    client::search_client::SearchClient,
    client::voyage_client::VoyageAiClientConfig,
    client::RateLimiter,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate embeddings for text
    Embed {
        /// Text to embed
        #[clap(short, long)]
        text: Vec<String>,

        /// Model to use for embeddings
        #[clap(short, long, default_value = "voyage-3-large")]
        model: String,
    },
    /// Rerank documents based on a query
    Rerank {
        /// Query to use for reranking
        #[clap(short, long)]
        query: String,

        /// Documents to rerank
        #[clap(short, long)]
        documents: Vec<String>,

        /// Number of top results to return
        #[clap(short, long)]
        top_k: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Get API key from environment
    let api_key = std::env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set");
    let config = VoyageConfig::new(api_key);
    
    // Create clients
    let embeddings_client = EmbeddingsClient::new(config.clone());
    let rerank_client = DefaultRerankClient::new(config.clone(), Arc::new(RateLimiter::new()));
    let search_client = SearchClient::new(embeddings_client.clone(), rerank_client.clone());
    
    // Create client config
    let client_config = VoyageAiClientConfig {
        config,
        embeddings_client: Arc::new(embeddings_client),
        rerank_client: Arc::new(rerank_client),
        search_client: Arc::new(search_client),
    };
    
    // Create the client
    let client = VoyageAiClient {
        config: client_config,
    };

    handle_command(&cli, &client).await?;
    Ok(())
}

async fn handle_command(cli: &Cli, client: &VoyageAiClient) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Embed { ref text, ref model } => {
            let _model = match model.as_str() {
                "voyage-3-large" => EmbeddingModel::Voyage3Large,
                "voyage-code-3" => EmbeddingModel::VoyageCode3,
                _ => EmbeddingModel::Voyage3Large,
            };

            // Use the embeddings client directly with our new API
            let embedding_vectors = client.embed_batch(text).await?;

            println!("Generated {} embeddings", embedding_vectors.len());
            for (i, embedding) in embedding_vectors.iter().enumerate() {
                println!("Embedding {}: {} dimensions", i, embedding.len());
            }
            Ok(())
        }

        Commands::Rerank {
            ref query,
            ref documents,
            top_k,
        } => {
            // Use the new find_similar_documents API
            println!("\nReranking documents by relevance to: {}", query);
            let mut similar_docs = client.find_similar_documents(query, documents.clone());
            
            // Process and display results
            println!("\nReranked documents by relevance:");
            let mut count = 0;
            while let Some(doc) = similar_docs.next().await {
                println!(
                    "Score {:.4}: {}",
                    doc.similarity, doc.document
                );
                
                count += 1;
                if let Some(k) = top_k {
                    if count >= k {
                        break;
                    }
                }
            }
            
            Ok(())
        }
    }
}
