use crate::{
    client::{
        embeddings_client::Client as EmbeddingsClient,
        rerank_client::DefaultRerankClient,
        search_client::SearchClient,
        RateLimiter,
        voyage_client::{VoyageAiClient, VoyageAiClientConfig},
    },
    config::VoyageConfig,
    errors::VoyageError,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct VoyageBuilder {
    config: Option<VoyageConfig>,
}

impl Default for VoyageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VoyageBuilder {
    pub fn new() -> VoyageBuilder {
        VoyageBuilder {
            config: None,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> VoyageBuilder {
        self.config = Some(VoyageConfig::new(api_key.into()));
        self
    }

    pub fn build(self) -> Result<VoyageAiClient, VoyageError> {
        let config = self.config.ok_or_else(|| VoyageError::BuilderError("API key is required".to_string()))?;
        let rate_limiter = Arc::new(RateLimiter::new());

        let embeddings_client = Arc::new(EmbeddingsClient::new(config.clone()));
        let rerank_client = Arc::new(DefaultRerankClient::new(
            config.clone(),
            rate_limiter.clone(),
        ));
        let search_client = Arc::new(SearchClient::new(
            (*embeddings_client).clone(),
            (*rerank_client).clone(),
        ));

        let client_config = VoyageAiClientConfig {
            config,
            embeddings_client,
            rerank_client,
            search_client,
        };

        Ok(VoyageAiClient {
            config: client_config,
        })
    }
}

