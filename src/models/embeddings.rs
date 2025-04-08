use crate::VoyageError;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputType {
    #[serde(rename = "query")]
    Query,
    #[serde(rename = "document")]
    Document,
    #[serde(rename = "code")]
    Code,
    #[serde(rename = "ast")]
    Ast,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Multiple(Vec<String>),
}

impl TryFrom<&[String]> for EmbeddingsInput {
    type Error = VoyageError;
    fn try_from(v: &[String]) -> Result<Self, Self::Error> {
        Ok(EmbeddingsInput::Multiple(v.to_vec()))
    }
}

impl TryFrom<Vec<&str>> for EmbeddingsInput {
    type Error = VoyageError;
    fn try_from(v: Vec<&str>) -> Result<Self, Self::Error> {
        Ok(EmbeddingsInput::Multiple(
            v.into_iter().map(String::from).collect(),
        ))
    }
}

impl From<&str> for EmbeddingsInput {
    fn from(s: &str) -> Self {
        EmbeddingsInput::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingsInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingsInput::Multiple(v)
    }
}

impl From<String> for EmbeddingsInput {
    fn from(s: String) -> Self {
        EmbeddingsInput::Single(s)
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingsInput,
    pub model: EmbeddingModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<InputType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    /// The type of object returned.
    #[serde(default)]
    pub object: String,
    /// A list of embedding data.
    pub data: Vec<EmbeddingData>,
    /// The model used for generating embeddings.
    #[serde(default)]
    pub model: String,
    /// Usage statistics for the request.
    pub usage: Usage,
}

/// Usage statistics for an embedding request.
#[derive(Debug, Deserialize)]
pub struct Usage {
    /// The total number of tokens used in the request.
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum EncodingFormat {
    #[serde(rename = "float")]
    Float,
    #[serde(rename = "base64")]
    Base64,
}

/// Supported embedding models by VoyageAI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbeddingModel {
    #[serde(rename = "voyage-3-large")]
    #[default]
    Voyage3Large,
    #[serde(rename = "voyage-code-3")]
    VoyageCode3,
}

impl EmbeddingModel {
    /// Returns the maximum context length for the model
    pub fn max_context_length(&self) -> usize {
        match self {
            Self::Voyage3Large | Self::VoyageCode3 => 32000,
        }
    }

    /// Returns the maximum number of tokens that can be processed in a single request
    pub fn max_tokens_per_request(&self) -> usize {
        match self {
            Self::Voyage3Large => 320_000,
            Self::VoyageCode3 => 320_000,
        }
    }

    /// Returns the embedding dimension for the model
    pub fn embedding_dimension(&self) -> usize {
        match self {
            Self::Voyage3Large => 2048,
            Self::VoyageCode3 => 1024,
        }
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Voyage3Large => write!(f, "voyage-3-large"),
            Self::VoyageCode3 => write!(f, "voyage-code-3"),
        }
    }
}

pub struct CodeEmbedding {
    pub text_embedding: Vec<f32>,
    pub ast_embedding: Vec<f32>,
}

