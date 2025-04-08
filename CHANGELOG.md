# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-04-07

### Changed

- **BREAKING**: Removed async_trait dependency, replacing it with the "Hidden Box/Pin" pattern
- API methods now return concrete types rather than Futures
- All trait methods are now synchronous, but contain async methods within their return types
- This update makes the API more efficient and aligns with project conventions

### Migration

- Old: `client.embeddings(request).await?`
- New: `client.embed_batch(&texts).await?`

## [0.1.0-alpha] - 2024-10-16

### Added

- Initial release of the VoyageAI Rust SDK
- Basic support for embeddings and reranking
- Simple examples and tests
