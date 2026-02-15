# Differences from Langfuse Python SDK

This document tracks major differences between ElasticDash Python SDK and the original Langfuse SDK.

## Overview

ElasticDash maintains API compatibility with core Langfuse concepts while extending functionality for the ElasticDash platform. This document helps developers understand where the implementations diverge.

---

## Architecture Changes

### API Endpoint
- **Langfuse:** Points to `https://cloud.langfuse.com` by default
- **ElasticDash:** Points to `https://logger.elasticdash.com` by default
- Configured via `ELASTICDASH_BASE_URL` environment variable

### OpenTelemetry Integration
- **ElasticDash:** Enhanced OpenTelemetry integration with additional span attributes and exporters
- More comprehensive distributed tracing support
- Additional metadata capture for production observability

---

## API Differences

### Client Initialization
- Environment variables use `ELASTICDASH_*` prefix instead of `LANGFUSE_*`
- `ELASTICDASH_PUBLIC_KEY`, `ELASTICDASH_SECRET_KEY`, `ELASTICDASH_BASE_URL`

### Additional Methods
*(To be documented as features diverge)*

---

## Feature Differences

### LLM Provider Support
- Core providers (OpenAI, Langchain) maintained from Langfuse
- *(Additional providers to be documented as added)*

### Instrumentation
- Enhanced automatic instrumentation capabilities
- Additional metadata collection for production monitoring
- Extended support for custom attributes

---

## Integration Differences

### Platform Integration
- **Langfuse:** Integrates with Langfuse Cloud/self-hosted platform
- **ElasticDash:** Integrates with ElasticDash Logger platform
- Different authentication and authorization mechanisms
- Different data storage and retrieval APIs

### API Client
- Auto-generated API client in `elasticdash/api/` directory
- Generated from ElasticDash OpenAPI specification
- May have different resource endpoints and types

---

## Removed Features

*(None documented yet - to be added as features are removed)*

---

## Added Features

### Enhanced Observability
- Additional OpenTelemetry span attributes for better trace analysis
- Enhanced metadata capture for LLM calls
- Improved error tracking and diagnostics

### Platform-Specific Features
- Integration with ElasticDash Logger UI
- Custom dashboards and analytics

*(More features to be documented as they are added)*

---

## Behavioral Differences

### Default Configuration
- Different default sampling rates
- Different flush intervals for batch processing
- Different retry policies for API calls

### Error Handling
- Enhanced error messages with ElasticDash-specific guidance
- Different error codes and responses from API

---

## Migration Notes

If migrating code from Langfuse to ElasticDash:

1. **Environment Variables:** Update all `LANGFUSE_*` to `ELASTICDASH_*`
2. **Import Statements:** Replace `langfuse` with `elasticdash` in imports
3. **API Endpoints:** Ensure `ELASTICDASH_BASE_URL` points to correct ElasticDash instance
4. **Authentication:** Obtain ElasticDash API keys (not compatible with Langfuse keys)
5. **Platform Features:** Some Langfuse Cloud features may not be available in ElasticDash

---

## Compatibility Notes

### What Remains Compatible
- Core tracing concepts (spans, generations, events)
- Basic OpenAI instrumentation patterns
- Langchain callback handler patterns
- Dataset management concepts

### What May Not Be Compatible
- API keys (separate authentication systems)
- Platform-specific features
- Some API endpoints and response formats
- UI and dashboard features

---

## Version Tracking

**Last Updated:** 2026-02-15
**ElasticDash Version:** 0.0.4
**Based on Langfuse Version:** (To be documented)

---

## Contributing

When making changes that diverge from Langfuse:
1. Document the change in this file
2. Explain the reasoning for the divergence
3. Update migration notes if it affects users
4. Consider whether the change should be contributed upstream to Langfuse

---

## Questions?

For questions about differences or migration:
- Open an issue on GitHub
- Refer to ElasticDash documentation
- Check Langfuse documentation for original concepts
