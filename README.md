# LangChain NVIDIA NIM RAG Agent Launchable

Learn how to build production-ready RAG (Retrieval-Augmented Generation) agents using LangChain and NVIDIA NIM inference microservices. This notebook demonstrates advanced RAG techniques including adaptive routing, self-correction, and hallucination detection.

## Why NVIDIA + LangChain?

**NVIDIA NIM** (NVIDIA Inference Microservices) provides optimized, production-ready AI models that integrate seamlessly with LangChain:

- **High-Performance Inference**: Access to NVIDIA-optimized models like LLaMA 3.1 (8B, 70B, 405B) running on accelerated infrastructure
- **Enterprise-Ready**: Deploy models on-premises or in the cloud with full control over your data and IP
- **Seamless Integration**: Native LangChain integrations (`ChatNVIDIA`, `NVIDIAEmbeddings`) work just like other LangChain components
- **State-of-the-Art Models**: Choose from a wide range of models optimized for different tasks (chat, embeddings, re-ranking)
- **Production Scale**: Built for enterprise workloads with consistent APIs and containerized deployment

## What You'll Learn

This notebook walks you through building a sophisticated RAG agent that combines techniques from multiple research papers:

- **Adaptive RAG**: Intelligently route queries to vectorstore or web search based on question type
- **Corrective RAG**: Automatically fall back to web search when documents aren't relevant
- **Self-RAG**: Self-correct hallucinations and ensure answers address the question
- **Structured Output**: Use Pydantic models for reliable, type-safe LLM responses
- **Tool Calling**: Leverage function calling capabilities for complex agent workflows

You'll see how to:
- Use NVIDIA's optimized embeddings (`NVIDIAEmbeddings`) for document indexing
- Query NVIDIA-hosted LLaMA models (`ChatNVIDIA`) for generation
- Build stateful agent workflows with LangGraph
- Integrate web search (Tavily) for real-time information retrieval
- Implement quality checks and hallucination detection

## Deploy

[![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-35cIYF1zpnEPrzNSK6RBmEs8GHQ)

## Setup Instructions

### 1. Deploy

Click the deploy badge above for one-click deployment. Everything is automated:
- GPU instance provisioning
- Python and Marimo installation
- All dependencies (LangChain, LangGraph, NVIDIA AI Endpoints, etc.)
- Notebook setup and Marimo service configuration

### 2. Access Marimo

Once deployment completes (usually 2-3 minutes), access Marimo at:
```
http://your-brev-instance-url:8080
```

The LangChain NVIDIA NIM notebook will be available in the workspace.

### 3. Get API Keys

You'll need two API keys, which the notebook will prompt you for:

- **NVIDIA API Key**: Get yours from [NVIDIA API Catalog](https://build.nvidia.com/)
- **Tavily API Key**: Get yours from [Tavily](https://tavily.com/) (for web search functionality)

The notebook will securely prompt you to enter these keys when you run the relevant cells. No need to set environment variables manually!

## Technical Stack

The notebook showcases:
- **LangGraph**: Build complex, stateful agent workflows with graph-based orchestration
- **NVIDIA NIM Models**: Access to LLaMA 3.1 models (8B, 70B) and NVIDIA embeddings via API
- **LangChain Integrations**: `ChatNVIDIA` and `NVIDIAEmbeddings` components that work seamlessly with the LangChain ecosystem
- **Advanced RAG Patterns**: Implement research-backed techniques for production RAG systems
- **Structured Outputs**: Use Pydantic models for reliable, validated LLM responses

## Useful Commands

```bash
# Check Marimo service status
sudo systemctl status marimo

# View Marimo logs
sudo journalctl -u marimo -f

# Restart Marimo service
sudo systemctl restart marimo

# Stop Marimo service
sudo systemctl stop marimo

# Start Marimo service
sudo systemctl start marimo
```

## Dependencies

The setup script installs:
- `marimo` - Interactive notebook environment
- `langchain-nvidia-ai-endpoints` - LangChain integration for NVIDIA NIM
- `langchain-community` - Community LangChain integrations
- `langchain` - Core LangChain library
- `langgraph` - LangGraph for building stateful agents
- `tavily-python` - Tavily API client for web search
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML processing
- Plus other common ML/data science packages

## Troubleshooting

### Marimo service not starting
```bash
# Check service status
sudo systemctl status marimo

# View detailed logs
sudo journalctl -u marimo -n 50
```

### Notebook not appearing
- Check that the notebook exists: `ls ~/langchaindev-launchable/langgraph_rag_agent_llama3_nvidia_nim.py`
- Verify the file exists and has content

### API Key Issues
- The notebook will prompt you for API keys when needed
- Make sure you have valid keys from [NVIDIA API Catalog](https://build.nvidia.com/) and [Tavily](https://tavily.com/)
- Keys are entered securely in the notebook cells, not as environment variables

### Port Access Issues
- Verify the port is open in Brev instance settings
- Check firewall rules: `sudo ufw status`

## References

- [LangChain NVIDIA Integration](https://github.com/langchain-ai/langchain-nvidia)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NVIDIA API Catalog](https://build.nvidia.com/)
- [Marimo Documentation](https://docs.marimo.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

This launchable is provided as-is. Please refer to the original notebook repository for license information.

