# LangChain NVIDIA NIM RAG Agent Launchable

A Brev launchable that sets up Marimo with the LangChain NVIDIA NIM RAG Agent notebook. This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) agent using LangGraph, LangChain, and NVIDIA NIM inference microservices.

## Overview

This launchable automatically:
- Sets up Marimo notebook environment
- Installs all required dependencies (LangChain, LangGraph, NVIDIA AI Endpoints, etc.)
- Downloads and converts the LangChain NVIDIA NIM notebook from Jupyter to Marimo format
- Configures Marimo to run as a systemd service

## GPU Requirements

Since this notebook uses **NVIDIA NIM API endpoints** (inference microservices), the models run on NVIDIA's infrastructure rather than locally. However, GPU recommendations are provided for local processing workloads:

### Minimum Requirements
- **GPU**: Any GPU (API-based inference, local GPU optional)
- **VRAM**: Not required for inference (handled by NVIDIA NIM API)
- **Use Case**: Suitable for testing and development

### Recommended Configuration
- **GPU**: NVIDIA RTX 3060 or better
- **VRAM**: 8GB+ VRAM
- **Use Case**: Local embeddings processing, document parsing, and general ML workloads

### Optimal Configuration
- **GPU**: NVIDIA RTX 3090, A10G, or better
- **VRAM**: 24GB+ VRAM
- **Use Case**: Heavy local processing, multiple concurrent requests, production workloads

### Notes
- The notebook uses NVIDIA NIM API endpoints, so model inference happens on NVIDIA's infrastructure
- Local GPU is beneficial for embeddings, document processing, and other computational tasks
- For production deployments with high throughput, consider NVIDIA A100 or H100 GPUs

## Setup Instructions

### 1. Create a Brev Instance

Create a new Brev instance and select a GPU configuration based on your needs (see GPU Requirements above).

### 2. Configure Environment Variables

Set the following environment variables in your Brev instance:

- `MARIMO_PORT`: Port for Marimo server (default: 8080)
- `MARIMO_REPO_URL`: Optional - URL to clone additional notebooks (default: marimo-team/examples)
- `MARIMO_NOTEBOOKS_DIR`: Optional - Directory name for notebooks (default: marimo-examples)

### 3. Run the Setup Script

The `oneshot.sh` script will automatically:
- Install Python and Marimo
- Install all required dependencies
- Download and convert the LangChain NVIDIA NIM notebook
- Set up Marimo as a systemd service

### 4. Configure API Keys

Before running the notebook, you'll need to set up API keys:

#### NVIDIA API Key (Required)
1. Get your NVIDIA API key from [NVIDIA API Catalog](https://build.nvidia.com/)
2. Set it as an environment variable:
   ```bash
   export NVIDIA_API_KEY="your-api-key-here"
   ```

#### Tavily API Key (Required for web search)
1. Get your Tavily API key from [Tavily](https://tavily.com/)
2. Set it as an environment variable:
   ```bash
   export TAVILY_API_KEY="your-api-key-here"
   ```

You can add these to your `~/.bashrc` or `~/.zshrc` to persist across sessions:
```bash
echo 'export NVIDIA_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export TAVILY_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### 5. Open Port on Brev

Make sure to open the Marimo port (default: 8080) in your Brev instance settings:
- Go to your Brev instance settings
- Open port `8080/tcp` (or your custom `MARIMO_PORT`)

### 6. Access Marimo

Once the setup is complete, access Marimo at:
```
http://your-brev-instance-url:8080
```

The LangChain NVIDIA NIM notebook will be available in the notebooks directory.

## Notebook Features

The notebook demonstrates:
- **RAG Agent**: Building a retrieval-augmented generation agent with LangGraph
- **NVIDIA NIM Integration**: Using NVIDIA NIM inference microservices for LLM inference
- **Web Search**: Integrating Tavily for real-time web search capabilities
- **Document Processing**: Loading and processing documents for RAG
- **Hallucination Detection**: Checking if generated content is grounded in retrieved documents

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
- Check that the notebook was downloaded: `ls ~/marimo-examples/langgraph_rag_agent_llama3_nvidia_nim.py`
- Verify conversion succeeded by checking the file exists and has content

### API Key Issues
- Ensure environment variables are set: `echo $NVIDIA_API_KEY`
- Restart the Marimo service after setting environment variables: `sudo systemctl restart marimo`

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

