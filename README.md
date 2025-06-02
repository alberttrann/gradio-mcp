# Gradio Multi-Component Pipeline RAG System

## Overview

This project implements an advanced multi-step Retrieval Augmented Generation (RAG) system designed for comprehensive technical research and analysis. It features a server-client architecture:

*   **Server (`mcp_server.py`)**: A Gradio-based application that exposes a powerful RAG pipeline. It leverages multiple AI models via OpenRouter and the Tavily search API for information retrieval and synthesis.
*   **Client (`mcp_client.py`)**: A rich command-line interface (CLI) that allows users to interact with the server, submit research queries, and view formatted results.

The system is designed to break down complex queries, gather relevant information from diverse sources, and synthesize it into detailed, well-structured technical reports.

## Features

### Server-Side (`mcp_server.py`)

*   **Primary RAG Pipeline (via `process_query` function):**
    *   Uses **Tavily API** for advanced, in-depth web searches, with a focus on academic and technical domains (e.g., `arxiv.org`, `github.com`).
    *   Employs a powerful language model (e.g., **Anthropic's Claude 3 Opus** via OpenRouter) for synthesizing research findings into a comprehensive answer.
    *   The synthesis prompt is engineered for detailed (1500-2000 words), academically rigorous content, including executive summaries, technical analysis, implementation guides, and critical evaluations.
    *   Structured Markdown output.
*   **Alternative Multi-Model RAG Pipeline (via `multi_step_rag_tool` - *defined but not directly exposed via Gradio by default*):**
    *   **Research Planning**: Uses an LLM (e.g., **DeepSeek R1**) to break down the user query into a detailed research plan with multiple areas.
    *   **Specialized Research Gathering**: Employs another LLM (e.g., **Google's Gemini 2.0 Flash**) to conduct focused research on each identified area.
    *   **Comprehensive Synthesis**: Utilizes a third LLM (e.g., **Qwen3 30B**) to synthesize all gathered findings into a detailed technical report.
*   **Robustness**: Implements retries with exponential backoff for API calls (`tenacity`).
*   **Configuration**: Easily configurable via a `.env` file for API keys and server settings.
*   **Logging**: Comprehensive logging (`logger_config.py`) to files and console for server operations, research, and synthesis steps. Log files are rotated daily.
*   **Gradio Interface**: Provides a web UI for interaction, showcasing query examples and system description.

### Client-Side (`mcp_client.py`)

*   **Rich CLI Experience**: Utilizes the `rich` library for an enhanced terminal UI, including formatted panels, markdown rendering, progress spinners, and syntax-highlighted code blocks.
*   **Real-time Progress**: Displays an animated progress indicator during the research process.
*   **Structured Output**: Clearly presents the research results from the server, maintaining formatting for titles, sections, and code.
*   **Error Handling**: Gracefully handles connection issues, timeouts (default 300s), and server-side errors, providing user-friendly messages and troubleshooting suggestions.
*   **Retry Options**: Offers to retry queries in case of timeouts or other recoverable errors.
*   **Configurable Server URL**: The target Gradio server URL can be set via the `MCP_SERVER_URL` environment variable.
*   **Debug Mode**: An optional `DEBUG` environment variable can be set to display full tracebacks for errors.

## Architecture

The system consists of:

1.  **`mcp_server.py`**: The core Gradio application.
    *   It initializes API clients (OpenRouter, Tavily).
    *   The main Gradio interface is wired to the `process_query` function, which executes a two-stage RAG:
        1.  `research_with_tavily()`: Fetches and processes research data.
        2.  `generate_answer_with_openrouter()`: Generates the final detailed report.
    *   It also contains the `multi_step_rag_tool` and its sub-functions (`craft_research_plan`, `gather_specialized_research`, `synthesize_findings`), representing a more granular, multi-model pipeline.
2.  **`mcp_client.py`**: A Python script that acts as a CLI to the Gradio server.
    *   It uses `gradio_client` to connect and send queries.
    *   Manages UI display and user interaction.
3.  **`logger_config.py`**: Sets up shared logging configurations used by both server (and potentially client if imported).
4.  **`utils.py`**: Contains helper functions for the server, such as response validation, parsing Tavily results, and formatting source information.
5.  **`.env` file (Server-side)**: Stores API keys and other sensitive configurations.
6.  **Requirement Files**:
    *   `requirements.txt`: Python dependencies for the server.
    *   `client_requirements.txt`: Python dependencies for the client.

## Prerequisites

*   Python 3.8+ (recommended, due to dependencies like `aiohttp`).
*   API Keys for:
    *   OpenRouter (for accessing various LLMs)
    *   Tavily API (for search)
*   A `.env` file correctly configured (see Configuration section).

## Setup & Installation

1.  **Clone the Repository** (Assuming the code is in a Git repository):
    ```bash
    git clone https://github.com/alberttrann/gradio-mcp.git
    cd gradio-mcp
    ```

2.  **Server Setup**:
    *   Navigate to the directory containing `mcp_server.py`.
    *   Create and activate a Python virtual environment:
        ```bash
        python -m venv venv_server
        source venv_server/bin/activate  # On Windows: venv_server\Scripts\activate
        ```
    *   Install server dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Create and configure the `.env` file (see Configuration section).

3.  **Client Setup**:
    *   Navigate to the directory containing `mcp_client.py`.
    *   Create and activate a Python virtual environment:
        ```bash
        python -m venv venv_client
        source venv_client/bin/activate  # On Windows: venv_client\Scripts\activate
        ```
    *   Install client dependencies:
        ```bash
        pip install -r client_requirements.txt
        ```

## Configuration

### Server (`.env` file)

Create a `.env` file in the same directory as `mcp_server.py` with the following content:

```env
OPENROUTER_API_KEY="your_openrouter_api_key"
TAVILY_API_KEY="your_tavily_api_key"
YOUR_SITE_URL="http://localhost:7860"  # URL for OpenRouter HTTP referer, adjust if server runs elsewhere
YOUR_SITE_NAME="My RAG MCP Server"     # Name for OpenRouter X-Title, customize as needed
```

Replace placeholder values with your actual API keys and desired settings.

### Client

The client can be configured using environment variables:

*   `MCP_SERVER_URL`: The URL of the running Gradio server. Defaults to `http://localhost:7860`.
    ```bash
    export MCP_SERVER_URL="http://your-server-address:port" # Linux/macOS
    set MCP_SERVER_URL="http://your-server-address:port"    # Windows CMD
    $env:MCP_SERVER_URL="http://your-server-address:port"   # Windows PowerShell
    ```
*   `DEBUG`: If set to any non-empty value (e.g., `true`), the client will print full Python tracebacks on error.
    ```bash
    export DEBUG=true # Linux/macOS
    ```

## Usage

1.  **Run the Server**:
    *   Ensure your server-side virtual environment is activated and the `.env` file is configured.
    *   Navigate to the server directory.
    *   Execute:
        ```bash
        python mcp_server.py
        ```
    *   The server will start, and you can typically access the Gradio web UI at `http://localhost:7860` (or the configured `YOUR_SITE_URL`).

2.  **Run the Client**:
    *   Ensure your client-side virtual environment is activated.
    *   (Optional) Set the `MCP_SERVER_URL` and `DEBUG` environment variables if needed.
    *   Navigate to the client directory.
    *   Execute:
        ```bash
        python mcp_client.py
        ```
    *   The client will start, clear the screen, and prompt you to enter a research query. Follow the on-screen instructions. Press Ctrl+C to exit.

## Code Structure

```
.
├── client_requirements.txt  # Client Python dependencies
├── requirements.txt         # Server Python dependencies
├── logger_config.py         # Logging setup
├── mcp_client.py            # CLI client application
├── mcp_server.py            # Gradio server application
├── utils.py                 # Utility functions for the server
├── .env.example             # Example environment file for server (you should create .env)
└── logs/                    # Directory for log files (created automatically)
```

## Note on the `multi_step_rag_tool`

The `mcp_server.py` script includes a sophisticated function named `multi_step_rag_tool` along with helper functions (`craft_research_plan`, `extract_research_areas`, `gather_specialized_research`, `synthesize_findings`). This tool outlines a more granular, three-stage RAG pipeline:
1.  **Plan**: Generate a research plan using a model like DeepSeek R1.
2.  **Gather**: Collect specialized information for each plan area using a model like Gemini 2.0 Flash.
3.  **Synthesize**: Compile all findings into a report using a model like Qwen3 30B.

Currently, the primary Gradio interface (`iface`) is wired to the `process_query` function, which uses a two-stage pipeline (Tavily search + Claude 3 Opus synthesis). The `multi_step_rag_tool` represents an alternative, potentially more advanced, pipeline that is defined within the codebase but would require changes to `mcp_server.py` to be the default endpoint for the Gradio interface.

## Potential Future Work

*   Wire the `multi_step_rag_tool` to the Gradio interface, perhaps as an alternative mode.
*   Allow selection of different LLMs for each step via the UI or client.
*   Add more sophisticated error handling and context passing between RAG steps.
*   Implement caching for research results.
