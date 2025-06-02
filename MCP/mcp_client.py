# mcp_client.py
import asyncio
import json
import os
import traceback
from gradio_client import Client

# --- Configuration ---
GRADIO_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:7860")

async def run_gradio_client_query_async(query: str):
    """
    Connects to the Gradio server and executes the RAG tool.
    """
    print(f"Connecting to Gradio server at: {GRADIO_SERVER_URL}")
    
    try:
        # Create a Gradio client for the server
        client = Client(GRADIO_SERVER_URL)
        print("Successfully connected to Gradio server.")
        
        print(f"Executing RAG tool with query: {query}")
        # The predict method will find the default function in the Interface
        result = client.predict(query)
        
        print("\n--- Result from RAG Tool ---")
        print(result)  # Result is already a string from the tool

    except ConnectionError:
        print(f"\nError: Could not connect to Gradio server at {GRADIO_SERVER_URL}")
        print("Make sure the server is running and the URL is correct.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Details:")
        traceback.print_exc()

if __name__ == "__main__":
    test_query = input("Enter your query for the RAG tool: ")
    if not test_query.strip():
        print("No query provided. Exiting.")
    else:
        asyncio.run(run_gradio_client_query_async(test_query))