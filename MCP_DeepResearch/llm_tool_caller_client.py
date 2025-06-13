# llm_tool_caller_client.py

import asyncio
from contextlib import AsyncExitStack
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

# --- 1. CONFIGURATION ---
load_dotenv()
MCP_SERVER_URL = "http://localhost:8000/sse"

# --- 2. CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self, server_url):
        self.server_url = server_url
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url=self.server_url)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*sse_transport)
        )
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    # --- THE DEFINITIVE FIX ---
    # This function now correctly awaits the SINGLE final result.
    # It does not stream.
    async def call_tool(self, tool_name, arguments):
        """
        Awaits the final result from the tool call.
        """
        print("\nExecuting tool on MCP server...")
        print("This will take several minutes. Please watch the SERVER terminal for real-time progress.")
        
        # Await the coroutine to get the final CallToolResult object
        result_object = await self.session.call_tool(tool_name, arguments=arguments)
        
        # The text is in the 'content' list of the final result object
        if result_object and result_object.content:
            return result_object.content[0].text
        else:
            raise Exception("Tool call did not return any content.")


# --- 3. DIRECT TOOL CALLER (Unchanged logic, but now it will work) ---
async def run_research_query(user_query: str, connection_manager: ConnectionManager):
    print(f"\n> Your Query: {user_query}")
    if not user_query.strip():
        print("Empty query, please try again.")
        return

    try:
        # Directly call the research tool
        tool_output_json = await connection_manager.call_tool(
            tool_name="conduct_deep_research",
            arguments={"query": user_query}
        )
        
        print("\n  - Tool execution complete on server.")
        
        tool_output_dict = json.loads(tool_output_json)
        
        print("\n--- RESEARCH COMPLETE ---")
        print(f"Status: {tool_output_dict.get('status')}")
        print("\n--- REPORT ---")
        print(tool_output_dict.get('report', 'No report was generated.'))

    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Error during tool execution: {e}")

# --- 4. MAIN EXECUTION ---
async def main():
    async with ConnectionManager(MCP_SERVER_URL) as manager:
        print("--- Direct Research Client Initialized ---")
        print("Your input will be sent directly to the 'conduct_deep_research' tool.")
        print("Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_query = input("\nResearch Query> ")
                if user_query.lower() in ['exit', 'quit']:
                    print("Exiting client.")
                    break
                
                await run_research_query(
                    user_query=user_query,
                    connection_manager=manager
                )

            except Exception as e:
                print(f"\nAn unexpected error occurred in the main loop: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")