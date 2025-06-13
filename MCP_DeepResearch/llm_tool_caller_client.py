# llm_tool_caller_client.py

import asyncio
from contextlib import AsyncExitStack
import json
import os
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.sse import sse_client

# --- 1. CONFIGURATION ---
load_dotenv()
MCP_SERVER_URL = "http://localhost:8000/sse"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)


# --- 2. DYNAMIC TOOL DEFINITION (Using Google's Native Format) ---
DEEP_RESEARCH_TOOL = FunctionDeclaration(
    name="conduct_deep_research",
    description=(
        "Performs comprehensive, multi-agent research on a given query. "
        "Use this for complex questions that require web searches, planning, "
        "and a detailed final report. Best for technical, scientific, or in-depth topics."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The specific and detailed research question or topic.",
            },
        },
        "required": ["query"],
    },
)

# --- 3. CONNECTION MANAGER (Unchanged) ---
class ConnectionManager:
    def __init__(self, server_url):
        self.server_url = server_url
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        sse_transport = await self.exit_stack.enter_async_context(sse_client(url=self.server_url))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*sse_transport))
        await self.session.initialize()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()


# --- 4. THE "SMART" PART: LETTING AN LLM CHOOSE THE TOOL ---
async def get_llm_tool_choice(user_query: str) -> dict | None:
    """
    Uses an LLM to decide which tool to call based on the user's query.
    """
    print("\n> Asking routing LLM to choose a tool...")

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[DEEP_RESEARCH_TOOL]
    )
    
    # --- THIS IS THE FIX ---
    # The correct keyword to force a tool call is 'ANY'.
    result = await model.generate_content_async(
        user_query,
        tool_config={'function_calling_config': 'ANY'}
    )
    
    if not result.candidates[0].content.parts or not result.candidates[0].content.parts[0].function_call:
        print("> LLM was forced to choose a tool but failed. This is unexpected.")
        return None

    function_call = result.candidates[0].content.parts[0].function_call
    
    tool_choice = {
        "tool_name": function_call.name,
        "arguments": {key: value for key, value in function_call.args.items()}
    }
    print(f"> LLM decided to call '{tool_choice['tool_name']}' with arguments: {tool_choice['arguments']}")
    return tool_choice

# --- 5. MAIN EXECUTION LOOP ---
async def main():
    async with ConnectionManager(MCP_SERVER_URL) as mcp_session:
        print("--- Smart Research Client Initialized ---")
        print("An LLM will route your query to the appropriate tool on the server.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            try:
                user_query = input("\nYour Request> ")
                if user_query.lower() in ['exit', 'quit']:
                    print("Exiting client.")
                    break
                if not user_query.strip():
                    continue

                tool_to_call = await get_llm_tool_choice(user_query)

                if not tool_to_call:
                    print("Could not determine an appropriate tool call. Please try rephrasing your request.")
                    continue

                print("\nExecuting tool on MCP server...")
                print("This may take several minutes. See the SERVER terminal for real-time progress.")

                result_object = await mcp_session.call_tool(
                    name=tool_to_call["tool_name"],
                    arguments=tool_to_call["arguments"]
                )
                
                if result_object and result_object.content:
                    final_result_json = result_object.content[0].text
                    final_result_dict = json.loads(final_result_json)
                    
                    print("\n--- ✅ RESEARCH COMPLETE ---")
                    print(f"Status: {final_result_dict.get('status')}")
                    print("\n--- 📝 FINAL REPORT ---")
                    print(final_result_dict.get('report', 'No report was generated.'))
                else:
                    print("\n--- ❌ ERROR ---")
                    print("Tool call did not return any content.")

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")