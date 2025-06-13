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


# --- NEW: HELPER FUNCTION TO SANITIZE SCHEMAS ---
def sanitize_schema(schema: dict) -> dict:
    """
    Recursively removes fields from a JSON schema that are not recognized
    by the Google GenAI library's strict validator (e.g., 'title').
    """
    if not isinstance(schema, dict):
        return schema

    # Known valid keys for Google's Schema/FunctionDeclaration parameters.
    # We filter out 'title' and other potential non-standard keys.
    VALID_KEYS = {"type", "description", "properties", "required", "items", "enum", "format"}
    
    sanitized = {}
    for key, value in schema.items():
        if key in VALID_KEYS:
            if key == "properties":
                sanitized[key] = {
                    prop_name: sanitize_schema(prop_value)
                    for prop_name, prop_value in value.items()
                }
            elif key == "items":
                sanitized[key] = sanitize_schema(value)
            else:
                sanitized[key] = value
                
    return sanitized


# --- CONNECTION MANAGER (Unchanged) ---
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


# --- "SMART" ROUTER (Unchanged) ---
async def get_llm_tool_choice(user_query: str, available_tools: list) -> dict | None:
    if not available_tools:
        print("\n> No tools available from the server to choose from.")
        return None
    print("\n> Asking routing LLM to choose a tool...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=available_tools)
    result = await model.generate_content_async(user_query, tool_config={'function_calling_config': 'ANY'})
    
    if not result.candidates[0].content.parts or not result.candidates[0].content.parts[0].function_call:
        print("> LLM did not choose a tool. It might have answered directly.")
        if result.candidates[0].content.parts and result.candidates[0].content.parts[0].text:
             print(f"\nLLM Response: {result.candidates[0].content.parts[0].text}")
        return None

    function_call = result.candidates[0].content.parts[0].function_call
    tool_choice = {"tool_name": function_call.name, "arguments": {key: value for key, value in function_call.args.items()}}
    print(f"> LLM decided to call '{tool_choice['tool_name']}' with arguments: {tool_choice['arguments']}")
    return tool_choice


# --- MAIN EXECUTION LOOP ---
async def main():
    async with ConnectionManager(MCP_SERVER_URL) as mcp_session:
        print("--- Dynamic MCP Client Initialized ---")
        print("--> Discovering available tools from the server...")
        gemini_tools = []
        try:
            list_tools_result = await mcp_session.list_tools()
            if list_tools_result and list_tools_result.tools:
                for mcp_tool in list_tools_result.tools:
                    # --- THIS IS THE FIX ---
                    # Sanitize the schema from the server before creating the Gemini tool definition
                    sanitized_parameters = sanitize_schema(mcp_tool.inputSchema)
                    
                    gemini_tool = FunctionDeclaration(
                        name=mcp_tool.name,
                        description=mcp_tool.description,
                        parameters=sanitized_parameters # Use the clean version
                    )
                    gemini_tools.append(gemini_tool)

                tool_names = [t.name for t in gemini_tools]
                print(f"--> Success! Found {len(gemini_tools)} tools: {tool_names}")
            else:
                 print("--> Warning: Server reported no available tools.")
        except Exception as e:
            print(f"--> Error discovering tools: {e}")
            print("--> Cannot proceed without tools. Exiting.")
            return

        print("An LLM will now route your query to the appropriate discovered tool.")
        print("Type 'exit'to stop.")

        while True:
            try:
                user_query = input("\nYour Request> ")
                if user_query.lower() in ['exit', 'quit']:
                    print("Exiting client.")
                    break
                if not user_query.strip():
                    continue

                tool_to_call = await get_llm_tool_choice(user_query, gemini_tools)

                if not tool_to_call:
                    continue

                print("\nExecuting tool on MCP server...")
                print("This may take several minutes. See the SERVER terminal for real-time progress.")
                result_object = await mcp_session.call_tool(name=tool_to_call["tool_name"], arguments=tool_to_call["arguments"])
                
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
