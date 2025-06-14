# llm_tool_caller_client.py

import asyncio
from contextlib import AsyncExitStack
import json
import os
import google.generativeai as genai
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


# --- 2. HELPER FUNCTIONS ---
def get_multiline_input() -> str:
    print("\nYour Request (type '(end)' on a new line to submit, or '/new', 'exit'):")
    lines = []
    while True:
        try:
            line = input()
            stripped_line = line.strip().lower()
            if stripped_line in ['exit', '/new']:
                return stripped_line
            if stripped_line == '(end)':
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            return 'exit'
    return "\n".join(lines)


# --- 3. CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self, server_url):
        self.server_url, self.session, self.exit_stack = server_url, None, AsyncExitStack()
    async def __aenter__(self):
        sse_transport = await self.exit_stack.enter_async_context(sse_client(url=self.server_url))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*sse_transport))
        await self.session.initialize()
        return self.session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()


# --- 4. THE CONTEXT-AWARE LOGIC ---
# --- THIS IS THE FIX: The function signature now correctly accepts all 4 arguments ---
async def process_turn(
    user_query: str, 
    text_history: list[str], 
    tool_manifest: str, 
    mcp_session: ClientSession
) -> list[str]:
    """
    Handles a single turn of the conversation using a simple text-based history.
    """
    # --- STEP 1: ROUTER ---
    router_model = genai.GenerativeModel("gemini-1.5-flash")
    history_context = "\n".join(text_history)
    router_prompt = f"""
    You are a smart router that analyzes a conversation and a user's latest request to decide which tool to call.

    # Conversation History (for context):
    ---
    {history_context if history_context else "No history yet."}
    ---

    # User's Latest Request:
    "{user_query}"

    # Available Tools (Menu):
    ---
    {tool_manifest}
    ---

    # Your Task:
    Analyze the user's LATEST request. Use the conversation history to understand context (like location, topic, or what "it" or "that" refers to).
    Then, respond with a single JSON object specifying the best tool to call and the arguments.
    The arguments should be self-contained and have all necessary information (e.g., include the location in the search query).
    If no tool is appropriate, set "tool_name" to "none".
    Respond ONLY with the JSON object.
    """

    print("\n> Thinking (Step 1/3: Choosing a tool)...")
    router_response = await router_model.generate_content_async(router_prompt)
    
    tool_to_call = None
    try:
        cleaned_text = router_response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(cleaned_text)
        if decision.get("tool_name") and decision.get("tool_name") != "none":
            tool_to_call = {"name": decision["tool_name"], "args": decision.get("arguments", {})}
    except Exception as e:
        print(f"> LLM failed to decide on a tool or gave an invalid response: {router_response.text}")
        return [f"User: {user_query}", f"Model: I'm sorry, I had trouble understanding how to proceed."]

    if not tool_to_call:
        print("> LLM decided no tool was necessary. Responding directly...")
        chat_model = genai.GenerativeModel("gemini-1.5-flash")
        chat_response = await chat_model.generate_content_async(f"Conversation History:\n{history_context}\n\nUser's Request:\n{user_query}\n\nRespond conversationally.")
        final_text = chat_response.text
        print(f"\nLLM Response: {final_text}")
        return [f"User: {user_query}", f"Model: {final_text}"]

    # --- STEP 2: EXECUTE ---
    print(f"> LLM decided to call '{tool_to_call['name']}' with arguments: {tool_to_call['args']}")
    print("\nExecuting tool on MCP server...")
    result_object = await mcp_session.call_tool(name=tool_to_call["name"], arguments=tool_to_call["args"])
    
    if not (result_object and result_object.content):
        error_message = "Tool call did not return any content."
        print(f"\n--- ❌ ERROR --- \n{error_message}")
        return [f"User: {user_query}", f"Model Error: {error_message}"]

    final_result_dict = json.loads(result_object.content[0].text)
    print("\n--- ✅ Tool Execution Complete ---")
    print(json.dumps(final_result_dict, indent=2))

    # --- STEP 3: CONDITIONAL SYNTHESIS ---
    direct_answer_keys = ["answer", "summary", "explanation", "report", "message"]
    for key in direct_answer_keys:
        if key in final_result_dict and final_result_dict.get(key):
            print(f"> Direct answer found in tool result. Presenting directly.")
            final_text = final_result_dict[key]
            print(f"\nLLM Response: {final_text}")
            return [f"User: {user_query}", f"Model: The tool '{tool_to_call['name']}' returned: {final_text}"]
    
    print("\n> Thinking (Step 2/2: Generating final summary)...")
    synthesis_model = genai.GenerativeModel("gemini-1.5-flash")
    synthesis_prompt = f"A user asked: \"{user_query}\". The '{tool_to_call['name']}' tool ran and produced this result: {json.dumps(final_result_dict)}. Please state this result to the user in a helpful, conversational way."
    final_response = await synthesis_model.generate_content_async(synthesis_prompt)
    final_text = final_response.text
    print(f"\nLLM Response: {final_text}")
    return [f"User: {user_query}", f"Model: {final_text}"]


# --- 5. MAIN EXECUTION LOOP ---
async def main():
    async with ConnectionManager(MCP_SERVER_URL) as mcp_session:
        print("--- Manual Tool-Calling MCP Client (with Text-Based Memory) ---")
        
        print("--> Discovering available tools from the server...")
        tool_manifest = ""
        try:
            list_tools_result = await mcp_session.list_tools()
            if list_tools_result and list_tools_result.tools:
                tool_descriptions = []
                for mcp_tool in list_tools_result.tools:
                    desc = (f"- Tool Name: `{mcp_tool.name}`\n"
                            f"  Description: {mcp_tool.description}\n"
                            f"  Arguments (JSON Schema): {json.dumps(mcp_tool.inputSchema)}")
                    tool_descriptions.append(desc)
                tool_manifest = "\n\n".join(tool_descriptions)
                print(f"--> Success! Found {len(list_tools_result.tools)} tools.")
            else:
                 print("--> Warning: Server reported no available tools.")
        except Exception as e:
            print(f"--> Error discovering tools: {e}")
            return
        
        text_history = []
        print("\nNew conversation started. Type `/new` to reset memory.")

        while True:
            try:
                user_query = get_multiline_input()

                if user_query.strip().lower() == 'exit':
                    print("Exiting client.")
                    break
                if user_query.strip().lower() == '/new':
                    text_history = []
                    print("\n✨ New conversation started. Memory has been cleared.")
                    continue
                if not user_query.strip():
                    continue
                
                # The call to process_turn now correctly passes all 4 arguments
                new_history_entries = await process_turn(user_query, text_history, tool_manifest, mcp_session)
                text_history.extend(new_history_entries)

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")