# mcp_client.py
import asyncio
import json
import os
import traceback
from gradio_client import Client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# --- Configuration ---
GRADIO_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:7860")
console = Console()

async def display_research_progress():
    """Displays an animated progress indicator during research"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        tasks = [
            progress.add_task("[cyan]Creating research plan...", total=None),
            progress.add_task("[yellow]Gathering research data...", total=None),
            progress.add_task("[green]Synthesizing findings...", total=None),
            progress.add_task("[blue]Formatting response...", total=None)
        ]
        
        while True:
            for task in tasks:
                progress.update(task, advance=0.1)
            await asyncio.sleep(0.5)

async def run_gradio_client_query_async(query: str):
    """Connects to the Gradio server and executes the RAG tool with enhanced output formatting."""
    try:
        # Show connection status
        console.print(f"Connecting to Gradio server at: {GRADIO_SERVER_URL}")
        
        # Create progress display task
        progress_task = asyncio.create_task(display_research_progress())
        
        # Create a Gradio client with proper error handling
        try:
            client = Client(GRADIO_SERVER_URL)
            console.print("Successfully connected to Gradio server.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gradio server: {str(e)}\nPlease check if the server is running at {GRADIO_SERVER_URL}")
        
        # Execute query with increased timeout and chunked processing
        console.print(f"Executing RAG tool with query: {query}\n")
        try:
            # Increase timeout to 300 seconds (5 minutes)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.predict(
                        query,
                        fn_index=0  # Use function index instead of api_name
                    )
                ),
                timeout=300
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Research operation timed out after 300 seconds.\n"
                "This could be due to:\n"
                "1. High server load\n"
                "2. Complex query requiring extensive processing\n"
                "3. Network connectivity issues\n"
                "Consider breaking down your query into smaller parts."
            )
        except Exception as e:
            if "Failed to gather research" in str(e):
                raise RuntimeError(
                    "Research phase failed. This could be due to:\n"
                    "1. Invalid or empty query\n"
                    "2. API rate limits exceeded\n"
                    "3. No relevant results found\n"
                    "4. Network connectivity issues\n"
                    f"Server message: {str(e)}"
                )
            raise RuntimeError(f"Error during research: {str(e)}")
            
        # Cancel progress display
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        # Clear screen for results
        console.clear()
        
        # Print header
        console.print(Panel.fit(
            "[bold green]Research Results[/]",
            subtitle="Multi-Step RAG Analysis"
        ))
        
        # Handle empty or invalid results
        if not result:
            raise ValueError("Server returned an empty response")
            
        if not isinstance(result, str):
            raise ValueError(
                "Received invalid response type from server.\n"
                f"Expected string, got: {type(result)}\n"
                f"Content preview: {str(result)[:100]}..."
            )

        # Check for error messages in the result
        error_indicators = [
            "Error performing research",
            "Error generating answer",
            "Failed to gather research",
            "No relevant information found"
        ]
        
        if any(indicator in result for indicator in error_indicators):
            raise RuntimeError(
                "Server encountered an error during processing:\n"
                f"{result}\n\n"
                "Suggestions:\n"
                "1. Check if your query is clear and specific\n"
                "2. Try breaking down the query into smaller parts\n"
                "3. Verify that all API keys are valid\n"
                "4. Check server logs for more details"
            )

        # Display the structured content
        sections = result.split("\n#")
        for section in sections:
            if not section.strip():
                continue
            
            # Handle section title
            if "\n" in section:
                title, content = section.split("\n", 1)
            else:
                title, content = section, ""
            
            # Print section header
            console.print(f"\n[bold blue]# {title.strip()}[/]\n")
            
            # Handle different content types
            if "```" in content:
                # Handle code blocks
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Regular text
                        if part.strip():
                            console.print(Markdown(part.strip()))
                    else:  # Code block
                        try:
                            lang = part.split("\n")[0] or "python"
                            code = "\n".join(part.split("\n")[1:])
                            if code.strip():
                                console.print(Syntax(code.strip(), lang, theme="monokai"))
                        except Exception as e:
                            console.print(f"[yellow]Warning: Failed to format code block: {str(e)}[/]")
                            console.print(part)
            else:
                # Regular markdown content
                try:
                    console.print(Markdown(content.strip()))
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to format markdown: {str(e)}[/]")
                    console.print(content.strip())
            
            console.print("\n" + "─" * 80 + "\n")  # Section separator

    except Exception as e:
        error_msg = str(e)
        
        # Create error panel with specific troubleshooting steps
        error_panel = Panel.fit(
            f"[bold red]Error during research:[/]\n{error_msg}\n\n"
            "[yellow]Troubleshooting Steps:[/]\n"
            "1. Server Connection:\n"
            f"   - Check if server is running at {GRADIO_SERVER_URL}\n"
            "   - Verify network connectivity\n\n"
            "2. API Configuration:\n"
            "   - Verify API keys in .env file\n"
            "   - Check for API rate limits\n\n"
            "3. Query Issues:\n"
            "   - Try breaking down complex queries\n"
            "   - Make query more specific\n"
            "   - Verify query format\n\n"
            "4. Debug Information:\n"
            "   - Check server logs for details\n"
            "   - Review error message carefully",
            title="Error Details",
            border_style="red"
        )
        
        console.print(error_panel)
        
        # Only show traceback for non-timeout errors in debug mode
        if os.getenv("DEBUG") and not isinstance(e, TimeoutError):
            console.print("\n[dim]Debug Traceback:[/]")
            console.print(Syntax(traceback.format_exc(), "python"))

        # Offer retry option with context-aware suggestions
        if isinstance(e, TimeoutError):
            if console.input("\n[yellow]Would you like to retry with a longer timeout? (y/n):[/] ").lower() == 'y':
                console.clear()
                return await run_gradio_client_query_async(query)
        elif "rate limit" in str(e).lower():
            console.print("\n[yellow]API rate limit reached. Please wait a few minutes before retrying.[/]")
        elif "no relevant information" in str(e).lower():
            console.print("\n[yellow]Try rephrasing your query or breaking it down into smaller parts.[/]")
        else:
            if console.input("\n[yellow]Would you like to retry the query? (y/n):[/] ").lower() == 'y':
                console.clear()
                return await run_gradio_client_query_async(query)

    # Add prompt for next action
    console.print("\n[bold blue]Press Enter for a new query or Ctrl+C to exit[/]")

if __name__ == "__main__":
    while True:
        try:
            # Clear screen and show welcome
            os.system('cls' if os.name == 'nt' else 'clear')
            console.print(Panel.fit(
                "[bold blue]Multi-Step RAG Research System[/]\n"
                "[dim]Advanced technical research and analysis[/]"
            ))
            
            # Get query
            query = input("\nEnter your research query: ").strip()
            if query:
                asyncio.run(run_gradio_client_query_async(query))
                input()  # Wait for Enter before next query
            else:
                console.print("[yellow]Please enter a query.[/]")
                
        except KeyboardInterrupt:
            console.print("\n[bold green]Goodbye![/]")
            break
