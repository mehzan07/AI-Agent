# mcp_agent.py:

import asyncio
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main():

    # Directory where this file is located
    project_dir = Path(__file__).resolve().parent

    # Python executable from the current virtual environment
    python_executable = sys.executable

    # Full path to the MCP server
    mcp_server_file = project_dir / "mcp_server.py"

    print("Starting MCP server...")
    print(f"Python: {python_executable}")
    print(f"MCP server: {mcp_server_file}")

    async with MCPServerStdio(
        name="Project MCP Server",

        params={
            "command": python_executable,
            "args": [
                str(mcp_server_file)
            ],
        },
    ) as server:

        print("MCP server connected.")
        print()
        print("MCP Agent is ready.")
        print("Type 'exit' to quit.")
        print()

        agent = Agent(
            name="MCP Agent",

            instructions="""
            You are a helpful AI assistant.

            You have access to tools provided by
            an MCP server.

            Use the MCP tools when they are useful
            for answering the user's question.

            Do not invent information returned by
            the MCP tools.
            """,

            mcp_servers=[
                server
            ],
        )

        while True:

            question = input("You: ").strip()

            if question.lower() == "exit":
                print()
                print("Goodbye!")
                break

            if not question:
                continue

            result = await Runner.run(
                agent,
                question
            )

            print()
            print("Agent:")
            print(result.final_output)
            print()


if __name__ == "__main__":
    asyncio.run(main())
