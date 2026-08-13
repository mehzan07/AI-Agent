import asyncio
from agents import Agent, Runner
from tools import calculate_total




agent = Agent(
    name="Software Helper",
        instructions="""
            You are a senior software architect.
            Explain software architecture concepts
            using practical examples.
            Prefer C# and .NET examples when appropriate.
            Keep explanations clear and understandable.
        """,
    tools=[calculate_total]
)

async def main():
    result = await Runner.run(
        agent,
        "Calculate the total cost of 15 products at €23.50 each.",
        
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())