import asyncio
from agents import Agent, Runner, function_tool
from tools import calculate_total

## adding calculate Tool to AI-Agent
@function_tool
def calculate_total(price: float, quantity: int) -> float:
    return price * quantity

agent = Agent(
    name="Software Helper",
        instructions="""
            You are a helpful software development assistant.
            When a calculation is required, use the calculator tool.
            Explain the result clearly to the user.
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