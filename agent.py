import asyncio
import json
import os

from agents import Agent, Runner, SQLiteSession


MEMORY_FILE = "memory.json"


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(memory):

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


memory = load_memory()


agent = Agent(
    name="Memory Agent",
    instructions="""
    You are a helpful AI assistant.

    Use the information provided in the user's long-term memory
    when it is relevant to the conversation.

    If the user tells you something useful about themselves,
    acknowledge it naturally.
    """
)


async def main():

    session = SQLiteSession("memory_demo")

    print("Memory Agent")
    print("Type 'exit' to stop.")
    print("Type 'remember:' followed by information to save it.")
    print()

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Save information to long-term memory
        if user_input.lower().startswith("remember:"):

            information = user_input[9:].strip()

            memory.setdefault("user_information", []).append(
                information
            )

            save_memory(memory)

            print("Agent: I will remember that.")
            print()

            continue

        # Add long-term memory to the Agent's input
        memory_text = json.dumps(memory, indent=2)

        prompt = f"""
Here is information remembered about the user:

{memory_text}

Use this information when it is relevant.

User message:
{user_input}
"""

        result = await Runner.run(
            agent,
            prompt,
            session=session
        )

        print("Agent:", result.final_output)
        print()


if __name__ == "__main__":
    asyncio.run(main())