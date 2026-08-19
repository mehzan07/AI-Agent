import asyncio
import json
import os
import sys
from pathlib import Path

from agents import Agent, Runner, SQLiteSession, function_tool
from agents.mcp import MCPServerStdio

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MEMORY_FILE = "memory.json"

# Folder containing your articles
ARTICLES_FOLDER = "articles"

# Persistent RAG database
RAG_DATABASE = "rag_db"

# Chroma collection name
RAG_COLLECTION = "articles"

# Number of article chunks to retrieve
RAG_TOP_K = 5

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LONG-TERM MEMORY
# ============================================================

def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_memory(memory):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False
        )


memory = load_memory()


# ============================================================
# RAG SYSTEM
# ============================================================

class ArticleRAG:

    def __init__(self):

        print("Loading RAG embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Persistent Chroma database
        self.client = chromadb.PersistentClient(
            path=RAG_DATABASE
        )

        self.collection = (
            self.client
            .get_or_create_collection(
                name=RAG_COLLECTION
            )
        )

        print("RAG system ready.")

    # --------------------------------------------------------
    # Split article into chunks
    # --------------------------------------------------------

    def chunk_text(
        self,
        text,
        chunk_size=500,
        overlap=100
    ):

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            if chunk.strip():

                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    # --------------------------------------------------------
    # Add one article
    # --------------------------------------------------------

    def add_article(
        self,
        article_id,
        title,
        text
    ):

        chunks = self.chunk_text(text)

        if not chunks:
            return

        embeddings = (
            self.embedding_model
            .encode(chunks)
            .tolist()
        )

        ids = []
        metadatas = []

        for i in range(len(chunks)):

            ids.append(
                f"{article_id}_chunk_{i}"
            )

            metadatas.append(
                {
                    "article_id": article_id,
                    "title": title,
                    "chunk": i
                }
            )

        # Upsert means that running the program again
        # will update existing article chunks instead
        # of creating duplicates.
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"RAG: indexed '{title}' "
            f"({len(chunks)} chunks)"
        )

    # --------------------------------------------------------
    # Load articles from folder
    # --------------------------------------------------------

    def load_articles(
        self,
        folder=ARTICLES_FOLDER
    ):

        if not os.path.exists(folder):

            print(
                f"RAG: article folder '{folder}' "
                f"does not exist."
            )

            return

        article_files = [
            file
            for file in os.listdir(folder)
            if file.lower().endswith(".txt")
        ]

        if not article_files:

            print(
                "RAG: no .txt articles found."
            )

            return

        for filename in article_files:

            path = os.path.join(
                folder,
                filename
            )

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    text = file.read()

                article_id = os.path.splitext(
                    filename
                )[0]

                self.add_article(
                    article_id=article_id,
                    title=filename,
                    text=text
                )

            except Exception as error:

                print(
                    f"RAG: could not load "
                    f"{filename}: {error}"
                )

    # --------------------------------------------------------
    # Search articles
    # --------------------------------------------------------

    def search(
        self,
        query,
        top_k=RAG_TOP_K
    ):

        # Nothing to search
        if self.collection.count() == 0:

            return []

        query_embedding = (
            self.embedding_model
            .encode([query])
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        retrieved = []

        for document, metadata in zip(
            documents,
            metadatas
        ):

            retrieved.append(
                {
                    "text": document,
                    "title": metadata.get(
                        "title",
                        "Unknown article"
                    ),
                    "article_id": metadata.get(
                        "article_id",
                        ""
                    )
                }
            )

        return retrieved


# ============================================================
# CREATE RAG DATABASE
# ============================================================

rag = ArticleRAG()

# Load/index articles when the program starts
rag.load_articles()


# ============================================================
# RAG TOOL FOR THE AGENT
# ============================================================

@function_tool
def search_articles(query: str) -> str:
    """
    Search the article knowledge base.

    Use this tool whenever the user's question could be
    answered using information from the stored articles.

    Args:
        query: The question or topic to search for.
    """

    results = rag.search(
        query,
        top_k=RAG_TOP_K
    )

    if not results:

        return (
            "No relevant information was found "
            "in the article knowledge base."
        )

    output = []

    for i, result in enumerate(
        results,
        start=1
    ):

        output.append(
            f"""
SOURCE {i}
ARTICLE: {result['title']}

{result['text']}
"""
        )

    return "\n".join(output)


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # MCP SERVER
    # --------------------------------------------------------

    project_dir = Path(__file__).resolve().parent

    python_executable = sys.executable

    mcp_server_file = project_dir / "mcp_server.py"

    print()
    print("==============================")
    print("Memory + RAG + MCP Agent")
    print("==============================")
    print()

    print("Starting MCP server...")
    print(f"Python: {python_executable}")
    print(f"MCP server: {mcp_server_file}")

    # Start the MCP server and keep it connected
    # while the agent is running.
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

        # ----------------------------------------------------
        # AGENT
        # ----------------------------------------------------

        agent = Agent(

            name="Memory, RAG and MCP Agent",

            instructions="""
You are a helpful AI assistant.

You have access to several capabilities:

1. Long-term user memory
2. Conversation memory through the session
3. An article knowledge base through the
   search_articles tool
4. Tools provided by an MCP server

==================================================
RAG RULES
==================================================

- When the user's question is related to information
  that could be contained in the articles, use the
  search_articles tool.

- Use the retrieved article information as your
  primary source for questions about the articles.

- Do not invent information that is not supported
  by the retrieved articles.

- If the article search does not contain enough
  information, clearly tell the user that the
  information was not found in the article
  knowledge base.

==================================================
MCP RULES
==================================================

- Use MCP tools when they are useful for answering
  the user's question.

- The MCP tools are provided by an external MCP server.

- Do not pretend that an MCP tool exists if it is
  not available.

- Do not invent information returned by MCP tools.

==================================================
MEMORY RULES
==================================================

- Long-term user memory is provided in the prompt.

- Conversation memory is maintained through the
  session.

- Use remembered information when it is relevant.

- If the user tells you something useful about
  themselves, acknowledge it naturally.

==================================================
GENERAL BEHAVIOR
==================================================

Choose the appropriate capability based on the
user's request.

You may combine information from memory, RAG,
MCP tools, and your general knowledge when
appropriate.

Always answer clearly and helpfully.
""",

            # Existing local RAG tool
            tools=[
                search_articles
            ],

            # New MCP connection
            mcp_servers=[
                server
            ]
        )

        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------

        session = SQLiteSession(
            "memory_demo"
        )

        print("Agent is ready.")
        print("Type 'exit' to stop.")
        print(
            "Type 'remember:' followed by "
            "information to save it."
        )
        print()

        # ----------------------------------------------------
        # INTERACTIVE LOOP
        # ----------------------------------------------------

        while True:

            user_input = input(
                "You: "
            ).strip()

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if user_input.lower() == "exit":

                print()
                print("Goodbye!")

                break

            # ------------------------------------------------
            # EMPTY INPUT
            # ------------------------------------------------

            if not user_input:

                continue

            # ------------------------------------------------
            # SAVE LONG-TERM MEMORY
            # ------------------------------------------------

            if user_input.lower().startswith(
                "remember:"
            ):

                information = (
                    user_input[9:].strip()
                )

                if information:

                    memory.setdefault(
                        "user_information",
                        []
                    ).append(
                        information
                    )

                    save_memory(
                        memory
                    )

                    print(
                        "Agent: I will remember that."
                    )

                print()

                continue

            # ------------------------------------------------
            # LOAD LONG-TERM MEMORY
            # ------------------------------------------------

            memory_text = json.dumps(
                memory,
                indent=2,
                ensure_ascii=False
            )

            # ------------------------------------------------
            # CREATE PROMPT
            # ------------------------------------------------

            prompt = f"""
Here is information remembered about the user:

{memory_text}

Use this information when it is relevant.

The agent has access to an article knowledge base
through the search_articles tool.

The agent also has access to tools provided by
an MCP server.

If the user's question relates to the articles,
use the search_articles tool.

If the user's question can be answered using an
MCP tool, use the appropriate MCP tool.

User message:

{user_input}
"""

            # ------------------------------------------------
            # RUN AGENT
            # ------------------------------------------------

            try:

                result = await Runner.run(
                    agent,
                    prompt,
                    session=session
                )

                print()
                print(
                    "Agent:",
                    result.final_output
                )

            except Exception as error:

                print()
                print(
                    "Agent error:",
                    error
                )

            print()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())