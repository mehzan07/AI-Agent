# python
import asyncio
import json
import os

from agents import Agent, Runner, SQLiteSession, function_tool

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
# AGENT
# ============================================================

agent = Agent(

    name="Memory and RAG Agent",

    instructions="""
You are a helpful AI assistant.

You have access to two types of memory:

1. Long-term user memory
2. Conversation memory through the session

You also have access to an article knowledge base
through the search_articles tool.

IMPORTANT RAG RULES:

- When the user's question is related to information
  that could be contained in the articles, use the
  search_articles tool.

- Use the retrieved article information as your
  primary source for questions about the articles.

- Do not invent information that is not supported
  by the retrieved articles.

- If the article search does not contain enough
  information, clearly tell the user that the
  information was not found in the article knowledge base.

- You may combine information from the articles,
  the conversation, and your general knowledge when
  appropriate.

- If the user asks a normal general question that
  clearly does not require the articles, you do not
  need to use the RAG tool.

LONG-TERM MEMORY:

The user's long-term memory is provided in the prompt.

Use it when it is relevant to the conversation.

If the user tells you something useful about themselves,
acknowledge it naturally.

Always answer clearly and helpfully.
""",

    tools=[
        search_articles
    ]
)


# ============================================================
# MAIN
# ============================================================

async def main():

    session = SQLiteSession(
        "memory_demo"
    )

    print()
    print("==============================")
    print("Memory + RAG Agent")
    print("==============================")
    print()
    print("Type 'exit' to stop.")
    print(
        "Type 'remember:' followed by "
        "information to save it."
    )
    print()

    while True:

        user_input = input(
            "You: "
        )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if user_input.lower() == "exit":

            break

        # ----------------------------------------------------
        # SAVE LONG-TERM MEMORY
        # ----------------------------------------------------

        if user_input.lower().startswith(
            "remember:"
        ):

            information = (
                user_input[9:].strip()
            )

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

        # ----------------------------------------------------
        # LOAD LONG-TERM MEMORY
        # ----------------------------------------------------

        memory_text = json.dumps(
            memory,
            indent=2,
            ensure_ascii=False
        )

        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------

        prompt = f"""
Here is information remembered about the user:

{memory_text}

Use this information when it is relevant.

The agent also has access to an article knowledge
base through the search_articles tool.

If the user's question relates to the articles,
use the search_articles tool.

User message:

{user_input}
"""

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        try:

            result = await Runner.run(
                agent,
                prompt,
                session=session
            )

            print(
                "Agent:",
                result.final_output
            )

        except Exception as error:

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

