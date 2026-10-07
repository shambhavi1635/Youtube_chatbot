

import os

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
RETRIEVER_K = 4

load_dotenv()


class PipelineError(Exception):
    """Something went wrong that the UI should show as a plain message."""


class TranscriptUnavailable(PipelineError):
    """No usable transcript for this video."""


class QuotaExceeded(PipelineError):
    """Gemini free-tier quota or rate limit hit."""


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def require_api_key() -> None:
    """Fail early with a readable message instead of deep inside an API call."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise PipelineError(
            "GOOGLE_API_KEY is not set. Add it to the .env file in the project root."
        )


def fetch_transcript(video_id: str, languages=("en",)) -> str:
    """Fetch the transcript and flatten it to plain text."""
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except TranscriptsDisabled:
        raise TranscriptUnavailable("This video has captions disabled.")
    except NoTranscriptFound:
        raise TranscriptUnavailable(
            "No transcript found for this video in the requested language."
        )
    except VideoUnavailable:
        raise TranscriptUnavailable("This video is unavailable.")
    except Exception as e:
        raise TranscriptUnavailable(f"Could not retrieve the transcript ({type(e).__name__}).")

    text = " ".join(chunk.text for chunk in transcript)
    if not text.strip():
        raise TranscriptUnavailable("The transcript for this video is empty.")
    return text


def split_transcript(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([transcript_text])


def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        if _is_quota_error(e):
            raise QuotaExceeded(
                "Gemini embedding quota reached. The free tier allows 100 embedding "
                "requests per minute. Wait a minute and try again, or try a shorter video."
            )
        raise PipelineError(f"Could not create embeddings ({type(e).__name__}).")


def build_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )


def format_docs(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)


def build_prompt():
    return PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question:{question}
""",
        input_variables=["context", "question"],
    )



def build_chain(retriever):
    """parallel_chain | prompt | llm | parser, exactly as in the notebook."""
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    return parallel_chain | build_prompt() | llm | StrOutputParser()


def process_video(video_id: str, progress=None):
    """Run the full indexing pipeline and return the ready-to-use chain.

    `progress` is an optional callback taking a short status string, so the
    Streamlit layer can report each stage without this module importing it.
    """
    def step(msg):
        if progress:
            progress(msg)

    require_api_key()

    step("Fetching transcript...")
    transcript_text = fetch_transcript(video_id)

    step("Splitting transcript into chunks...")
    chunks = split_transcript(transcript_text)

    step(f"Generating embeddings for {len(chunks)} chunks...")
    vector_store = build_vector_store(chunks)

    step("Building retriever...")
    retriever = build_retriever(vector_store)

    step("Assembling chain...")
    chain = build_chain(retriever)

    return {
        "transcript_chars": len(transcript_text),
        "num_chunks": len(chunks),
        "vector_store": vector_store,
        "retriever": retriever,
        "chain": chain,
    }


def answer_question(chain, question: str) -> str:
    """Ask the chain a question, translating API failures into clean messages."""
    if not question or not question.strip():
        raise PipelineError("Please type a question.")
    try:
        return chain.invoke(question.strip())
    except Exception as e:
        if _is_quota_error(e):
            raise QuotaExceeded(
                "Gemini API quota or rate limit reached. Please wait a moment and try again."
            )
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            raise PipelineError(
                "The Gemini model is temporarily overloaded. Please try again in a moment."
            )
        raise PipelineError(f"Could not generate an answer ({type(e).__name__}).")
