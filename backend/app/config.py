import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base directory points to `backend/`
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from root workspace and/or backend folder
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "multi")
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found in environment variables")
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# --- Serper API Configuration ---
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
if not SERPER_API_KEY:
    logger.error("SERPER_API_KEY not found in environment variables")
    raise ValueError("SERPER_API_KEY not found in environment variables")

# --- Model Configuration ---
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "2048"))
# bge-reranker-v2-m3 is Pinecone's currently-supported hosted rerank model
# (cohere-rerank-3.5 and pinecone-rerank-v0 were deprecated).
RERANK_MODEL = os.environ.get("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")

# --- Upload / safety limits (EDGE CASE: unbounded uploads / runaway prompts) ---
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB default
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "400"))
STALE_PROCESSING_SECONDS = int(os.environ.get("STALE_PROCESSING_SECONDS", str(30 * 60)))  # 30 min

# Rough token estimate: ~4 chars/token. Used to cap how much text we ever
# stuff into a single LLM prompt, regardless of chunk *count*.
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_ANSWER_PROMPT_TOKENS = int(os.environ.get("MAX_ANSWER_PROMPT_TOKENS", "12000"))
MAX_ANSWER_PROMPT_CHARS = MAX_ANSWER_PROMPT_TOKENS * CHARS_PER_TOKEN_ESTIMATE
MAX_TABLE_HTML_CHARS = 4000  # truncate any single huge table before prompting
MAX_SUMMARY_DOC_CHARS = 60000  # cap for whole-document summarization map step

# CORS origins (EDGE CASE: hardcoded localhost breaks any non-dev deployment)
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

# --- SQLite setup (replaces in-memory global_document_storage) ---
# Anchored to backend/ directory so existing data is preserved
DB_PATH = str(BASE_DIR / "app_data.db")
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)
