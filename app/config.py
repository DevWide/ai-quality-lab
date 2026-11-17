import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", ROOT / "data" / "docs"))

RAG_MODEL = os.getenv("RAG_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
