"""Run this once to ingest sample docs into ChromaDB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import chromadb
from chromadb.utils import embedding_functions
import vvid

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'sample_docx')