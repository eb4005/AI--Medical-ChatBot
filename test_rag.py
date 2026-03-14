import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.embeddings import get_embedding_model
import traceback

print("Testing embedding model load...")
try:
    model = get_embedding_model()
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:")
    traceback.print_exc()
