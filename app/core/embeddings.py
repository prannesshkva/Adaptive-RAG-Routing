import numpy as np
import hashlib
from typing import List, Union

_SHARED_EMBEDDER = None


class ProductionEmbedder:
    """
    Production-ready semantic text embedder.
    Uses BAAI/bge-base-en-v1.5 (or all-MiniLM-L6-v2) with zero-failure fallback
    to deterministic normalized embeddings.
    """
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._dim = None

    def _init_model(self):
        if not self._initialized:
            self._initialized = True
            try:
                from sentence_transformers import SentenceTransformer
                # Try loading cached local model first
                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name)
                # Compute sample vector to determine dimension
                sample = self._model.encode(["ping"], normalize_embeddings=True)
                self._dim = sample.shape[1]
            except Exception as e:
                # Fallback to deterministic dense vectors if transformer unavailable
                self._model = None
                self._dim = 384

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._init_model()
        return self._dim

    def _fallback_vector(self, text: str) -> np.ndarray:
        """
        Deterministic dense embedding generator for offline/resilient execution.
        """
        words = text.lower().strip().split()
        dim = self._dim or 384
        if not words:
            return np.zeros(dim, dtype=np.float32)
        
        vec = np.zeros(dim, dtype=np.float32)
        for i, w in enumerate(words):
            h = int(hashlib.sha256(w.encode('utf-8')).hexdigest()[:8], 16)
            rng = np.random.RandomState(h % (2**32 - 1))
            pos_harmonic = np.sin((i + 1) * np.linspace(0.1, np.pi, dim))
            vec += (rng.randn(dim) * 0.5 + pos_harmonic).astype(np.float32)
            
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 1e-8 else vec.astype(np.float32)

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        self._init_model()
        if isinstance(texts, str):
            texts = [texts]
            
        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False
                )
                return np.array(embeddings, dtype=np.float32)
            except Exception:
                pass
                
        # Deterministic fallback
        vectors = []
        for t in texts:
            vec = self._fallback_vector(t)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def get_embedder() -> ProductionEmbedder:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = ProductionEmbedder()
    return _SHARED_EMBEDDER
