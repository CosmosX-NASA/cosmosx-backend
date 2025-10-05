from __future__ import annotations

import logging
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from model import Research


class _EnsureCodeFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "code"):
            record.code = "-"
        return True


_root_logger = logging.getLogger()
for _h in list(_root_logger.handlers):
    _h.addFilter(_EnsureCodeFilter())


class ResearchRagRepository:
    """
    Lightweight, dependency-minimized retriever.

    Implementation now uses an in-memory TF-IDF (pure NumPy/Python) over
    title+abstract for ~500 docs, which is fast and has no heavyweight deps.

    Args:
        db (Session): SQLAlchemy session.
        csv_path (str): Path to the paper metadata CSV (immutable per requirement).
        store_dir (str): Kept for API compatibility but unused.
        device (str): Kept for API compatibility but unused.
        cross_encoder_model (str): Kept for API compatibility but unused.
    """

    def __init__(
        self,
        db: Session,
        csv_path: str = "db/rag_store/info.csv",
        store_dir: str = "db/rag_store",
    ):
        self.db = db
        self.csv_path = csv_path
        self.store_dir = store_dir  # unused, kept for compatibility

        if not os.path.exists(self.csv_path):
            raise Exception("Rag info.csv not found")

        # Load CSV once (immutable by requirement)
        self.papers_df = pd.read_csv(self.csv_path)
        self.papers = self.papers_df.to_dict("records")

        # Build an in-memory TF‑IDF index (no persistence / rebuild detection)
        self._build_tfidf_index()

    # ---------- Internal utilities ----------
    @staticmethod
    def _l2_normalize_dict(vec: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
        norm = np.sqrt(sum(v * v for v in vec.values()))
        if norm < eps:
            return vec
        inv = 1.0 / norm
        return {k: v * inv for k, v in vec.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple alnum tokenizer, lowercase; lightweight and fast
        if not isinstance(text, str):
            return []
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    def _paper_text(self, p: dict) -> str:
        title = (p.get("title") or "").strip()
        body = (p.get("abstract") or "").strip()
        if title and body:
            return f"{title} - {body}"
        return title or body

    # ---------- TF‑IDF index build (in-memory only) ----------
    def _build_tfidf_index(self):
        docs_tokens: List[List[str]] = []
        for p in self.papers:
            docs_tokens.append(self._tokenize(self._paper_text(p)))

        self.n_docs = len(docs_tokens)
        # Document frequency
        df_counts: Dict[str, int] = {}
        for toks in docs_tokens:
            for t in set(toks):
                df_counts[t] = df_counts.get(t, 0) + 1

        # IDF with smoothing
        self.idf: Dict[str, float] = {}
        for term, df in df_counts.items():
            self.idf[term] = np.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
        # Also define default IDF for unseen query terms
        self.default_idf = np.log((1.0 + self.n_docs) / 1.0) + 1.0

        # Per-document normalized TF‑IDF vectors (sparse dicts)
        self.doc_vectors: List[Dict[str, float]] = []
        for toks in docs_tokens:
            tf: Dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0.0) + 1.0
            # raw TF * IDF
            tfidf = {t: tf[t] * self.idf.get(t, self.default_idf) for t in tf}
            self.doc_vectors.append(self._l2_normalize_dict(tfidf))

        # Keep PMCID list aligned with doc_vectors order
        self.doc_pmcids: List[str] = [
            str((p.get("PMCID") or "").strip()) for p in self.papers]

    # ---------- Query encoding & search (no Torch) ----------
    def _encode_query(self, query: str) -> Dict[str, float]:
        toks = self._tokenize(query)
        if not toks:
            return {}
        tf: Dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        tfidf = {t: tf[t] * self.idf.get(t, self.default_idf) for t in tf}
        return self._l2_normalize_dict(tfidf)

    def _scores_against_docs(self, q_vec: Dict[str, float]) -> np.ndarray:
        # Cosine over L2-normalized sparse dicts => dot product over shared keys
        if not q_vec:
            return np.zeros(self.n_docs, dtype=np.float32)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for i, dvec in enumerate(self.doc_vectors):
            s = 0.0
            # iterate over smaller of the two dicts for speed
            if len(q_vec) <= len(dvec):
                for t, wq in q_vec.items():
                    wd = dvec.get(t)
                    if wd is not None:
                        s += wq * wd
            else:
                for t, wd in dvec.items():
                    wq = q_vec.get(t)
                    if wq is not None:
                        s += wq * wd
            scores[i] = s
        return scores

    def _get_by_pmc_ids(self, pmc_ids: List[str]) -> list[Research]:
        return (
            self.db.query(Research)
            .filter(Research.pmc_id.in_(pmc_ids))
            .all()
        )

    def _fetch_researches_ordered(self, pmc_ids: List[str]) -> list[Research]:
        if not pmc_ids:
            return []
        researches = self._get_by_pmc_ids(pmc_ids)
        lookup = {r.pmc_id: r for r in researches}
        return [lookup[pmc_id] for pmc_id in pmc_ids if pmc_id in lookup]

    # ---------- Search API (reranking removed) ----------
    def find_by_user_search(self, search: str, pageSize: int = 5, ce_rerank_k: int = 0):
        # ce_rerank_k is ignored (reranking removed)
        q_vec = self._encode_query(search)
        scores = self._scores_against_docs(q_vec)

        if scores.size == 0:
            return []

        k = min(pageSize, scores.shape[0])
        idx_part = np.argpartition(scores, -k)[-k:]
        idx_sorted = idx_part[np.argsort(scores[idx_part])[::-1]]

        pmc_ids: List[str] = []
        for j in idx_sorted:
            pmc_id = self.doc_pmcids[int(j)]
            if pmc_id:
                pmc_ids.append(pmc_id)

        return self._fetch_researches_ordered(pmc_ids)


if __name__ == "__main__":
    from db.db import get_db_session

    rag = ResearchRagRepository(
        db=get_db_session(),
        csv_path="db/rag_store/info.csv",
        store_dir="db/rag_store",
    )

    q = "The impact of solar wind on humans in space"
    hits = rag.find_by_user_search(q, pageSize=5)
    print([getattr(r, "pmc_id", None) for r in hits])
