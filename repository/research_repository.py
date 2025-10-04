# uv add transformers adapter-transformers torch
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from adapters import AutoAdapterModel
from sqlalchemy.orm import Session
import torch
import numpy as np
import pandas as pd
import os
import json

import logging

# --- Patch logging to avoid KeyError: 'code' in formatters ---


class _EnsureCodeFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "code"):
            record.code = "-"
        return True


# Attach the filter to all existing handlers
_root_logger = logging.getLogger()
for _h in list(_root_logger.handlers):
    _h.addFilter(_EnsureCodeFilter())

# Silence overly chatty logs from the adapters library which triggered the original message
logging.getLogger("adapters").setLevel(logging.WARNING)

from sqlalchemy import Column, String, Integer
from db.db_base import db_base
from typing import List
from model import Research


class ResearchRagRepository:
    """
    - 최초 실행 시 info.csv를 임베딩해 인덱스를 저장하고,
      이후에는 저장된 인덱스를 불러와 빠르게 검색함.
    - 검색은 Numpy 기반 코사인 유사도(FAISS 미사용)로 수행하여 libomp 충돌을 회피함.
      - 논문의 개수는 529개로 FAISS를 사용하는 것이 오버해드가 더 크기에 미사용.
      - (옵션) 초기 벡터 검색 상위 N(기본 50)개를 Cross-Encoder로 재랭킹.
    - 임베딩과 재랭킹에 사용하는 문서 텍스트는 CSV의 제목과 초록만을 활용.

    Args:
        csv_path (str): 논문 메타데이터 CSV 경로. 
        store_dir (str): 인덱스(임베딩)를 저장할 경로. 기본 ".rag_store"
        device (str): 임베딩 및 재랭킹 모델이 동작할 디바이스(cpu/gpu).
        cross_encoder_model (str): HF 허브 Cross-Encoder 모델 이름/경로. 기본 "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """

    def __init__(
        self,
        db: Session,
        csv_path: str = "db/rag_store/info.csv",
        store_dir: str = "db/rag_store",
        device: str = "cpu",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.db = db
        self.csv_path = csv_path
        self.store_dir = store_dir
        self.device = device
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder = None  # Lazy-load for reranking
        self.ce_tok = None
        self.ce_model = None

        os.makedirs(self.store_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            raise Exception('Rag info.csv not found')

        # 모델/토크나이저 로드 (Hugging Face Hub)
        # SPECTER2 토크나이저
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        # 어댑터 지원 베이스 모델
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter_plus_plus")
        # 문서 임베딩용(Proximity)과 질의 임베딩용(Query) 어댑터를 각각 로드
        self.model.load_adapter("allenai/specter2", load_as="proximity")
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", load_as="query")
        self.model.eval()

        # CSV 로드
        self.papers_df = pd.read_csv(self.csv_path)
        self.papers = self.papers_df.to_dict("records")

        # 인덱스 로드 또는 생성
        self.index = None  # L2-normalized doc embeddings
        self._build_or_load_index()

    # ---------- 내부 유틸 ----------
    @staticmethod
    def _l2_normalize(a, axis=1, eps=1e-12):
        norm = np.linalg.norm(a, axis=axis, keepdims=True)
        return a / (norm + eps)

    def _isEnglish(self, s: str) -> bool:
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        return True

    def _translate2EN(self, query: str) -> str:
        return query  # 영어가 아니라면 영어로 변역해서 입력하는 것이 적합.

    def _cfg_path(self) -> str:
        return os.path.join(self.store_dir, "cfg.json")

    def _index_path(self) -> str:
        return os.path.join(self.store_dir, "doc_index.npy")

    def _paper_to_text(self, p: dict) -> str:
        """Join title and abstract for Cross-Encoder input."""
        title = (p.get("title") or "").strip()
        body = (p.get("abstract") or "").strip()
        if not title and not body:
            return ""
        if title and body:
            return f"{title} {self.tok.sep_token} {body}"
        return title or body

    def _csv_signature(self):
        """CSV 변경 여부 확인용 간단한 시그니처."""
        try:
            st = os.stat(self.csv_path)
            return {
                "path": os.path.abspath(self.csv_path),
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
                "rows": int(len(self.papers)),
            }
        except FileNotFoundError:
            return None

    def _build_or_load_index(self):
        cfg_path = self._cfg_path()
        idx_path = self._index_path()
        sig = self._csv_signature()

        # 저장된 인덱스 로드 시도
        if os.path.exists(cfg_path) and os.path.exists(idx_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                idx = np.load(idx_path).astype("float32")

                expected = {k: sig[k]
                            for k in ("path", "size", "rows")} if sig else None
                if (
                    expected is not None
                    and cfg.get("csv_signature", {}) == expected
                    and cfg.get("content_source") == "abstract"
                    and idx.shape[0] == len(self.papers)
                ):
                    self.index = idx
                    print("[RAG] Loaded existing index from store.")
                    return
            except Exception:
                pass  # 손상되었거나 호환 불가인 경우 재생성

        # 인덱스 재생성
        print("[RAG] Building index from CSV...")
        doc_emb = self.encode_papers(self.papers)
        self.index = self._l2_normalize(doc_emb, axis=1)

        # 저장
        np.save(idx_path, self.index)
        if sig is None:
            raise FileNotFoundError(
                f"CSV metadata not found at {self.csv_path}")
        cfg = {
            "csv_signature": {k: sig[k] for k in ("path", "size", "rows")},
            "dim": int(self.index.shape[1]),
            "content_source": "abstract",
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        print("[RAG] Index built and saved.")

    # ---------- 임베딩 ----------
    @torch.no_grad()
    def encode_papers(self, papers, batch_size: int = 32, max_length: int = 512):
        self.model.set_active_adapters("proximity")
        self.model.to(self.device)
        out = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i: i + batch_size]
            texts = []
            for p in batch:
                title = (p.get("title") or "").strip()
                body = (p.get("abstract") or "").strip()
                if title and body:
                    texts.append(f"{title}{self.tok.sep_token}{body}")
                else:
                    texts.append(title or body)
            enc = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)
            last_hidden = self.model(**enc).last_hidden_state  # [B, L, H]
            cls = last_hidden[:, 0, :]  # [B, H] -> 768-d
            out.append(cls.cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.empty((0, 768), dtype="float32")

    @torch.no_grad()
    def encode_queries(self, queries, max_length: int = 64):
        self.model.set_active_adapters("query")
        self.model.to(self.device)
        enc = self.tok(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        q = self.model(
            **enc).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        return self._l2_normalize(q, axis=1)

    def _load_cross_encoder(self):
        if self.ce_model is None:
            self.ce_tok = AutoTokenizer.from_pretrained(
                self.cross_encoder_model)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                self.cross_encoder_model)
            self.ce_model.to(self.device)
            self.ce_model.eval()

    @torch.no_grad()
    def _ce_predict(self, pairs, batch_size: int = 32, max_length: int = 512):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts = [p[0] for p in batch]
            docs = [p[1] for p in batch]
            enc = self.ce_tok(
                texts,
                text_pair=docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.ce_model(**enc)
            logits = out.logits  # [B, 1] or [B, 2]
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1).detach().cpu().numpy()
            else:
                s = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            scores.extend(s.tolist())
        return np.array(scores, dtype="float32")

    def _get_by_pmc_ids(self, pmc_ids: List[str]) -> list[Research]:
        return (self.db.query(Research)
                .filter(Research.pmc_id.in_(pmc_ids))
                .all())

    def _fetch_researches_ordered(self, pmc_ids: list[str]) -> list[Research]:
        if not pmc_ids:
            return []
        researches = self._get_by_pmc_ids(pmc_ids)
        lookup = {r.pmc_id: r for r in researches}
        return [lookup[pmc_id] for pmc_id in pmc_ids if pmc_id in lookup]

    # ---------- 검색 ----------

    def find_by_user_search(self, search: str, pageSize: int = 5, ce_rerank_k: int = 50):
        if self.index is None:
            self._build_or_load_index()

        if not self._isEnglish(search):
            search = self._translate2EN(search)

        # 1) Dense retrieval with SPECTER2 search encoder
        q = self.encode_queries([search])  # [1, 768]
        scores = np.dot(self.index, q[0])  # cosine similarity (정규화된 내적)

        # 가져올 후보 개수 (재랭킹 후보는 기본 50, 최소 k)
        n_candidates = max(
            pageSize, ce_rerank_k if ce_rerank_k and ce_rerank_k > 0 else pageSize)
        n_candidates = min(n_candidates, scores.shape[0])

        # Top-n 후보 고르기 (partial sort)
        idx_part = np.argpartition(scores, -n_candidates)[-n_candidates:]
        idx_sorted_dense = idx_part[np.argsort(scores[idx_part])[::-1]]

        # 2) (옵션) Cross-Encoder 재랭킹
        if ce_rerank_k and ce_rerank_k > 0:
            self._load_cross_encoder()
            pairs = [(search, self._paper_to_text(self.papers[int(j)]))
                     for j in idx_sorted_dense]
            ce_scores = self._ce_predict(pairs, batch_size=32)

            order = np.argsort(ce_scores)[::-1]
            final_idx = [int(idx_sorted_dense[o]) for o in order[:pageSize]]
            pmc_ids = []
            for j in final_idx:
                rec = self.papers[int(j)]
                pmc_id = rec.get("PMCID")
                if pmc_id:
                    pmc_ids.append(pmc_id)
            return self._fetch_researches_ordered(pmc_ids)

        # 3) 재랭킹 비활성화 시: dense 결과 상위 k개 반환
        final_idx = idx_sorted_dense[:pageSize]
        pmc_ids = []
        for j in final_idx:
            rec = self.papers[int(j)]
            pmc_id = rec.get('PMCID')
            if pmc_id:
                pmc_ids.append(pmc_id)
        return self._fetch_researches_ordered(pmc_ids)


if __name__ == "__main__":
    import time
    from db.db import get_db_session

    # 기본 경로들(모두 변경 가능)
    rag = ResearchRagRepository(
        db=get_db_session(),
        csv_path="db/rag_store/info.csv",
        store_dir="db/rag_store",
    )

    # 예시 질의
    q = "The impact of solar wind on humans in space"
    hits = rag.find_by_user_search(q, pageSize=5, ce_rerank_k=50)
    for r in hits:
        if "ce_score" in r:
            print(
                f"[dense={r.get('score'):.4f} | ce={r.get('ce_score'):.4f}] {r.get('PMCID')} {r.get('title')}")
        else:
            print(
                f"[dense={r.get('score'):.4f}] {r.get('PMCID')} {r.get('title')}")

    # 이미 로드된 상태에서는 얼마나 빠른가보자.
    time.sleep(1)
    print("\nQuestion: Microgravity will dilate human blood vessels and promote growth.\n\n")

    q = "Microgravity will dilate human blood vessels and promote growth."
    hits = rag.find_by_user_search(q, pageSize=5, ce_rerank_k=50)
    print([r.get('PMCID') for r in hits])
    for r in hits:
        if "ce_score" in r:
            print(
                f"[dense={r.get('score'):.4f} | ce={r.get('ce_score'):.4f}] {r.get('PMCID')} {r.get('title')}")
        else:
            print(
                f"[dense={r.get('score'):.4f}] {r.get('PMCID')} {r.get('title')}")

"""
[dense=0.7909 | ce=1.7362] PMC11988870 Microgravity and Cellular Biology: Insights into Cellular Responses and Implications for Human Health
[dense=0.7515 | ce=0.4570] PMC7787258 Prolonged Exposure to Microgravity Reduces Cardiac Contractility and Initiates Remodeling in Drosophila
[dense=0.7518 | ce=-0.3368] PMC4110898 Fifteen Days Microgravity Causes Growth in Calvaria of Mice
[dense=0.7859 | ce=-2.1094] PMC7339929 Simulated Microgravity Induces Regionally Distinct Neurovascular and Structural Remodeling of Skeletal Muscle and Cutaneous Arteries in the Rat
[dense=0.7582 | ce=-2.3219] PMC6275019 Synergistic Effects of Weightlessness  Isoproterenol  and Radiation on DNA Damage Response and Cytokine Production in Immune Cells
"""
