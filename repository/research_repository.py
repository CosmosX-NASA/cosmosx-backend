# uv add transformers adapter-transformers torch
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from adapters import AutoAdapterModel
import torch
import pandas as pd
from sqlalchemy.orm import Session
import numpy as np
import os
import json
import logging
import warnings


# ========== 로깅 에러 완전 해결 ==========

class SafeFilter(logging.Filter):
    """누락된 필드를 기본값으로 채워주는 필터"""

    def filter(self, record):
        if not hasattr(record, 'code'):
            record.code = '-'
        if not hasattr(record, 'funcName'):
            record.funcName = '-'
        if not hasattr(record, 'lineno'):
            record.lineno = 0
        return True


# 루트 로거와 모든 핸들러에 필터 적용
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(SafeFilter())

# 추가: 새로 생성되는 핸들러에도 필터 자동 적용
_original_add_handler = logging.Logger.addHandler


def _patched_add_handler(self, handler):
    handler.addFilter(SafeFilter())
    _original_add_handler(self, handler)


logging.Logger.addHandler = _patched_add_handler

# adapters 라이브러리 로그 레벨 조정
logging.getLogger("adapters").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================

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
        self.cross_encoder = None
        self.ce_tok = None
        self.ce_model = None

        os.makedirs(self.store_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'RAG info.csv not found at {csv_path}')

        print("[RAG] Loading SPECTER2 models...")
        # SPECTER2 토크나이저
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        # 어댑터 지원 베이스 모델
        self.model = AutoAdapterModel.from_pretrained("allenai/specter_plus_plus")

        # 문서 임베딩용(Proximity)과 질의 임베딩용(Query) 어댑터를 각각 로드
        self.model.load_adapter("allenai/specter2", load_as="proximity")
        self.model.load_adapter("allenai/specter2_adhoc_query", load_as="query")
        self.model.eval()
        print("[RAG] Models loaded successfully.")

        # CSV 로드
        self.papers_df = pd.read_csv(self.csv_path)
        self.papers = self.papers_df.to_dict("records")

        # 인덱스 로드 또는 생성
        self.index = None
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
        return query

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

                expected = {k: sig[k] for k in ("path", "size", "rows")} if sig else None
                if (
                        expected is not None
                        and cfg.get("csv_signature", {}) == expected
                        and cfg.get("content_source") == "abstract"
                        and idx.shape[0] == len(self.papers)
                ):
                    self.index = idx
                    print("[RAG] Loaded existing index from store.")
                    return
            except Exception as e:
                print(f"[RAG] Failed to load index: {e}, rebuilding...")

        # 인덱스 재생성
        print("[RAG] Building index from CSV...")
        doc_emb = self.encode_papers(self.papers)
        self.index = self._l2_normalize(doc_emb, axis=1)

        # 저장
        np.save(idx_path, self.index)
        if sig is None:
            raise FileNotFoundError(f"CSV metadata not found at {self.csv_path}")
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
        # 어댑터 활성화를 try-except로 감싸서 로깅 에러 무시
        try:
            self.model.set_active_adapters("proximity")
        except Exception:
            pass  # 로깅 에러 무시

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
            last_hidden = self.model(**enc).last_hidden_state
            cls = last_hidden[:, 0, :]
            out.append(cls.cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.empty((0, 768), dtype="float32")

    @torch.no_grad()
    def encode_queries(self, queries, max_length: int = 64):
        try:
            self.model.set_active_adapters("query")
        except Exception:
            pass

        self.model.to(self.device)
        enc = self.tok(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        q = self.model(**enc).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        return self._l2_normalize(q, axis=1)

    def _load_cross_encoder(self):
        if self.ce_model is None:
            self.ce_tok = AutoTokenizer.from_pretrained(self.cross_encoder_model)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                self.cross_encoder_model
            )
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
            logits = out.logits
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1).detach().cpu().numpy()
            else:
                s = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            scores.extend(s.tolist())
        return np.array(scores, dtype="float32")

    def _get_by_pmc_ids(self, pmc_ids: List[str]) -> list[Research]:
        return self.db.query(Research).filter(Research.pmc_id.in_(pmc_ids)).all()

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

        q = self.encode_queries([search])
        scores = np.dot(self.index, q[0])

        n_candidates = max(
            pageSize, ce_rerank_k if ce_rerank_k and ce_rerank_k > 0 else pageSize
        )
        n_candidates = min(n_candidates, scores.shape[0])

        idx_part = np.argpartition(scores, -n_candidates)[-n_candidates:]
        idx_sorted_dense = idx_part[np.argsort(scores[idx_part])[::-1]]

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

        final_idx = idx_sorted_dense[:pageSize]
        pmc_ids = []
        for j in final_idx:
            rec = self.papers[int(j)]
            pmc_id = rec.get('PMCID')
            if pmc_id:
                pmc_ids.append(pmc_id)
        return self._fetch_researches_ordered(pmc_ids)