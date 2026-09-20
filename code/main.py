#!/usr/bin/env python3
__version__ = "4.0.0"
__build_date__ = "2026-07-06"

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import json
import gc
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
if not hasattr(torch, '_real_original_load'):
    setattr(torch, '_real_original_load', torch.load)
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return getattr(torch, '_real_original_load')(*args, **kwargs)
    torch.load = _patched_torch_load  # type: ignore[assignment]
    print("PyTorch 2.6+ compatibility patch applied")


@dataclass
class Config:
    data_path: str = "prometheus"
    target_per_level: int = 5000
    output_dir: str = "./outputs"
    experiment_name: str = "jade"
    seed: int = 42
    max_user_chars: int = 512
    max_response_chars: int = 600
    max_context_chars: int = 1000
    max_input_tokens: int = 768
    max_output_tokens: int = 192
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    orli_hidden_dim: int = 512
    orli_epochs: int = 20
    orli_batch_size: int = 32
    orli_lr: float = 1e-4
    orli_patience: int = 3
    # ORLI quality-signal fixes (see ORLI_DEEP_DIVE.md / ORLI_FIX_STRATEGY.md):
    orli_feature: str = "concat"        # 'pair'(orig) | 'response' | 'concat'(query+response separate channels)
    orli_loss: str = "huber"            # 'huber' | 'mse'
    orli_balance: bool = True           # inverse-frequency class weighting (counter the 4-5 star skew)
    protect_high_ratings: bool = False  # rating-aware reward: imitate gold MORE on already-high-rated sources
    alpha_start: float = 1.0
    alpha_end: float = 0.3
    beta_start: float = 0.1
    beta_end: float = 0.6
    gamma_start: float = 0.1
    gamma_end: float = 0.3
    batch_size: int = 1
    epochs: int = 3
    batches_per_epoch: int = 500
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    logprob_chunk_size: int = 4
    log_every: int = 25          # console summary + training_history.json snapshot cadence (steps)
    checkpoint_every: int = 0    # save actor_latest every N steps (0 = off) -> weights survive a crash
    warmup_ratio: float = 0.05
    grpo_k: int = 16
    grad_clip: float = 1.0
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-6
    max_buffer_per_bucket: int = 1500
    retrieval_k: int = 5
    rating_aware_retrieval: bool = True
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: str = "auto"
    use_contrastive: bool = True
    quantization: str = "auto" 
    use_armo: bool = True        
    max_test_per_tier: int = 500 
    eval_baseline: bool = False   
    load_actor: str = ""          
    load_orli: str = ""           
    use_retrieval: bool = True   
    orli_val_frac: float = 0.1    
    keep_checkpoints: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir: str = field(default="", init=False)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = f"{self.output_dir}/{self.experiment_name}_K{self.grpo_k}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(f"{self.run_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.run_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.run_dir}/figures", exist_ok=True)


def setup_logging(config: Config) -> logging.Logger:
    logger = logging.getLogger(f"JADE_{config.grpo_k}_{config.seed}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(f"{config.run_dir}/logs/training.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def safe_truncate(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    text = str(text)
    return text[:max_chars-3] + "..." if len(text) > max_chars else text


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory() -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{allocated:.1f}GB / {total:.1f}GB"
    return "CPU"


def json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)

INTENT_PATTERNS = {
    'Coding': ['code', 'python', 'program', 'function', 'algorithm', 'debug',
               'javascript', 'sql', 'html', 'css', 'api', 'class', 'implement',
               'compile', 'syntax', 'variable', 'loop', 'array', 'script',
               'software', 'developer', 'programming', 'git', 'database'],
    'Reasoning': ['calculate', 'solve', 'math', 'equation', 'proof', 'probability',
                  'statistics', 'logic', 'derive', 'theorem', 'formula', 'compute',
                  'number', 'arithmetic', 'percentage'],
    'Creative Writing': ['write a story', 'write a poem', 'creative', 'fiction',
                         'narrative', 'compose', 'screenplay', 'dialogue', 'novel',
                         'essay', 'poem', 'lyrics', 'write me'],
    'Explanation': ['explain', 'what is', 'what are', 'how does', 'define',
                    'describe', 'difference between', 'meaning of', 'concept of',
                    'tell me about', 'elaborate'],
    'Analysis': ['analyze', 'compare', 'evaluate', 'assess', 'critique',
                 'review', 'pros and cons', 'advantages', 'disadvantages',
                 'strengths', 'weaknesses'],
    'Summarization': ['summarize', 'summary', 'tl;dr', 'brief overview',
                      'key points', 'main ideas', 'condense'],
    'Task Completion': ['list', 'give me', 'provide', 'create a', 'generate',
                        'make a', 'design', 'plan', 'outline', 'draft',
                        'suggest', 'recommend'],
    'Question Answering': ['who ', 'what ', 'where ', 'when ', 'why ', 'how ',
                           'which ', 'is it true', 'can you tell me', 'does '],
    'Evaluation': ['rate', 'score', 'assess', 'grade', 'judge', 'rank',
                   'evaluate the', 'quality of', 'how well', 'how good'],
}


def classify_intent(text: str) -> str:
    """Classify instruction text into an intent category."""
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            return intent
    return "Question Answering" if '?' in text else "General"

KNOWN_DATASETS = {
    'prometheus':    'prometheus-eval/Feedback-Collection',
    'helpsteer2':    'nvidia/HelpSteer2',
    'ultrafeedback': 'openbmb/UltraFeedback',
}


def load_prometheus(target_per_level: int, seed: int, logger: logging.Logger) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not installed. Run: pip install datasets")
        raise RuntimeError("Missing dependency: pip install datasets")

    logger.info("Loading Prometheus Feedback Collection from HuggingFace...")
    logger.info("Dataset: prometheus-eval/Feedback-Collection (KAIST, ICLR 2024)")
    logger.info("(First run downloads ~200MB; subsequent runs use HF cache)")

    ds = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    logger.info(f"Downloaded {len(ds):,} raw samples")

    records = []
    skipped = 0
    for i, entry in enumerate(ds):
        instruction = (entry.get('orig_instruction', '') or '').strip()
        response = (entry.get('orig_response', '') or '').strip()
        score_raw = entry.get('orig_score', None)

        if len(instruction) < 10 or len(response) < 20:
            skipped += 1
            continue
        try:
            score = int(str(score_raw).strip())
            if not 1 <= score <= 5:
                skipped += 1
                continue
        except (ValueError, TypeError):
            skipped += 1
            continue

        records.append({
            'sample_id': f"prom_{i:06d}",
            'user': instruction,
            'chatgpt_after': response,
            'chatgpt_before': '',
            'rating': score,
            'intent_final': classify_intent(instruction),
        })

    df = pd.DataFrame(records)
    logger.info(f"Valid samples: {len(df):,} (skipped {skipped:,})")

    # Balance: subsample to target_per_level per rating
    if target_per_level > 0:
        balanced_dfs = []
        for rating in sorted(df['rating'].unique()):
            rdf = df[df['rating'] == rating]
            n = min(len(rdf), target_per_level)
            balanced_dfs.append(rdf.sample(n=n, random_state=seed))
            logger.info(f"  Rating {rating}: {len(rdf):,} available -> sampled {n:,}")
        df = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Final dataset: {len(df):,} samples")
    return df


def load_helpsteer2(target_per_level: int, seed: int, logger: logging.Logger) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Missing dependency: pip install datasets")

    logger.info("Loading HelpSteer2 from HuggingFace...")
    logger.info("Dataset: nvidia/HelpSteer2 (human-annotated, ~21K response pairs)")
    logger.info("(First run downloads ~500MB; subsequent runs use HF cache)")

    ds = load_dataset("nvidia/HelpSteer2", split="train")
    logger.info(f"Downloaded {len(ds):,} raw samples")

    records = []
    skipped = 0
    for i, entry in enumerate(ds):
        prompt = (entry.get('prompt', '') or '').strip()
        response = (entry.get('response', '') or '').strip()
        helpfulness = entry.get('helpfulness', None)

        if len(prompt) < 10 or len(response) < 20:
            skipped += 1
            continue
        if helpfulness is None:
            skipped += 1
            continue
        try:
            rating = int(helpfulness) + 1  # 0-4 → 1-5
            if not 1 <= rating <= 5:
                skipped += 1
                continue
        except (ValueError, TypeError):
            skipped += 1
            continue

        records.append({
            'sample_id': f"hs2_{i:06d}",
            'user': prompt,
            'chatgpt_after': response,
            'chatgpt_before': '',
            'rating': rating,
            'intent_final': classify_intent(prompt),
        })

    df = pd.DataFrame(records)
    logger.info(f"Valid samples: {len(df):,} (skipped {skipped:,})")

    if target_per_level > 0:
        balanced_dfs = []
        for rating in sorted(df['rating'].unique()):
            rdf = df[df['rating'] == rating]
            n = min(len(rdf), target_per_level)
            balanced_dfs.append(rdf.sample(n=n, random_state=seed))
            logger.info(f"  Rating {rating}: {len(rdf):,} available -> sampled {n:,}")
        df = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Final HelpSteer2 dataset: {len(df):,} samples")
    return df


def load_ultrafeedback(target_per_level: int, seed: int, logger: logging.Logger) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Missing dependency: pip install datasets")

    logger.info("Loading UltraFeedback from HuggingFace...")
    logger.info("Dataset: openbmb/UltraFeedback (~64K prompts, GPT-4 rated 1-5)")
    logger.info("(First run downloads ~2GB; subsequent runs use HF cache)")

    ds = load_dataset("openbmb/UltraFeedback", split="train")
    logger.info(f"Downloaded {len(ds):,} prompts")

    ANNOTATION_KEYS = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    RATING_FIELDS = ['Rating', 'rating', 'overall_rating', 'score']

    def _extract_dim_rating(ann_dict):
        if not isinstance(ann_dict, dict):
            return None
        for key in RATING_FIELDS:
            if key in ann_dict:
                try:
                    val = str(ann_dict[key]).strip()
                    if '/' in val:
                        val = val.split('/')[0]
                    r = float(val)
                    if 1 <= r <= 5:
                        return r
                except (ValueError, TypeError):
                    continue
        return None

    def _composite_rating(annotations):
        if not annotations or not isinstance(annotations, dict):
            return None
        ratings = [_extract_dim_rating(annotations.get(k)) for k in ANNOTATION_KEYS]
        ratings = [r for r in ratings if r is not None]
        if not ratings:
            return None
        return int(np.clip(round(np.mean(ratings)), 1, 5))

    records = []
    skipped = 0
    for i, entry in enumerate(ds):
        if i % 10000 == 0 and i > 0:
            logger.info(f"  Processed {i:,}/{len(ds):,} ({len(records):,} valid pairs so far)")

        instruction = (entry.get('instruction', '') or '').strip()
        completions = entry.get('completions', [])
        if len(instruction) < 10 or not completions:
            skipped += max(1, len(completions) if completions else 1)
            continue

        for j, completion in enumerate(completions):
            response = (completion.get('output', '') or completion.get('response', '') or '').strip()
            if len(response) < 50 or len(response) > 2000:
                skipped += 1
                continue
            rating = _composite_rating(completion.get('annotations', {}))
            if rating is None:
                skipped += 1
                continue
            records.append({
                'sample_id': f"uf_{i:06d}_{j}",
                'user': instruction,
                'chatgpt_after': response,
                'chatgpt_before': '',
                'rating': rating,
                'intent_final': classify_intent(instruction),
            })

    df = pd.DataFrame(records)
    logger.info(f"Valid pairs extracted: {len(records):,} (skipped {skipped:,})")

    if target_per_level > 0:
        balanced_dfs = []
        for rating in sorted(df['rating'].unique()):
            rdf = df[df['rating'] == rating]
            n = min(len(rdf), target_per_level)
            balanced_dfs.append(rdf.sample(n=n, random_state=seed))
            logger.info(f"  Rating {rating}: {len(rdf):,} available -> sampled {n:,}")
        df = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Final UltraFeedback dataset: {len(df):,} samples")
    return df


def load_and_split_data(config: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    # ---- Route: HuggingFace streaming or local CSV ----
    data_key = config.data_path.lower().strip()
    if data_key in KNOWN_DATASETS:
        logger.info(f"Dataset mode: HuggingFace streaming ({data_key})")
        if data_key == 'helpsteer2':
            df = load_helpsteer2(
                target_per_level=config.target_per_level,
                seed=config.seed,
                logger=logger,
            )
        elif data_key == 'ultrafeedback':
            df = load_ultrafeedback(
                target_per_level=config.target_per_level,
                seed=config.seed,
                logger=logger,
            )
        else:  # prometheus (default)
            df = load_prometheus(
                target_per_level=config.target_per_level,
                seed=config.seed,
                logger=logger,
            )
    elif os.path.isfile(config.data_path):
        logger.info(f"Dataset mode: local CSV ({config.data_path})")
        df = pd.read_csv(config.data_path)
        logger.info(f"Loaded {len(df)} samples")
    else:
        raise FileNotFoundError(
            f"Dataset not found: '{config.data_path}'\n"
            f"Streaming options: --data_path helpsteer2 | ultrafeedback | prometheus\n"
            f"Local CSV:         --data_path ./path/to/file.csv  (e.g. Recipe4U)"
        )

    # ---- Common validation ----
    required = ['sample_id', 'user', 'chatgpt_after', 'rating', 'intent_final']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df.dropna(subset=required)
    df['rating'] = df['rating'].astype(int)
    if 'chatgpt_before' not in df.columns:
        df['chatgpt_before'] = ''
    df['chatgpt_before'] = df['chatgpt_before'].fillna('').astype(str)
    df = df[df['rating'].between(1, 5)]
    logger.info(f"After cleaning: {len(df)} samples")
    logger.info("Rating distribution:")
    for r in sorted(df['rating'].unique()):
        count = len(df[df['rating'] == r])
        pct = count / len(df) * 100
        logger.info(f"  {r}: {count:5d} ({pct:5.1f}%)")

    train_dfs, orli_val_dfs, test_dfs = [], [], []
    for rating in sorted(df['rating'].unique()):
        rating_df = df[df['rating'] == rating]
        if len(rating_df) < 3:
            train_dfs.append(rating_df)
            continue
        n_test = max(1, int(len(rating_df) * 0.2))
        if n_test > config.max_test_per_tier:
            n_test = config.max_test_per_tier
        train_pool, test_r = train_test_split(rating_df, test_size=n_test, random_state=config.seed)
        n_val = max(1, int(len(train_pool) * config.orli_val_frac))
        if n_val >= len(train_pool):
            n_val = max(1, len(train_pool) - 1)
        train_r, val_r = train_test_split(train_pool, test_size=n_val, random_state=config.seed)
        train_dfs.append(train_r)
        orli_val_dfs.append(val_r)
        test_dfs.append(test_r)
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=config.seed).reset_index(drop=True)
    if not test_dfs:
        raise ValueError("Not enough data for test split (need >=3 samples per rating)")
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=config.seed).reset_index(drop=True)
    orli_val_df = (pd.concat(orli_val_dfs).sample(frac=1, random_state=config.seed).reset_index(drop=True)
                   if orli_val_dfs else test_df)
    logger.info(f"Split: {len(train_df)} train / {len(orli_val_df)} orli_val / {len(test_df)} test")
    return train_df, orli_val_df, test_df


class Encoder:
    def __init__(self, config: Config, logger: logging.Logger):
        from sentence_transformers import SentenceTransformer
        logger.info("=" * 70)
        logger.info("LOADING ENCODER")
        logger.info("=" * 70)
        self.config = config
        self.logger = logger
        self.device = config.device
        logger.info(f"Model: {config.embedding_model}")
        try:
            self.model = SentenceTransformer(config.embedding_model, device=self.device)
        except Exception as e:
            logger.warning(f"Failed to load {config.embedding_model}: {e}")
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Using fallback: {fallback}")
            self.model = SentenceTransformer(fallback, device=self.device)
            config.embedding_model = fallback
        self.dim: int  # set below after None check
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Could not determine embedding dimension for {config.embedding_model}")
        self.dim = dim
        if self.dim != config.embedding_dim:
            logger.warning(f"Updating embedding_dim: {config.embedding_dim} -> {self.dim}")
            config.embedding_dim = self.dim
        logger.info(f"Encoder loaded (dim={self.dim})")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        texts = [safe_truncate(t, self.config.max_user_chars) for t in texts]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class RatingAwareMemory:
    def __init__(self, encoder: Encoder, train_df: pd.DataFrame, config: Config, logger: logging.Logger):
        import faiss
        logger.info("=" * 70)
        logger.info("BUILDING MEMORY")
        logger.info("=" * 70)
        self.encoder = encoder
        self.config = config
        self.logger = logger
        self.indices = {}
        self.samples = {}
        for rating in sorted(train_df['rating'].unique()):
            rating_df = train_df[train_df['rating'] == rating]
            queries = rating_df['user'].tolist()
            embeddings = encoder.encode(queries, show_progress=False)
            index = faiss.IndexFlatIP(config.embedding_dim)
            index.add(embeddings)
            self.indices[rating] = index
            samples = []
            for _, row in rating_df.iterrows():
                samples.append({
                    'query': safe_truncate(str(row['user']), config.max_user_chars),
                    'response': safe_truncate(str(row['chatgpt_after']), config.max_response_chars),
                    'rating': int(row['rating']),
                    'intent': str(row['intent_final']),
                    'prev_msg': safe_truncate(str(row.get('chatgpt_before', '')), 300)
                })
            self.samples[rating] = samples
            logger.info(f"  {rating}: {len(samples)} samples indexed")
        high_df = train_df[train_df['rating'] >= 4]
        high_queries = high_df['user'].tolist()
        high_embeddings = encoder.encode(high_queries, show_progress=False)
        self.high_quality_index = faiss.IndexFlatIP(config.embedding_dim)
        self.high_quality_index.add(high_embeddings.astype(np.float32))
        self.high_quality_samples = []
        for _, row in high_df.iterrows():
            self.high_quality_samples.append({
                'query': safe_truncate(str(row['user']), config.max_user_chars),
                'response': safe_truncate(str(row['chatgpt_after']), config.max_response_chars),
                'rating': int(row['rating'])
            })
        logger.info(f"  High-quality index: {len(self.high_quality_samples)} samples")
        logger.info("Memory built")

    def retrieve(self, query: str, source_rating: Optional[int] = None, k: Optional[int] = None) -> Dict:
        k = k or self.config.retrieval_k
        query_emb = self.encoder.encode_single(query)
        retrieved = []
        negative_examples = []
        if self.config.rating_aware_retrieval and source_rating is not None:
            if source_rating <= 2 and source_rating in self.indices:
                D, I = self.indices[source_rating].search(query_emb.reshape(1, -1), k)
                for idx, score in zip(I[0], D[0]):
                    if idx < len(self.samples[source_rating]):
                        sample = self.samples[source_rating][idx]
                        negative_examples.append({**sample, 'score': float(score)})
        D, I = self.high_quality_index.search(query_emb.reshape(1, -1), k * 2)
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.high_quality_samples):
                sample = self.high_quality_samples[idx]
                retrieved.append({**sample, 'score': float(score)})
                if len(retrieved) >= k:
                    break
        context_parts = []
        for r in retrieved[:k]:
            context_parts.append(f"[Good example] Q: {r['query'][:80]}... A: {r['response'][:120]}...")
        context_str = "\n".join(context_parts)
        return {
            'positive_samples': retrieved,
            'negative_samples': negative_examples,
            'context_str': safe_truncate(context_str, self.config.max_context_chars),
            'state': 'engaged' if source_rating and source_rating >= 4 else 'neutral'
        }

    def encode_response_pair(self, query: str, response: str) -> np.ndarray:
        # Delegates to build_orli_features so TRAINING (ORLIDataset) and INFERENCE (here) use
        # byte-identical feature construction and can never drift apart.
        return build_orli_features(self.encoder, [query], [response], self.config)[0]


def orli_feature_dim(config: "Config") -> int:
    """Input width the ORLI judge expects given the feature mode.
    'pair'/'response' -> one embedding; 'concat' -> query+response embeddings stacked."""
    return config.embedding_dim * (2 if config.orli_feature == "concat" else 1)


def build_orli_features(encoder: "Encoder", queries, responses, config: "Config") -> np.ndarray:
    """SINGLE source of truth for ORLI input features (ORLIDataset at train time and
    encode_response_pair at inference). Root-cause fixes vs the original 'pair' mode:
      'response' — embed only the response (drop the topic-dominating query).
      'concat'   — embed query and response SEPARATELY and concatenate, so the response gets
                   its own channel instead of being averaged into the query's topic.
    Returns float32 array of shape (N, orli_feature_dim(config))."""
    qs = [safe_truncate(str(q), config.max_user_chars // 2) for q in queries]
    rs = [safe_truncate(str(r), config.max_response_chars // 2) for r in responses]
    mode = config.orli_feature
    if mode == "response":
        return encoder.encode([f"Response: {r}" for r in rs]).astype(np.float32)
    if mode == "concat":
        q_emb = encoder.encode([f"Query: {q}" for q in qs])
        r_emb = encoder.encode([f"Response: {r}" for r in rs])
        return np.concatenate([q_emb, r_emb], axis=1).astype(np.float32)
    # 'pair' — original behaviour
    return encoder.encode([f"Query: {q}\nResponse: {r}" for q, r in zip(qs, rs)]).astype(np.float32)


def _ridge_probe(train_ds, val_ds, device, lam: float = 10.0):
    """Closed-form ridge regression on the FROZEN features -> the linear 'ceiling' of how much
    quality signal the features carry. If even this is weak, no MLP or loss tweak can help and
    the fix must be the features/encoder. Cheap (one linear solve), no training loop."""
    Xtr = torch.stack([s["emb"] for s in train_ds.samples]).to(device)
    ytr = torch.tensor([s["rating"] for s in train_ds.samples], device=device).float()
    Xva = torch.stack([s["emb"] for s in val_ds.samples]).to(device)
    yva = np.array([s["rating"] for s in val_ds.samples], dtype=np.float64)
    Xtr1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=device)], dim=1)
    Xva1 = torch.cat([Xva, torch.ones(Xva.shape[0], 1, device=device)], dim=1)
    A = Xtr1.T @ Xtr1 + lam * torch.eye(Xtr1.shape[1], device=device)
    w = torch.linalg.solve(A, Xtr1.T @ ytr)
    pv = (Xva1 @ w).detach().cpu().numpy()
    mae = float(np.mean(np.abs(pv - yva)))
    try:
        from scipy.stats import spearmanr
        sp = float(spearmanr(pv, yva)[0])  # [0] = correlation; robust across scipy versions
    except Exception:
        sp = float("nan")
    return mae, sp


class ORLIJudge(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        dim = orli_feature_dim(config)  # 1024 ('pair'/'response') or 2048 ('concat')
        h1 = config.orli_hidden_dim  # 512
        h2 = h1 // 2  # 256

        # Layer 1: input projection (dim → h1)
        self.proj1 = nn.Linear(dim, h1)
        self.ln1 = nn.LayerNorm(h1)

        # Layer 2: refine at h1 (residual connection)
        self.proj2 = nn.Linear(h1, h1)
        self.ln2 = nn.LayerNorm(h1)

        # Layer 3: downsample (h1 → h2)
        self.proj3 = nn.Linear(h1, h2)
        self.ln3 = nn.LayerNorm(h2)

        # Layer 4: refine at h2 (residual connection)
        self.proj4 = nn.Linear(h2, h2)
        self.ln4 = nn.LayerNorm(h2)

        # Layer 5: output head
        self.head = nn.Linear(h2, 1)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.15)
        self.drop3 = nn.Dropout(0.15)
        self.drop4 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: project to h1
        h = self.drop1(F.gelu(self.ln1(self.proj1(x))))
        # Layer 2: residual refinement at h1
        h = h + self.drop2(F.gelu(self.ln2(self.proj2(h))))
        # Layer 3: downsample to h2
        h = self.drop3(F.gelu(self.ln3(self.proj3(h))))
        # Layer 4: residual refinement at h2
        h = h + self.drop4(F.gelu(self.ln4(self.proj4(h))))
        # Layer 5: scalar output
        return self.head(h).squeeze(-1)


class ORLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: Encoder, config: Config):
        self.samples = []
        all_embs = build_orli_features(encoder, df['user'].tolist(), df['chatgpt_after'].tolist(), config)
        for i, (_, row) in enumerate(df.iterrows()):
            self.samples.append({
                'emb': torch.tensor(all_embs[i], dtype=torch.float),
                'rating': float(row['rating'])
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['emb'], torch.tensor(s['rating'], dtype=torch.float)


def train_orli_judge(judge: ORLIJudge, train_df: pd.DataFrame, val_df: pd.DataFrame, encoder: Encoder, config: Config, logger: logging.Logger) -> Tuple[ORLIJudge, Dict]:
    logger.info("=" * 70)
    logger.info("TRAINING ORLI JUDGE")
    logger.info("=" * 70)
    train_ds = ORLIDataset(train_df, encoder, config)
    val_ds = ORLIDataset(val_df, encoder, config)
    train_loader = DataLoader(train_ds, batch_size=config.orli_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.orli_batch_size, shuffle=False)
    optimizer = AdamW(judge.parameters(), lr=config.orli_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.orli_epochs)
    device = config.device
    judge = judge.to(device)
    tier_w = {}
    if config.orli_balance:
        vc = train_df['rating'].astype(int).value_counts().to_dict()
        n_tiers = max(1, len(vc)); total = sum(vc.values())
        tier_w = {int(r): total / (n_tiers * c) for r, c in vc.items()}
        logger.info("ORLI class weights (inverse-freq): "
                    + ", ".join(f"{k}*={tier_w[k]:.2f}" for k in sorted(tier_w)))
    logger.info(f"ORLI feature={config.orli_feature} (input dim {orli_feature_dim(config)}), "
                f"loss={config.orli_loss}, balance={config.orli_balance}")
    best_mae = float('inf')
    history: Dict[str, Any] = {'train_loss': [], 'val_mae': []}
    save_path = f"{config.run_dir}/checkpoints/orli_judge.pt"
    patience_counter = 0
    for epoch in range(config.orli_epochs):
        judge.train()
        total_loss = 0
        for emb, rating in train_loader:
            emb, rating = emb.to(device), rating.to(device)
            optimizer.zero_grad()
            pred = judge(emb)
            if config.orli_loss == 'mse':
                per_sample = (pred - rating) ** 2
            else:
                per_sample = F.huber_loss(pred, rating, delta=1.0, reduction='none')
            if tier_w:
                wts = torch.tensor([tier_w.get(int(round(float(r))), 1.0) for r in rating.detach().cpu()],
                                   device=device, dtype=torch.float)
                loss = (per_sample * wts).mean()
            else:
                loss = per_sample.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        judge.eval()
        preds, targets = [], []
        with torch.no_grad():
            for emb, rating in val_loader:
                emb = emb.to(device)
                pred = judge(emb)
                preds.extend(pred.cpu().numpy())
                targets.extend(rating.numpy())
        mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
        history['train_loss'].append(avg_loss)
        history['val_mae'].append(mae)
        logger.info(f"  Epoch {epoch+1:2d}/{config.orli_epochs}: Loss={avg_loss:.4f} | Val MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            torch.save({'model_state_dict': judge.state_dict(), 'best_mae': best_mae, 'epoch': epoch}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.orli_patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {config.orli_patience} epochs)")
                break
    checkpoint = torch.load(save_path, map_location=device)
    judge.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"ORLI Judge trained. Best MAE: {best_mae:.3f} (stopped at epoch {checkpoint['epoch']+1})")

    # Per-tier MAE — exposes calibration quality at rating extremes
    logger.info("Computing per-tier ORLI MAE on validation set...")
    judge.eval()
    per_tier_mae = {}
    with torch.no_grad():
        for tier_rating in sorted(val_df['rating'].unique()):
            tier_df = val_df[val_df['rating'] == tier_rating]
            if len(tier_df) == 0:
                continue
            tier_ds = ORLIDataset(tier_df, encoder, config)
            tier_loader = DataLoader(tier_ds, batch_size=config.orli_batch_size, shuffle=False)
            preds, targets = [], []
            for emb, r in tier_loader:
                pred = judge(emb.to(device))
                preds.extend(pred.cpu().numpy())
                targets.extend(r.numpy())
            tier_mae = float(np.mean(np.abs(np.array(preds) - np.array(targets))))
            per_tier_mae[int(tier_rating)] = tier_mae
            logger.info(f"  {tier_rating}-star MAE: {tier_mae:.3f}  (N={len(tier_df)})")
    history['per_tier_mae'] = per_tier_mae

    judge.eval()
    vp, vt = [], []
    with torch.no_grad():
        for emb, rating in val_loader:
            vp.extend(judge(emb.to(device)).cpu().numpy().tolist())
            vt.extend(rating.numpy().tolist())
    vp_arr, vt_arr = np.array(vp), np.array(vt)
    pred_std = float(vp_arr.std()); pred_mean = float(vp_arr.mean())
    mean_baseline_mae = float(np.mean(np.abs(vt_arr.mean() - vt_arr)))
    try:
        from scipy.stats import spearmanr
        sp = float(spearmanr(vp_arr, vt_arr)[0])  # [0] = correlation
    except Exception:
        sp = float('nan')
    probe_mae, probe_sp = _ridge_probe(train_ds, val_ds, device)
    history['diagnostics'] = {
        'spearman': sp, 'pred_mean': pred_mean, 'pred_std': pred_std,
        'orli_val_mae': float(best_mae), 'predict_mean_mae': mean_baseline_mae,
        'linear_probe_mae': probe_mae, 'linear_probe_spearman': probe_sp,
    }
    logger.info("ORLI DIAGNOSTIC METRICS:")
    logger.info(f"  Spearman(pred, rating)   = {sp:+.3f}")
    logger.info(f"  pred mean / std          = {pred_mean:.2f} / {pred_std:.2f}")
    logger.info(f"  ORLI MAE                 = {best_mae:.3f}  (predict-the-mean MAE = {mean_baseline_mae:.3f})")
    logger.info(f"  linear-probe ceiling     : MAE={probe_mae:.3f}  Spearman={probe_sp:+.3f}")
    return judge, history


class Actor:
    def __init__(self, config: Config, logger: logging.Logger):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        logger.info("=" * 70)
        logger.info("LOADING ACTOR")
        logger.info("=" * 70)
        self.config = config
        self.logger = logger
        self.device = config.device
        logger.info(f"Model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # ---- Smart quantization selection based on VRAM ----
        quant_mode = config.quantization.lower()
        if quant_mode == "auto":
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 80:
                    quant_mode = "bf16"
                elif vram_gb >= 24:
                    quant_mode = "8bit"
                else:
                    quant_mode = "4bit"
                logger.info(f"Auto-detected VRAM: {vram_gb:.1f} GB -> quantization={quant_mode}")
            else:
                quant_mode = "fp32"
                logger.info("No GPU detected -> fp32 (CPU)")

        if quant_mode == "4bit":
            logger.info("Loading with 4-bit quantization (QLoRA)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        elif quant_mode == "8bit":
            logger.info("Loading with 8-bit quantization...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        elif quant_mode == "bf16":
            logger.info("Loading in bf16 (no quantization — full precision)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True
            )
        elif quant_mode == "fp16":
            logger.info("Loading in fp16 (no quantization)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True
            )
        else:  # fp32 / CPU fallback
            logger.info("Loading in fp32 (CPU)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        logger.info(f"GPU memory after model load: {get_gpu_memory()}")

        # ---- Smart LoRA target selection ----
        if config.lora_targets.lower() == "auto":
            lora_target_modules = ["q_proj", "v_proj"]
            # For larger models (>=1B params), expand to all 7 projections
            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params >= 1_000_000_000:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
                    "gate_proj", "up_proj", "down_proj"       # FFN
                ]
                logger.info(f"Large model ({total_params/1e9:.1f}B params) -> all 7 LoRA targets")
            logger.info(f"Auto LoRA targets: {lora_target_modules}")
        else:
            lora_target_modules = [t.strip() for t in config.lora_targets.split(",")]
            logger.info(f"Manual LoRA targets: {lora_target_modules}")

        logger.info("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Actor loaded. Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        logger.info(f"GPU memory after LoRA: {get_gpu_memory()}")

    def format_prompt(self, query: str, context: str = "", state: str = "neutral", prev_msg: str = "") -> str:
        query = safe_truncate(query, self.config.max_user_chars)
        context = safe_truncate(context, self.config.max_context_chars)
        data_path_lower = self.config.data_path.lower()
        if 'recipe' in data_path_lower:
            role = "a helpful academic writing tutor"
            state_instructions = {
                "engaged": "The student is progressing well. Provide encouraging guidance.",
                "confused": "The student seems confused. Be extra clear and supportive.",
                "neutral": "Help the student clearly and thoroughly."
            }
        else:
            # General-purpose assistant (UltraFeedback, etc.)
            role = "a helpful, accurate, and thorough assistant"
            state_instructions = {
                "engaged": "The user is engaged. Provide detailed, high-quality guidance.",
                "confused": "Be extra clear, structured, and step-by-step in your explanation.",
                "neutral": "Provide a clear, helpful, and complete response."
            }
        instruction = state_instructions.get(state, "Help the user with their question.")
        system_content = f"You are {role}. {instruction}"
        if context:
            system_content += f"\n\nReference:\n{context}"
        messages = [{"role": "system", "content": system_content}]
        if prev_msg:
            messages.append({"role": "assistant", "content": safe_truncate(prev_msg, 300)})
        messages.append({"role": "user", "content": query})
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"System: {system_content}\nUser: {query}\nAssistant:"
        return prompt

    @torch.no_grad()
    def generate_k(self, prompts: List[str], k: Optional[int] = None) -> List[List[str]]:
        k = k or self.config.grpo_k
        self.model.eval()
        batch_responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_input_tokens, padding=False)
            input_ids = inputs['input_ids'].to(self.device)
            responses = []
            for i in range(k):
                temp = min(1.8, 0.7 + (i * 0.05))
                try:
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.config.max_output_tokens,
                        temperature=temp,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    new_tokens = output_ids[0][input_ids.shape[1]:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    response = safe_truncate(response.strip(), self.config.max_response_chars)
                except Exception:
                    response = "I apologize, but I couldn't generate a response."
                responses.append(response)
            batch_responses.append(responses)
        self.model.train()
        return batch_responses

    def compute_log_probs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        log_probs = []
        for prompt, response in zip(prompts, responses):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_input_tokens - 50, padding=False)
            prompt_len = prompt_tokens['input_ids'].shape[1]
            full_text = prompt + response
            full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.max_input_tokens + self.config.max_output_tokens, padding=False)
            input_ids = full_tokens['input_ids'].to(self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            shift_logits = logits[:, prompt_len-1:-1, :]
            shift_labels = input_ids[:, prompt_len:]
            log_prob = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_prob.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_log_prob = token_log_probs.sum(dim=1) / max(shift_labels.shape[1], 1)
            log_probs.append(seq_log_prob)
        return torch.cat(log_probs)

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_adapter(self, path: str):
        self.model.load_adapter(path, adapter_name="loaded", is_trainable=False)
        self.model.set_adapter("loaded")
        self.logger.info(f"Loaded trained actor LoRA from {path} (active adapter = 'loaded')")


class PERBuffer:
    def __init__(self, train_df: pd.DataFrame, config: Config, logger: logging.Logger):
        logger.info("=" * 70)
        logger.info("INITIALIZING PER BUFFER")
        logger.info("=" * 70)
        self.config = config
        self.buckets = defaultdict(list)
        self.priorities = defaultdict(list)
        self.beta = config.per_beta_start
        for _, row in train_df.iterrows():
            rating = int(row['rating'])
            sample = {
                'query': safe_truncate(str(row['user']), config.max_user_chars),
                'gold_response': safe_truncate(str(row['chatgpt_after']), config.max_response_chars),
                'rating': rating,
                'intent': str(row['intent_final']),
                'prev_msg': safe_truncate(str(row.get('chatgpt_before', '')), 300)
            }
            self.buckets[rating].append(sample)
            base_priority = (6 - rating) / 5 + 0.2
            self.priorities[rating].append(base_priority)
        for rating in self.buckets:
            if len(self.buckets[rating]) > config.max_buffer_per_bucket:
                indices = np.random.choice(len(self.buckets[rating]), config.max_buffer_per_bucket, replace=False)
                self.buckets[rating] = [self.buckets[rating][i] for i in indices]
                self.priorities[rating] = [self.priorities[rating][i] for i in indices]
        for rating in sorted(self.buckets.keys()):
            logger.info(f"  {rating}: {len(self.buckets[rating])} samples")
        logger.info("PER Buffer initialized")

    def sample(self, batch_size: int) -> Tuple[List[Dict], List[float], List[Tuple]]:
        all_samples, all_priorities, all_indices = [], [], []
        for rating in self.buckets:
            for i, (sample, priority) in enumerate(zip(self.buckets[rating], self.priorities[rating])):
                all_samples.append(sample)
                all_priorities.append(priority)
                all_indices.append((rating, i))
        if not all_samples:
            return [], [], []
        priorities = np.array(all_priorities) ** self.config.per_alpha
        probs = priorities / priorities.sum()
        n_samples = min(batch_size, len(all_samples))
        indices = np.random.choice(len(all_samples), n_samples, replace=False, p=probs)
        N = len(all_samples)
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return [all_samples[i] for i in indices], weights.tolist(), [all_indices[i] for i in indices]

    def update_priorities(self, per_indices: List[Tuple], new_priorities: List[float]):
        for (rating, idx), priority in zip(per_indices, new_priorities):
            if idx < len(self.priorities[rating]):
                self.priorities[rating][idx] = priority + self.config.per_epsilon

    def anneal_beta(self, progress: float):
        self.beta = self.config.per_beta_start + progress * (self.config.per_beta_end - self.config.per_beta_start)


class GRPOTrainer:
    def __init__(self, actor: Actor, judge: ORLIJudge, memory: RatingAwareMemory, buffer: PERBuffer, config: Config, logger: logging.Logger):
        self.actor = actor
        self.judge = judge
        self.memory = memory
        self.buffer = buffer
        self.config = config
        self.logger = logger
        self.device = config.device
        self.optimizer = AdamW(actor.get_trainable_params(), lr=config.learning_rate, weight_decay=0.01)
        # Cosine LR scheduler with linear warmup
        # IMPORTANT: total_optimizer_steps counts the number of times
        # scheduler.step() is called (once per accum cycle), NOT total batches.
        # Using total batches here would cause the cosine to only half-decay.
        total_batches = config.epochs * config.batches_per_epoch
        total_optimizer_steps = total_batches // config.gradient_accumulation_steps
        warmup_steps = max(1, int(total_optimizer_steps * config.warmup_ratio))
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.accum_steps = config.gradient_accumulation_steps
        self.global_step = 0
        self.micro_step = 0  # counts forward passes (for grad accum)
        self._last_grad_norm = 0.0
        self.history: Dict[str, Any] = {
            'step': [], 'epoch': [], 'loss': [], 'reward': [],
            'reward_sim': [], 'reward_orli': [], 'contrastive_reward': [],
            'alpha': [], 'beta': [], 'gamma': [],
            'advantage_mean': [], 'advantage_std': [],
            'grad_norm': [], 'gen_len': [], 'lr': [],
            'per_rating_reward': {1: [], 2: [], 3: [], 4: [], 5: []},
        }
        self.judge.eval()
        for p in self.judge.parameters():
            p.requires_grad = False

    def get_curriculum_weights(self) -> Tuple[float, float, float]:
        total_steps = self.config.epochs * self.config.batches_per_epoch
        progress = min(1.0, self.global_step / total_steps)
        alpha = self.config.alpha_start + progress * (self.config.alpha_end - self.config.alpha_start)
        beta = self.config.beta_start + progress * (self.config.beta_end - self.config.beta_start)
        gamma = self.config.gamma_start + progress * (self.config.gamma_end - self.config.gamma_start)
        return alpha, beta, gamma

    def compute_contrastive_reward(self, response_emb: np.ndarray, gold_emb: np.ndarray, negative_embs: List[np.ndarray]) -> float:
        """Compute contrastive reward as: pos_sim - mean(neg_sims).
        Returns a scalar reward per candidate (not a loss, not differentiable).
        Signal reaches Actor through GRPO advantages → policy gradient.
        """
        pos_sim = float(np.dot(response_emb, gold_emb))
        if not negative_embs:
            return pos_sim
        neg_sims = [float(np.dot(response_emb, neg)) for neg in negative_embs]
        return float(pos_sim - np.mean(neg_sims))

    def train_step(self, samples: List[Dict], is_weights: List[float], per_indices: List[Tuple], current_epoch: int) -> Dict:
        self.actor.model.train()
        alpha, beta, gamma = self.get_curriculum_weights()
        prompts = []
        gold_responses = []
        sample_negative_embs = []  # per-sample negative embeddings
        for sample in samples:
            # --no_retrieval disables the FAISS scaffold (context + contrastive negatives)
            # so the no-retrieval ablation trains consistently with how it is evaluated.
            if self.config.use_retrieval:
                ret = self.memory.retrieve(sample['query'], source_rating=sample['rating'])
                context_str, state, neg_samples = ret['context_str'], ret['state'], ret['negative_samples']
            else:
                context_str, state, neg_samples = "", "neutral", []
            prompt = self.actor.format_prompt(query=sample['query'], context=context_str, state=state, prev_msg=sample['prev_msg'])
            prompts.append(prompt)
            gold_responses.append(sample['gold_response'])
            neg_embs = []
            if neg_samples:
                for neg in neg_samples[:2]:
                    # Use encode_single (not encode_response_pair) so embeddings
                    # are in the same space as resp_emb and gold_emb
                    neg_emb = self.memory.encoder.encode_single(neg['response'])
                    neg_embs.append(neg_emb)
            sample_negative_embs.append(neg_embs)
        all_responses = self.actor.generate_k(prompts)
        all_rewards = []
        all_contrastive_rewards = []
        all_sim_rewards = []
        all_orli_rewards = []
        flat_prompts = []
        flat_responses = []
        per_rating_rewards = {1: [], 2: [], 3: [], 4: [], 5: []}
        for i, (sample, responses) in enumerate(zip(samples, all_responses)):
            gold_emb = self.memory.encoder.encode_single(sample['gold_response'])
            for response in responses:
                resp_emb = self.memory.encoder.encode_single(response)
                sim_reward = float(np.dot(resp_emb, gold_emb))
                pair_emb = self.memory.encode_response_pair(sample['query'], response)
                pair_tensor = torch.tensor(pair_emb, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    orli_reward = self.judge(pair_tensor).item()
                # Contrastive as reward component (not loss)
                contrastive_r = 0.0
                if self.config.use_contrastive and sample_negative_embs[i]:
                    contrastive_r = self.compute_contrastive_reward(resp_emb, gold_emb, sample_negative_embs[i])
                all_contrastive_rewards.append(contrastive_r)
                all_sim_rewards.append(sim_reward)
                all_orli_rewards.append(orli_reward)
                alpha_eff = alpha * (0.6 + 0.1 * sample['rating']) if self.config.protect_high_ratings else alpha
                final_reward = (alpha_eff * sim_reward) + (beta * orli_reward) + (gamma * contrastive_r)
                all_rewards.append(final_reward)
                flat_prompts.append(prompts[i])
                flat_responses.append(response)
                per_rating_rewards[sample['rating']].append(final_reward)
        rewards_tensor = torch.tensor(all_rewards, device=self.device)
        rewards_grouped = rewards_tensor.view(-1, self.config.grpo_k)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean) / std).view(-1)
        is_weights_expanded = torch.tensor(is_weights, device=self.device, dtype=torch.float).repeat_interleave(self.config.grpo_k)
        N = len(flat_responses)  # = (#samples in batch) * K
        chunk = max(1, self.config.logprob_chunk_size)
        adv_detached = advantages.detach()
        policy_loss_value = 0.0  # reconstructs the original (unscaled) mean policy loss
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            chunk_logps = self.actor.compute_log_probs(flat_prompts[start:end], flat_responses[start:end])
            # Normalize by the GLOBAL N (so summing chunks == mean over all N) and by
            # accum_steps (gradient accumulation), matching the original loss exactly.
            chunk_loss = -(is_weights_expanded[start:end] * adv_detached[start:end] * chunk_logps).sum() / N / self.accum_steps
            chunk_loss.backward()
            policy_loss_value += chunk_loss.item() * self.accum_steps  # undo /accum_steps for logging
            del chunk_logps, chunk_loss
        self.micro_step += 1
        # Only step optimizer after accumulating enough gradients
        if self.micro_step % self.accum_steps == 0:
            gn = torch.nn.utils.clip_grad_norm_(self.actor.get_trainable_params(), self.config.grad_clip)
            self._last_grad_norm = float(gn)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        new_priorities = []
        for i, sample in enumerate(samples):
            pred_reward = rewards_grouped[i].mean().item()
            error = abs(pred_reward - sample['rating'])
            rating_weight = (6 - sample['rating']) / 5
            priority = error + 0.3 * rating_weight + self.config.per_epsilon
            new_priorities.append(priority)
        self.buffer.update_priorities(per_indices, new_priorities)
        self.global_step += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        avg_contrastive = float(np.mean(all_contrastive_rewards)) if all_contrastive_rewards else 0.0
        avg_sim = float(np.mean(all_sim_rewards)) if all_sim_rewards else 0.0
        avg_orli = float(np.mean(all_orli_rewards)) if all_orli_rewards else 0.0
        gen_len = float(np.mean([len(r) for r in flat_responses])) if flat_responses else 0.0
        mean_reward = rewards_tensor.mean().item()
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        self.history['step'].append(self.global_step)
        self.history['epoch'].append(current_epoch)
        self.history['loss'].append(policy_loss_value)
        self.history['reward'].append(mean_reward)
        self.history['reward_sim'].append(avg_sim)
        self.history['reward_orli'].append(avg_orli)
        self.history['contrastive_reward'].append(avg_contrastive)
        self.history['alpha'].append(alpha)
        self.history['beta'].append(beta)
        self.history['gamma'].append(gamma)
        self.history['advantage_mean'].append(adv_mean)
        self.history['advantage_std'].append(adv_std)
        self.history['grad_norm'].append(self._last_grad_norm)
        self.history['gen_len'].append(gen_len)
        self.history['lr'].append(current_lr)
        for rating in range(1, 6):
            if per_rating_rewards[rating]:
                self.history['per_rating_reward'][rating].append(float(np.mean(per_rating_rewards[rating])))
        # Fix 3: free per-step scratch and any fragmented cache EVERY step (was every 20).
        clear_memory()
        return {
            'loss': policy_loss_value, 'reward': mean_reward,
            'reward_sim': avg_sim, 'reward_orli': avg_orli, 'contrastive': avg_contrastive,
            'adv_mean': adv_mean, 'adv_std': adv_std, 'grad_norm': self._last_grad_norm,
            'gen_len': gen_len, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lr': current_lr,
        }

    def _log_step_memory(self, mem_csv: str, epoch: int):
        if not torch.cuda.is_available():
            return
        end_alloc = torch.cuda.memory_allocated() / 1e9
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        with open(mem_csv, "a") as f:
            f.write(f"{self.global_step},{epoch},{end_alloc:.3f},{peak_alloc:.3f},{reserved:.3f}\n")

    def _log_step_metrics(self, mcsv: str, m: Dict, epoch: int):
        """Append a full per-step row to metrics.csv. Flushed EVERY step, so even a crash
        leaves the complete training trace to make decisions from (not just text logs)."""
        with open(mcsv, "a") as f:
            f.write(f"{self.global_step},{epoch},{m['loss']:.5f},{m['reward']:.5f},"
                    f"{m['reward_sim']:.5f},{m['reward_orli']:.5f},{m['contrastive']:.5f},"
                    f"{m['adv_mean']:.5f},{m['adv_std']:.5f},{m['grad_norm']:.4f},"
                    f"{m['lr']:.3e},{m['alpha']:.4f},{m['beta']:.4f},{m['gamma']:.4f},{m['gen_len']:.0f}\n")

    def _rolling_per_rating(self, window: int = 50) -> Dict[int, float]:
        out = {}
        for r in range(1, 6):
            vals = self.history['per_rating_reward'][r][-window:]
            if vals:
                out[r] = float(np.mean(vals))
        return out

    def _format_progress(self, m: Dict, epoch: int) -> str:
        pr = self._rolling_per_rating()
        pr_str = "  ".join(f"{r}*={pr[r]:+.2f}" for r in sorted(pr)) if pr else "n/a"
        return (f"[step {self.global_step} | epoch {epoch}/{self.config.epochs}]  "
                f"loss={m['loss']:.3f}  reward={m['reward']:+.3f} "
                f"(sim={m['reward_sim']:+.2f} orli={m['reward_orli']:+.2f} con={m['contrastive']:+.2f})\n"
                f"        adv mu={m['adv_mean']:+.2f} sd={m['adv_std']:.2f} | grad={m['grad_norm']:.2f} | "
                f"lr={m['lr']:.1e} | a={m['alpha']:.2f} b={m['beta']:.2f} g={m['gamma']:.2f} | "
                f"genlen={m['gen_len']:.0f} | GPU={get_gpu_memory()}\n"
                f"        per-rating reward (roll50): {pr_str}")

    def _flush_history(self, epoch_completed: int = 0):
        """Write training_history.json now so a mid-epoch crash still leaves a snapshot."""
        path = os.path.join(self.config.run_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump({
                'global_step': self.global_step,
                'epoch_completed': epoch_completed,
                'epochs_total': self.config.epochs,
                'best_epoch': getattr(self, '_best_epoch', 0),
                'best_reward': getattr(self, '_best_reward', float('-inf')),
                'history': {k: (v if not isinstance(v, dict) else {str(r): vals for r, vals in v.items()})
                            for k, v in self.history.items()},
            }, f, indent=2, default=json_serializable)

    def train(self) -> Dict:
        self.logger.info("=" * 70)
        self.logger.info(f"GRPO TRAINING (K={self.config.grpo_k}, {self.config.epochs} epochs)")
        self.logger.info("=" * 70)
        self.logger.info(f"Curriculum: alpha {self.config.alpha_start}->{self.config.alpha_end}, beta {self.config.beta_start}->{self.config.beta_end}, gamma {self.config.gamma_start}->{self.config.gamma_end}")
        self.logger.info(f"Contrastive: {'Reward component' if self.config.use_contrastive else 'Disabled'}")
        self.logger.info(f"Gradient accumulation: {self.accum_steps} (effective batch = {self.config.batch_size * self.accum_steps})")
        self.logger.info(f"LR schedule: cosine with {self.config.warmup_ratio*100:.0f}% warmup")
        self.logger.info(f"Log-prob chunk size: {self.config.logprob_chunk_size} "
                         f"(at most this many full-model graphs alive at once)")
        mem_csv = os.path.join(self.config.run_dir, "logs", "memory.csv")
        with open(mem_csv, "w") as _mf:
            _mf.write("step,epoch,end_alloc_gb,peak_alloc_gb,reserved_gb\n")
        metrics_csv = os.path.join(self.config.run_dir, "logs", "metrics.csv")
        with open(metrics_csv, "w") as _mf:
            _mf.write("step,epoch,loss,reward,reward_sim,reward_orli,reward_contrastive,"
                      "adv_mean,adv_std,grad_norm,lr,alpha,beta,gamma,gen_len\n")
        self.logger.info(f"Observability: per-step -> logs/metrics.csv & logs/memory.csv; "
                         f"snapshot training_history.json + console summary every {self.config.log_every} steps"
                         + (f"; actor_latest checkpoint every {self.config.checkpoint_every} steps"
                            if self.config.checkpoint_every > 0 else ""))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Zero gradients once at the start for accumulation
        self.optimizer.zero_grad()
        # Best-actor tracking
        self._best_reward: float = float('-inf')
        self._best_epoch: int = 0
        self._best_actor_path: str = ""
        # Track per-epoch checkpoint paths for pruning
        saved_epoch_checkpoints: List[str] = []
        for epoch in range(self.config.epochs):
            epoch_losses = []
            epoch_rewards = []
            pbar = tqdm(range(self.config.batches_per_epoch), desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch_idx in pbar:
                total_steps = self.config.epochs * self.config.batches_per_epoch
                progress = (epoch * self.config.batches_per_epoch + batch_idx) / total_steps
                self.buffer.anneal_beta(progress)
                samples, is_weights, per_indices = self.buffer.sample(self.config.batch_size)
                if not samples:
                    continue
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()  # measure THIS step's peak
                try:
                    metrics = self.train_step(samples, is_weights, per_indices, epoch)
                    epoch_losses.append(metrics['loss'])
                    epoch_rewards.append(metrics['reward'])
                    self._log_step_memory(mem_csv, epoch + 1)
                    self._log_step_metrics(metrics_csv, metrics, epoch + 1)
                    if self.global_step % self.config.log_every == 0:
                        self.logger.info(self._format_progress(metrics, epoch + 1))
                        self._flush_history(epoch_completed=epoch)  # mid-epoch snapshot
                    if self.config.checkpoint_every > 0 and self.global_step % self.config.checkpoint_every == 0:
                        self.actor.save(os.path.join(self.config.run_dir, "checkpoints", "actor_latest"))
                    pbar.set_postfix({'L': f"{metrics['loss']:.3f}", 'R': f"{metrics['reward']:.2f}",
                                      'orli': f"{metrics['reward_orli']:.2f}", 'grad': f"{metrics['grad_norm']:.1f}",
                                      'lr': f"{metrics['lr']:.1e}"})
                except Exception as e:
                    self.logger.warning(f"Batch error: {e}")
                    clear_memory()
                    continue
            # Flush any residual accumulated gradients at end of epoch
            # (matters when batches_per_epoch is not divisible by accum_steps)
            if self.micro_step % self.accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.actor.get_trainable_params(), self.config.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            delta_reward = avg_reward - getattr(self, '_prev_epoch_reward', avg_reward)
            self._prev_epoch_reward = avg_reward
            pr = self._rolling_per_rating(window=self.config.batches_per_epoch)
            pr_str = "  ".join(f"{r}*={pr[r]:+.2f}" for r in sorted(pr)) if pr else "n/a"
            n_ok = len(epoch_losses)
            self.logger.info("-" * 70)
            self.logger.info(f"EPOCH {epoch+1}/{self.config.epochs} SUMMARY  ({n_ok}/{self.config.batches_per_epoch} batches ok)")
            self.logger.info(f"  loss={avg_loss:.4f}  reward={avg_reward:+.3f}  (delta vs last epoch {delta_reward:+.3f})")
            self.logger.info(f"  per-rating reward: {pr_str}")
            self.logger.info(f"  GPU={get_gpu_memory()}")
            if epoch + 1 == self.config.epochs and delta_reward > 0.01:
                self.logger.info(f"  NOTE: reward still rising at the final epoch (+{delta_reward:.3f}) -> "
                                 f"more --epochs (now {self.config.epochs}) would likely help.")
            self.logger.info("-" * 70)

            # Save per-epoch actor checkpoint (LoRA weights via save_pretrained)
            ckpt_path = os.path.join(self.config.run_dir, "checkpoints", f"actor_epoch{epoch+1}")
            self.actor.save(ckpt_path)
            saved_epoch_checkpoints.append(ckpt_path)
            self.logger.info(f"  Checkpoint saved: checkpoints/actor_epoch{epoch+1}/")

            # Track best epoch by mean reward
            if avg_reward > self._best_reward:
                self._best_reward = avg_reward
                self._best_epoch = epoch + 1
                self._best_actor_path = ckpt_path

            # Prune old checkpoints if keep_checkpoints > 0
            if self.config.keep_checkpoints > 0 and len(saved_epoch_checkpoints) > self.config.keep_checkpoints:
                import shutil
                to_delete = saved_epoch_checkpoints[:-self.config.keep_checkpoints]
                for old_path in to_delete:
                    # Never delete the current best
                    if old_path != self._best_actor_path and os.path.isdir(old_path):
                        shutil.rmtree(old_path)
                        self.logger.info(f"  Pruned old checkpoint: {os.path.basename(old_path)}/")
                saved_epoch_checkpoints = saved_epoch_checkpoints[-self.config.keep_checkpoints:]

            # Snapshot training history at epoch end (also flushed every log_every steps mid-epoch).
            self._flush_history(epoch_completed=epoch + 1)

            clear_memory()
        self.logger.info("Training complete")
        # Save final best actor (highest mean reward across all epochs)
        if self._best_actor_path and os.path.isdir(self._best_actor_path):
            best_dest = os.path.join(self.config.run_dir, "checkpoints", "actor_best")
            if self._best_actor_path != best_dest:
                import shutil
                if os.path.exists(best_dest):
                    shutil.rmtree(best_dest)
                shutil.copytree(self._best_actor_path, best_dest)
            self.logger.info(f"Best actor (epoch {self._best_epoch}, reward={self._best_reward:.4f}) "
                             f"saved to checkpoints/actor_best/")
        return self.history


def apply_armorm_compat_shim(logger: Optional[logging.Logger] = None) -> bool:
    """Make ArmoRM's remote code importable on modern transformers.

    ArmoRM's modeling_custom.py does `from transformers.models.llama.modeling_llama import
    LLAMA_INPUTS_DOCSTRING` — a docstring constant the transformers >=4.47 attention/docstring
    refactor removed. It has NO runtime effect, and it is the ONLY removed symbol ArmoRM needs
    (LlamaModel / LlamaPreTrainedModel / add_start_docstrings_to_model_forward still exist).
    Injecting a stub fixes the load WITHOUT downgrading transformers (which would break Qwen2.5).
    This is exactly why the 2026-06-30 run's ArmoRM failed — see new_outputs/RESULTS_ANALYSIS.md.
    """
    try:
        import transformers.models.llama.modeling_llama as _llm
        if not hasattr(_llm, "LLAMA_INPUTS_DOCSTRING"):
            _llm.LLAMA_INPUTS_DOCSTRING = ""
            if logger:
                logger.info("ArmoRM compat shim: stubbed LLAMA_INPUTS_DOCSTRING (transformers>=4.47)")
            return True
    except Exception as e:
        if logger:
            logger.warning(f"ArmoRM compat shim could not run: {e}")
    return False


def chat_template_input_ids(tokenizer, messages, max_length: int) -> torch.Tensor:
    enc = tokenizer.apply_chat_template(
        messages, return_tensors="pt", truncation=True, max_length=max_length,
    )
    if torch.is_tensor(enc):
        return enc
    ids = enc["input_ids"] if "input_ids" in enc else None
    if ids is None:
        raise TypeError(f"apply_chat_template returned unexpected type {type(enc)!r} "
                        f"with keys {list(getattr(enc, 'keys', lambda: [])())}")
    return ids if torch.is_tensor(ids) else torch.tensor(ids)


class ArmoRMEvaluator:

    def __init__(self, config: Config, logger: logging.Logger):
        logger.info("=" * 70)
        logger.info("LOADING ArmoRM EVALUATOR (independent metric)")
        logger.info("=" * 70)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise RuntimeError("transformers package required: pip install transformers")

        # Must run BEFORE from_pretrained triggers ArmoRM's remote-code import.
        apply_armorm_compat_shim(logger)

        model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        logger.info(f"Model: {model_id}")
        logger.info("Loading in bf16 (~16GB VRAM)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": 0},
        ).eval()
        self.max_length = 4096
        self.config = config
        self.logger = logger
        # Failure tracking — a silently-failing independent metric is worse than none,
        # because a broken metric reads as a clean null. We count and surface failures.
        self.n_total = 0
        self.n_fail = 0
        self._warned = False
        logger.info(f"ArmoRM loaded. GPU memory after load: {get_gpu_memory()}")
        _t = self.score("What is the capital of France?", "The capital of France is Paris.")
        self.n_total = 0; self.n_fail = 0; self._warned = False  # reset stats after the self-test
        if _t is None:
            raise RuntimeError(
                "ArmoRM loaded but a TEST score FAILED — almost always a transformers version "
                "mismatch (e.g. apply_chat_template returning a BatchEncoding). Run "
                "'python check_armorm.py', then rebuild the image and confirm 'python main.py "
                "--version' shows the latest build. Aborting now instead of wasting the eval."
            )
        logger.info(f"ArmoRM self-test OK (test score={_t:.4f}).")

    @torch.no_grad()
    def score(self, query: str, response: str) -> Optional[float]:

        self.n_total += 1
        messages = [
            {"role": "user", "content": safe_truncate(query, 1500)},
            {"role": "assistant", "content": safe_truncate(response, 1500)},
        ]
        try:
            input_ids = chat_template_input_ids(
                self.tokenizer, messages, self.max_length,
            ).to(self.model.device)
            output = self.model(input_ids)
            return float(output.score.float().item())
        except Exception as e:
            self.n_fail += 1
            if not self._warned:
                self.logger.warning(f"ArmoRM scoring failed on an example: {e} "
                                    f"(further failures counted silently)")
                self._warned = True
            return None

    def failure_summary(self) -> str:
        if self.n_total == 0:
            return "ArmoRM: no examples scored"
        pct = 100.0 * self.n_fail / self.n_total
        return f"ArmoRM scoring: {self.n_fail}/{self.n_total} failed ({pct:.1f}%)"


class Evaluator:
    def __init__(self, encoder: Encoder, judge: ORLIJudge, memory: RatingAwareMemory, config: Config, logger: logging.Logger):
        self.encoder = encoder
        self.judge = judge
        self.memory = memory
        self.config = config
        self.logger = logger
        self.device = config.device
        self.judge.eval()

    @torch.no_grad()
    def evaluate(self, actor: Actor, test_df: pd.DataFrame,
                 armo: Optional['ArmoRMEvaluator'] = None) -> Dict:
        self.logger.info("=" * 70)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 70)
        if armo is not None:
            self.logger.info("ArmoRM independent scoring: ENABLED")
        use_retrieval = self.config.use_retrieval
        self.logger.info(f"Retrieval scaffold at eval: {'ENABLED' if use_retrieval else 'DISABLED (clean null)'}")
        actor.model.eval()
        results = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            query = safe_truncate(str(row['user']), self.config.max_user_chars)
            gold_response = safe_truncate(str(row['chatgpt_after']), self.config.max_response_chars)
            prev_msg = safe_truncate(str(row.get('chatgpt_before', '')), 300)
            rating = int(row['rating'])
            if use_retrieval:
                ret = self.memory.retrieve(query, source_rating=rating)
                context_str, state = ret['context_str'], ret['state']
            else:
                context_str, state = "", "neutral"
            prompt = actor.format_prompt(query=query, context=context_str, state=state, prev_msg=prev_msg)
            responses = actor.generate_k([prompt], k=1)
            generated = responses[0][0] if responses and responses[0] else ""

            # ORLI scoring (primary metric — same scorer used in training)
            gen_emb = self.encoder.encode_single(generated)
            gold_emb = self.encoder.encode_single(gold_response)
            similarity = float(np.dot(gen_emb, gold_emb))
            gen_pair_emb = self.memory.encode_response_pair(query, generated)
            gold_pair_emb = self.memory.encode_response_pair(query, gold_response)
            gen_orli = self.judge(torch.tensor(gen_pair_emb, device=self.device).unsqueeze(0)).item()
            gold_orli = self.judge(torch.tensor(gold_pair_emb, device=self.device).unsqueeze(0)).item()

            # ArmoRM scoring (independent metric — never used in training).
            # score() returns None on failure; armo_delta only defined when both succeed.
            gen_armo = armo.score(query, generated) if armo is not None else None
            gold_armo = armo.score(query, gold_response) if armo is not None else None
            armo_delta = (gen_armo - gold_armo) if (gen_armo is not None and gold_armo is not None) else None

            results.append({
                'sample_id': str(row.get('sample_id', '')),
                'rating': rating,
                'similarity': similarity,
                'gen_orli': gen_orli,
                'gold_orli': gold_orli,
                'delta': gen_orli - gold_orli,
                'generated': generated,
                'gen_armo': gen_armo,
                'gold_armo': gold_armo,
                'armo_delta': armo_delta,
            })
        actor.model.train()
        if armo is not None:
            self.logger.info(armo.failure_summary())
            if armo.n_fail > 0.5 * max(1, armo.n_total):
                self.logger.warning("ArmoRM failed on >50% of examples — treat the independent "
                                    "metric as UNRELIABLE for this run.")
        return self._aggregate(results)

    def _aggregate(self, results: List[Dict]) -> Dict:
        from scipy import stats as scipy_stats

        # ArmoRM may be absent (--no_armo) or fail on individual examples (None).
        # Treat the metric as present only if at least one example scored successfully,
        # and everywhere below filter out None so a few failures don't poison the means.
        has_armo = any(r.get('gen_armo') is not None for r in results)

        # Seed the bootstrap so CIs are reproducible across reruns (independent of the
        # nondeterministic training RNG state that precedes evaluation).
        rng = np.random.RandomState(self.config.seed)

        def _bootstrap_ci(data: List[float], n_boot: int = 5000) -> Tuple[float, float]:
            data = [d for d in data if d is not None and not np.isnan(d)]
            if len(data) < 2:
                return float('nan'), float('nan')
            arr = np.array(data)
            boot = np.array([np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
            return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        def _wilcoxon_p(data: List[float]) -> float:
            data = [d for d in data if d is not None and not np.isnan(d)]
            if len(data) < 10:
                return float('nan')
            try:
                _, p = scipy_stats.wilcoxon(data, alternative='greater')
                return float(p)
            except Exception:
                return float('nan')

        def _mean(vals: List) -> float:
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else float('nan')

        # ---- Objective diagnostic metrics, saved to results.json (interpretation -> paper) ----
        # gen ORLI spread + its correlation with the input tier and with output length.
        def _pearson(a: List, b: List) -> float:
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            if len(a) < 2 or a.std() == 0 or b.std() == 0:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])
        gen_orlis = [r['gen_orli'] for r in results]
        gold_orlis = [r['gold_orli'] for r in results]
        ratings = [r['rating'] for r in results]
        gen_lens = [len(str(r.get('generated', ''))) for r in results]

        # Overall metrics
        overall: Dict = {
            'n': len(results),
            'armo_available': bool(has_armo),   # explicit flag: was the independent metric scored?
            'similarity': float(np.mean([r['similarity'] for r in results])),
            'gen_orli': float(np.mean(gen_orlis)),
            'gold_orli': float(np.mean(gold_orlis)),
            'delta': float(np.mean([r['delta'] for r in results])),
            # score-spread diagnostics
            'gen_orli_std': float(np.std(gen_orlis)),
            'gold_orli_std': float(np.std(gold_orlis)),
            'pearson_gen_orli_rating': _pearson(gen_orlis, ratings),
            'pearson_gold_orli_rating': _pearson(gold_orlis, ratings),
            # length diagnostics
            'gen_len_mean': float(np.mean(gen_lens)) if gen_lens else 0.0,
            'gen_len_max': int(np.max(gen_lens)) if gen_lens else 0,
            'pearson_gen_orli_len': _pearson(gen_orlis, gen_lens),
        }
        if has_armo:
            overall['n_armo'] = sum(1 for r in results if r['armo_delta'] is not None)
            overall['gen_armo'] = _mean([r['gen_armo'] for r in results])
            overall['gold_armo'] = _mean([r['gold_armo'] for r in results])
            overall['armo_delta'] = _mean([r['armo_delta'] for r in results])

        # Per-tier metrics with bootstrap CIs and Wilcoxon tests
        per_rating: Dict = {}
        all_orli_pvals: List[float] = []
        all_armo_pvals: List[float] = []
        tier_order: List[int] = []

        for rating in sorted(set(r['rating'] for r in results)):
            rr = [r for r in results if r['rating'] == rating]
            deltas = [r['delta'] for r in rr]
            ci_lo, ci_hi = _bootstrap_ci(deltas)
            p_orli = _wilcoxon_p(deltas)

            entry: Dict = {
                'n': len(rr),
                'similarity': float(np.mean([r['similarity'] for r in rr])),
                'gen_orli': float(np.mean([r['gen_orli'] for r in rr])),
                'gold_orli': float(np.mean([r['gold_orli'] for r in rr])),
                'delta': float(np.mean(deltas)),
                'delta_std': float(np.std(deltas)),
                'delta_ci95_low': ci_lo,
                'delta_ci95_high': ci_hi,
                'wilcoxon_p': p_orli,
            }

            if has_armo:
                armo_deltas = [r['armo_delta'] for r in rr if r['armo_delta'] is not None]
                a_lo, a_hi = _bootstrap_ci(armo_deltas)
                p_armo = _wilcoxon_p(armo_deltas)
                entry.update({
                    'n_armo': len(armo_deltas),
                    'gen_armo': _mean([r['gen_armo'] for r in rr]),
                    'gold_armo': _mean([r['gold_armo'] for r in rr]),
                    'armo_delta': _mean([r['armo_delta'] for r in rr]),
                    'armo_delta_ci95_low': a_lo,
                    'armo_delta_ci95_high': a_hi,
                    'armo_wilcoxon_p': p_armo,
                })
                all_armo_pvals.append(p_armo)

            per_rating[rating] = entry
            all_orli_pvals.append(p_orli)
            tier_order.append(rating)

        # Holm-Bonferroni correction across tiers
        try:
            from statsmodels.stats.multitest import multipletests
            for pvals, key_p, key_sig in [
                (all_orli_pvals, 'wilcoxon_p_holm', 'holm_sig'),
                (all_armo_pvals, 'armo_wilcoxon_p_holm', 'armo_holm_sig'),
            ]:
                if not pvals:
                    continue
                valid = [(i, p) for i, p in enumerate(pvals) if not np.isnan(p)]
                if valid:
                    idxs, ps = zip(*valid)
                    reject, p_corr, _, _ = multipletests(list(ps), method='holm')
                    for rank, (orig_i, _) in enumerate(valid):
                        rating = tier_order[orig_i]
                        per_rating[rating][key_p] = float(p_corr[rank])
                        per_rating[rating][key_sig] = bool(reject[rank])
        except ImportError:
            pass  # statsmodels optional; raw p-values still saved

        return {'overall': overall, 'per_rating': per_rating, 'raw': results}

    def print_results(self, results: Dict):
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RESULTS")
        self.logger.info("=" * 70)
        o = results['overall']
        has_armo = 'gen_armo' in o

        self.logger.info(f"\nOVERALL (n={o['n']}):")
        self.logger.info(f"  Similarity:    {o['similarity']:.3f}")
        self.logger.info(f"  Gen ORLI:      {o['gen_orli']:.3f}")
        self.logger.info(f"  Gold ORLI:     {o['gold_orli']:.3f}")
        self.logger.info(f"  ORLI Delta:    {o['delta']:+.3f}")
        if has_armo:
            self.logger.info(f"  Gen ArmoRM:    {o['gen_armo']:.4f}")
            self.logger.info(f"  Gold ArmoRM:   {o['gold_armo']:.4f}")
            self.logger.info(f"  ArmoRM Delta:  {o['armo_delta']:+.4f}")
        else:
            self.logger.info("  ArmoRM:        not available for this run (scored by ORLI only)")

        # Objective diagnostic metrics (also saved in results.json['overall']).
        self.logger.info("\nDIAGNOSTIC METRICS:")
        self.logger.info(f"  Gen ORLI std:               {o.get('gen_orli_std', float('nan')):.3f}")
        self.logger.info(f"  Pearson(gen ORLI, rating):  {o.get('pearson_gen_orli_rating', float('nan')):+.3f}")
        self.logger.info(f"  Pearson(gold ORLI, rating): {o.get('pearson_gold_orli_rating', float('nan')):+.3f}")
        self.logger.info(f"  Gen length mean/max:        {o.get('gen_len_mean', 0):.0f} / {o.get('gen_len_max', 0)} chars")
        self.logger.info(f"  Pearson(gen ORLI, length):  {o.get('pearson_gen_orli_len', float('nan')):+.3f}")

        self.logger.info("\nPER-RATING (ORLI):")
        hdr = f"{'★':<6} {'N':<6} {'GenORLI':<10} {'GoldORLI':<10} {'Δ ORLI':<10} {'95% CI':<20} {'p(Holm)':<10}"
        self.logger.info(hdr)
        self.logger.info("-" * 75)
        for rating in sorted(results['per_rating'].keys()):
            m = results['per_rating'][rating]
            ci = f"[{m.get('delta_ci95_low', float('nan')):+.3f}, {m.get('delta_ci95_high', float('nan')):+.3f}]"
            p_str = f"{m.get('wilcoxon_p_holm', m.get('wilcoxon_p', float('nan'))):.3f}"
            sig = "*" if m.get('holm_sig', False) else ""
            self.logger.info(
                f"{rating:<6} {m['n']:<6} {m['gen_orli']:<10.3f} {m['gold_orli']:<10.3f} "
                f"{m['delta']:<+10.3f} {ci:<20} {p_str}{sig}"
            )

        if has_armo:
            self.logger.info("\nPER-RATING (ArmoRM — independent metric):")
            hdr2 = f"{'★':<6} {'N':<6} {'GenArmo':<12} {'GoldArmo':<12} {'Δ ArmoRM':<12} {'95% CI':<24} {'p(Holm)':<10}"
            self.logger.info(hdr2)
            self.logger.info("-" * 85)
            for rating in sorted(results['per_rating'].keys()):
                m = results['per_rating'][rating]
                a_ci = f"[{m.get('armo_delta_ci95_low', float('nan')):+.4f}, {m.get('armo_delta_ci95_high', float('nan')):+.4f}]"
                a_p = f"{m.get('armo_wilcoxon_p_holm', m.get('armo_wilcoxon_p', float('nan'))):.3f}"
                a_sig = "*" if m.get('armo_holm_sig', False) else ""
                self.logger.info(
                    f"{rating:<6} {m['n']:<6} {m['gen_armo']:<12.4f} {m['gold_armo']:<12.4f} "
                    f"{m['armo_delta']:<+12.4f} {a_ci:<24} {a_p}{a_sig}"
                )


def create_visualizations(history: Dict, results: Dict, config: Config, logger: logging.Logger):
    logger.info("Creating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = history['step']
    ax = axes[0, 0]
    ax.plot(steps, history['loss'], 'b-', alpha=0.5, linewidth=1)
    window = min(20, len(history['loss']) // 5) or 1
    if window > 1 and len(history['loss']) > window:
        smoothed = pd.Series(history['loss']).rolling(window=window, center=True).mean()
        ax.plot(steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (w={window})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'(a) Training Loss (K={config.grpo_k})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.plot(steps, history['reward'], 'g-', alpha=0.5, linewidth=1)
    if window > 1 and len(history['reward']) > window:
        smoothed = pd.Series(history['reward']).rolling(window=window, center=True).mean()
        ax.plot(steps, smoothed, 'g-', linewidth=2, label='Smoothed')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('(b) Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(steps, history['alpha'], 'b-', linewidth=2, label='alpha (Similarity)')
    ax.plot(steps, history['beta'], 'r-', linewidth=2, label='beta (Quality)')
    ax.plot(steps, history['gamma'], 'm-', linewidth=2, label='gamma (Contrastive)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Weight')
    ax.set_title('(c) Curriculum Schedule')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(steps, history['contrastive_reward'], 'purple', alpha=0.5, linewidth=1)
    if window > 1 and len(history['contrastive_reward']) > window:
        smoothed = pd.Series(history['contrastive_reward']).rolling(window=window, center=True).mean()
        ax.plot(steps, smoothed, 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Contrastive Reward')
    ax.set_title('(d) Contrastive Reward')
    ax.grid(True, alpha=0.3)
    plt.suptitle(f'JADE Training Dynamics (K={config.grpo_k})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.run_dir}/figures/fig1_training.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{config.run_dir}/figures/fig1_training.pdf", bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ratings = sorted(results['per_rating'].keys())
    gen_orli = [results['per_rating'][r]['gen_orli'] for r in ratings]
    gold_orli = [results['per_rating'][r]['gold_orli'] for r in ratings]
    deltas = [results['per_rating'][r]['delta'] for r in ratings]
    ax = axes[0]
    x = np.arange(len(ratings))
    width = 0.35
    ax.bar(x - width/2, gen_orli, width, label='Generated', color='#3498db')
    ax.bar(x + width/2, gold_orli, width, label='Gold', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}' for r in ratings])
    ax.set_ylabel('ORLI Score')
    ax.set_title('(a) Generated vs Gold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax = axes[1]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
    ax.bar(x, deltas, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}' for r in ratings])
    ax.set_ylabel('Delta (Gen - Gold)')
    ax.set_title('(b) Improvement by Rating')
    for i, d in enumerate(deltas):
        ax.annotate(f'{d:+.3f}', xy=(i, d), ha='center', va='bottom' if d > 0 else 'top', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax = axes[2]
    heatmap_data = pd.DataFrame({
        'Gen ORLI': gen_orli,
        'Gold ORLI': gold_orli,
        'Delta': deltas,
        'Similarity': [results['per_rating'][r]['similarity'] for r in ratings]
    }, index=[f'{r}' for r in ratings])
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('(c) Performance Heatmap')
    plt.suptitle(f'JADE Per-Rating Analysis (K={config.grpo_k})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.run_dir}/figures/fig2_per_rating.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{config.run_dir}/figures/fig2_per_rating.pdf", bbox_inches='tight')
    plt.close()
    logger.info(f"Figures saved to {config.run_dir}/figures/")


def run_experiment(config: Config):
    # ---- Version Banner (prevents stale Docker confusion) ----
    print("\n" + "=" * 70)
    print(f"  JADE v{__version__} (built {__build_date__})")
    print("=" * 70)
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! This will be extremely slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
        print(f"  Quantization: {config.quantization}")
    print(f"  Dataset: {config.data_path}")
    print(f"  Base Model: {config.base_model}")
    print(f"  K={config.grpo_k}, Epochs={config.epochs}, Batch={config.batch_size}, GradAccum={config.gradient_accumulation_steps}")
    print(f"  LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    print("=" * 70 + "\n")
    set_seed(config.seed)
    logger = setup_logging(config)
    logger.info("=" * 70)
    logger.info(f"JADE v{__version__} (built {__build_date__})")
    logger.info("=" * 70)
    logger.info(f"Dataset: {config.data_path}")
    logger.info(f"K = {config.grpo_k}")
    logger.info(f"Epochs = {config.epochs}")
    logger.info(f"Batch size = {config.batch_size} (effective={config.batch_size * config.gradient_accumulation_steps})")
    logger.info(f"Grad accumulation = {config.gradient_accumulation_steps}")
    logger.info(f"Learning rate = {config.learning_rate} (cosine, {config.warmup_ratio*100:.0f}% warmup)")
    logger.info(f"LoRA rank = {config.lora_rank}, alpha = {config.lora_alpha}")
    logger.info(f"Base model = {config.base_model}")
    logger.info(f"Quantization = {config.quantization}")
    logger.info(f"ArmoRM: {'ENABLED' if config.use_armo else 'DISABLED'}")
    logger.info(f"Retrieval scaffold: {'ENABLED' if config.use_retrieval else 'DISABLED (clean null)'}")
    logger.info(f"Baseline mode: {'YES (no GRPO training)' if config.eval_baseline else 'NO'}")
    if config.eval_baseline and config.use_retrieval:
        logger.info("NOTE: baseline WITH retrieval = base model + scaffold. For the rawest "
                    "regression-to-the-mean null, add --no_retrieval.")
    logger.info(f"Max test per tier: {config.max_test_per_tier}")
    logger.info(f"Output: {config.run_dir}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Memory: {get_gpu_memory()}")
    with open(os.path.join(config.run_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2, default=json_serializable)
    start_time = datetime.now()
    try:
        train_df, orli_val_df, test_df = load_and_split_data(config, logger)
        encoder = Encoder(config, logger)
        memory = RatingAwareMemory(encoder, train_df, config, logger)
        judge = ORLIJudge(config)
        if config.load_orli:
            # Reuse a saved judge (e.g. an eval-only re-run reusing the original run's ORLI)
            ckpt = torch.load(config.load_orli, map_location=config.device)
            judge.load_state_dict(ckpt['model_state_dict'])
            judge = judge.to(config.device)
            orli_history = {'loaded_from': config.load_orli, 'best_mae': ckpt.get('best_mae')}
            logger.info(f"Loaded ORLI judge from {config.load_orli} (skipped ORLI training)")
        else:
            # ORLI is early-stopped and MAE-reported on orli_val_df (held out from both
            # the GRPO train set and the test set) — addresses the val==test leakage.
            judge, orli_history = train_orli_judge(judge, train_df, orli_val_df, encoder, config, logger)
        actor = Actor(config, logger)
        if config.load_actor:
            actor.load_adapter(config.load_actor)

        history: Dict = {}
        # skip_training is true for the null baseline AND for eval-only (--load_actor).
        skip_training = config.eval_baseline or bool(config.load_actor)
        if config.eval_baseline:
            # Baseline mode: evaluate unmodified base model, no GRPO.
            # Use this run to establish the regression-to-the-mean null condition.
            logger.info("=" * 70)
            logger.info("BASELINE MODE — skipping GRPO training")
            logger.info("Evaluating unmodified base model to establish null condition")
            logger.info("=" * 70)
        elif config.load_actor:
            logger.info("=" * 70)
            logger.info("EVAL-ONLY MODE — loaded a trained actor, skipping GRPO training")
            logger.info(f"Actor adapter: {config.load_actor}")
            logger.info("=" * 70)
        else:
            buffer = PERBuffer(train_df, config, logger)
            trainer = GRPOTrainer(actor, judge, memory, buffer, config, logger)
            history = trainer.train()

        # Load ArmoRM after actor training to avoid VRAM contention during training.
        # If ArmoRM was requested (default) but can't load or score, ABORT loudly here — do NOT
        # silently fall back to an ORLI-only result after hours of eval. Use --no_armo to skip
        # the independent metric on purpose.
        armo: Optional[ArmoRMEvaluator] = None
        if config.use_armo:
            armo = ArmoRMEvaluator(config, logger)

        evaluator = Evaluator(encoder, judge, memory, config, logger)
        results = evaluator.evaluate(actor, test_df, armo=armo)
        evaluator.print_results(results)

        if not skip_training:
            create_visualizations(history, results, config, logger)

        run_mode = 'baseline' if config.eval_baseline else ('eval_only' if config.load_actor else 'full')
        output = {
            'version': __version__,
            'build_date': __build_date__,
            'config': asdict(config),
            'mode': run_mode,
            'results': {'overall': results['overall'], 'per_rating': results['per_rating']},
            'training_history': history,
            'orli_history': orli_history,
            'runtime_seconds': (datetime.now() - start_time).total_seconds()
        }
        with open(os.path.join(config.run_dir, "results.json"), 'w') as f:
            json.dump(output, f, indent=2, default=json_serializable)
        # Per-example predictions (keyed by sample_id) so a baseline run and a JADE run
        # can be aligned for the per-tier cross-condition delta (see compare_conditions.py).
        with open(os.path.join(config.run_dir, "predictions.json"), 'w') as f:
            json.dump({
                'mode': output['mode'],
                'data_path': config.data_path,
                'seed': config.seed,
                'use_retrieval': config.use_retrieval,
                'predictions': results.get('raw', []),
            }, f, indent=2, default=json_serializable)
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Runtime: {datetime.now() - start_time}")
        logger.info(f"Results saved to: {config.run_dir}")
        return output
    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(
        description='JADE v4: Judge-free Alignment via Data Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )
    # Core
    parser.add_argument('--k', type=int, default=16, help='GRPO group size K')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batches_per_epoch', type=int, default=500, help='Batches per epoch')
    parser.add_argument('--data_path', type=str, default='helpsteer2',
                        help='Dataset key (helpsteer2/ultrafeedback/prometheus) or local CSV path')
    parser.add_argument('--target_per_level', type=int, default=5000,
                        help='Samples per rating level when streaming from HF (0 = keep all)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size (prompts per step). Keep at 1 on a single GPU: '
                             'compute_log_probs holds batch_size*K autograd graphs at once.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--logprob_chunk_size', type=int, default=4,
                        help='How many candidates are back-propagated together in the policy update. '
                             'Peak training VRAM is proportional to this, NOT to K. Lower it (e.g. 2) if you still OOM.')
    parser.add_argument('--log_every', type=int, default=25,
                        help='Console metrics summary + training_history.json snapshot every N steps')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save checkpoints/actor_latest every N steps (0=off). Gives recoverable weights '
                             'mid-run if a long job crashes.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Actor learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='LR warmup ratio')
    parser.add_argument('--no_contrastive', action='store_true', help='Disable contrastive reward component')
    parser.add_argument('--max_buffer', type=int, default=1500, help='Max samples per rating bucket in PER buffer')
    # Curriculum weights (override for SFT-only ablation: set beta/gamma to 0)
    parser.add_argument('--alpha_start', type=float, default=1.0, help='Curriculum alpha start (similarity weight)')
    parser.add_argument('--alpha_end',   type=float, default=0.3, help='Curriculum alpha end')
    parser.add_argument('--beta_start',  type=float, default=0.1, help='Curriculum beta start (ORLI weight)')
    parser.add_argument('--beta_end',    type=float, default=0.6, help='Curriculum beta end')
    parser.add_argument('--gamma_start', type=float, default=0.1, help='Curriculum gamma start (contrastive weight)')
    parser.add_argument('--gamma_end',   type=float, default=0.3, help='Curriculum gamma end')
    # ORLI
    parser.add_argument('--orli_epochs', type=int, default=20, help='ORLI judge training epochs')
    parser.add_argument('--orli_patience', type=int, default=3, help='ORLI early stopping patience')
    parser.add_argument('--orli_feature', choices=['pair', 'response', 'concat'], default='concat',
                        help="ORLI features: 'concat' embeds query & response on SEPARATE channels "
                             "(fixes the query dominating the vector); 'response' = response only; "
                             "'pair' = original combined string.")
    parser.add_argument('--orli_loss', choices=['huber', 'mse'], default='huber', help='ORLI regression loss')
    parser.add_argument('--no_orli_balance', dest='orli_balance', action='store_false',
                        help='disable inverse-frequency class weighting in ORLI (it is ON by default)')
    parser.set_defaults(orli_balance=True)
    parser.add_argument('--protect_high_ratings', action='store_true',
                        help='rating-aware reward: imitate gold MORE on already-high-rated sources (protect 5-star)')
    # Model / quantization
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Base model for actor')
    parser.add_argument('--quantization', type=str, default='auto',
                        choices=['auto', '4bit', '8bit', 'fp16', 'bf16'],
                        help='Quantization mode (auto = detect from VRAM; H200 will pick bf16)')
    parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha (typically 2x rank)')
    parser.add_argument('--lora_targets', type=str, default='auto',
                        help='LoRA target modules: "auto" or comma-separated')
    # Tokenization / generation
    parser.add_argument('--max_output_tokens', type=int, default=192, help='Max generation tokens')
    parser.add_argument('--max_response_chars', type=int, default=600, help='Max response char truncation')
    parser.add_argument('--max_user_chars', type=int, default=512, help='Max query char truncation')
    # Evaluation
    parser.add_argument('--max_test_per_tier', type=int, default=500,
                        help='Max test samples per rating tier (default: 500, was 150)')
    parser.add_argument('--no_armo', action='store_true',
                        help='Disable ArmoRM independent evaluation (saves ~16GB VRAM)')
    parser.add_argument('--eval_baseline', action='store_true',
                        help='Skip GRPO training; evaluate unmodified base model (null/control condition)')
    parser.add_argument('--load_actor', type=str, default='',
                        help='Path to a trained LoRA adapter (e.g. outputs/<run>/checkpoints/actor_best). '
                             'EVAL-ONLY: loads it and skips the ~37 h GRPO retrain — used to add the ArmoRM '
                             'column / null baseline to an existing checkpoint in ~3 h.')
    parser.add_argument('--load_orli', type=str, default='',
                        help='Path to a saved orli_judge.pt to reuse instead of retraining ORLI '
                             '(pair with --load_actor to reproduce the original run scoring exactly).')
    parser.add_argument('--no_retrieval', action='store_true',
                        help='Disable FAISS retrieval scaffold (context + contrastive negatives). '
                             'Combine with --eval_baseline for the rawest regression-to-the-mean null.')
    parser.add_argument('--orli_val_frac', type=float, default=0.1,
                        help='Fraction of TRAIN held out for ORLI early-stopping/MAE (default: 0.1)')
    parser.add_argument('--keep_checkpoints', type=int, default=0,
                        help='Keep N most recent per-epoch actor checkpoints (0 = keep all). '
                             'actor_best/ is always kept regardless of this setting.')
    # Meta
    parser.add_argument('--version', action='store_true', help='Print version and exit')
    args = parser.parse_args()

    if args.version:
        print(f"JADE v{__version__} (built {__build_date__})")
        sys.exit(0)

    config = Config(
        grpo_k=args.k,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        data_path=args.data_path,
        target_per_level=args.target_per_level,
        output_dir=args.output_dir,
        seed=args.seed,
        use_contrastive=not args.no_contrastive,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logprob_chunk_size=args.logprob_chunk_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        warmup_ratio=args.warmup_ratio,
        max_buffer_per_bucket=args.max_buffer,
        max_output_tokens=args.max_output_tokens,
        max_response_chars=args.max_response_chars,
        max_user_chars=args.max_user_chars,
        learning_rate=args.learning_rate,
        orli_epochs=args.orli_epochs,
        orli_patience=args.orli_patience,
        orli_feature=args.orli_feature,
        orli_loss=args.orli_loss,
        orli_balance=args.orli_balance,
        protect_high_ratings=args.protect_high_ratings,
        base_model=args.base_model,
        quantization=args.quantization,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        gamma_start=args.gamma_start,
        gamma_end=args.gamma_end,
        use_armo=not args.no_armo,
        max_test_per_tier=args.max_test_per_tier,
        eval_baseline=args.eval_baseline,
        load_actor=args.load_actor,
        load_orli=args.load_orli,
        use_retrieval=not args.no_retrieval,
        orli_val_frac=args.orli_val_frac,
        keep_checkpoints=args.keep_checkpoints,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
