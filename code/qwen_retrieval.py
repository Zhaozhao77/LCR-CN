import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        bsz = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


# ---------- 数据读取 ----------
def data_Path(datafile):
    datalist = []
    with open(datafile,'r',encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            # 构造候选集合
            highlaw = ''.join(str(item) for item in rec['high_level_laws'])
            clauses = [d['title'] + ' ' + d['clause'] for d in rec['distractors']]
            clauses.append(highlaw)
            datalist.append({
                "id" : rec['id'],
                "title" : rec['title'],
                "level" : rec['level'],
                "category" : rec['category'],
                "content" : rec['content'],
                "golden_content" : rec['golden_content'],
                "high_level_laws" : highlaw,
                "conflict_type" : rec['conflict_type'],
                "conflict_description" : rec['conflict_description'],
                "clause_list": clauses
            })
    return datalist

@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 8192,
    batch_size: int = 16
) -> np.ndarray:
    """返回 (N, D) 的 L2-归一化向量"""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)  
        vecs.append(emb.cpu())
    return torch.cat(vecs, dim=0).numpy()  


def compute_metrics_for_one(
    scores: np.ndarray,
    candidates: List[str],
    golden: str,
    Ks: Tuple[int, ...] = (1, 3, 5),
):
    ranked_idx = np.argsort(-scores)

    rank_of_golden = None
    for r, idx in enumerate(ranked_idx):
        if candidates[idx] == golden:
            rank_of_golden = r
            break
    rr = 1.0 / (rank_of_golden + 1) if rank_of_golden is not None else 0.0

    hits = {}
    precs = {}
    recs = {}
    ndcgs = {}
    for K in Ks:
        hit = 1 if (rank_of_golden is not None and rank_of_golden < K) else 0
        hits[K] = hit
        precs[K] = hit / K                 
        recs[K] = hit                     
        ndcgs[K] = (1.0 / np.log2(rank_of_golden + 2)) if hit else 0.0

    return rr, hits, precs, recs, ndcgs

def evaluate_qwen3_embedding(
    data: List[Dict[str, Any]],
    model_id: str = "Qwen3-Embedding-0.6B",
    device_str: str = "cuda:0",
    Ks: Tuple[int, ...] = (1, 3, 5),
    batch_size: int = 16,
    max_length: int = 8192,
    task_desc: str = "请检索出该法条的上位法",
):
    device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,               
        device_map="auto",
        max_memory={0: "22GiB", 1: "22GiB", "cpu": "30GiB"}  
    )
    model.eval()

    N = len(data)
    sum_rr = 0.0
    hits = {K: 0 for K in Ks}
    precs = {K: 0.0 for K in Ks}
    recs = {K: 0.0 for K in Ks}
    ndcgs = {K: 0.0 for K in Ks}

    for rec in tqdm(data, desc="Evaluating"):
        query = rec["content"]
        cands = rec["clause_list"]
        golden = rec["high_level_laws"]

        q_text = get_detailed_instruct(task_desc, query)

        q_emb = embed_texts([q_text], tokenizer, model, device, max_length=max_length, batch_size=1)   # (1, D)
        d_emb = embed_texts(cands,   tokenizer, model, device, max_length=max_length, batch_size=batch_size)  # (M, D)

        sims = (q_emb @ d_emb.T).reshape(-1)  # (M,)

        rr, h, p, r, n = compute_metrics_for_one(sims, cands, golden, Ks)
        sum_rr += rr
        for K in Ks:
            hits[K]  += h[K]
            precs[K] += p[K]
            recs[K]  += r[K]
            ndcgs[K] += n[K]

    print(f"Dataset size: {N}")
    print(f"MRR: {sum_rr / N:.4f}\n")
    for K in Ks:
        print(f"--- Metrics @ K={K} ---")
        print(f"Accuracy@{K}: {hits[K] / N:.4f}")
        print(f"Precision@{K}: {precs[K] / N:.4f}")
        print(f"Recall@{K}:    {recs[K] / N:.4f}")
        print(f"nDCG@{K}:      {ndcgs[K] / N:.4f}\n")

    return {
        "MRR": sum_rr / N,
        **{f"Accuracy@{K}": hits[K]  / N for K in Ks},
        **{f"Precision@{K}": precs[K] / N for K in Ks},
        **{f"Recall@{K}":    recs[K]  / N for K in Ks},
        **{f"nDCG@{K}":      ndcgs[K] / N for K in Ks},
    }


if __name__ == "__main__":
    datafile = "dataset/test.jsonl"
    datalist = data_Path(datafile)
    # 选择模型（可换成 0.6B/4B/8B ）
    model_id = "Qwen3-Embedding-0.6B"
    metrics = evaluate_qwen3_embedding(
        datalist,
        model_id=model_id,
        device_str="cuda:0",
        Ks=(1, 3, 5),
        batch_size=16,
        max_length=8192,
        task_desc="请检索出该法条的上位法",
    )
