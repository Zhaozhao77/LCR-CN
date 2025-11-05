import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba
from transformers import (
    DPRQuestionEncoder, DPRContextEncoder,
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
)
import math
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

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

def evaluate_retrieval(model, datalist, Ks=(1,3,5), batch_size=32):
    """
    对 datalist 做检索评估：
      - Accuracy@K, Precision@K, Recall@K, nDCG@K for each K in Ks
      - MRR（与 K 无关）
    """
    N = len(datalist)
    # 累计指标
    hits = {K: 0 for K in Ks}
    precisions = {K: 0.0 for K in Ks}
    recalls    = {K: 0.0 for K in Ks}
    ndcgs      = {K: 0.0 for K in Ks}
    sum_rr = 0.0

    for rec in tqdm(datalist, desc="Evaluating Retrieval"):
        query = rec['content']
        candidates = rec['clause_list']
        golden = rec['high_level_laws']

        emb_q = model.encode([query],
                             device=model.device,
                             normalize_embeddings=True)
        emb_c = model.encode(candidates,
                             batch_size=batch_size,
                             device=model.device,
                             normalize_embeddings=True)

        sims = (emb_q @ emb_c.T)[0]                
        ranked_idx = np.argsort(-sims)
        
        ranks = np.where([candidates[i]==golden for i in ranked_idx])[0]
        rank_of_golden = ranks[0] if len(ranks)>0 else None

        if rank_of_golden is not None:
            sum_rr += 1.0 / (rank_of_golden + 1)
    
        for K in Ks:
            topk = ranked_idx[:K]
            hit = 1 if (rank_of_golden is not None and rank_of_golden < K) else 0
            hits[K] += hit
            precisions[K] += hit / K
            recalls[K] += hit
            if rank_of_golden is not None and rank_of_golden < K:
                ndcgs[K] += 1.0 / np.log2(rank_of_golden + 2)

    mrr = sum_rr / N
    print(f"Dataset size: {N}")
    print(f"MRR: {mrr:.4f}\n")

    for K in Ks:
        acc = hits[K] / N
        prec = precisions[K] / N
        rec = recalls[K] / N  
        ndcg = ndcgs[K] / N
        print(f"--- Metrics @ K={K} ---")
        print(f"Accuracy@{K}: {acc:.4f}")
        print(f"Precision@{K}: {prec:.4f}")
        print(f"Recall@{K}:    {rec:.4f}")
        print(f"nDCG@{K}:      {ndcg:.4f}\n")

    return {
        "MRR": mrr,
        **{f"Accuracy@{K}": hits[K]/N for K in Ks},
        **{f"Precision@{K}": precisions[K]/N for K in Ks},
        **{f"Recall@{K}": recalls[K]/N for K in Ks},
        **{f"nDCG@{K}": ndcgs[K]/N for K in Ks},
    }

def evaluate_bm25(datalist, Ks=(1,3,5)):
    N = len(datalist)
    acc = {K: 0 for K in Ks}
    prec = {K: 0.0 for K in Ks}
    rec  = {K: 0.0 for K in Ks}
    ndcg = {K: 0.0 for K in Ks}
    mrr_list = []

    for item in tqdm(datalist, desc="BM25 evaluate"):
        q = item['content']
        cands = item['clause_list']
        golden = item['high_level_laws']

        tokenized_cands = [jieba.lcut(c) for c in cands]
        tokenized_q     = jieba.lcut(q)

        bm25 = BM25Okapi(tokenized_cands)
        scores = bm25.get_scores(tokenized_q)  
        ranked_idx = np.argsort(scores)[::-1]

        ranks = np.where(np.array(cands)[ranked_idx] == golden)[0]
        if len(ranks):
            rank0 = ranks[0]
            mrr_list.append(1.0 / (rank0 + 1))
        else:
            mrr_list.append(0.0)

        for K in Ks:
            topk = ranked_idx[:K]
            hit_indices = [i for i in topk if cands[i] == golden]
            if hit_indices:
                acc[K]  += 1
                prec[K] += len(hit_indices) / K
                rec[K]  += 1
                rank0 = topk.tolist().index(hit_indices[0])
                ndcg[K] += 1.0 / np.log2(rank0 + 2)

    print(f"Dataset size: {N}")
    print(f"MRR: {np.mean(mrr_list):.4f}\n")

    for K in Ks:
        print(f"--- Metrics @ K={K} ---")
        print(f"Accuracy@{K}: {acc[K]/N:.4f}")
        print(f"Precision@{K}: {prec[K]/N:.4f}")
        print(f"Recall@{K}:    {rec[K]/N:.4f}")
        print(f"nDCG@{K}:      {ndcg[K]/N:.4f}\n")
        
if __name__ == "__main__":
    datafile = "dataset/test.jsonl"
    datalist = data_Path(datafile)
    evaluate_bm25(datalist)