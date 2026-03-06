"""Retrieval evaluation script.

Usage:
    python eval.py
    python eval.py --api http://localhost:8000 --dataset eval_dataset.json --k 5

Metrics reported:
    hit@k       — fraction of questions where expected_document appears in top-k citations
    keyword@k   — fraction of questions where >=1 expected keyword appears in top-k chunk text
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse


def query(api_base: str, question: str) -> list[dict]:
    """Call the non-streaming query endpoint and return citations."""
    url = f"{api_base}/query/"
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body.get("citations", [])


def evaluate(api_base: str, dataset_path: str, k: int) -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)

    pairs = dataset.get("pairs", [])
    if not pairs:
        print("No evaluation pairs found in dataset.")
        sys.exit(1)

    hit_count = 0
    kw_count = 0

    for pair in pairs:
        question = pair["question"]
        expected_doc = pair.get("expected_document", "").lower()
        expected_kws = [kw.lower() for kw in pair.get("expected_keywords", [])]

        print(f"\nQ: {question}")
        try:
            citations = query(api_base, question)[:k]
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        docs_returned = [c["document"].lower() for c in citations]
        text_blob = " ".join(c["chunk_text"].lower() for c in citations)

        hit = expected_doc in docs_returned if expected_doc else True
        kw_hit = any(kw in text_blob for kw in expected_kws) if expected_kws else True

        if hit:
            hit_count += 1
        if kw_hit:
            kw_count += 1

        print(f"  hit@{k}: {'YES' if hit else 'NO'} | keyword@{k}: {'YES' if kw_hit else 'NO'}")
        print(f"  returned: {docs_returned}")

    n = len(pairs)
    print(f"\n--- Results ({n} questions, k={k}) ---")
    print(f"hit@{k}:     {hit_count}/{n} = {hit_count/n:.1%}")
    print(f"keyword@{k}: {kw_count}/{n} = {kw_count/n:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.api, args.dataset, args.k)
