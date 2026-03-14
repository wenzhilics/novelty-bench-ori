# 对每个问题提取的关键点语义去重，同时统计每个answer贡献了多少个新的关键点

import asyncio
import json
import functools
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@functools.cache
def load_deberta_tokenizer_and_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def maybe_test_equality(s0: str, s1: str) -> bool | None:
    unigram_0 = s0.strip().lower().split()
    unigram_1 = s1.strip().lower().split()
    max_len = max(len(unigram_0), len(unigram_1))
    if max_len <= 5:
        common_unigrams = set(unigram_0) & set(unigram_1)
        return len(common_unigrams) * 2 >= max_len
    return None


@torch.inference_mode()
def classifier_score_sync(s1: str, s2: str) -> float:
    tokenizer, model = load_deberta_tokenizer_and_model()
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=64,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
    prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


async def equivalence_check(s0: str, s1: str) -> bool:
    equality = maybe_test_equality(s0, s1)
    if equality is not None:
        return equality
    loop = asyncio.get_event_loop()
    score = await loop.run_in_executor(None, classifier_score_sync, s0, s1)
    return score > 0.102


async def deduplicate_points(all_points: list[list[str]]) -> list[int]:
    """
    Deduplicate points across all answers for a given question.
    Returns a list of the count of new points contributed by each answer.

    Args:
        all_points: A list of lists, where each inner list contains the points for an answer.
                    Example: [[p1, p2, p3], [p4, p2, p5], ...]

    Returns:
        A list of integers representing the number of unique points contributed by each answer.
        Example: [3, 2, 1, ...]
    """
    seen_points = []  # Global seen points. Leader Algorithm
    new_counts = []

    for points in all_points:
        flat_points = []
        for p in points:
            if isinstance(p, list):
                flat_points.extend([str(x) for x in p])
            else:
                flat_points.append(str(p))

        new_count = 0
        for point in flat_points:
            is_new = True
            for seen in seen_points:
                if await equivalence_check(point, seen):
                    is_new = False
                    break
            if is_new:
                seen_points.append(point)
                new_count += 1
        new_counts.append(new_count)

    return new_counts


async def process_instance(instance: dict) -> dict:
    all_points = instance["points"]  # list of lists
    new_counts = await deduplicate_points(all_points)
    return {
        **instance,
        "new_point_counts": new_counts,  # Number of new points contributed by each answer
        "total_unique_points": sum(new_counts),  # Total number of unique points in the question
    }


async def main():
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="points.jsonl from point extractor")
    parser.add_argument("--output-file", required=True, help="output jsonl with new_point_counts")
    args = parser.parse_args()

    instances = []
    with open(args.input_file) as f:
        for line in f:
            instances.append(json.loads(line))
    results = []

    async with aio_open(args.output_file, "w", buffering=1) as f:
        for instance in tqdm(instances, total=len(instances)):
            result = await process_instance(instance)
            results.append(result)
            await f.write(json.dumps(result) + "\n")

    # Count the number of new points contributed by each answer position
    num_generations = len(results[0]["new_point_counts"])
    counts_per_position = np.zeros(num_generations)
    for result in results:
        for pos, count in enumerate(result["new_point_counts"]):
            counts_per_position[pos] += count

if __name__ == "__main__":
    asyncio.run(main())