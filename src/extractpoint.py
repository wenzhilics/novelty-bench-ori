# 对同一个问题的多个answer，提取其中的关键点（point）

import asyncio
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from aiofiles import open as aio_open
from datasets import load_dataset
from tqdm.asyncio import tqdm
import functools


@functools.cache
def load_extractor_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded!")
    return tokenizer, model


def extract_points_sync(question: str, answer: str, model_name: str) -> list[str]:
    tokenizer, model = load_extractor_model(model_name)

    messages = [
        {
            "role": "system",
            "content": (
                "Given a question and an answer, extract 2-3 key points from the answer. "
                "Each point should be:\n"
                "- A single, atomic claim or fact\n"
                "- Self-contained and understandable without the full answer\n"
                "- Distinct from the other points\n"
                "- Concise, no more than 10 words\n\n"
                "Return a JSON list of strings only, e.g. [\"point1\", \"point2\"]. "
                "No other text."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nAnswer: {answer}",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.shape[1]:]
    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        # 尝试直接解析
        points = json.loads(raw)
        assert isinstance(points, list)
        return points
    except Exception:
        # 如果有多余文字，尝试提取 [...] 部分
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            points = json.loads(raw[start:end])
            assert isinstance(points, list)
            return points
        except Exception as e:
            print(f"Failed to parse points: {raw}, error: {e}")
            return []


async def extract_points(question: str, answer: str, model_name: str) -> list[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, extract_points_sync, question, answer, model_name
    )


async def extract_points_for_instance(instance: dict, model_name: str) -> dict:
    question = instance["prompt"]
    generations = instance["generations"]

    # 串行处理，避免并发调用同一个模型
    all_points = []
    for answer in generations:
        points = await extract_points(question, answer, model_name)
        all_points.append(points)

    return {
        **instance,
        "points": all_points,
    }


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    instances = load_dataset("json", data_files=args.input_file, split="train")

    async with aio_open(args.output_file, "w", buffering=1) as f:
        for instance in tqdm(instances, total=len(instances)):
            result = await extract_points_for_instance(instance, args.model)
            await f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    asyncio.run(main())