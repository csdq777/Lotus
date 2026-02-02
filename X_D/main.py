import argparse
import os
import json
import asyncio
import random
import numpy as np
import re

from dictionary import *
from corpus import *
from model import *
from tokenizer import *
from prompt import *

# ==== vLLM 并发依赖 ====
try:
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    SamplingParams = None
    AsyncLLMEngine = None
    AsyncEngineArgs = None


# ========== 文本清洗函数（当前不调用，仅保留）==========
def clean_pred(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if "\n" in s:
        s = s.split("\n", 1)[0]
    if "<|endoftext|>" in s:
        s = s.split("<|endoftext|>", 1)[0].strip()
    if "<|im_end|>" in s:
        s = s.split("<|im_end|>", 1)[0].strip()
    return s.strip()


# ========== 提取 ###…### 中内容 ==========
def extract_enclosed_text(text, boundary="###"):
    text = "" if text is None else str(text)
    splits = text.split(boundary)
    if len(splits) < 3:
        return ""  # 不回退 clean_pred，保持你当前策略
    return splits[-2].strip()


# ========== vLLM 并发推理函数 ==========
async def _gen_one(engine, prompt, sampling_params, req_id, semaphore):
    async with semaphore:
        last = None
        async for out in engine.generate(prompt, sampling_params, request_id=req_id):
            last = out
        if last and getattr(last, "outputs", None):
            return req_id, last.outputs[0].text
        return req_id, ""


async def run_vllm_concurrent(prompts, sampling_params, model_path, n_gpu=1, concurrency=8, dtype="bfloat16", trust_remote_code=True):
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=n_gpu,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for i, p in enumerate(prompts):
        req_id = f"req-{i}"
        tasks.append(_gen_one(engine, p, sampling_params, req_id, semaphore))

    results = [None] * len(prompts)
    for coro in asyncio.as_completed(tasks):
        req_id, text = await coro
        idx = int(req_id.split("-")[-1])
        results[idx] = text
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # linguistic resources
    parser.add_argument('--src_lang', type=str, default='zh')
    parser.add_argument('--tgt_lang', type=str, default='mn')
    parser.add_argument('--dict_path', type=str, default='../data/mn_zh_dict.jsonl')
    parser.add_argument('--corpus_path', type=str, default='../data/zh_mn_train.json')
    parser.add_argument('--test_data_path', type=str, default='../data/zh_mn_test.json')

    # model
    parser.add_argument('--model_name', type=str, default='Llama-3.1-8B-Instruct')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Lotus/models/Llama-3.1-8B-Instruct')
    parser.add_argument('--chat_mode', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--no_vllm', action='store_true')

    # generation config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=1024)  
    parser.add_argument('--concurrency', type=int, default=200)      

    # config for prompt
    parser.add_argument('--prompt_type', type=str, default='za2zh')
    parser.add_argument('--num_parallel_sent', type=int, default=5)

    # output path
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    # load dictionary
    dictionary = WordDictionary(args.src_lang, args.tgt_lang, args.dict_path)

    # load corpus
    parallel_corpus = ParallelCorpus(args.src_lang, args.tgt_lang, args.corpus_path)

    # load test data
    test_data = json.load(open(args.test_data_path, "r"))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # sampling params
    if SamplingParams is None and not args.no_vllm:
        raise RuntimeError("未检测到 vLLM，请安装后去掉 --no_vllm；或使用 --no_vllm 走 HF 路径。")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens) if not args.no_vllm else None

    # construct prompt func
    prompt_type_to_prompt_func = {
        'za2zh': construct_prompt_za2zh,
        'zh2za': construct_prompt_zh2za,
    }
    if args.prompt_type not in prompt_type_to_prompt_func:
        raise NotImplementedError("Unsupported prompt type!")
    prompt_func = prompt_type_to_prompt_func[args.prompt_type]

    # output path
    if args.output_path is None:
        args.output_path = f"../output/{args.model_name}_{args.prompt_type}_parallel{args.num_parallel_sent}.jsonl"
    while os.path.exists(args.output_path):
        args.output_path = args.output_path + ".new.jsonl"
    print("output_path:", args.output_path)
    fout = open(args.output_path, "w", encoding="utf-8")

    from tqdm import tqdm

    if args.no_vllm:
        # ===== 普通 HuggingFace 路径 =====
        llm, tokenizer = load_model(args.model_name, args.model_path, args.n_gpu, use_vllm=False)

        for item in tqdm(test_data):
            src_sentence = item[args.src_lang]
            prompt = prompt_func(src_sentence, dictionary, parallel_corpus, args)
            raw_pred = get_pred_no_vllm(llm, tokenizer, prompt, args)   # 模型完整输出
            pred = extract_enclosed_text(raw_pred)                       # 仅提取 ###…### 中内容

            print("input:", src_sentence)
            print("gold:", item[args.tgt_lang])
            print("pred:", pred)

            fout.write(json.dumps({
                "query": src_sentence,
                "pred": pred,
                "gold": item[args.tgt_lang],
                "prompt": prompt,
                "source": item.get('source', None),
                "coversation": str(raw_pred)   # 新增：原始完整输出
            }, ensure_ascii=False) + "\n")

    else:
        # ===== vLLM 并发推理路径 =====
        prompts, src_sents, tgts, sources = [], [], [], []
        for item in test_data:
            src_sentence = item[args.src_lang]
            p = prompt_func(src_sentence, dictionary, parallel_corpus, args)
            prompts.append(p)
            src_sents.append(src_sentence)
            tgts.append(item[args.tgt_lang])
            sources.append(item.get('source', None))

        # 批量并发推理
        texts = asyncio.run(
            run_vllm_concurrent(
                prompts=prompts,
                sampling_params=sampling_params,
                model_path=args.model_path,
                n_gpu=args.n_gpu,
                concurrency=args.concurrency,
            )
        )

        for src_sentence, gold, prompt, pred_text, source in zip(src_sents, tgts, prompts, texts, sources):
            pred = extract_enclosed_text(pred_text)  # 仅提取 ###…###
            print("input:", src_sentence)
            print("gold:", gold)
            print("pred:", pred)

            fout.write(json.dumps({
                "query": src_sentence,
                "pred": pred,
                "gold": gold,
                "prompt": prompt,
                "source": source,
                "conversation": str(pred_text)  # 新增：原始完整输出
            }, ensure_ascii=False) + "\n")

    fout.close()
