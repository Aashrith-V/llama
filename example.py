# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import nvidia_smi

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

mem = 0

def B2G(num):
    return round(num/(1024**3),2)

def print_memory(name, handle, pre_used):
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used = info.used
    print(f'{name}: {B2G(used)}')
    print(f'This step use: {B2G(used-pre_used)}')
    print('------------')
    return used

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    global mem
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cuda:0")

    print(ckpt_path, local_rank, world_size)
    mem = print_memory("checkpoint", handle, mem)

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    mem = print_memory("modelargs", handle, mem)

    tokenizer = Tokenizer(model_path=tokenizer_path)

    mem = print_memory("tokenizer", handle, mem)

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)

    mem = print_memory("transformer", handle, mem)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    
    mem = print_memory("load state dict", handle, mem)

    generator = LLaMA(model, tokenizer)

    mem = print_memory("generator", handle, mem)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def load_the_model(    
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")   
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    return generator

def infer_the_model(
    generator, 
    prompts,
    max_gen_len: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    results = generator.generate(
        prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )
    return results

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    max_gen_len: int = 256
):
    global mem
    mem = print_memory('Start', handle, 0)

    generator = load_the_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    prompt = input("enter prompt > ")
    prompts = [prompt]
    results = infer_the_model(generator, prompts, max_gen_len, temperature, top_p)

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
