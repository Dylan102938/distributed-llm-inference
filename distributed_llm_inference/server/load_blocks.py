import json

import safetensors
import torch
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig
from transformers.utils import cached_file

from distributed_llm_inference.models.llama import LlamaBlock

INDEX_FILE_PATTERNS = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def get_sharded_block_state_from_file(file: str, block_prefix: str):
    sharded_state_dict = {}

    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(block_prefix):
                sharded_state_dict[key[len(block_prefix):]] = f.get_tensor(key)
        
    return sharded_state_dict


def get_block_state_dict(repo: str, block_idx: int, cache_dir: str | None = None, token: str | bool = False):
    for pattern in INDEX_FILE_PATTERNS:
        index_file = cached_file(repo, pattern, cache_dir=cache_dir, token=token)
        if index_file is not None:
            break
    
    if index_file is None:
        raise FileNotFoundError("Could not find index file for the provided repository")
    
    index = json.load(open(index_file))
    if 'weight_map' not in index:
        raise ValueError("Index file does not contain a weight map")

    prefix = f"model.layers.{block_idx}."
    weight_files = set()
    for key, value in index["weight_map"].items():
        if key.startswith(prefix):
            weight_files.add(value)
    
    state_dict = {}
    for file in weight_files:
        local_file = cached_file(repo, file, cache_dir=cache_dir, token=token)
        sharded_state_dict = get_sharded_block_state_from_file(local_file, prefix)
        state_dict.update(sharded_state_dict)
    
    return state_dict
    

def load_block(model_name: str, block_idx: int, cache_dir: str | None = None, token: str | bool = False) -> LlamaBlock:
    print(f"Downloading model config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, token=token)
    
    print(f"Downloading state dict for block {block_idx}...")
    state_dict = get_block_state_dict(model_name, block_idx, cache_dir=cache_dir, token=token)
    
    block = LlamaBlock(config)

    for name, _ in block.named_parameters():
        assert name in state_dict, f"Parameter {name} not found in state_dict"
        parameters = state_dict[name]
        if not str(parameters.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            parameters = parameters.to(torch.float16)
        
        set_module_tensor_to_device(block, name, "cpu", value=parameters)
    
    return block
