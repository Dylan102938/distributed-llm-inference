from typing import Dict, List, TypedDict

from hivemind import ModuleBackend
from transformers.models.llama.modeling_llama import LlamaAttention

from distributed_llm_inference.server.backend import TransformerBackend
from distributed_llm_inference.server.task_pool import TaskPool


class Block(TypedDict):
    block_index: int
    block_id: str


class InferenceWorker:
    def __init__(
        self,
        model: str,
        block_index_start: int, 
        block_index_end: int,
        block_
    ):
        self.blocks: Dict[str, TransformerBackend] = {}
        for block in self.block_ids:
            self.blocks[block["block_id"]] = download_pretrained_block()

    def run(self):

