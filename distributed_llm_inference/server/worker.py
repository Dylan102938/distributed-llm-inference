from typing import TypedDict


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
        # self.blocks: Dict[str, TransformerBackend] = {}
        # for block in self.block_ids:
        #     self.blocks[block["block_id"]] = load_block()
        pass

    def run(self):
        pass
