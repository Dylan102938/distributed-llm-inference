from typing import Dict, Tuple, Union

import torch
from hivemind import ModuleBackend
from hivemind.moe.server.task_pool import TaskPool
from hivemind.utils.nested import nested_map
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor
from torch import nn


class InferenceBackend(ModuleBackend):
    def __init__(
        self,
        name: str,
        module: nn.Module,
        *,
        args_schema: Tuple[BatchTensorDescriptor, ...] = None,
        kwargs_schema: Dict[str, BatchTensorDescriptor] = None,
        outputs_schema: Union[BatchTensorDescriptor, Tuple[BatchTensorDescriptor, ...]] = None,
        **kwargs,
    ):
        self.name, self.module, self.optimizer, self.scheduler = name, module, None, None

        self.args_schema = args_schema = tuple(args_schema or ())
        self.kwargs_schema = kwargs_schema = dict(kwargs_schema or {})
        assert args_schema or kwargs_schema, (
            "Module must take at least one positional or keyword input."
            " Did you forget to provide args_schema/kwargs_schema?"
        )

        if outputs_schema is None:
            dummy_args = tuple(sample.make_zeros(DUMMY_BATCH_SIZE) for sample in args_schema)
            dummy_kwargs = {key: sample.make_zeros(DUMMY_BATCH_SIZE) for key, sample in kwargs_schema.items()}
            dummy_outputs = self.module(*dummy_args, **dummy_kwargs)
            outputs_schema = nested_map(BatchTensorDescriptor.from_tensor, dummy_outputs)

        self.forward_schema = (self.args_schema, self.kwargs_schema)  # inputs for forward
        self.outputs_schema = outputs_schema  # outputs from forward

        self.backward_schema = (self.forward_schema, self.outputs_schema)  # inputs to backward
        self.grad_inputs_schema = self.forward_schema  # outputs from backward
        self.inference_pool = TaskPool(self.forward, name=f"{self.name}_inference", **kwargs)

    def backward(self, *inputs: torch.Tensor):
        raise NotImplementedError("InferenceBackend does not support backward pass")
    
    def on_backward(self, batch_size: int) -> None:
        raise NotImplementedError("InferenceBackend does not support backward pass")

    def get_pools(self):
       return self.inference_pool