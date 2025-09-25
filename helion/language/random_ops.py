from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.device_function import SymbolArgument
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["rand"]


@_decorators.api(tiles_as_sizes=True)
def rand(
    shape: list[object],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    The main propose of ``hl.rand`` is to explicitly pass a seed arg for deterministic
    randomness in helion kernels, whereas ``torch.rand_like`` doesn't take seed arg
    (though it can seeded globally)`. ``hl.rand`` lower to ``tl.rand(seed, offset)`` with ``offset``
    built from a linear range over the allocation and reshaped to the given shape.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For host allocations, use ``torch.rand()``.

    Args:
        shape: A list of sizes
        seed: int seed for the random number generator
        dtype: currently only float32 supported

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with random values

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(x)
                (m,) = x.shape
                for (tile_m,) in hl.tile([m]):
                    output[tile_m] = hl.rand([tile_m], seed=seed)
                return output

    """
    raise NotInsideKernel


@_decorators.register_fake(rand)
def _rand_fake(
    shape: list[int | torch.SymInt],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.codegen(rand)
def _rand_codegen(state: CodegenState) -> ast.AST:
    """
    Generate tl.rand() code with global indices for deterministic RNG per element.
    """
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)

    tensor_shape = fake_value.size()
    ndim = len(tensor_shape)
    if ndim == 0:
        raise ValueError("hl.rand() requires at least one dimension")

    seed_ast = state.ast_arg(1)
    env = CompileEnvironment.current()

    symbol_args = []
    rdim_args = {}
    for arg in state.device_function.arguments:
        if isinstance(arg, SymbolArgument) and arg.name != "seed":
            symbol_args.append(arg.name)
        elif arg.name.startswith("_RDIM_SIZE_"):
            rdim_args[arg.name] = arg

    index_vars = []
    size_names = []
    used_rdims = set()
    symbol_idx = 0

    for i in range(ndim):
        block_id = env.get_block_id(tensor_shape[i])
        if block_id is not None:
            rdim_name = f"_RDIM_SIZE_{block_id}"
            if rdim_name in rdim_args:
                index_vars.append(f"tl.arange(0, {rdim_name})")
                size_names.append(rdim_name)
                used_rdims.add(rdim_name)
                continue

        if block_id is not None:
            index_vars.append(state.codegen.index_var(block_id))
            if symbol_idx < len(symbol_args):
                size_names.append(symbol_args[symbol_idx])
                symbol_idx += 1
            else:
                size_names.append(str(tensor_shape[i]))
            continue

        available_rdims = [name for name in rdim_args if name not in used_rdims]
        if available_rdims:
            rdim_name = available_rdims[0]
            index_vars.append(f"tl.arange(0, {rdim_name})")
            size_names.append(rdim_name)
            used_rdims.add(rdim_name)
        else:
            raise RuntimeError(
                "hl.rand() requires tiled dimensions. "
                "Use hl.rand() inside hl.tile() loops with tile variables."
            )

    if ndim == 1:
        offset_expr = expr_from_string(index_vars[0])
    else:
        broadcast_slices = []
        for i in range(ndim):
            slice_parts = ["None"] * ndim
            slice_parts[i] = ":"
            broadcast_slices.append(f"[{', '.join(slice_parts)}]")

        offset_parts = []
        for i in range(ndim):
            broadcasted_index = f"{index_vars[i]}{broadcast_slices[i]}"

            if i < ndim - 1:
                stride_expr = " * ".join(size_names[i + 1 :])
                offset_parts.append(f"{broadcasted_index} * {stride_expr}")
            else:
                offset_parts.append(broadcasted_index)

        offset_expr = expr_from_string(" + ".join(offset_parts))

    return expr_from_string(
        "tl.rand({seed}, {offset})", seed=seed_ast, offset=offset_expr
    )


@_decorators.get_masked_value(rand)
def _(
    node: torch.fx.Node,
) -> float:
    return 0


@_decorators.ref(rand)
def _(
    shape: list[int | RefTile],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    processed_shape: list[int] = []
    for s in shape:
        if isinstance(s, RefTile):
            processed_shape.append(s.end - s.begin)
        else:
            processed_shape.append(int(s))
    env = CompileEnvironment.current()
    gen = torch.Generator(device=env.device if device is None else device)
    gen.manual_seed(seed)
    return torch.rand(
        processed_shape,
        dtype=dtype,
        generator=gen,
        device=env.device if device is None else device,
    )
