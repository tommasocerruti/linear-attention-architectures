"""Monkey-patches for Megatron training_log and setup_model_and_optimizer.

All state lives in the module-level ``_STATE`` dict so the wrappers can share
it without leaking into Megatron.
"""

from __future__ import annotations

import math
import os
import time
import json
from pathlib import Path
from typing import Any

from . import phase_timer
from .mfu import compute_mfu
from .writer import JsonLogger

_STATE: dict[str, Any] = {
    "writer": None,
    "model": None,
    "n_params_total": None,
    "n_params_active": None,
    "n_layers": None,
    "hidden": None,
    "seq_len": None,
    "last_wall": None,
    "last_iter": None,
    "loss_history": [],
    "loss_spike_k": 3.0,
    "log_per_layer_grads": False,
    "log_act_stats": False,
    "log_loss_spikes": False,
    "log_top1_acc": False,
    "act_accum": {},
    "top1_correct": 0.0,
    "top1_total": 0.0,
    "cler_gamma_log_path": None,
    "cler_gamma_log_announced": False,
    "cler_residual_log_path": None,
    "cler_residual_log_announced": False,
    "cler_value_log_path": None,
    "cler_value_log_announced": False,
}


def configure(writer: JsonLogger, cfg: dict[str, Any]) -> None:
    _STATE["writer"] = writer
    _STATE["log_per_layer_grads"] = cfg.get("log_per_layer_grads", False)
    _STATE["log_act_stats"] = cfg.get("log_act_stats", False)
    _STATE["log_loss_spikes"] = cfg.get("log_loss_spikes", False)
    _STATE["log_top1_acc"] = cfg.get("log_top1_acc", False)
    _STATE["loss_spike_k"] = cfg.get("loss_spike_k", 3.0)


def _count_params(model: Any) -> tuple[int, int]:
    """Return (total, active) parameter counts for the whole model.

    Megatron passes a list of model chunks (one per virtual pipeline stage), but
    each rank only holds its local TP/PP/EP shard. To recover the true total we
    sum locally and then all-reduce across the whole world, dividing out the
    replicating factors (DP replicates the model; CP only splits the sequence,
    so params are duplicated across CP ranks too).
    """
    chunks = model if isinstance(model, list) else [model]
    local = sum(p.numel() for chunk in chunks for p in chunk.parameters())

    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return local, local

    if not (dist.is_available() and dist.is_initialized()):
        return local, local

    try:
        from megatron.core import parallel_state as ps
        dp = ps.get_data_parallel_world_size(with_context_parallel=False)
        cp = ps.get_context_parallel_world_size()
    except Exception:
        dp, cp = 1, 1

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor([local], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total = int(t.item()) // max(dp * cp, 1)
    return total, total


def _register_act_hooks(model: Any) -> None:
    """Attach forward hooks that record output norm and max per transformer layer."""
    import torch

    accum: dict[str, dict[str, float]] = _STATE["act_accum"]

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            with torch.no_grad():
                entry = accum.setdefault(name, {"norm": 0.0, "max": 0.0, "n": 0})
                entry["norm"] += float(t.detach().float().norm().item())
                entry["max"] = max(entry["max"], float(t.detach().float().abs().max().item()))
                entry["n"] += 1
        return hook

    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if name.endswith(".self_attention") or name.endswith(".mlp"):
                module.register_forward_hook(make_hook(name))


def _drain_act_accum() -> dict[str, dict[str, float]]:
    accum = _STATE["act_accum"]
    snapshot: dict[str, dict[str, float]] = {}
    for name, entry in accum.items():
        n = max(1, entry["n"])
        snapshot[name] = {"norm_mean": entry["norm"] / n, "max": entry["max"]}
    _STATE["act_accum"] = {}
    return snapshot


def _per_layer_grad_norms(model: Any) -> dict[str, float]:
    chunks = model if isinstance(model, list) else [model]
    out: dict[str, float] = {}
    for chunk in chunks:
        for name, p in chunk.named_parameters():
            if p.grad is None:
                continue
            out[name] = float(p.grad.detach().float().norm().item())
    return out


def _collect_cler_gamma_stats(model: Any) -> dict[str, Any]:
    chunks = model if isinstance(model, list) else [model]
    gammas: dict[str, float] = {}
    for chunk in chunks:
        for name, p in chunk.named_parameters():
            if not name.endswith("cler_gamma"):
                continue
            values = p.detach().float().reshape(-1)
            if values.numel() == 1:
                gammas[name] = float(values.item())
            else:
                for index, value in enumerate(values.tolist()):
                    gammas[f"{name}.{index}"] = float(value)

    if not gammas:
        return {}

    values = list(gammas.values())
    abs_values = [abs(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "abs_mean": sum(abs_values) / len(abs_values),
        "max_abs": max(abs_values),
        "l2": math.sqrt(sum(value * value for value in values)),
        "values": gammas,
    }


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return {}

    with torch.no_grad():
        data = tensor.detach()
        values = data.float().reshape(-1)
        numel = int(values.numel())
        if numel == 0:
            return {
                "shape": list(data.shape),
                "dtype": str(data.dtype).replace("torch.", ""),
                "numel": 0,
            }

        finite = torch.isfinite(values)
        finite_count = int(finite.sum().item())
        if finite_count == 0:
            return {
                "shape": list(data.shape),
                "dtype": str(data.dtype).replace("torch.", ""),
                "numel": numel,
                "finite_ratio": 0.0,
            }
        if finite_count != numel:
            values = values[finite]

        abs_values = values.abs()
        square_mean = values.square().mean()
        return {
            "shape": list(data.shape),
            "dtype": str(data.dtype).replace("torch.", ""),
            "numel": numel,
            "finite_ratio": finite_count / numel,
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "abs_mean": float(abs_values.mean().item()),
            "max_abs": float(abs_values.max().item()),
            "rms": float(square_mean.sqrt().item()),
            "l2": float(values.norm().item()),
        }


def _collect_cler_route_query_stats(model: Any) -> dict[str, Any]:
    chunks = model if isinstance(model, list) else [model]
    queries: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for name, p in chunk.named_parameters():
            if "cler_route_queries" not in name:
                continue
            stats = _tensor_summary(p)
            if stats:
                queries[name] = stats

    if not queries:
        return {}

    abs_means = [
        float(stats["abs_mean"]) for stats in queries.values() if "abs_mean" in stats
    ]
    max_abs_values = [
        float(stats["max_abs"]) for stats in queries.values() if "max_abs" in stats
    ]
    payload: dict[str, Any] = {
        "count": len(queries),
        "queries": queries,
    }
    if abs_means:
        payload["abs_mean_mean"] = sum(abs_means) / len(abs_means)
    if max_abs_values:
        payload["max_abs"] = max(max_abs_values)
    return payload


def _collect_attn_res_query_stats(model: Any) -> dict[str, Any]:
    chunks = model if isinstance(model, list) else [model]
    queries: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for name, p in chunk.named_parameters():
            if "attn_res_queries" not in name:
                continue
            stats = _tensor_summary(p)
            if stats:
                queries[name] = stats

    if not queries:
        return {}

    abs_means = [float(s["abs_mean"]) for s in queries.values() if "abs_mean" in s]
    max_abs_values = [float(s["max_abs"]) for s in queries.values() if "max_abs" in s]
    payload: dict[str, Any] = {"count": len(queries), "queries": queries}
    if abs_means:
        payload["abs_mean_mean"] = sum(abs_means) / len(abs_means)
    if max_abs_values:
        payload["max_abs"] = max(max_abs_values)
    return payload


def _collect_cler_residual_stats(model: Any) -> dict[str, Any]:
    chunks = model if isinstance(model, list) else [model]
    layers: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if not name.endswith(".self_attention"):
                continue
            residual = getattr(module, "cler_residual", None)
            if residual is None:
                continue
            stats = _tensor_summary(residual)
            if stats:
                layers[f"{name}.cler_residual"] = stats

    if not layers:
        return {}

    abs_means = [
        float(stats["abs_mean"]) for stats in layers.values() if "abs_mean" in stats
    ]
    max_abs_values = [
        float(stats["max_abs"]) for stats in layers.values() if "max_abs" in stats
    ]
    payload: dict[str, Any] = {
        "count": len(layers),
        "layers": layers,
    }
    if abs_means:
        payload["abs_mean_mean"] = sum(abs_means) / len(abs_means)
    if max_abs_values:
        payload["max_abs"] = max(max_abs_values)
    return payload


def _collect_cler_value_stats(model: Any) -> dict[str, Any]:
    """Summarize the GDN value tensor v (stashed as cler_value), parallel to the residual stats.
    Lets us compare ||v|| vs ||r|| for CLER-V vs CLER-H runs (the magnitude question)."""
    chunks = model if isinstance(model, list) else [model]
    layers: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if not name.endswith(".self_attention"):
                continue
            value = getattr(module, "cler_value", None)
            if value is None:
                continue
            stats = _tensor_summary(value)
            if stats:
                layers[f"{name}.cler_value"] = stats
    if not layers:
        return {}
    abs_means = [float(s["abs_mean"]) for s in layers.values() if "abs_mean" in s]
    payload: dict[str, Any] = {"count": len(layers), "layers": layers}
    if abs_means:
        payload["abs_mean_mean"] = sum(abs_means) / len(abs_means)
    return payload


def _collect_cler_gate_stats(model: Any) -> dict[str, Any]:
    """Summarize the data-dependent CLER dynamic-gate activations sigmoid(Linear(value))."""
    chunks = model if isinstance(model, list) else [model]
    layers: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if not name.endswith(".self_attention"):
                continue
            gate = getattr(module, "cler_gate_activation", None)
            if gate is None:
                continue
            stats = _tensor_summary(gate)
            if stats:
                layers[f"{name}.cler_gate"] = stats

    if not layers:
        return {}

    means = [float(s["mean"]) for s in layers.values() if "mean" in s]
    max_abs_values = [float(s["max_abs"]) for s in layers.values() if "max_abs" in s]
    payload: dict[str, Any] = {"count": len(layers), "layers": layers}
    if means:
        payload["mean_mean"] = sum(means) / len(means)
    if max_abs_values:
        payload["max_abs"] = max(max_abs_values)
    return payload


def _append_cler_gate_log(iteration: int) -> dict[str, Any]:
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    model = _STATE.get("model")
    if model is None:
        return {}
    try:
        interval = int(os.environ.get("CLER_GATE_LOG_INTERVAL", "10"))
    except ValueError:
        interval = 10
    interval = max(interval, 1)
    if int(iteration) % interval != 0 and int(iteration) != 1:
        return {}

    stats = _collect_cler_gate_stats(model)
    if not stats:
        return {}

    path = _STATE.get("cler_gate_log_path")
    if path is None:
        log_dir = Path(os.environ.get("APERTUS_LOG_DIR", "_research/results/performance"))
        run_name = os.environ.get("APERTUS_RUN_NAME", "run")
        path = str(log_dir / f"{run_name}.cler_gate.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("")
        _STATE["cler_gate_log_path"] = path

    if not _STATE.get("cler_gate_log_announced"):
        print(f"[cler] dynamic-gate stats log: {path}", flush=True)
        _STATE["cler_gate_log_announced"] = True

    payload = {"step": int(iteration), "wall": time.time(), **stats}
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        fh.write("\n")
    return stats


def _append_cler_residual_log(iteration: int) -> dict[str, Any]:
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    if os.environ.get("CLER_RESIDUAL_LOG_ENABLED", "1") == "0":
        return {}
    model = _STATE.get("model")
    if model is None:
        return {}
    try:
        interval = int(os.environ.get("CLER_RESIDUAL_LOG_INTERVAL", "10"))
    except ValueError:
        interval = 10
    interval = max(interval, 1)
    if int(iteration) % interval != 0 and int(iteration) != 1:
        return {}

    stats = _collect_cler_residual_stats(model)
    if not stats:
        return {}

    path = _STATE.get("cler_residual_log_path")
    if path is None:
        log_dir = Path(os.environ.get("APERTUS_LOG_DIR", "_research/results/performance"))
        run_name = os.environ.get("APERTUS_RUN_NAME", "run")
        path = str(log_dir / f"{run_name}.cler_residual.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("")
        _STATE["cler_residual_log_path"] = path

    if not _STATE.get("cler_residual_log_announced"):
        print(f"[cler] residual stats log: {path}", flush=True)
        _STATE["cler_residual_log_announced"] = True

    payload = {
        "step": int(iteration),
        "wall": time.time(),
        **stats,
    }
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        fh.write("\n")
    return stats


def _append_cler_value_log(iteration: int) -> dict[str, Any]:
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    if os.environ.get("CLER_RESIDUAL_LOG_ENABLED", "1") == "0":
        return {}
    model = _STATE.get("model")
    if model is None:
        return {}
    try:
        interval = int(os.environ.get("CLER_RESIDUAL_LOG_INTERVAL", "10"))
    except ValueError:
        interval = 10
    interval = max(interval, 1)
    if int(iteration) % interval != 0 and int(iteration) != 1:
        return {}

    stats = _collect_cler_value_stats(model)
    if not stats:
        return {}

    path = _STATE.get("cler_value_log_path")
    if path is None:
        log_dir = Path(os.environ.get("APERTUS_LOG_DIR", "_research/results/performance"))
        run_name = os.environ.get("APERTUS_RUN_NAME", "run")
        path = str(log_dir / f"{run_name}.cler_value.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("")
        _STATE["cler_value_log_path"] = path

    if not _STATE.get("cler_value_log_announced"):
        print(f"[cler] value stats log: {path}", flush=True)
        _STATE["cler_value_log_announced"] = True

    payload = {"step": int(iteration), "wall": time.time(), **stats}
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        fh.write("\n")
    return stats


def _append_attn_res_log(iteration: int) -> dict[str, Any]:
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    model = _STATE.get("model")
    if model is None:
        return {}
    try:
        interval = int(os.environ.get("ATTN_RES_LOG_INTERVAL", "100"))
    except ValueError:
        interval = 100
    interval = max(interval, 1)
    if int(iteration) % interval != 0 and int(iteration) != 1:
        return {}

    stats = _collect_attn_res_query_stats(model)
    if not stats:
        return {}

    path = _STATE.get("attn_res_log_path")
    if path is None:
        log_dir = Path(os.environ.get("APERTUS_LOG_DIR", "_research/results/performance"))
        run_name = os.environ.get("APERTUS_RUN_NAME", "run")
        path = str(log_dir / f"{run_name}.attn_res.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("")
        _STATE["attn_res_log_path"] = path

    if not _STATE.get("attn_res_log_announced"):
        print(f"[attn_res] query stats log: {path}", flush=True)
        _STATE["attn_res_log_announced"] = True

    payload = {"step": int(iteration), "wall": time.time(), **stats}
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        fh.write("\n")
    return stats


def _append_cler_gamma_log(iteration: int) -> dict[str, Any]:
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    model = _STATE.get("model")
    if model is None:
        return {}
    try:
        interval = int(os.environ.get("CLER_GAMMA_LOG_INTERVAL", "10"))
    except ValueError:
        interval = 10
    interval = max(interval, 1)
    if int(iteration) % interval != 0 and int(iteration) != 1:
        return {}

    stats = _collect_cler_gamma_stats(model)
    if not stats:
        stats = {}

    route_query_stats = _collect_cler_route_query_stats(model)
    if route_query_stats:
        stats["route_queries"] = route_query_stats

    if not stats:
        return {}

    path = _STATE.get("cler_gamma_log_path")
    if path is None:
        log_dir = Path(os.environ.get("APERTUS_LOG_DIR", "_research/results/performance"))
        run_name = os.environ.get("APERTUS_RUN_NAME", "run")
        path = str(log_dir / f"{run_name}.cler_gamma.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("")
        _STATE["cler_gamma_log_path"] = path

    if not _STATE.get("cler_gamma_log_announced"):
        print(f"[cler] gamma stats log: {path}", flush=True)
        _STATE["cler_gamma_log_announced"] = True

    payload = {
        "step": int(iteration),
        "wall": time.time(),
        **stats,
    }
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        fh.write("\n")
    return stats


def _is_spike(loss: float) -> bool:
    history = _STATE["loss_history"]
    history.append(loss)
    if len(history) > 200:
        history.pop(0)
    if len(history) < 30:
        return False
    mean = sum(history[:-1]) / (len(history) - 1)
    var = sum((x - mean) ** 2 for x in history[:-1]) / (len(history) - 1)
    std = math.sqrt(max(var, 1e-12))
    return (loss - mean) > _STATE["loss_spike_k"] * std


def patch_setup_model_and_optimizer() -> None:
    """Wrap setup_model_and_optimizer to capture the model and (optionally) hooks."""
    from megatron.training import training as mtt

    original = mtt.setup_model_and_optimizer

    def wrapped(*args, **kwargs):
        phase_timer.stamp("before_model_build")
        result = original(*args, **kwargs)
        phase_timer.stamp("after_model_build")
        model = result[0] if isinstance(result, tuple) else result
        _STATE["model"] = model
        total, active = _count_params(model)
        _STATE["n_params_total"] = total
        _STATE["n_params_active"] = active
        if _STATE["log_act_stats"]:
            _register_act_hooks(model)
        writer = _STATE["writer"]
        if writer is not None:
            writer.set(n_params_total=total, n_params_active=active)
        return result

    mtt.setup_model_and_optimizer = wrapped


def patch_compute_language_model_loss() -> None:
    """Wrap LanguageModule.compute_language_model_loss to also compute top-1 accuracy.

    Logits are TP-vocab-parallel: each rank holds ``[s, b, local_vocab]``. We find
    the global argmax by reducing local max values across the TP group, masking to
    the rank that owns the winner, then all-reducing the winner index. Only TP
    rank 0 accumulates to avoid TP-wise over-counting.
    """
    from megatron.core.models.common.language_module import \
        language_module as lm

    original = lm.LanguageModule.compute_language_model_loss

    def wrapped(self, labels, logits):
        loss = original(self, labels, logits)
        if not _STATE["log_top1_acc"]:
            return loss

        try:
            import torch
            import torch.distributed as dist
            from megatron.core import parallel_state as ps

            with torch.no_grad():
                tp_size = ps.get_tensor_model_parallel_world_size()
                tp_rank = ps.get_tensor_model_parallel_rank()
                tp_group = ps.get_tensor_model_parallel_group()

                local_max_val, local_argmax = logits.max(dim=-1)
                labels_sb = labels.transpose(0, 1).contiguous()

                vocab_start = tp_rank * logits.size(-1)
                global_argmax = (local_argmax + vocab_start).long()

                if tp_size > 1:
                    global_max_val = local_max_val.clone()
                    dist.all_reduce(global_max_val, op=dist.ReduceOp.MAX, group=tp_group)
                    neg_one = torch.full_like(global_argmax, -1)
                    winner_idx = torch.where(local_max_val == global_max_val, global_argmax, neg_one)
                    dist.all_reduce(winner_idx, op=dist.ReduceOp.MAX, group=tp_group)
                else:
                    winner_idx = global_argmax

                if tp_rank == 0:
                    correct = (winner_idx == labels_sb).sum().float().item()
                    total = float(labels_sb.numel())
                    _STATE["top1_correct"] += correct
                    _STATE["top1_total"] += total
        except Exception:
            pass

        return loss

    lm.LanguageModule.compute_language_model_loss = wrapped


def _emit_top1_accuracy(row: dict[str, Any], iteration: int) -> None:
    """All-reduce accumulated top-1 counts across the world, emit to row and wandb.

    Must be called on every rank since it performs a collective. Only rank 0
    writes to wandb (the writer layer handles its own rank-0 gating).
    """
    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return

    if not (dist.is_available() and dist.is_initialized()):
        if _STATE["top1_total"] > 0:
            acc = _STATE["top1_correct"] / _STATE["top1_total"]
            row["top1_accuracy"] = acc
        _STATE["top1_correct"] = 0.0
        _STATE["top1_total"] = 0.0
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    counts = torch.tensor(
        [_STATE["top1_correct"], _STATE["top1_total"]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    _STATE["top1_correct"] = 0.0
    _STATE["top1_total"] = 0.0

    total = float(counts[1].item())
    if total <= 0:
        return
    acc = float(counts[0].item()) / total
    row["top1_accuracy"] = acc

    if int(os.environ.get("RANK", "0")) != 0:
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"top1_accuracy": acc}, step=int(iteration))
    except Exception:
        pass


def patch_training_log() -> None:
    """Wrap training_log to append one row per call to the JSON writer."""
    from megatron.training import training as mtt

    original = mtt.training_log

    def wrapped(
        loss_dict, total_loss_dict, learning_rate, iteration, loss_scale,
        report_memory_flag, skipped_iter, grad_norm, params_norm,
        num_zeros_in_grad, max_attention_logit,
        pg_collection=None, is_first_iteration=False,
        **kwargs,
    ):
        ret = original(
            loss_dict, total_loss_dict, learning_rate, iteration, loss_scale,
            report_memory_flag, skipped_iter, grad_norm, params_norm,
            num_zeros_in_grad, max_attention_logit,
            pg_collection=pg_collection, is_first_iteration=is_first_iteration,
            **kwargs,
        )

        writer = _STATE["writer"]
        if writer is None:
            return ret

        from megatron.training import get_args
        args = get_args()

        now = time.time()
        last_wall = _STATE["last_wall"]
        last_iter = _STATE["last_iter"]
        _STATE["last_wall"] = now
        _STATE["last_iter"] = iteration

        tokens_per_sec_per_gpu = None
        if last_wall is not None and last_iter is not None and iteration > last_iter:
            dt = max(now - last_wall, 1e-9)
            tokens_this_interval = (iteration - last_iter) * args.global_batch_size * args.seq_length
            tokens_per_sec_per_gpu = tokens_this_interval / dt / max(1, args.world_size)

        if tokens_per_sec_per_gpu is not None and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[apertus] iter {int(iteration):5d} | tokens/s/GPU: {tokens_per_sec_per_gpu:10.1f}",
                flush=True,
            )

        row: dict[str, Any] = {
            "step": int(iteration),
            "wall": now,
        }
        def _pick_loss(d):
            if not d:
                return None
            k = next((k for k in d if "lm loss" in k.lower() or k == "lm loss"), None)
            if k is None:
                return None
            v = d[k]
            if isinstance(v, (list, tuple)) and len(v) >= 1:
                v = v[0]
            try:
                return float(v.item() if hasattr(v, "item") else v)
            except Exception:
                return None

        train_loss = _pick_loss(loss_dict)
        if train_loss is None:
            train_loss = _pick_loss(total_loss_dict)
        if train_loss is not None:
            row["train_loss"] = train_loss
        if learning_rate is not None:
            row["lr"] = float(learning_rate)
        if grad_norm is not None:
            row["grad_norm"] = float(grad_norm)
        if params_norm is not None:
            row["params_norm"] = float(params_norm)
        if tokens_per_sec_per_gpu is not None:
            row["tput"] = float(tokens_per_sec_per_gpu)
        cler_gamma_stats = _append_cler_gamma_log(int(iteration))
        if cler_gamma_stats:
            row["cler_gamma"] = cler_gamma_stats
        _append_cler_value_log(int(iteration))
        cler_residual_stats = _append_cler_residual_log(int(iteration))
        if cler_residual_stats:
            row["cler_residual"] = cler_residual_stats
        attn_res_stats = _append_attn_res_log(int(iteration))
        if attn_res_stats:
            row["attn_res_queries"] = attn_res_stats
        cler_gate_stats = _append_cler_gate_log(int(iteration))
        if cler_gate_stats:
            row["cler_gate"] = cler_gate_stats

        if _STATE["log_per_layer_grads"] and _STATE["model"] is not None:
            row["per_layer_grad_norm"] = _per_layer_grad_norms(_STATE["model"])

        if _STATE["log_act_stats"]:
            row["act_stats"] = _drain_act_accum()

        if _STATE["log_loss_spikes"] and "train_loss" in row:
            row["loss_spike"] = _is_spike(row["train_loss"])

        if _STATE["log_top1_acc"]:
            _emit_top1_accuracy(row, int(iteration))

        writer.append(**row)

        n_layers = _STATE.get("n_layers")
        hidden = _STATE.get("hidden")
        seq_len = _STATE.get("seq_len")
        n_active = _STATE.get("n_params_active")
        if (
            tokens_per_sec_per_gpu is not None
            and n_layers and hidden and seq_len and n_active
        ):
            mfu = compute_mfu(n_active, n_layers, seq_len, hidden, tokens_per_sec_per_gpu)
            writer.set(
                tokens_per_sec_per_gpu=float(tokens_per_sec_per_gpu),
                mfu=float(mfu),
                steps=int(iteration),
                tokens=int(iteration * args.global_batch_size * args.seq_length),
            )

        return ret

    mtt.training_log = wrapped


def patch_train_step() -> None:
    """Wrap train_step to stamp iter 1/2 wall-clock boundaries.

    Only the first two calls emit stamps; subsequent calls are a plain pass-
    through with zero overhead beyond a single int increment.
    """
    from megatron.training import training as mtt

    original = mtt.train_step

    def wrapped(*args, **kwargs):
        phase_timer.note_train_step_start()
        try:
            return original(*args, **kwargs)
        finally:
            phase_timer.note_train_step_end()

    mtt.train_step = wrapped
