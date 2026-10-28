"""
Tensor Parallel (TP=2) inference for OPT-6.7B
Launch with: torchrun --nproc_per_node=2 tp_inference.py

All 5 systems:
  1. Baseline (GPU-only)
  2. DejaVu (CPU offload)
  3. H2O (eviction)
  4. DejaVu + Prefetch (pinned memory + CUDA streams)
  5. Hybrid (H2O + DejaVu + Prefetch)
"""

import torch
import torch.distributed as dist
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import transformers
from datasets import load_dataset

transformers.utils.logging.disable_progress_bar()


# ═════════════════════════════════════════════
# DISTRIBUTED SETUP
# ═════════════════════════════════════════════

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
torch.cuda.set_per_process_memory_fraction(0.65)

if rank == 0:
    print(f"World size: {world_size}")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# ═════════════════════════════════════════════
# MODEL + TOKENIZER
# ═════════════════════════════════════════════

model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    torch_dtype=torch.float16,
).to(device)
model.eval()

if rank == 0:
    print(f"\nModel: {model_name}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num heads: {model.config.num_attention_heads}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Head dim: {model.config.hidden_size // model.config.num_attention_heads}")


# ═════════════════════════════════════════════
# TENSOR PARALLELISM: SHARD ATTENTION WEIGHTS
# ═════════════════════════════════════════════

def shard_attention_weights(model, rank, world_size):
    num_heads = model.config.num_attention_heads          # 32
    head_dim = model.config.hidden_size // num_heads      # 128
    heads_per_rank = num_heads // world_size              # 16

    start = rank * heads_per_rank * head_dim
    end = start + heads_per_rank * head_dim

    for layer in model.model.decoder.layers:
        attn = layer.self_attn

        # Q, K, V projections — slice rows (output dim)
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            proj.weight.data = proj.weight.data[start:end, :].contiguous()
            if proj.bias is not None:
                proj.bias.data = proj.bias.data[start:end].contiguous()
            proj.out_features = heads_per_rank * head_dim

        # out_proj — slice columns (input dim)
        attn.out_proj.weight.data = attn.out_proj.weight.data[:, start:end].contiguous()
        attn.out_proj.in_features = heads_per_rank * head_dim
        # Divide bias by world_size so allreduce sums to correct value
        if attn.out_proj.bias is not None:
            attn.out_proj.bias.data = attn.out_proj.bias.data / world_size

        # Update head count
        attn.num_heads = heads_per_rank

    if rank == 0:
        print(f"\nWeights sharded: {heads_per_rank} heads per rank")
        print(f"  Q/K/V: [{heads_per_rank * head_dim}, {model.config.hidden_size}]")
        print(f"  out_proj: [{model.config.hidden_size}, {heads_per_rank * head_dim}]")

shard_attention_weights(model, rank, world_size)


# ═════════════════════════════════════════════
# TENSOR PARALLELISM: ALLREDUCE HOOKS
# ═════════════════════════════════════════════

def add_allreduce_hooks(model):
    hooks = []
    for layer in model.model.decoder.layers:
        def make_hook():
            def hook(module, input, output):
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                return output
            return hook
        h = layer.self_attn.out_proj.register_forward_hook(make_hook())
        hooks.append(h)
    if rank == 0:
        print(f"Allreduce hooks added to {len(hooks)} layers")
    return hooks

hooks = add_allreduce_hooks(model)


# ═════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
long_prompts = []
for item in dataset:
    text = item["text"].strip()
    tokens = tokenizer.encode(text)
    if len(tokens) > 200:
        long_prompts.append(text)
    if len(long_prompts) >= 32:
        break

if rank == 0:
    print(f"\nFound {len(long_prompts)} prompts")


# ═════════════════════════════════════════════
# KV CACHE HELPERS (same as your notebook)
# ═════════════════════════════════════════════

def unpack_kv_cache(kv_cache):
    if kv_cache is None:
        return None
    if hasattr(kv_cache, "layers") and isinstance(kv_cache.layers, list):
        return [(layer.keys, layer.values) for layer in kv_cache.layers]
    if isinstance(kv_cache, (tuple, list)):
        return kv_cache
    raise TypeError(f"Unsupported KV cache type: {type(kv_cache)}")

def get_seq_len(kv_cache):
    layers = unpack_kv_cache(kv_cache)
    if layers is None or len(layers) == 0:
        return 0
    return layers[0][0].shape[-2]

def kv_cache_size_mb(kv_cache):
    layers = unpack_kv_cache(kv_cache)
    if layers is None:
        return 0
    total = 0
    for k, v in layers:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total / (1024 ** 2)

def offload_kv_to_cpu_pinned(kv_cache):
    layers = unpack_kv_cache(kv_cache)
    return tuple(
        (k.detach().contiguous().cpu().pin_memory(),
         v.detach().contiguous().cpu().pin_memory())
        for k, v in layers
    )

def extract_new_kv_chunk_pinned(full_kv_cache, old_seq_len):
    layers = unpack_kv_cache(full_kv_cache)
    return tuple(
        (k[:, :, old_seq_len:, :].detach().contiguous().cpu().pin_memory(),
         v[:, :, old_seq_len:, :].detach().contiguous().cpu().pin_memory())
        for k, v in layers
    )

def append_kv_chunk_pinned(cpu_kv_cache, new_chunk):
    return tuple(
        (torch.cat([old_k, new_k], dim=-2).contiguous().pin_memory(),
         torch.cat([old_v, new_v], dim=-2).contiguous().pin_memory())
        for (old_k, old_v), (new_k, new_v) in zip(cpu_kv_cache, new_chunk)
    )

def build_kv_cache(kv_layers):
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_layers):
        cache.update(k, v, layer_idx)
    return cache

# Older-style helpers used by DejaVu vanilla
def get_kv_layers(kv_cache):
    return unpack_kv_cache(kv_cache)

def get_cache_seq_len(kv_cache):
    return get_seq_len(kv_cache)

def move_full_kv_to_device_timed(kv_cache, target_device):
    start = time.perf_counter()
    dev_layers = [(k.to(target_device), v.to(target_device)) for k, v in kv_cache]
    cache = build_kv_cache(dev_layers)
    end = time.perf_counter()
    return cache, (end - start)

def extract_new_kv_chunk(full_kv_cache, old_seq_len):
    layers = get_kv_layers(full_kv_cache)
    new_layers = []
    for k, v in layers:
        new_k = k[:, :, old_seq_len:, :].detach().cpu()
        new_v = v[:, :, old_seq_len:, :].detach().cpu()
        new_layers.append((new_k, new_v))
    return tuple(new_layers)

def append_kv_caches(cpu_kv_cache, new_chunk_cache):
    merged = []
    for (old_k, old_v), (new_k, new_v) in zip(cpu_kv_cache, new_chunk_cache):
        merged_k = torch.cat([old_k, new_k], dim=-2)
        merged_v = torch.cat([old_v, new_v], dim=-2)
        merged.append((merged_k, merged_v))
    return tuple(merged)

def build_dynamic_cache(kv_layers):
    return build_kv_cache(kv_layers)


# ═════════════════════════════════════════════
# SYSTEM 1: BASELINE (GPU-only)
# ═════════════════════════════════════════════

def run_baseline_benchmark():
    microbatches_count = [8, 16, 32]
    n_tokens = [256, 512]
    results = []

    total_vram_mb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2)

    for D in microbatches_count:
        for N in n_tokens:
            try:
                prompts = long_prompts[:D]

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                total_start = time.perf_counter()
                with torch.no_grad():
                    kv_caches = []
                    next_tokens = []

                    # PREFILL
                    for i in range(D):
                        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
                        outputs = model(input_ids=inputs.input_ids, use_cache=True)
                        kv_caches.append(outputs.past_key_values)
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)
                        next_tokens.append(next_token)

                    torch.cuda.reset_peak_memory_stats()
                    # DECODE
                    for step in range(N - 1):
                        for i in range(D):
                            outputs = model(
                                input_ids=next_tokens[i],
                                past_key_values=kv_caches[i],
                                use_cache=True,
                            )
                            kv_caches[i] = outputs.past_key_values
                            next_tokens[i] = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)

                total_end = time.perf_counter()

                latency = total_end - total_start
                total_tokens = D * N
                throughput = total_tokens / latency
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_util = (peak_vram_mb / total_vram_mb) * 100

                total_kv_bytes = 0
                for kv in kv_caches:
                    for layer in kv.layers:
                        total_kv_bytes += layer.keys.nelement() * layer.keys.element_size()
                        total_kv_bytes += layer.values.nelement() * layer.values.element_size()
                kv_cache_mb = total_kv_bytes / (1024 ** 2)

                if rank == 0:
                    print(f"D={D:<4} N={N:<4} | Latency={latency:.3f}s | Throughput={throughput:.1f} tok/s | Peak VRAM={peak_vram_mb:.1f}MB | KV Cache={kv_cache_mb:.1f}MB | VRAM%={vram_util:.1f}%")

                results.append({
                    "Microbatches (D)": D,
                    "Tokens Generated (N)": N,
                    "Latency (s)": latency,
                    "Throughput (tok/s)": throughput,
                    "Peak VRAM (MB)": peak_vram_mb,
                    "KV Cache Total (MB)": kv_cache_mb,
                    "VRAM Util (%)": vram_util,
                })

                del kv_caches, next_tokens
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"D={D:<4} N={N:<4} | OOM")
                    results.append({
                        "Microbatches (D)": D,
                        "Tokens Generated (N)": N,
                        "Latency (s)": "OOM",
                        "Throughput (tok/s)": "OOM",
                        "Peak VRAM (MB)": "OOM",
                        "KV Cache Total (MB)": "OOM",
                        "VRAM Util (%)": "OOM",
                    })
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    return pd.DataFrame(results)


# ═════════════════════════════════════════════
# SYSTEM 2: DEJAVU (vanilla CPU offload)
# ═════════════════════════════════════════════

def run_dejavu_benchmark():
    microbatches_count = [8, 16, 32]
    n_tokens = [256, 512]
    results = []

    total_vram_mb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2)

    for D in microbatches_count:
        for N in n_tokens:
            try:
                prompts = long_prompts[:D]

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                total_start = time.perf_counter()

                with torch.no_grad():
                    # PREFILL
                    cpu_kv_caches = []
                    next_tokens = []
                    for i in range(D):
                        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
                        outputs = model(input_ids=inputs.input_ids, use_cache=True)
                        next_tokens.append(torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1))
                        layers = get_kv_layers(outputs.past_key_values)
                        cpu_kv_caches.append([(k.detach().cpu(), v.detach().cpu()) for k, v in layers])
                        del outputs
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    transfer_time = 0.0
                    # DECODE
                    for step in range(N - 1):
                        for i in range(D):
                            gpu_cache, t_upload = move_full_kv_to_device_timed(cpu_kv_caches[i], device)
                            old_seq_length = get_cache_seq_len(gpu_cache)
                            outputs = model(
                                input_ids=next_tokens[i],
                                past_key_values=gpu_cache,
                                use_cache=True,
                            )
                            new_kv_chunk = extract_new_kv_chunk(outputs.past_key_values, old_seq_length)

                            t_start = time.perf_counter()
                            cpu_kv_caches[i] = append_kv_caches(cpu_kv_caches[i], new_kv_chunk)
                            t_stop = time.perf_counter()
                            t_offload = t_stop - t_start

                            next_tokens[i] = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)

                            transfer_time += t_upload
                            transfer_time += t_offload

                            torch.cuda.empty_cache()

                    del gpu_cache, new_kv_chunk
                    torch.cuda.empty_cache()
                total_end = time.perf_counter()

                latency = total_end - total_start
                total_tokens = D * N
                throughput = total_tokens / latency
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_util = (peak_vram_mb / total_vram_mb) * 100
                transfer_overhead = (transfer_time / latency) * 100

                total_kv_bytes = 0
                for cache in cpu_kv_caches:
                    for k, v in cache:
                        total_kv_bytes += k.nelement() * k.element_size()
                        total_kv_bytes += v.nelement() * v.element_size()
                cpu_kv_cache_mb = total_kv_bytes / (1024 ** 2)

                if rank == 0:
                    print(f"D={D:<4} N={N:<4} | Latency={latency:.3f}s | Throughput={throughput:.1f} tok/s | Peak VRAM={peak_vram_mb:.1f}MB | VRAM%={vram_util:.1f}% | CPU KV Cache={cpu_kv_cache_mb:.1f}MB | Transfer Time={transfer_time:.2f}s | Transfer Overhead={transfer_overhead:.2f}%")

                results.append({
                    "Microbatches (D)": D,
                    "Tokens Generated (N)": N,
                    "Latency (s)": latency,
                    "Throughput (tok/s)": throughput,
                    "Peak VRAM (MB)": peak_vram_mb,
                    "VRAM Util (%)": vram_util,
                    "CPU KV Cache Total (MB)": cpu_kv_cache_mb,
                    "Transfer Time (s)": transfer_time,
                    "Transfer Overhead (%)": transfer_overhead,
                })

                del next_tokens
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"D={D:<4} N={N:<4} | OOM")
                    results.append({
                        "Microbatches (D)": D,
                        "Tokens Generated (N)": N,
                        "Latency (s)": "OOM",
                        "Throughput (tok/s)": "OOM",
                        "Peak VRAM (MB)": "OOM",
                        "VRAM Util (%)": "OOM",
                        "CPU KV Cache Total (MB)": "OOM",
                        "Transfer Time (s)": "OOM",
                        "Transfer Overhead (%)": "OOM",
                    })
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    return pd.DataFrame(results)


# ═════════════════════════════════════════════
# SYSTEM 3: H2O (eviction, GPU-only)
# ═════════════════════════════════════════════

def run_h2o_benchmark(budget_ratio=0.4, recent_ratio=0.5):
    microbatches_count = [8, 16, 32]
    n_tokens = [256, 512]
    results = []

    total_vram_mb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2)

    for D in microbatches_count:
        for N in n_tokens:
            try:
                prompts = long_prompts[:D]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                sample_len = tokenizer(prompts[0], return_tensors="pt").input_ids.shape[1]
                budget = int((sample_len + N) * budget_ratio)
                budget = max(budget, sample_len)
                recent_window = int(budget * recent_ratio)

                total_start = time.perf_counter()

                with torch.no_grad():
                    # PREFILL
                    next_tokens = []
                    cum_scores_list = []
                    kv_caches = []
                    for i in range(D):
                        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
                        outputs = model(input_ids=inputs.input_ids, use_cache=True, output_attentions=True)
                        next_tokens.append(torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1))

                        seq_len = outputs.past_key_values.get_seq_length()
                        scores = torch.zeros(seq_len, device=device)
                        for attn in outputs.attentions:
                            scores += attn[0].sum(dim=0).sum(dim=0)
                        cum_scores_list.append(scores)
                        kv_caches.append(outputs.past_key_values)

                    torch.cuda.reset_peak_memory_stats()

                    # DECODE
                    for step in range(N - 1):
                        for i in range(D):
                            outputs = model(
                                input_ids=next_tokens[i],
                                past_key_values=kv_caches[i],
                                use_cache=True,
                                output_attentions=True,
                            )
                            kv_caches[i] = outputs.past_key_values
                            next_tokens[i] = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)

                            current_len = kv_caches[i].get_seq_length()
                            new_score = torch.zeros(1, device=device)
                            cum_scores_list[i] = torch.cat([cum_scores_list[i], new_score])
                            for attn in outputs.attentions:
                                step_scores = attn[0].sum(dim=0).sum(dim=0)
                                if len(step_scores) == len(cum_scores_list[i]):
                                    cum_scores_list[i] += step_scores

                            # H2O eviction
                            while current_len > budget:
                                evict_candidates = cum_scores_list[i][:-recent_window].clone()
                                if len(evict_candidates) == 0:
                                    break

                                evict_pos = evict_candidates.argmin().item()

                                keep_mask = torch.ones(current_len, dtype=torch.bool, device=device)
                                keep_mask[evict_pos] = False
                                keep_indices = torch.where(keep_mask)[0]

                                kv_layers = unpack_kv_cache(kv_caches[i])
                                kept_layers = [(k[:, :, keep_indices, :], v[:, :, keep_indices, :]) for k, v in kv_layers]
                                kv_caches[i] = build_kv_cache(kept_layers)

                                cum_scores_list[i] = cum_scores_list[i][keep_mask]
                                current_len -= 1

                            del outputs

                total_end = time.perf_counter()

                latency = total_end - total_start
                throughput = (D * N) / latency
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_util = (peak_vram_mb / total_vram_mb) * 100

                kv_bytes = 0
                for cache in kv_caches:
                    kv_layers = unpack_kv_cache(cache)
                    for k, v in kv_layers:
                        kv_bytes += k.nelement() * k.element_size()
                        kv_bytes += v.nelement() * v.element_size()
                kv_mb = kv_bytes / (1024 ** 2)

                if rank == 0:
                    print(f"D={D:<4} N={N:<4} | Budget={budget:<4} | Latency={latency:.3f}s | Throughput={throughput:.1f} tok/s | "
                          f"Peak VRAM={peak_vram_mb:.1f}MB | VRAM%={vram_util:.1f}% | GPU KV={kv_mb:.1f}MB")

                results.append({
                    "Microbatches (D)": D,
                    "Tokens Generated (N)": N,
                    "Budget": budget,
                    "Latency (s)": latency,
                    "Throughput (tok/s)": throughput,
                    "Peak VRAM (MB)": peak_vram_mb,
                    "VRAM Util (%)": vram_util,
                    "GPU KV (MB)": kv_mb,
                })

                del kv_caches, next_tokens, cum_scores_list
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"D={D:<4} N={N:<4} | OOM")
                    results.append({
                        "Microbatches (D)": D,
                        "Tokens Generated (N)": N,
                        "Budget": budget,
                        "Latency (s)": "OOM",
                        "Throughput (tok/s)": "OOM",
                        "Peak VRAM (MB)": "OOM",
                        "VRAM Util (%)": "OOM",
                        "GPU KV (MB)": "OOM",
                    })
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    return pd.DataFrame(results)


# ═════════════════════════════════════════════
# SYSTEM 4: DEJAVU + PREFETCH (pinned + streams)
# ═════════════════════════════════════════════

def run_dejavu_prefetch_benchmark():
    microbatches_count = [8, 16, 32]
    n_tokens = [256, 512]
    results = []

    total_vram_mb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2)

    for D in microbatches_count:
        for N in n_tokens:
            try:
                prompts = long_prompts[:D]

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                total_start = time.perf_counter()

                with torch.no_grad():
                    # PREFILL
                    cpu_kv_caches = []
                    next_tokens = []
                    for i in range(D):
                        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
                        outputs = model(input_ids=inputs.input_ids, use_cache=True)
                        next_tokens.append(torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1))
                        cpu_kv_caches.append(offload_kv_to_cpu_pinned(outputs.past_key_values))
                        del outputs
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    prefetch_stream = torch.cuda.Stream(device=device)
                    offload_stream = torch.cuda.Stream(device=device)

                    # Prefetch first microbatch
                    with torch.cuda.stream(prefetch_stream):
                        prefetched = [(k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                                      for k, v in cpu_kv_caches[0]]

                    # DECODE
                    for step in range(N - 1):
                        for i in range(D):
                            next_i = (i + 1) % D

                            prefetch_stream.synchronize()
                            offload_stream.synchronize()

                            gpu_layers = prefetched

                            with torch.cuda.stream(prefetch_stream):
                                prefetched = [(k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                                              for k, v in cpu_kv_caches[next_i]]

                            gpu_cache = build_kv_cache(gpu_layers)
                            old_seq_len = gpu_cache.get_seq_length()
                            outputs = model(
                                input_ids=next_tokens[i],
                                past_key_values=gpu_cache,
                                use_cache=True,
                            )

                            with torch.cuda.stream(offload_stream):
                                new_chunk = extract_new_kv_chunk_pinned(outputs.past_key_values, old_seq_len)
                                cpu_kv_caches[i] = append_kv_chunk_pinned(cpu_kv_caches[i], new_chunk)

                            next_tokens[i] = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)

                    try:
                        del gpu_cache, new_chunk
                    except:
                        pass
                    torch.cuda.empty_cache()

                total_end = time.perf_counter()

                latency = total_end - total_start
                throughput = (D * N) / latency
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_util = (peak_vram_mb / total_vram_mb) * 100
                cpu_kv_mb = sum(kv_cache_size_mb(cache) for cache in cpu_kv_caches)

                if rank == 0:
                    print(f"D={D:<4} N={N:<4} | Latency={latency:.3f}s | Throughput={throughput:.1f} tok/s | "
                          f"Peak VRAM={peak_vram_mb:.1f}MB | VRAM%={vram_util:.1f}% | CPU KV Cache={cpu_kv_mb:.1f}MB")

                results.append({
                    "Microbatches (D)": D,
                    "Tokens Generated (N)": N,
                    "Latency (s)": latency,
                    "Throughput (tok/s)": throughput,
                    "Peak VRAM (MB)": peak_vram_mb,
                    "VRAM Util (%)": vram_util,
                    "CPU KV Cache Total (MB)": cpu_kv_mb,
                })

                del next_tokens
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"D={D:<4} N={N:<4} | OOM")
                    results.append({
                        "Microbatches (D)": D,
                        "Tokens Generated (N)": N,
                        "Latency (s)": "OOM",
                        "Throughput (tok/s)": "OOM",
                        "Peak VRAM (MB)": "OOM",
                        "VRAM Util (%)": "OOM",
                        "CPU KV Cache Total (MB)": "OOM",
                    })
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    return pd.DataFrame(results)


# ═════════════════════════════════════════════
# SYSTEM 5: HYBRID (H2O + DejaVu + Prefetch)
# ═════════════════════════════════════════════

def run_h2o_dejavu_prefetch_benchmark(budget_ratio=0.4, recent_ratio=0.5):
    microbatches_count = [8, 16, 32]
    n_tokens = [256, 512]
    results = []

    total_vram_mb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 2)

    for D in microbatches_count:
        for N in n_tokens:
            try:
                prompts = long_prompts[:D]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                sample_len = tokenizer(prompts[0], return_tensors="pt").input_ids.shape[1]
                budget = int((sample_len + N) * budget_ratio)
                budget = max(budget, sample_len)
                recent_window = int(budget * recent_ratio)

                total_start = time.perf_counter()

                with torch.no_grad():
                    next_tokens = []
                    cpu_kv_caches = []
                    cum_scores_list = []

                    # PREFILL
                    for i in range(D):
                        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
                        outputs = model(
                            input_ids=inputs.input_ids,
                            use_cache=True,
                            output_attentions=True,
                        )
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)
                        next_tokens.append(next_token)

                        seq_len = outputs.past_key_values.get_seq_length()
                        scores = torch.zeros(seq_len, device=device)
                        for attn in outputs.attentions:
                            scores += attn[0].sum(dim=0).sum(dim=0)
                        cum_scores_list.append(scores)

                        cpu_kv_caches.append(offload_kv_to_cpu_pinned(outputs.past_key_values))

                    prefetch_stream = torch.cuda.Stream(device=device)
                    offload_stream = torch.cuda.Stream(device=device)

                    # Prefetch first microbatch
                    with torch.cuda.stream(prefetch_stream):
                        prefetched = tuple(
                            (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                            for k, v in cpu_kv_caches[0]
                        )

                    # DECODE
                    for step in range(N - 1):
                        for i in range(D):
                            next_i = (i + 1) % D

                            prefetch_stream.synchronize()
                            offload_stream.synchronize()

                            gpu_layers = prefetched

                            # Prefetch next microbatch
                            with torch.cuda.stream(prefetch_stream):
                                prefetched = tuple(
                                    (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                                    for k, v in cpu_kv_caches[next_i]
                                )

                            # Build cache and decode
                            gpu_cache = build_kv_cache(gpu_layers)
                            outputs = model(
                                input_ids=next_tokens[i],
                                past_key_values=gpu_cache,
                                use_cache=True,
                                output_attentions=True,
                            )
                            next_tokens[i] = torch.argmax(outputs.logits[:, -1, :], dim=1).unsqueeze(1)

                            # Update cumulative scores
                            kv = outputs.past_key_values
                            current_len = kv.get_seq_length()
                            new_score = torch.zeros(1, device=device)
                            cum_scores_list[i] = torch.cat([cum_scores_list[i], new_score])
                            for attn in outputs.attentions:
                                step_scores = attn[0].sum(dim=0).sum(dim=0)
                                if len(step_scores) == len(cum_scores_list[i]):
                                    cum_scores_list[i] += step_scores

                            # H2O eviction
                            while current_len > budget:
                                evict_candidates = cum_scores_list[i][:-recent_window].clone()
                                if len(evict_candidates) == 0:
                                    break

                                evict_pos = evict_candidates.argmin().item()

                                keep_mask = torch.ones(current_len, dtype=torch.bool, device=device)
                                keep_mask[evict_pos] = False
                                keep_indices = torch.where(keep_mask)[0]

                                kv_layers = unpack_kv_cache(kv)
                                kept_layers = [(k[:, :, keep_indices, :], v[:, :, keep_indices, :]) for k, v in kv_layers]
                                kv = build_kv_cache(kept_layers)

                                cum_scores_list[i] = cum_scores_list[i][keep_mask]
                                current_len -= 1

                            # Offload evicted cache to pinned CPU
                            with torch.cuda.stream(offload_stream):
                                cpu_kv_caches[i] = tuple(
                                    (k.detach().contiguous().cpu().pin_memory(),
                                     v.detach().contiguous().cpu().pin_memory())
                                    for k, v in unpack_kv_cache(kv)
                                )

                            del gpu_layers, gpu_cache, outputs, kv

                total_end = time.perf_counter()

                latency = total_end - total_start
                throughput = (D * N) / latency
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                vram_util = (peak_vram_mb / total_vram_mb) * 100

                cpu_kv_mb = sum(
                    k.nelement() * k.element_size() + v.nelement() * v.element_size()
                    for cache in cpu_kv_caches for k, v in cache
                ) / (1024 ** 2)

                if rank == 0:
                    print(f"D={D:<4} N={N:<4} | Budget={budget:<4} | Latency={latency:.3f}s | Throughput={throughput:.1f} tok/s | "
                          f"Peak VRAM={peak_vram_mb:.1f}MB | VRAM%={vram_util:.1f}% | CPU KV={cpu_kv_mb:.1f}MB")

                results.append({
                    "Microbatches (D)": D,
                    "Tokens Generated (N)": N,
                    "Budget": budget,
                    "Latency (s)": latency,
                    "Throughput (tok/s)": throughput,
                    "Peak VRAM (MB)": peak_vram_mb,
                    "VRAM Util (%)": vram_util,
                    "CPU KV (MB)": cpu_kv_mb,
                })

                del cpu_kv_caches, next_tokens, cum_scores_list, prefetched
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"D={D:<4} N={N:<4} | OOM")
                    results.append({
                        "Microbatches (D)": D,
                        "Tokens Generated (N)": N,
                        "Budget": budget if 'budget' in dir() else "N/A",
                        "Latency (s)": "OOM",
                        "Throughput (tok/s)": "OOM",
                        "Peak VRAM (MB)": "OOM",
                        "VRAM Util (%)": "OOM",
                        "CPU KV (MB)": "OOM",
                    })
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

    return pd.DataFrame(results)


# ═════════════════════════════════════════════
# RUN ALL BENCHMARKS
# ═════════════════════════════════════════════

if __name__ == "__main__":

    # Quick sanity check: generate a few tokens
    if rank == 0:
        print("\n" + "="*60)
        print("SANITY CHECK: generating text")
        print("="*60)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    if rank == 0:
        print(f"Prompt: {prompt}")
        print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Check KV cache shape
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model(input_ids=inputs.input_ids, use_cache=True)
        kv = out.past_key_values
        layer0_k = kv.key_cache[0] if hasattr(kv, 'key_cache') else kv[0][0]
        if rank == 0:
            print(f"KV cache shape: {layer0_k.shape}  (expect 16 heads with TP=2)")
    del out, kv
    torch.cuda.empty_cache()

    dist.barrier()

    # --- Run benchmarks ---

    if rank == 0:
        print("\n" + "="*60)
        print("SYSTEM 1: BASELINE (GPU-only, TP=2)")
        print("="*60)
    dist.barrier()
    df_baseline = run_baseline_benchmark()

    if rank == 0:
        print("\n" + "="*60)
        print("SYSTEM 2: DEJAVU (CPU offload, TP=2)")
        print("="*60)
    dist.barrier()
    df_dejavu = run_dejavu_benchmark()

    if rank == 0:
        print("\n" + "="*60)
        print("SYSTEM 3: H2O (eviction, TP=2)")
        print("="*60)
    dist.barrier()
    df_h2o = run_h2o_benchmark()

    if rank == 0:
        print("\n" + "="*60)
        print("SYSTEM 4: DEJAVU + PREFETCH (TP=2)")
        print("="*60)
    dist.barrier()
    df_prefetch = run_dejavu_prefetch_benchmark()

    if rank == 0:
        print("\n" + "="*60)
        print("SYSTEM 5: HYBRID H2O + DEJAVU + PREFETCH (TP=2)")
        print("="*60)
    dist.barrier()
    df_hybrid = run_h2o_dejavu_prefetch_benchmark()

    # Save results (rank 0 only)
    if rank == 0:
        df_baseline.to_csv("results_tp2_baseline.csv", index=False)
        df_dejavu.to_csv("results_tp2_dejavu.csv", index=False)
        df_h2o.to_csv("results_tp2_h2o.csv", index=False)
        df_prefetch.to_csv("results_tp2_prefetch.csv", index=False)
        df_hybrid.to_csv("results_tp2_hybrid.csv", index=False)
        print("\n" + "="*60)
        print("ALL DONE. Results saved to CSV files.")
        print("="*60)

    dist.barrier()
    dist.destroy_process_group()
