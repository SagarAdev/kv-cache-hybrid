# Hybrid KV Cache Eviction + Offloading for Memory-Constrained LLM Inference

CS2510 (Operating Systems), Spring 2026, University of Pittsburgh.
Sagar Arun Kumar, Dharma Swaroop, Fareed Durgam.

Full write-up: [`report/OS_finalProject_report.pdf`](report/OS_finalProject_report.pdf)

## Summary

The KV cache grows with batch size and sequence length and is the main memory bottleneck in LLM decoding. H2O-style eviction and DejaVu-style CPU offloading are usually treated as alternatives. This project composes them: H2O evicts low-attention tokens first, then the smaller cache is offloaded to CPU and prefetched back with CUDA streams and pinned memory.

Five systems are benchmarked on OPT-6.7B (WikiText-2 prompts, D in {8, 16, 32} sequences, N in {256, 512} generated tokens):

1. Baseline (everything on GPU)
2. DejaVu (synchronous CPU offload)
3. H2O (GPU-only eviction)
4. DejaVu + Prefetch (CUDA streams + pinned memory)
5. Hybrid (H2O + DejaVu + Prefetch)

## Results

**Single GPU** (A100, memory capped at 18 GB)

| System | Throughput (tok/s) | OOM |
|---|---|---|
| Baseline | ~37 | D=16, N=512 and beyond |
| H2O (GPU-only) | ~31 | large configs |
| DejaVu | 7.6 - 9.6 | never |
| DejaVu + Prefetch | 16.1 - 18.7 (1.9x over DejaVu) | never |
| Hybrid | 12.8 - 14.6 | never |

- Vanilla DejaVu spends 67-73% of time on CPU<->GPU transfers.
- The hybrid uses 62% less CPU KV memory than the offload-only systems (4.7 GB vs 12.4 GB at D=32, N=512).
- On a single GPU the hybrid is slower than DejaVu + Prefetch, because of the cost of attention score tracking and the eviction loop.

**Tensor parallel, TP=2** (2x RTX A5000, 24 GB each, manual implementation)

| System | Throughput (tok/s) |
|---|---|
| Baseline | ~50 |
| DejaVu + Prefetch | 17.5 - 20.7 |
| Hybrid | 22.7 - 25.3 |

The ordering reverses at TP=2: the hybrid is faster than DejaVu + Prefetch (e.g. 23.1 vs 18.0 tok/s at D=8, N=512).

Ordering of the systems was also checked on Mistral-7B and Llama-3.1-8B (report Section 5.4).

## Repo layout

```
report/                          project report (PDF)
src/tp_inference.py              TP=2 implementation, all five systems
notebooks/
  phase1_gpu_baseline.ipynb      early GPU baseline
  phase2_dejavu_final.ipynb      early DejaVu simulation
  01_baseline_and_h2o.ipynb      baseline + H2O
  02_dejavu.ipynb                vanilla DejaVu
  03_dejavu_prefetch_pinned.ipynb  prefetch with/without pinned memory, profiling
  04_hybrid_h2o_dejavu.ipynb     hybrid pipeline
  05_h2o_validation_perplexity.ipynb  H2O output validation and perplexity
  06_figures.ipynb               report plots
  experiments/hotcold_rescore.ipynb   hot/cold rescoring experiment (not in report)
  archive/                       original, unsplit notebooks
```

The numbered notebooks were split from `archive/OS_MAIN.ipynb` and `archive/OS_VISUALIZATIONS.ipynb` without changing any cell. Saved outputs are from the original runs.

## Running

Single GPU: open the numbered notebooks on a CUDA GPU (A100 used here). The notebooks cap memory with `torch.cuda.set_per_process_memory_fraction(0.45)` to simulate an 18 GB card. Models are downloaded from Hugging Face (`facebook/opt-6.7b`).

Several cells use `LONG_PROMPTS`, while the dataset cell builds `long_prompts`. Set `LONG_PROMPTS = long_prompts` after the dataset cell before running.

TP=2 (needs 2 GPUs, NCCL):

```bash
torchrun --nproc_per_node=2 src/tp_inference.py
```

Tested with Transformers 4.50.3. The `DynamicCache` internals (`key_cache`/`value_cache` vs `.layers`) differ between Transformers versions, so check your installed version.

## Limitations

- The TP=2 pipeline was not profiled. The report does not break down eviction, transfer, allreduce and compute time, so the explanation for the reversal is a mechanistic argument, not a measurement.
- Only attention is tensor-parallel. FFN, embeddings and `lm_head` are replicated, so weights are not distributed across GPUs.
- The "DejaVu" system here is vanilla CPU offload. The paper's design also includes prefetch, pinned memory and swap-aware scheduling, which is what the DejaVu + Prefetch system adds.
- H2O scores are summed across all heads and layers, which loses per-head specialization. Perplexity was used as a fluency check and does not show task accuracy under eviction.
- Not implemented: persistent pinned-memory pool, pre-allocated KV buffers (chunked buffers gave ~30% at chunk size 3 but ran out of memory at high D), transposed buffer layout, KV quantization before offload.

## Future work

Elastic tensor parallelism: scale the TP degree at runtime based on KV cache memory pressure, with the hybrid eviction + offload pipeline as the last fallback.

## References

1. Zhang et al., H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs, NeurIPS 2023.
2. Strati et al., DejaVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving, ICML 2024.
3. Kwon et al., PagedAttention, SOSP 2023.
4. Lee et al., InfiniGen, OSDI 2024.
5. Stojkovic et al., DynamoLLM, HPCA 2025.
