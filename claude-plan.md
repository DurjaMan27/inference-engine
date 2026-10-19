# Steerable Inference Engine — 6-Week Plan

Oct 18, 2026 · @Varun Rao

## Overview

Build a C++/CUDA inference engine for Qwen3-0.6B that can take guidance mid-reasoning, tune the model to follow it, and serve it behind an agent. Done by about December 1 at 10 hrs/week, in time for January applications.

- **Weeks 1–4 (systems):** forward pass, KV cache, CUDA kernels, steering. This closes the C++/GPU/inference gap.
- **Weeks 5–6 (keywords):** LoRA post-training on Ray, then an OpenAI-compatible server with tool calling and a supervisor agent. This makes it legible to AI engineer screens.
- **GPU:** RunPod only, in weeks 3, 5 and 6. Everything else runs on your laptop.

Write the forward pass, decode loop and kernels yourself, even if AI tools help with boilerplate. Those are the parts interviewers will make you explain line by line.

## Week 1 — CPU forward pass

**Done when:** your C++ logits for a fixed prompt match Hugging Face to about 1e-3.

- Download Qwen3-0.6B and read its `config.json`. Note the GQA head counts, head dim, RoPE theta, tied embeddings, and QK-norm (RMSNorm on q and k per head). QK-norm is the main difference from Llama, and the easiest thing to miss.
- Write a safetensors loader: an 8-byte header length, a JSON header, then raw tensors. Convert bf16 to fp32 on load for the CPU path.
- Implement embedding, RMSNorm, RoPE, GQA attention (full sequence, causal mask), SwiGLU MLP, and the final norm and LM head. Use llama2.c as a structural reference only.
- Tokenize in Python with the HF tokenizer and pass token IDs to the engine. Don't write BPE.
- Write a Python script that dumps per-layer activations from HF, so you can compare layer by layer when the final logits disagree.

**Keywords:** C++20, transformer architecture, GQA, RoPE, RMSNorm, numerical validation

## Week 2 — KV cache and decode loop

**Done when:** greedy decoding matches HF `generate()` token for token for 200 tokens, and you have a CPU tokens/sec number.

- Split the forward pass into **prefill** (many tokens, fills the cache) and **decode** (one token, reads the cache). Week 4 depends on prefill working for any chunk length at any position.
- Preallocate the KV cache as \[layer\]\[kv\_head\]\[position\]\[head\_dim\], and track the current length explicitly.
- Add a sampler: greedy, temperature, top-p, and a seed for reproducibility.
- Apply the Qwen3 chat template, including the `<think>` tags, so the model actually reasons.
- Parallelize the matmuls with OpenMP and record the speedup.
- Set up CMake, a `tests/` folder (GoogleTest or plain asserts), and a CI workflow that runs the parity test on a tiny slice.

**Keywords:** KV cache, prefill vs. decode, sampling, OpenMP, CMake, unit tests, CI

## Week 3 — CUDA on RunPod

**Done when:** the GPU path passes the same parity test, and you have tokens/sec and time-to-first-token compared against HF transformers on the same GPU, plus an Nsight breakdown of where decode time goes.

- Rent a single RTX 4090 or similar. Keep weights and your build on a network volume, and stop the pod whenever you step away.
- Write the code on your laptop, and use pod time only to compile, test and profile.
- Use cuBLAS for the GEMMs, in fp16 or bf16.
- Write your own kernels for RMSNorm (a warp-level reduction), RoPE, SwiGLU, and decode attention: one query over the cache, with an online-softmax running max.
- Profile with Nsight Systems, then optimize the biggest non-GEMM cost, usually kernel launch overhead or attention. Record before and after.
- Write a short paragraph explaining why decode is memory-bandwidth-bound. Interviewers will ask.

**Keywords:** CUDA, cuBLAS, custom kernels, warp reductions, mixed precision, Nsight Systems, profiling, throughput, TTFT

## Week 4 — Mid-generation steering

**Done when:** `inject(text)` changes a live generation, and you have a latency table (inject vs. full re-prefill) and a baseline compliance rate for the untuned model.

- Run the decode loop on its own thread. A controller pushes guidance into a lock-protected queue, and the loop checks it at each token boundary.
- On inject: wrap the text in a fixed marker (for example `\n[Guidance: …]\n`), tokenize it, prefill just those k tokens at the next positions, and resume sampling. No restart and no recompute.
- Support two timing policies, "next token" and "next sentence boundary", and compare how coherent the output stays under each.
- Benchmark inject vs. re-prefilling the whole context at 512, 1k, 2k and 4k tokens. That table is your headline number.
- Build a 100-prompt compliance eval: math or logic problems with a scripted injection ("use the second method," "check the units"), auto-graded on whether the answer follows it. Expect a low score. That's the setup for week 5.

**Keywords:** inference-time steering, incremental prefill, concurrency, latency benchmarking, evals

## Week 5 — Post-training on Ray

**Done when:** compliance rises from the week 4 baseline, a standard benchmark doesn't regress, and the merged LoRA weights run in your own C++ engine.

- **Synthetic data:** use a larger model (an API model, or Qwen3-8B on the pod) to write 1–2k reasoning traces where guidance appears mid-thought and the continuation adopts it. Filter with an automatic check that the final answer follows the guidance.
- **Loss masking:** compute loss only on the model's own continuation, never on the prompt or the injected span. The model learns to respond to guidance, not to produce it.
- **Training:** LoRA SFT with TRL and PEFT, wrapped in **Ray Train** across 2 GPUs (data-parallel), with metrics logged to Weights & Biases. This is the Ray/distributed gap Tower, Cohere and Millennium named, at small scale.
- **Eval:** rerun the week 4 compliance suite, plus a GSM8K subset to check general reasoning didn't drop.
- **Close the loop:** merge the adapter, export to safetensors, and load it in your engine. Train in Python, serve in C++.

**Keywords:** LoRA, PEFT, SFT, TRL, synthetic data generation, loss masking, Ray Train, distributed data-parallel, W&B, evaluation

## Week 6 — Serving, tool calling and a supervisor agent

**Done when:** any OpenAI-client script can talk to your engine, a supervisor agent steers it live, and the README has numbers and a 60-second demo video.

- **Bindings:** expose the engine to Python with pybind11 (generate, stream, inject).
- **Server:** FastAPI with an OpenAI-compatible `/v1/chat/completions` that streams over SSE, plus `/v1/sessions/{id}/inject`. Being OpenAI-compatible means existing SDKs work unchanged.
- **Tool calling:** pass JSON-schema tool definitions, parse Qwen3's tool-call output, validate it against the schema, and retry on failure. Report the malformed-call rate.
- **Supervisor agent:** a second model reads the worker's streaming thoughts and has one tool, `steer(text)`. Compare task success on the week 4 problems with and without the supervisor, and count how often it intervenes.
- **Ship it:** a Dockerfile, a README leading with the latency, compliance and success tables, and the demo video.
- **Stretch:** JSON-schema-constrained decoding by masking invalid tokens in your C++ sampler, which drives malformed tool calls to zero.

**Keywords:** pybind11, FastAPI, OpenAI-compatible API, SSE streaming, function/tool calling, structured outputs, agents, multi-agent supervision, evals, Docker

## Resume entry, budget and cuts

Target bullets to replace TreeTrader. Uncomment them only once the numbers are real, and fill in X, Y, A and B from your own results.

> **Steerable LLM Inference Engine** | C++, CUDA, Python, Ray
>
> - Built a C++/CUDA inference engine for Qwen3-0.6B (custom RMSNorm, RoPE and attention kernels, KV cache, cuBLAS GEMMs), validated against Hugging Face logits and profiled with Nsight; added mid-generation steering via incremental KV-cache prefill (X ms vs. Y ms full re-prefill at 4k context)
> - LoRA-tuned the model with Ray Train on synthetic steering data (compliance A% → B%, no GSM8K regression) and served it behind an OpenAI-compatible streaming API with tool calling and a supervisor agent that steers reasoning in real time

**GPU budget:** roughly 10 hrs in week 3, 8 hrs on 2 GPUs in week 5, and a few hours in week 6. At typical RunPod rates that's about $30–60 total, but check current prices before you rent.

**If a week slips:** drop the Docker and stretch items first, then the sentence-boundary policy. Never drop the parity tests or the benchmark tables, because the numbers are what make the resume bullet credible.

**Cut on purpose:** writing your own BPE, int8 quantization, paged KV cache, batching across requests, pretraining, MCP, and multi-agent orchestration beyond the supervisor.
