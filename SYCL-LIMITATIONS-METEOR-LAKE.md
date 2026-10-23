# SYCL limitations on Meteor Lake (ThinkPad T14 Gen 5, Core Ultra 7 165U)

Notes from profiling and optimization work on the Intel iGPU SYCL backend with
Ternary-Bonsai-2 27B PTQ1_0. Numbers are from this machine (build `build-rebase`,
oneAPI 2026.1, compute-runtime 1.17.39758 / IGC 2.41.9). Re-measure on other
hardware before assuming any of this applies there.

## Hardware facts (measured)

- Memory: socketed DDR5-5600 SO-DIMM (2/2 slots, 2x16 GB), 89.6 GB/s theoretical.
  Not LPDDR5X: the T14 Gen 5 has SO-DIMM slots. EDAC sysfs reports the type as
  "Unbuffered-DDR3" (bogus label on this platform); trust slot count and sizes.
- Sustained GPU bandwidth from device memory: ~74 GB/s read, ~67 GB/s mixed
  read+write, ~70 GB/s write. Host<->device copies over shared memory: ~26 GB/s.
  A 10-core CPU stream test saturates at the same DRAM roof.
- GPU: Xe-LPG, 4 Xe-cores, no XMX (`dpas_exec_size` is 0, `has_dpas16` false).
  Register file is 128 GRFs; at SIMD16 that is ~256 bytes (64 floats) per
  work-item. Kernels needing more spill to scratch memory.
- Thermal variance on sustained runs is about +/-20 percent. Always benchmark
  with `-n 200+`, cooldowns of 45-55 s between runs, and interleaved A/B.

## Rooflines (why the remaining headroom is small)

- Decode (batch 1): 6.5 GB of traffic per token (5.9 GB PTQ1_0 weights + ~0.6 GB
  GDN state read+write). At the 67 GB/s mixed roof the ceiling is ~10.3 t/s.
  Measured 8.67 t/s = 84 percent of the roofline. The mat-vec kernel itself runs
  at 76-80 percent of the pure read roof; sweeping `nr` (2,3,4,6) and
  `GGML_SYCL_MMV_Y` (1,2,4) found the shipped config (nr=4, MMV_Y=1) optimal.
- Prefill: compute bound. The MKL int8 GEMMs reach an effective 6-7 TFLOPS,
  close to the ALU limit without XMX. pp512 ~86 t/s, of which ~74 percent is
  the w8a8 GEMM path (not improvable in-tree).

## SYCL graph capture: not viable on this hardware

- The device reports `ext_oneapi_limited_graph=1` but `ext_oneapi_graph`
  (updatable graphs) = 0. Without updatable graphs, every replayed round must
  `finalize()` the ~1000-kernel graph again, which costs more than the saved
  submit time. Measured with DFlash spec decoding: capture ON 6.5-7.3 t/s vs
  OFF 7.4-8.1 t/s (8-15 percent regression).
- Untraced enqueue cost is ~14-16 us per node (unitrace inflates it 2x).
- Blockers found while experimenting (fixes are small but were reverted):
  - `check_graph_compatibility` rejects hinted Hadamard rotations
    (`GGML_HINT_SRC0_IS_HADAMARD`); safe to relax for F32/F16 contiguous.
  - `scratch_get()` in `ptq1-t2.cpp` waits on pool growth; warmup only.
  - `fattn-mkl.cpp` calls `stream->wait()` before every MKL GEMM: a permanent
    capture blocker for any graph containing MKL flash attention.
- Re-test on Xe2/B580-class parts that report `ext_oneapi_graph` support before
  writing capture off everywhere.

## Flash attention

- `flash_attn_tile<DKQ=256>` (the fallback kernel for head size 256) spills
  ~160 registers: 1815 scratch references in the ISA, ~47 GFLOPS effective,
  91 ms per attention layer in pp512. It is a last resort on this GPU.
- The MKL GEMM path is ~200x faster for prefill-sized batches but had three
  pre-existing bugs that its old `n_kv >= 1024` gate kept hidden (tests max out
  at kv=512):
  1. permuted or head-major padded K/V views SIGSEGV in the dequant
     (a null call inside the oneMKL dispatcher, PC=0);
  2. Q batch planes beyond the first (Q->ne[3] > 1 vs K single batch) give
     wrong results (ERR ~1.3);
  3. only dense rows or packed interleaved heads are handled. The real
     llama.cpp KV cache is packed interleaved (nb[2] == row size,
     nb[1] == row size * n_heads), which is why inference works.
  The gate added in commit "sycl: MKL flash attention for prefill chunks below
  1024 KV tokens" encodes exactly these constraints. If the MKL kernel is ever
  extended to more layouts, re-run the full FLASH_ATTN_EXT suite: the crashes
  are sequence-dependent (the same case passes when run alone).
- MKL flash attention also serializes on the host (`stream->wait()` per GEMM),
  which is why it is only worth it for prefill-sized Q batches, not decode.

## GDN (gated delta net)

- `gdn_blocked_sycl` with NC=8 spills ~162 registers at SIMD16 on the 128-GRF
  file. The B580 auto-heuristic assumed Xe3's larger register file; the no-XMX
  branch added in commit "sycl: GDN blocked-kernel variant for GPUs without
  XMX" picks NC=4/T=8 instead (85.9 vs 79.2 t/s pp512). Variants:
  `GGML_SYCL_GDN_BLK_VARIANT` (1-10), disable with `GGML_SYCL_GDN_BLOCKED_OFF`.
- The blocked kernel only runs for n_tokens >= `GGML_SYCL_GDN_BLK_MIN` (16):
  decode always uses the scalar kernel; variant changes do not affect tg.

## w8a8 prefill matmuls

- The w8a8 path re-converts all weights to int8 on every prefill pass: ~27 GB
  written + 2x 5.9 GB read (~600 ms of a 6 s pp512 pass). The int8 copy is too
  large to cache (27 GB vs 32 GB system RAM). This is still the best option:
  GemmMT and the s4 oneDNN GEMM both measure 32 t/s pp512 vs 87 for w8a8.
- Do not "fix" the conversion cost by switching GEMM backends on this machine.

## Speculative decoding (DFlash draft)

- Draft acceptance is content dependent: 0.74-0.80 (mean len 4.1) on simple
  reasoning tasks vs 0.29-0.37 (mean len 2.2-2.5) on long reasoning plus
  structured code. Break-even vs plain decode is mean len ~3.3 at the current
  ~410 ms round cost, so the draft model is a net loss on code-heavy prompts
  until PrismML trains a better one.
- The GDN model has no partial rollback (`n_rs_seq` clamps to 0), so spec
  decoding takes 166 MB checkpoints; these fire only ~2 times per 15 rounds
  and are not a per-round cost.

## The w8a8 int8 GEMM is lossy for PTQ1_0 (fixed by chunking)

- The generic w8a8 path (and the s4 GEMM) folds the f16 group scales into the
  int8 weights as `round(127 * d_g / D)` (D = row max). For Bonsai-2's PTQ1_0
  weights the within-row scale dynamic range is wide, so small-scale groups get
  crushed: measured 3-170 percent dot errors on ~10 percent of outputs. Every
  kernel in the path is bit-exact against its own definition - the deviation is
  the approximation itself, not a memory bug.
- Engagement cliff: matmuls with more than PTQ1_0_ACT_MAX_COLS (256) columns
  that are not T2-repacked (the GDN/attention projections; the FFN is T2 by
  name) fall off the MMVQ act kernels into w8a8. Prompts processed in chunks
  over 256 tokens produced garbage; perplexity 248320 vs 4.88 on CPU. Prompts
  at or under 256 tokens were fine. This predates the rebase: the original
  llama-ui runs with 381-token prompts were already past the cliff.
- Fix (commit "sycl: chunk PTQ1_0 prefill matmuls through the MMVQ act
  kernels"): PTQ1_0 with act-eligible weights always chunks through the MMVQ
  act kernels. Correct, but prefill drops from 87 t/s (wrong numbers) to
  ~28.8 t/s.
- The fast exact path was attempted and failed: oneDNN `set_scales` with
  per-group f32 scales is numerically exact in a standalone repro (errors at
  accumulation-order noise level, works with negative and zero scales), but in
  the real pipeline it kills the device (UR_RESULT_ERROR_DEVICE_LOST) around
  the m=17408 call, on both VMM and legacy pools, with private and shared
  scratchpads. Root cause unknown; revisit on a newer oneDNN.
- PQ2_0 models use the same lossy ratio scheme and were left on the old path
  (no model on hand to validate a change).
- Follow-up option: wider act-kernel column tiles (the switch in
  `mul_mat_vec_ptq1_0_q8_1_act_sycl` stops at 8 columns; the weight re-reads
  per 8-column chunk are what limits the chunked path to ~29 t/s).

## Pre-existing prism bugs found, not fixed (report upstream)

- MUL_MAT_ID (k = 3*blck) intermittent NaN race (q4_0/q4_1/q5_0/q5_1/q8_0).
- MUL_MAT_HADAMARD (blk=1024, f16, swiglu=1): error 2.2e-7 vs 1e-7 tolerance.
- SET_ROWS into PTQ1_0/PQ2_0 aborts.

## Profiling recipes on this machine

- Op-level GPU time: `GGML_SYCL_OP_TIMING=1` (draft-sized graphs only; the
  node/row caps in `sycl_op_timing::draft_sized` exclude the target decode
  graph - patch them temporarily to profile it, then revert).
- Kernel timeline: `/home/tglozar/dev/pti-gpu/tools/unitrace/unitrace
  --chrome-kernel-logging -o out.json <cmd>` (writes `out.json` into the
  process cwd if -o is not given; do not pass `--`).
- IGC assembly dumps: `NEO_CACHE_MAX_SIZE=0 IGC_ShaderDumpEnableAll=1
  IGC_DumpToCurrentDir=1` in an empty dir (the Level Zero disk cache
  otherwise skips IGC entirely). Find the kernel's `.spv` by `strings`, then
  `ocloc compile -file X.spv -device mtl-p -spirv_input -out_dir out` and
  `ocloc disasm -file out/X_mtl.bin -dump d -device mtl-p`. Spill warnings
  print during compile ("allocated N regs and spilled around M"). Raw ISA:
  extract the `.text.*` ELF section and `iga64 -p 12p71 -d` (Xe-LPG shares
  the XeHPG ISA).
- Backtraces without gdb: systemd-coredump + `coredumpctl dump` + `eu-stack
  --core`. oneMKL crashes often show PC=0 (null dispatch entry).
- `GGML_LOG_INFO` from ggml-sycl is filtered by the llama-cli/server log
  handlers: use `fprintf(stderr, ...)` for temporary debug prints.
- Bandwidth microbenchmarks: `/tmp/bw` (SYCL) and `/tmp/bw/cpubw` (CPU).
  Watch for dead-code elimination in read benchmarks (consume every loaded
  element) and for latency-bound CPU loops (unroll with many accumulators).

## Environment quirks

- Source `/opt/intel/oneapi/setvars.sh` in every shell; missing it produces
  confusing build errors.
- `taskset -c 0-3` for all runs (4 cores reserved for inference).
- The shell is Czech-locale: `pkill -f llama-cli` kills the invoking bash
  itself. Use `pkill -x llama-cli`.
- Untracked dump directories (`igc/`, `igc-dump/`, `metrics_*`, `trace*`,
  `vt_ic_*`, `llms/`) in the repo root are artifacts from this work; the
  useful IGC dumps are in `/tmp/igcdump2`.
