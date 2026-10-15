# CascadeSnap stage snapshot (thesis-stage1-20261015)

This is a stage snapshot of the CascadeSnap code (layer-group pipelined checkpointing on top of PCcheck).

| Path | Content |
|---|---|
| `checkpoint_eval/pccheck/multistream_checkpoint.py` | CascadeSave: grouped update / save scheduling, D2H pipeline, int8 moments, error latch |
| `checkpoint_eval/pccheck/main_ssd_memory.cpp` | Storage engine: O_DIRECT + native AIO, slot publish (CAS), `abort_checkpoint` |
| `checkpoint_eval/pccheck/multistream_pipeline_utils.py` | DeepSpeed PP glue (zero-copy stage checkpoint) |
| `cascadesnap/transformers_patches/` | HF-transformers integration as a `git format-patch` series against upstream `ee88ae5994` (trainers, PP runners, async DCP baseline) |
| `cascadesnap/tests/` | Regression and benchmark scripts |
| `cascadesnap/docs/P0_FIXES_20261014.md` | What the 2026-10-14 correctness fixes changed and how they were verified |

The CascadeLoad side (restore pipeline) lives in the `pipelayer` repository, on a branch with the same name.

## Build

```bash
cd checkpoint_eval/pccheck && make libtest_ssd.so     # needs libaio, libpmem, CUDA
```

## Apply the transformers patches

```bash
git clone https://github.com/huggingface/transformers && cd transformers
git checkout ee88ae5994 && git am /path/to/cascadesnap/transformers_patches/*.patch
```

## Regression suite

```bash
python cascadesnap/tests/test_p0_fixes.py --repo <this repo> --ckdir <dir on the target SSD>
```

Each test runs in its own process. The C++ library keeps its slot queue in process-wide statics, so the suite must not create several writers in one process.

| Test | Checks |
|---|---|
| T1 | Grouped update matches `optimizer.step()` bit for bit (torch Adam/AdamW foreach/fused/for-loop, SGD, DeepSpeed FusedAdam; two param groups) |
| T2 | sqrt-domain int8 quantization never produces a zero code, and the update magnitude stays bounded |
| T3 | PP zero-copy path writes fp32 / int8 checkpoint content that is bit-exact with the expected values |
| T4 | Write-after-read guard holds under a delayed copy thread (the old guard is shown to tear) |
| T5 | Injected write failures: nothing is published, no slot leaks, and the next save is correct |
| T6 | Single-GPU path with fused AdamW and two groups: update and file content are both bit-exact |

`bench_load_roundtrip.py --phase {save,load} --config {fp32,quant}_{split,atomic}` runs the save→restore roundtrip in a fresh process. It needs the pipelayer repository. Paths inside the bench scripts are the original workstation defaults; override them with the CLI flags.
