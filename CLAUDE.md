# Traffic Sign Recognition — ViT Project

## Project

**Traffic Sign Recognition with Vision Transformer**

Fine-tune ViT (`google/vit-base-patch16-224`) on the German Traffic Sign Recognition Benchmark (GTSRB). Benchmark against a ResNet50 baseline. Visualize attention maps for interpretability. Deploy via FastAPI and Hugging Face Spaces.

---

## Hardware constraints

- GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM
- Native Windows (NOT WSL)
- Python 3.11.1 in `.venv` (uv-managed)
- Stack: torch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 5.12.0, datasets 5.0.0

VRAM-aware decisions: use mixed precision (fp16/bf16), batch size cap ~32 for ViT-base, gradient checkpointing if VRAM gets tight.

---

## Stack (locked — do not upgrade mid-project)

Python 3.11.1 | torch 2.5.1+cu121 | torchvision 0.20.1+cu121 | transformers 5.12.0 | datasets 5.0.0 | FastAPI | uv

---

## Project structure

```
traffic_sign_vit/
├── CLAUDE.md                  # this file
├── README.md                  # user-facing project doc
├── docs/
│   └── DECISIONS.md           # handoff + decision log (updated after every step)
├── requirements.txt
├── .gitignore
├── data/                      # data download, dataset class, transforms
├── models/                    # ViT + ResNet50 model definitions
├── training/                  # fine-tuning scripts, train loop
├── evaluation/                # metrics, attention visualization
├── api/                       # FastAPI inference endpoint
├── demo/                      # HF Spaces app
└── notebooks/                 # exploration scripts (plain .py preferred over .ipynb)
```

---

## Hard rules — enforced every session, no exceptions

1. Work block by block. Do not jump ahead. The user signals "next block" or "approved, move on."
2. For each major decision, give a 2-3 line WHY. Then wait for approval before generating files.
3. Mark assumptions explicitly with `ASSUMPTION:` prefix.
4. No emojis anywhere. Keyboard punctuation only — no em-dashes, en-dashes, smart quotes, ellipsis characters.
5. Production-quality code: type hints, docstrings, error handling. But no over-engineering.
6. If a step seems too complex for the goal, propose a simpler version and ask before implementing.
7. Do not delete files. Do not modify the venv. Do not edit `requirements.txt` without user approval.
8. When a script is run-and-monitor (training), output progress with `tqdm` or clear logging, not silent.
9. Maintain `docs/DECISIONS.md` after every completed step without being asked. Update it with: what was built, why the decision was made, what was assumed, and what the next step is. It must be self-contained — an AI with zero context must be able to read it and pick up exactly where the session left off.
10. If in doubt, ask. Get it clarified before proceeding. No ambiguous steps.
11. If a decision is architecturally significant and uncertain, flag it explicitly and ask the user before implementing. Do not make a wrong or vague decision to avoid asking.
12. Every step taken must be logically sound and based on full knowledge of the current state.
13. Code must be debug-friendly. If anything breaks, the user must be able to locate and fix it without AI assistance.
14. All code must be modular and follow standard software engineering practices.
15. No bare `except` clauses. Always catch specific exceptions.

---

## Output discipline — enforced every response, no exceptions

- No preamble. Start with the answer or action directly.
- No filler: "I'll now...", "Let me...", "Great question", "Sure!"
- No restating the task before doing it.
- No summary at the end repeating what was just done.
- Explain only when explanation adds information not already obvious from context.
- Every sentence must earn its place. If removing it loses nothing, remove it.

---

## Coding conventions

- Imports grouped: stdlib, third-party, local. Sorted alphabetically within group.
- Type hints on all function signatures.
- Docstrings: Google or NumPy style, brief but present.
- Logging via Python `logging` module, not `print` for anything important.
- Random seeds set explicitly for reproducibility. Default seed: 42.
- Configurable parameters live in a config dataclass or YAML, not hardcoded in functions.
- No bare `except` clauses — always catch specific exceptions.

---

## Environment setup

```bash
uv sync
.venv\Scripts\activate          # Windows (native)
```

---

## Verification commands — run after every change

```bash
python -m pytest tests/ -v
python -m mypy src/ --ignore-missing-imports
python -m ruff check src/
```

---

## Known gotchas

- **VRAM OOM:** RTX 3060 6GB fills fast with ViT-base. If OOM: reduce batch size first, then enable `model.gradient_checkpointing_enable()`, then switch to fp16.
- **HuggingFace cache:** Models download to `~/.cache/huggingface/`. Set `HF_HOME` env var if disk space is a concern.
- **GTSRB dataset:** HuggingFace `datasets` version may affect split structure. Always verify `train`/`test` keys before writing the Dataset class.
- **torch.amp API:** Use `torch.amp.autocast("cuda")` (torch 2.x API) — not the deprecated `torch.cuda.amp.autocast()`.
- **transformers ViT normalization:** `ViTForImageClassification` expects pixel values in `[0, 1]` normalized with ImageNet mean/std. Do not double-normalize.

---

## Out of scope

- Multi-GPU training
- Distillation, quantization, or model compression
- ONNX / TensorRT export
- TPU / cloud GPU training
- Model serving frameworks beyond FastAPI (no Triton, no vLLM, etc.)
- Custom CUDA kernels

---

## Build blocks (in order)

1. Data pipeline + EDA (download, Dataset class, transforms, smoke test)
2. ViT fine-tuning script
3. Run training, save checkpoint
4. ResNet50 baseline (mirrors block 2-3)
5. Evaluation: confusion matrix, per-class accuracy, comparison table, latency benchmark
6. Attention map visualization (for ViT only)
7. FastAPI inference endpoint
8. Hugging Face Spaces demo
9. README final pass

The user will signal which block to start. Do not assume.
