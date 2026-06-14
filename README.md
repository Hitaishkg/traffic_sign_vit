# Traffic Sign Recognition with Vision Transformer

**Status:** In active development. Target completion: Wed Jun 17 2026.

## What this is

A Vision Transformer fine-tuned on the German Traffic Sign Recognition Benchmark (GTSRB), benchmarked against a ResNet50 baseline. Includes attention-map visualization for interpretability and a FastAPI inference endpoint.

## Why ViT vs CNN for this task

Traffic sign recognition has historically been a CNN-dominated benchmark. This project compares modern attention-based architectures against the convolutional baseline on accuracy, latency, and qualitative interpretability — relevant for ADAS systems where understanding model focus matters.

## Stack

- Models: google/vit-base-patch16-224, torchvision ResNet50
- Framework: PyTorch + Hugging Face transformers
- Dataset: GTSRB (43 classes, 50,000+ images)
- Inference: FastAPI
- Demo: Hugging Face Spaces

## Project timeline

- Day 1 (Sat): Data pipeline + EDA
- Day 2 (Sun): ViT + ResNet50 fine-tuning + attention visualization
- Day 3 (Mon): FastAPI inference endpoint
- Day 4 (Tue): HF Spaces deployment + documentation

## Repo structure (planned)

├── data/                  # Dataset loaders + transforms
├── models/                # ViT + ResNet50 configurations
├── training/              # Fine-tuning scripts
├── evaluation/            # Metrics, confusion matrices, attention visualization
├── api/                   # FastAPI inference endpoint
├── demo/                  # HF Spaces app
├── notebooks/             # EDA, training analysis
└── README.md