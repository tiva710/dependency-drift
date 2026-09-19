# Dependency Drift in Open-Source ML Models on Hugging Face

Measures how often Hugging Face model repositories reference a base model,
dataset, or library version that has since become invalid or stale, and what
forms that dependency drift takes.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml data-collection/metadata-collection/config.yaml
# then add your HF key
```
