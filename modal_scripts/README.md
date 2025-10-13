# MTEB Modal Scripts

This directory contains Modal.com scripts for running MTEB (Massive Text Embedding Benchmark) evaluations on ViDoRe benchmarks in the cloud. These scripts allow you to evaluate embedding models at scale using Modal's serverless GPU infrastructure.

## Scripts Overview

### 1. `modal_mteb_run.py`

**Remote MTEB Evaluation from the GitHub Repo**

This script clones the MTEB repository from GitHub and runs evaluations using the remote codebase. It's useful when you want to use the latest version of MTEB from the repository.

**Features:**

- Clones MTEB from `adithya-s-k/mteb` (nayana-bench branch)
- Uses Modal's L40S GPU instances
- Supports both single model and batch evaluation
- Automatic result saving to local JSON files
- Built-in caching for models and datasets

### 2. `modal_mteb_local.py`

**Local MTEB Evaluation with Custom Code**

This script uses your local MTEB directory, allowing you to run evaluations with custom modifications or local changes to the MTEB codebase.

**Features:**

- Uses local MTEB directory (uploaded to Modal container)
- Includes ColPali engine with Gemma3 features
- Supports custom benchmarks and modifications
- Same GPU and caching capabilities as the remote version
- HuggingFace authentication support

## Prerequisites

1. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal setup
   ```
2. **HuggingFace Token** (for `modal_mteb_local.py`): Create a Modal secret named `huggingface-secret`
   ```bash
   modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

### Basic Single Model Evaluation

**Using Remote MTEB:**

```bash
cd mteb/modal_scripts
modal run modal_mteb_run.py --model "vidore/colqwen2.5-v0.2 --benchmarks "ViDoRe(v2)"
```

**Using Local MTEB:**

```bash
cd mteb/modal_scripts
modal run modal_mteb_local.py --model "vidore/colqwen2.5-v0.2 --benchmarks "ViDoRe(v2)"
```

### Custom Benchmarks

Specify specific benchmarks instead of default ViDoRe v1 and v2:

```bash
modal run modal_mteb_local.py --model "vidore/colpali-v1.3" --benchmarks "NayanaIR-Bench"

modal run modal_mteb_local.py --model "vidore/colpali-v1.3" --benchmarks "NayanaIR-Bench"
```

### Batch Evaluation (Multiple Models)

Evaluate multiple models in a single run:

```bash
modal run modal_mteb_local.py \
  --model "vidore/colpali-v1.3,vidore/colqwen2.5-v0.2,google/gemma-2-9b" \
  --batch-mode true
```

### Custom Output Directory

```bash
modal run modal_mteb_local.py \
  --model "your-model-name" \
  --output-folder "./custom_results"
```

### Custom Batch Size

For memory optimization:

```bash
modal run modal_mteb_local.py \
  --model "large-model-name" \
  --batch-size 16
```

### List Available Benchmarks

```bash
modal run modal_mteb_local.py::list_benchmarks
```

## Command Line Options

| Option            | Description                                                                  | Default                    |
| ----------------- | ---------------------------------------------------------------------------- | -------------------------- |
| `--model`         | Model name(s) from HuggingFace Hub. For batch mode, use comma-separated list | `"vidore/colqwen2.5-v0.2"` |
| `--benchmarks`    | Comma-separated benchmark names. Empty for default ViDoRe v1,v2              | `""` (uses ViDoRe v1,v2)   |
| `--output-folder` | Local directory to save results                                              | `"../../results"`          |
| `--batch-mode`    | Enable batch evaluation for multiple models                                  | `false`                    |
| `--batch-size`    | Batch size for model inference                                               | `None` (auto)              |

## Output Format

Results are saved as JSON files in the specified output folder with the naming convention:

- Single model: `local_{model_name}_{timestamp}.json`
- Batch mode: One file per model

### Example Output Structure

```json
{
  "model": "vidore/colpali-v1.3",
  "benchmarks": ["ViDoRe(v1)", "ViDoRe(v2)"],
  "status": "completed",
  "results": {
    "task_results": [...]
  },
  "timestamp": "2024-10-11T14:30:00",
  "mteb_source": "local"
}
```