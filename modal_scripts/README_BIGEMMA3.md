# Running BiGemma3 on MTEB Vidore Benchmark with Modal

This guide explains how to evaluate the BiGemma3 vision-language model on the Vidore document retrieval benchmark using Modal's GPU infrastructure.

## Overview

**BiGemma3** is a dense embedding model based on Google's Gemma3-4B-IT, adapted to generate 2560-dimensional embeddings for:
- Text-only inputs (queries)
- Image-only inputs (document pages)
- Multimodal inputs (text + image)

The model has been integrated into the MTEB framework and can be evaluated on the Vidore benchmark, which tests document understanding and retrieval capabilities.

## Prerequisites

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Set up Modal Authentication

```bash
modal setup
```

### 3. Configure Secrets

Create a HuggingFace secret in Modal:

```bash
modal secret create huggingface-secret \
  HUGGINGFACE_TOKEN=<your-hf-token>
```

### 4. Repository Setup

Ensure you have the local MTEB repository with BiGemma3 integration:

```bash
cd /path/to/nayana-ir/mteb
```

The Modal script automatically:
- Mounts the local MTEB directory to `/mteb` in the container
- Installs the custom colpali library with BiGemma3 support
- Installs MTEB from the local directory

## File Structure

```
mteb/
├── mteb/
│   └── models/
│       └── bigemma_models.py          # BiGemma3 wrapper implementation
├── modal_scripts/
│   ├── modal_mteb_local.py            # Main Modal evaluation script
│   └── README_BIGEMMA3.md             # This file
└── results/                           # Evaluation results will be saved here
```

## Quick Start

### 1. List Available Benchmarks

```bash
modal run modal_scripts/modal_mteb_local.py::list_benchmarks
```

This will show all available benchmarks including:
- ViDoRe(v1) - Original Vidore benchmark
- ViDoRe(v2) - Updated Vidore benchmark

### 2. Run BiGemma3 Evaluation

#### Basic Evaluation (Default: ViDoRe v1 and v2)

```bash
modal run modal_scripts/modal_mteb_local.py \
  --model bigemma3-4b-embedding
```

This will:
- Deploy to Modal with L40S GPU
- Load BiGemma3 model (google/gemma-3-4b-it)
- Run evaluation on all Vidore tasks
- Save results to `../../results/`

#### Custom Batch Size (Recommended for GPU Memory Management)

```bash
modal run modal_scripts/modal_mteb_local.py \
  --model bigemma3-4b-embedding \
  --batch-size 8
```

Recommended batch sizes:
- **8-12**: Safe for 24GB VRAM (L40S)
- **16**: Optimal balance of speed and memory
- **4**: If running out of memory

#### Specific Benchmark

```bash
modal run modal_scripts/modal_mteb_local.py \
  --model bigemma3-4b-embedding \
  --benchmarks "ViDoRe(v1)"
```

#### Custom Output Folder

```bash
modal run modal_scripts/modal_mteb_local.py \
  --model bigemma3-4b-embedding \
  --output-folder "results/bigemma3_exp1"
```

### 3. Batch Evaluation (Compare Multiple Models)

```bash
modal run modal_scripts/modal_mteb_local.py \
  --model "bigemma3-4b-embedding,vidore/colqwen2.5-v0.2,google/siglip-so400m-patch14-384" \
  --batch-mode True \
  --batch-size 8
```

This will evaluate all three models sequentially and save results for each.

## Understanding the Vidore Benchmark

### Tasks Included

The Vidore benchmark evaluates document retrieval on 8 different tasks:

1. **VidoreArxivQARetrieval**: Academic paper Q&A
   - 500 queries, 500 documents
   - Domain: Scientific papers

2. **VidoreDocVQARetrieval**: Document visual Q&A
   - 500 queries, 500 documents
   - Domain: General documents

3. **VidoreInfoVQARetrieval**: Infographic Q&A
   - 500 queries, 500 documents
   - Domain: Infographics

4. **VidoreTabfquadRetrieval**: Table fact-checking
   - 280 queries, 70 documents
   - Domain: Tables

5. **VidoreTatdqaRetrieval**: Table Q&A
   - 1663 queries, 277 documents
   - Domain: Financial tables

6. **VidoreShiftProjectRetrieval**: Project document retrieval
   - 100 queries, 1000 documents
   - Domain: Project documentation

7. **VidoreSyntheticDocQAAI**: Synthetic AI documents
   - 100 queries, 968 documents
   - Domain: Artificial Intelligence

8. **VidoreSyntheticDocQAEnergy**: Synthetic energy documents
   - 100 queries, 977 documents
   - Domain: Energy sector

### Evaluation Metrics

Primary metric: **NDCG@5** (Normalized Discounted Cumulative Gain at 5)

Also reported:
- Recall@K (K=1,5,10,20,100)
- Precision@K
- MAP (Mean Average Precision)

### How BiGemma3 is Evaluated

For each task:

1. **Query Encoding**: Text queries are encoded using `get_text_embeddings()`
   - Queries are processed through BiGemmaProcessor3
   - Model generates 2560-dim embeddings
   - Embeddings are L2-normalized

2. **Document Encoding**: Document images are encoded using `get_image_embeddings()`
   - Images are processed through BiGemmaProcessor3
   - Model generates 2560-dim embeddings
   - Embeddings are L2-normalized

3. **Similarity Scoring**: Cosine similarity computed via `calculate_probs()`
   - Uses BiGemmaProcessor3.score() method
   - Matrix of (n_queries × n_docs) similarities

4. **Ranking**: Documents ranked by similarity for each query

5. **Metrics**: NDCG@5 and other metrics computed from rankings

## Expected Performance

### Resource Usage

- **GPU**: L40S (24GB VRAM)
- **Memory**: ~8-12GB VRAM with batch_size=8
- **Runtime**: ~30-60 minutes for full Vidore benchmark
- **Compute Cost**: ~$0.50-1.00 per evaluation (Modal L40S pricing)

### Baseline Comparisons

Expected NDCG@5 ranges for different model types:

| Model Type | Example | Approx. NDCG@5 |
|------------|---------|----------------|
| Late Interaction | ColQwen2.5 | 75-85% |
| Dense Embedding | BiGemma3 | 60-75% |
| CLIP-style | SigLIP | 50-65% |
| GME-V | GME-Qwen2-VL-2B | 70-80% |

*Note: These are rough estimates. Actual performance depends on training data and task specifics.*

## Monitoring Evaluation

### Real-time Logs

Modal streams logs in real-time:

```
Starting evaluation for model: bigemma3-4b-embedding
Loading BiGemma3 model: google/gemma-3-4b-it
Loading BiGemmaProcessor3 for: google/gemma-3-4b-it
Model loaded on device: cuda:0
Embedding dimension: 2560

Evaluating on VidoreArxivQARetrieval:
  Encoding texts: 100%|██████████| 500/500
  Encoding images: 100%|██████████| 500/500
  Computing scores...
  NDCG@5: 0.6543

Evaluating on VidoreDocVQARetrieval:
  ...
```

### Check Modal Dashboard

Visit https://modal.com/apps to see:
- Active function calls
- GPU utilization
- Logs and errors
- Cost tracking

## Results Output

### JSON Results

Results are saved to:
```
results/local_bigemma3-4b-embedding_YYYYMMDD_HHMMSS.json
```

Structure:
```json
{
  "model": "bigemma3-4b-embedding",
  "benchmarks": ["ViDoRe(v1)", "ViDoRe(v2)"],
  "status": "completed",
  "results": {
    "task_results": [
      {
        "task_name": "VidoreArxivQARetrieval",
        "scores": {
          "ndcg_at_5": 0.6543,
          "recall_at_5": 0.8234,
          ...
        }
      },
      ...
    ]
  },
  "timestamp": "2025-10-10T12:00:00"
}
```

### Analyzing Results

Use the provided results analyzer:

```python
import json

with open('results/local_bigemma3-4b-embedding_*.json') as f:
    results = json.load(f)

# Extract NDCG@5 scores
for task in results['results']['task_results']:
    task_name = task['task_name']
    ndcg5 = task['scores']['ndcg_at_5']
    print(f"{task_name}: {ndcg5:.4f}")

# Average NDCG@5
ndcg_scores = [t['scores']['ndcg_at_5'] for t in results['results']['task_results']]
avg_ndcg5 = sum(ndcg_scores) / len(ndcg_scores)
print(f"\nAverage NDCG@5: {avg_ndcg5:.4f}")
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
modal run modal_scripts/modal_mteb_local.py \
  --model bigemma3-4b-embedding \
  --batch-size 4  # Reduce batch size
```

#### 2. Model Loading Fails

**Symptom**: `ModuleNotFoundError: No module named 'colpali_engine'`

**Solution**: Check that the Modal image includes:
```python
.run_commands(
    "uv pip install --python $(command -v python) git+https://github.com/adithya-s-k/colpali.git@feat/gemma3"
)
```

#### 3. BiGemma3 Model Not Found

**Symptom**: `KeyError: 'bigemma3-4b-embedding'`

**Solution**: Ensure `bigemma_models.py` is registered in `mteb/models/overview.py`:
```python
from mteb.models import bigemma_models
# ...
model_modules = [
    # ...
    bigemma_models,
    # ...
]
```

#### 4. Slow Image Encoding

**Symptom**: Image encoding taking very long

**Explanation**: BiGemma3 processes each image individually to avoid batch size mismatches (variable image sizes).

**Optimization**: This is expected behavior. For faster evaluation, consider:
- Using a larger GPU (A100)
- Increasing timeout in Modal function decorator

### Debug Mode

Enable verbose logging:

```bash
# In modal_mteb_local.py, line 226:
eval_kwargs = {"output_folder": output_folder, "verbosity": 3}  # Max verbosity
```

## Advanced Usage

### Custom BiGemma3 Configuration

Modify `mteb/models/bigemma_models.py`:

```python
# Use different embedding dimension
bigemma3_4b_custom = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="google/gemma-3-4b-it",
        embedding_dim=1536,  # Instead of 2560
    ),
    name="bigemma3-4b-1536dim",
    # ... rest of config
)
```

Then run:
```bash
modal run modal_scripts/modal_mteb_local.py --model bigemma3-4b-1536dim
```

### Evaluate on Custom Checkpoint

If you fine-tuned BiGemma3:

```python
# In bigemma_models.py
bigemma3_finetuned = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="your-hf-username/bigemma3-finetuned",  # Your checkpoint
        embedding_dim=2560,
    ),
    name="bigemma3-finetuned",
    # ...
)
```

### Multimodal Fusion Evaluation

Test different fusion strategies:

```python
# In BiGemma3Wrapper.get_fused_embeddings()
# fusion_mode="sum"   # Add text + image embeddings
# fusion_mode="mean"  # Average text + image embeddings
```

## Performance Optimization Tips

### 1. Use Larger GPU for Faster Evaluation

Edit `modal_scripts/modal_mteb_local.py`:

```python
@app.function(
    image=image,
    gpu="A100",  # Instead of L40S
    timeout=2 * 60 * 60,  # 2 hours
    ...
)
```

### 2. Parallel Task Evaluation

Currently tasks run sequentially. For parallel evaluation, modify the Modal script to spawn separate functions per task.

### 3. Cache Model Weights

Use Modal Volume to cache model weights:

```python
volumes={
    "/cache": modal.Volume.from_name("mteb-cache", create_if_missing=True)
}
```

Model will download once and reuse on subsequent runs.

## Comparison Script

Compare BiGemma3 with other models:

```bash
# Evaluate multiple models
modal run modal_scripts/modal_mteb_local.py \
  --model "bigemma3-4b-embedding,Alibaba-NLP/gme-Qwen2-VL-2B-Instruct,vidore/colqwen2.5-v0.2" \
  --batch-mode True

# Then compare results
python -c "
import json
import glob

models = ['bigemma3-4b-embedding', 'gme-Qwen2-VL-2B-Instruct', 'colqwen2.5-v0.2']
for model in models:
    files = glob.glob(f'results/*{model}*.json')
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
        ndcg = [t['scores']['ndcg_at_5'] for t in data['results']['task_results']]
        print(f'{model}: NDCG@5={sum(ndcg)/len(ndcg):.4f}')
"
```

## Citation

If you use BiGemma3 in your research, please cite:

```bibtex
@misc{bigemma3,
  title={BiGemma3: Dense Vision-Language Embeddings for Document Retrieval},
  author={Your Name},
  year={2025},
  url={https://github.com/adithya-s-k/colpali}
}
```

And the Vidore benchmark:

```bibtex
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
```

## Support

For issues:
- BiGemma3 model: https://github.com/adithya-s-k/colpali/issues
- MTEB integration: https://github.com/embeddings-benchmark/mteb/issues
- Modal infrastructure: https://modal.com/docs

## Next Steps

After evaluation:

1. **Analyze Results**: Compare NDCG@5 across tasks to identify strengths/weaknesses
2. **Error Analysis**: Look at failed retrievals to understand limitations
3. **Fine-tuning**: Use results to guide model improvements
4. **Submit to MTEB**: If performance is strong, submit to official leaderboard

---

**Happy Benchmarking!** 🚀
