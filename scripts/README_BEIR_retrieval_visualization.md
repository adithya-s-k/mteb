# BEIR Retrieval Visualization Tool

An interactive visualization tool for debugging and analyzing model performance on BEIR format datasets. This tool embeds queries and documents using vision-language models (ColPali, BiGemma3), computes similarity scores, calculates retrieval metrics (NDCG@K, Recall@K), and generates an HTML dashboard for visual analysis.

## Features

✨ **Key Features:**

- 🔢 **Embedding Generation**: Batch-processes queries and corpus documents with GPU acceleration
- 📊 **Metric Calculation**: Computes per-query NDCG@1, NDCG@5, NDCG@10, NDCG@100, Recall@5, Recall@10, Recall@100
- 🎨 **Interactive Visualization**: HTML dashboard with query-by-query analysis
- 🔍 **Ground Truth Comparison**: Side-by-side comparison of model predictions vs ground truth
- 📈 **Similarity Heatmaps**: Visual distribution of similarity scores across entire corpus
- 🎯 **Filtering**: Filter queries by NDCG score ranges to focus on poor/medium/good performers
- 💾 **Persistent Storage**: Saves embeddings, rankings, and metrics in Modal Volume

## Architecture

```
Modal Function (GPU L40S)
├── Load BEIR Dataset from HuggingFace
├── Load Model (ColPali/BiGemma3)
├── Embed Queries & Corpus (batched)
├── Compute Similarities (cosine or max_sim)
├── Calculate Per-Query Metrics (NDCG, Recall)
├── Generate HTML Visualization
└── Save to Modal Volume
    ├── embeddings/ (query & corpus embeddings)
    ├── results/ (similarities, rankings, metrics)
    ├── images/ (corpus document images)
    └── index.html (interactive dashboard)
```

## Installation & Setup

### Prerequisites

1. Modal account with GPU access
2. HuggingFace token with dataset access
3. Modal secrets configured:
   ```bash
   modal secret create adithya-hf-wandb HUGGINGFACE_TOKEN=hf_...
   ```

### Required Packages

The Modal image automatically installs:

- `datasets`, `pillow`, `numpy`, `torch`, `torchvision`
- `scikit-learn`, `requests`, `tqdm`, `pytrec_eval`
- ColPali library from GitHub
- Local MTEB library

## Usage

### Basic Usage

```bash
# ColPali model with max_sim
modal run mteb/scripts/visualize_beir_retrieval_modal.py \
    --model "vidore/colpali-v1.2" \
    --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
    --similarity "max_sim" \
    --top-k 10

# BiGemma3 model with cosine similarity
modal run mteb/scripts/visualize_beir_retrieval_modal.py \
    --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
    --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
    --similarity "cosine" \
    --top-k 10
```

### Parameters

| Parameter      | Default  | Description                                 |
| -------------- | -------- | ------------------------------------------- |
| `--model`      | Required | HuggingFace model name                      |
| `--dataset`    | Required | HuggingFace BEIR dataset name               |
| `--similarity` | `cosine` | Similarity function: `cosine` or `max_sim`  |
| `--top-k`      | `10`     | Number of top results to retrieve per query |
| `--batch-size` | `8`      | Batch size for embedding                    |

### Download Results

```bash
# List available outputs
modal volume ls mteb-viz-cache

# Download specific output
modal volume get mteb-viz-cache viz_output_vidore_colpali-v1.2_nayana-beir-eval-multilang_v2_20251023_120000

# Open visualization
open ./viz_output_vidore_colpali-v1.2_nayana-beir-eval-multilang_v2_20251023_120000/index.html  # macOS
start ./viz_output_vidore_colpali-v1.2_nayana-beir-eval-multilang_v2_20251023_120000/index.html  # Windows
```

## Output Structure

```
viz_output_model_dataset_timestamp/
├── index.html                      # Main interactive visualization
├── summary.json                    # Overall stats and metrics
├── embeddings/
│   ├── query_embeddings.npy        # Query embeddings (single-vector)
│   ├── query_embeddings.npz        # Query embeddings (multi-vector)
│   ├── corpus_embeddings.npy       # Corpus embeddings (single-vector)
│   └── corpus_embeddings.npz       # Corpus embeddings (multi-vector)
├── results/
│   ├── similarities.npy            # Full similarity matrix
│   ├── rankings.json               # Top-K rankings per query
│   ├── per_query_metrics.json      # NDCG, Recall per query
│   └── overall_metrics.json        # Averaged metrics
└── images/
    ├── corpus_<id>.png             # Corpus document images
    └── ...
```

## HTML Visualization Features

### Header Section

- **Model Info**: Model name, dataset, similarity function
- **Overall Metrics**: NDCG@5, Recall@5 at a glance

### Query Controls

- **Query Selector**: Dropdown with NDCG@5 scores
- **Navigation**: Previous/Next buttons, keyboard arrows (← →)
- **Filtering**: Filter by NDCG@5 ranges (Poor < 0.3, Medium 0.3-0.7, Good > 0.7)
- **Search**: Text search for queries

### Per-Query View

#### Metrics Dashboard

- NDCG@1, NDCG@5, NDCG@10 with color coding:
  - 🟢 Green: > 0.7 (Good)
  - 🟡 Yellow: 0.3-0.7 (Medium)
  - 🔴 Red: < 0.3 (Poor)
- Recall@5

#### Side-by-Side Comparison

**Left Column - Ground Truth:**

- Documents from qrels with relevance scores
- Sorted by ground truth score

**Right Column - Model Predictions:**

- Top-K retrieved documents
- Similarity scores
- ✅ Badge if in ground truth (correct)
- ❌ Badge if not in ground truth (incorrect)
- Ground truth score if applicable

#### Similarity Heatmap

- Visual representation of similarity scores across entire corpus
- Blue (low similarity) to Red (high similarity)
- Hover to see exact scores

### Image Modal

- Click any document image for full-size view
- Press ESC or click outside to close

## Supported Models

### Multi-Vector Models (max_sim)

- `vidore/colpali-v1.1`
- `vidore/colpali-v1.2`
- `vidore/colpali-v1.3`

### Single-Vector Models (cosine)

- `Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-*`
- `Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-*`
- `Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-*`
- Any BiGemma3-based model

## Metrics Explained

### NDCG (Normalized Discounted Cumulative Gain)

- Measures ranking quality considering position and relevance
- Range: 0.0 (worst) to 1.0 (perfect)
- NDCG@K: Only considers top-K results

### Recall@K

- Fraction of relevant documents found in top-K
- Range: 0.0 (none found) to 1.0 (all found)

## Performance Tips

### GPU Selection

Default: `L40S` (recommended for large models)

- ColPali: L40S or A100
- BiGemma3: L40S sufficient

### Batch Size

- **ColPali**: 4-8 (memory intensive)
- **BiGemma3**: 8-12 (more efficient)

### Timeout

Default: 4 hours

- Small datasets (<100 queries): < 30 min
- Medium datasets (100-500 queries): 30-60 min
- Large datasets (>500 queries): 1-2 hours

## Example Workflows

### Workflow 1: Debug Poor Performance

```bash
# 1. Run visualization
modal run mteb/scripts/visualize_beir_retrieval_modal.py \
    --model "vidore/colpali-v1.2" \
    --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
    --similarity "max_sim"

# 2. Download results
modal volume get mteb-viz-cache viz_output_... ./viz

# 3. Open HTML, filter by "Poor" (NDCG@5 < 0.3)

# 4. Analyze:
#    - What types of queries fail?
#    - Are images unclear/ambiguous?
#    - Is text in ground truth but model retrieves different language?
#    - Check similarity heatmap: Are correct docs getting low scores?
```

### Workflow 2: Compare Models

```bash
# Run for multiple models
for model in "vidore/colpali-v1.2" "vidore/colpali-v1.3"; do
    modal run mteb/scripts/visualize_beir_retrieval_modal.py \
        --model "$model" \
        --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
        --similarity "max_sim"
done

# Compare overall_metrics.json files
# Compare per-query performance on same queries
```

### Workflow 3: Analyze Cross-Lingual Performance

```bash
# Use multilingual dataset
modal run mteb/scripts/visualize_beir_retrieval_modal.py \
    --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
    --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
    --similarity "cosine"

# In HTML:
# - Filter queries by language (in query metadata)
# - Check if model retrieves correct language variants
# - Analyze similarity distribution across languages
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
modal run ... --batch-size 4
```

### Modal Timeout

```bash
# Edit timeout in script (line ~88):
timeout=8 * 60 * 60,  # 8 hours
```

### Missing Images

- Check if dataset has `image` column
- Verify images are PIL.Image objects
- Some corpus documents may not have images (shows "No image available")

### Slow Embedding

- ColPali multi-vector embeddings are slower than single-vector
- Use smaller batch size if GPU memory is bottleneck
- Consider sampling dataset for quick tests (TODO feature)

## Future Enhancements (TODO)

- [ ] **Sampling**: Sample N queries for quick testing
- [ ] **Embedding Space Visualization**: PCA/t-SNE plots
- [ ] **Error Analysis**: Automated categorization of failures
- [ ] **Batch Comparison**: Compare multiple models in single HTML
- [ ] **Export**: Export filtered results to BEIR format
- [ ] **Caching**: Reuse embeddings across runs
- [ ] **Streaming**: Process large datasets in chunks

## Technical Details

### Similarity Functions

**Cosine Similarity** (for single-vector models):

```python
similarity = (query_emb @ corpus_emb.T) / (||query_emb|| * ||corpus_emb||)
```

**Max-Sim** (for multi-vector models like ColPali):

```python
# For each query token, find max similarity with any corpus token
# Average across query tokens
max_sim = mean([max(sim(q_token, c_token) for c_token in corpus_tokens)
                for q_token in query_tokens])
```

### File Formats

**Embeddings**:

- `.npy`: NumPy array (single-vector)
- `.npz`: Compressed NumPy arrays (multi-vector, one array per sample)

**Results**:

- `similarities.npy`: [num_queries, num_corpus] float32 array
- `rankings.json`: {query_id: {corpus_id: similarity_score}}

## Support

For issues or questions:

1. Check Modal logs: `modal app logs beir-retrieval-visualizer`
2. Verify dataset format matches BEIR spec
3. Ensure model is compatible (ColPali or BiGemma3)

## License

Same as parent MTEB project.
