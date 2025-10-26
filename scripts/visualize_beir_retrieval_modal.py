#!/usr/bin/env python3
"""BEIR Retrieval Visualization with Model Embeddings.

Creates an interactive HTML visualization showing model performance on BEIR datasets.
Embeds queries and documents, computes similarities, calculates NDCG scores, and visualizes
retrieval results with ground truth comparison.

Usage:
    # Run with ColPali model (max_sim)
    modal run visualize_beir_retrieval_modal.py::visualize_retrieval \
        --model "vidore/colpali-v1.2" \
        --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
        --similarity "max_sim" \
        --top-k 10

    # Run with BiGemma3 model (cosine)
    modal run visualize_beir_retrieval_modal.py::visualize_retrieval \
        --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
        --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
        --similarity "cosine" \
        --top-k 10

    # Download results
    modal volume get mteb-viz-cache viz_output_model_dataset ./local_viz_output
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import modal

# Create the Modal app
app = modal.App("beir-retrieval-visualizer")

# Create a volume for persistent storage
volume = modal.Volume.from_name("mteb-viz-cache", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the Modal image with required dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "clang")
    .uv_pip_install(
        [
            "datasets",
            "pillow",
            "numpy",
            "torch",
            "torchvision",
            "scikit-learn",
            "requests",
            "hf_transfer",
            "tqdm",
            "pytrec_eval",
        ]
    )
    .add_local_dir(
        local_path="../../mteb",
        remote_path="/mteb",
        copy=True,
    )
    .run_commands("cd /mteb && uv pip install --python $(command -v python) -e .")
    .run_commands(
        "uv pip install --python $(command -v python) git+https://github.com/adithya-s-k/colpali.git@feat/gemma3"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    memory=76800,
)
def visualize_retrieval(
    model_name: str,
    dataset_name: str,
    similarity: str = "cosine",
    top_k: int = 10,
    batch_size: int = 16,
    output_folder: str = None,
):
    """Main function to generate retrieval visualization.

    Args:
        model_name: HuggingFace model name (e.g., "vidore/colpali-v1.2")
        dataset_name: BEIR dataset name (e.g., "Nayana-cognitivelab/nayana-beir-eval-multilang_v2")
        similarity: Similarity function ("cosine" or "max_sim")
        top_k: Number of top results to retrieve per query
        batch_size: Batch size for embedding
        output_folder: Output folder name (auto-generated if None)
    """
    import sys

    sys.path.insert(0, "/mteb")

    import numpy as np
    from datasets import load_dataset

    print("=" * 80)
    print("BEIR Retrieval Visualization")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Similarity: {similarity}")
    print(f"Top-K: {top_k}")
    print(f"Batch Size: {batch_size}")
    print("=" * 80)

    # Set cache directories
    os.environ["HF_HOME"] = "/data/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/cache"
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Generate output folder name
    if output_folder is None:
        model_slug = model_name.replace("/", "_")
        dataset_slug = dataset_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"viz_output_{model_slug}_{dataset_slug}_{timestamp}"

    output_path = Path("/data") / output_folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories (no images dir - we'll use HF dataset locally)
    embeddings_dir = output_path / "embeddings"
    results_dir = output_path / "results"

    embeddings_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"\n📁 Output directory: {output_path}")
    print("  ℹ️  Images will be loaded from HuggingFace dataset locally")

    # Load dataset
    print("\n📊 Loading BEIR dataset...")
    corpus = load_dataset(dataset_name, "corpus", split="test")
    queries = load_dataset(dataset_name, "queries", split="test")
    qrels = load_dataset(dataset_name, "qrels", split="test")

    print(f"  ✓ Loaded {len(queries)} queries")
    print(f"  ✓ Loaded {len(corpus)} corpus documents")
    print(f"  ✓ Loaded {len(qrels)} qrels")

    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    model, processor = load_model(model_name)

    # Determine if multi-vector or single-vector
    is_multi_vector = "colpali" in model_name.lower()
    print(f"  ✓ Model type: {'Multi-vector' if is_multi_vector else 'Single-vector'}")
    print(f"  ✓ Similarity function: {similarity}")

    # Embed queries
    print("\n🔢 Embedding queries...")
    query_embeddings, query_data = embed_queries(
        queries, model, processor, batch_size, is_multi_vector
    )
    print(
        f"  ✓ Query embeddings shape: {query_embeddings.shape if not is_multi_vector else 'Multi-vector'}"
    )

    # Save query embeddings
    if is_multi_vector:
        np.savez_compressed(
            embeddings_dir / "query_embeddings.npz",
            **{
                f"query_{i}": emb.cpu().float().numpy()
                for i, emb in enumerate(query_embeddings)
            },
        )
    else:
        np.save(
            embeddings_dir / "query_embeddings.npy",
            query_embeddings.cpu().float().numpy(),
        )

    # Embed corpus
    print("\n🔢 Embedding corpus...")
    corpus_embeddings, corpus_data = embed_corpus(
        corpus, model, processor, batch_size, is_multi_vector
    )
    print(
        f"  ✓ Corpus embeddings shape: {corpus_embeddings.shape if not is_multi_vector else 'Multi-vector'}"
    )

    # Save corpus embeddings
    if is_multi_vector:
        np.savez_compressed(
            embeddings_dir / "corpus_embeddings.npz",
            **{
                f"corpus_{i}": emb.cpu().float().numpy()
                for i, emb in enumerate(corpus_embeddings)
            },
        )
    else:
        np.save(
            embeddings_dir / "corpus_embeddings.npy",
            corpus_embeddings.cpu().float().numpy(),
        )

    # Compute similarities and rankings
    print("\n🔍 Computing similarities...")
    similarities, rankings = compute_similarities_and_rankings(
        query_embeddings,
        corpus_embeddings,
        query_data,
        corpus_data,
        similarity,
        top_k,
        is_multi_vector,
        processor if is_multi_vector else None,
    )

    # Save similarities
    np.save(results_dir / "similarities.npy", similarities)

    # Save rankings
    with open(results_dir / "rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)

    # Process qrels
    print("\n📋 Processing qrels...")
    qrels_dict = process_qrels(qrels, query_data, corpus_data)

    # Calculate metrics
    print("\n📊 Calculating metrics...")
    per_query_metrics, overall_metrics = calculate_metrics(rankings, qrels_dict, top_k)

    # Save metrics
    with open(results_dir / "per_query_metrics.json", "w") as f:
        json.dump(per_query_metrics, f, indent=2)

    with open(results_dir / "overall_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\n📈 Overall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Generate HTML visualization
    print("\n🎨 Generating HTML visualization...")
    generate_html_visualization(
        query_data,
        corpus_data,
        rankings,
        qrels_dict,
        per_query_metrics,
        overall_metrics,
        similarities,
        output_path,
        model_name,
        dataset_name,
        similarity,
        top_k,
    )

    # Save summary
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "similarity": similarity,
        "top_k": top_k,
        "num_queries": len(query_data),
        "num_corpus": len(corpus_data),
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": overall_metrics,
        "output_folder": output_folder,
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate local viewer script
    generate_local_viewer_script(output_path, dataset_name)

    volume.commit()

    print("\n✅ Visualization complete!")
    print(f"📁 Output saved to: /data/{output_folder}")
    print("\n💾 To download locally, run:")
    print(f"   modal volume get mteb-viz-cache {output_folder} ./local_{output_folder}")
    print("\n🖼️  To view with images locally:")
    print(f"   cd ./local_{output_folder}")
    print("   python view_visualization.py")

    return {
        "output_folder": output_folder,
        "overall_metrics": overall_metrics,
        "num_queries": len(query_data),
        "num_corpus": len(corpus_data),
    }


def load_model(model_name: str):
    """Load model and processor."""
    import torch

    # Determine model type and load accordingly
    if "colpali" in model_name.lower():
        from colpali_engine.models import ColPali, ColPaliProcessor

        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_name)

    elif "bigemma" in model_name.lower() or "gemma" in model_name.lower():
        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        model = BiGemma3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        processor = BiGemmaProcessor3.from_pretrained(model_name)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, processor


def embed_queries(queries, model, processor, batch_size, is_multi_vector):
    """Embed all queries."""
    import torch
    from tqdm import tqdm

    query_data = []
    all_embeddings = []

    queries_list = list(queries)

    for i in tqdm(range(0, len(queries_list), batch_size), desc="Embedding queries"):
        batch = queries_list[i : i + batch_size]

        # Extract query texts
        query_texts = [q["query"] for q in batch]

        # Process queries
        if is_multi_vector:
            # ColPali: Add query prefix and augmentation tokens
            augmented_texts = [
                processor.query_prefix + text + processor.query_augmentation_token * 10
                for text in query_texts
            ]
            inputs = processor.process_texts(augmented_texts).to(model.device)
        else:
            # BiGemma3: Process texts directly
            inputs = processor.process_texts(query_texts).to(model.device)

        # Forward pass
        with torch.no_grad():
            embeddings = model(**inputs)

        if is_multi_vector:
            # Multi-vector: Keep as list of tensors
            for j, emb in enumerate(embeddings):
                all_embeddings.append(emb.cpu())
                query_data.append(
                    {
                        "query_id": batch[j]["query-id"],
                        "query_text": batch[j]["query"],
                        "metadata": {
                            k: v
                            for k, v in batch[j].items()
                            if k not in ["query-id", "query"]
                        },
                        "index": i + j,
                    }
                )
        else:
            # Single-vector: Stack into tensor
            all_embeddings.append(embeddings.cpu())
            for j, q in enumerate(batch):
                query_data.append(
                    {
                        "query_id": q["query-id"],
                        "query_text": q["query"],
                        "metadata": {
                            k: v for k, v in q.items() if k not in ["query-id", "query"]
                        },
                        "index": i + j,
                    }
                )

    if not is_multi_vector:
        all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings, query_data


def embed_corpus(corpus, model, processor, batch_size, is_multi_vector):
    """Embed all corpus documents (images NOT saved - use HF dataset locally)."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    corpus_data = []
    all_embeddings = []

    corpus_list = list(corpus)

    for i in tqdm(range(0, len(corpus_list), batch_size), desc="Embedding corpus"):
        batch = corpus_list[i : i + batch_size]

        # Extract images
        images = []
        for doc in batch:
            if "image" in doc and doc["image"] is not None:
                img = doc["image"]
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                else:
                    images.append(Image.new("RGB", (32, 32), color="gray"))
            else:
                images.append(Image.new("RGB", (32, 32), color="gray"))

        # Process images
        inputs = processor.process_images(images).to(model.device)

        # Forward pass
        with torch.no_grad():
            embeddings = model(**inputs)

        if is_multi_vector:
            # Multi-vector: Keep as list of tensors
            for j, emb in enumerate(embeddings):
                all_embeddings.append(emb.cpu())

                corpus_id = batch[j]["corpus-id"]
                has_image = "image" in batch[j] and batch[j]["image"] is not None

                corpus_data.append(
                    {
                        "corpus_id": corpus_id,
                        "has_image": has_image,
                        "metadata": {
                            k: v
                            for k, v in batch[j].items()
                            if k not in ["corpus-id", "image"]
                        },
                        "index": i + j,
                    }
                )
        else:
            # Single-vector: Stack into tensor
            all_embeddings.append(embeddings.cpu())
            for j, doc in enumerate(batch):
                corpus_id = doc["corpus-id"]
                has_image = "image" in doc and doc["image"] is not None

                corpus_data.append(
                    {
                        "corpus_id": corpus_id,
                        "has_image": has_image,
                        "metadata": {
                            k: v
                            for k, v in doc.items()
                            if k not in ["corpus-id", "image"]
                        },
                        "index": i + j,
                    }
                )

    if not is_multi_vector:
        all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings, corpus_data


def compute_similarities_and_rankings(
    query_embeddings,
    corpus_embeddings,
    query_data,
    corpus_data,
    similarity_fn,
    top_k,
    is_multi_vector,
    processor=None,
):
    """Compute similarities between queries and corpus, return rankings."""
    import numpy as np
    import torch
    from tqdm import tqdm

    num_queries = len(query_data)
    num_corpus = len(corpus_data)

    similarities = np.zeros((num_queries, num_corpus), dtype=np.float32)
    rankings = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(num_queries), desc="Computing similarities"):
        query_id = query_data[i]["query_id"]

        if is_multi_vector:
            # Multi-vector similarity (max_sim for ColPali)
            query_emb = query_embeddings[i].to(device)

            scores = []
            for j in range(num_corpus):
                corpus_emb = corpus_embeddings[j].to(device)

                if similarity_fn == "max_sim":
                    # ColBERT-style max sim
                    score = processor.score([query_emb], [corpus_emb])[0].item()
                else:
                    # Fallback to average cosine similarity
                    sim_matrix = torch.nn.functional.cosine_similarity(
                        query_emb.unsqueeze(1), corpus_emb.unsqueeze(0), dim=-1
                    )
                    score = sim_matrix.max(dim=1)[0].mean().item()

                scores.append(score)

            similarities[i] = np.array(scores)
        else:
            # Single-vector similarity
            query_emb = query_embeddings[i].unsqueeze(0).to(device)
            corpus_emb = corpus_embeddings.to(device)

            if similarity_fn == "cosine":
                # Cosine similarity
                query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                corpus_norm = torch.nn.functional.normalize(corpus_emb, p=2, dim=1)
                scores = torch.mm(query_norm, corpus_norm.t())[0]
            else:
                # Dot product
                scores = torch.mm(query_emb, corpus_emb.t())[0]

            similarities[i] = scores.cpu().float().numpy()

        # Get top-k rankings
        top_indices = np.argsort(similarities[i])[::-1][:top_k]

        rankings[query_id] = {
            corpus_data[idx]["corpus_id"]: float(similarities[i][idx])
            for idx in top_indices
        }

    return similarities, rankings


def process_qrels(qrels, query_data, corpus_data):
    """Process qrels into dict format."""
    qrels_dict = {}

    for qrel in qrels:
        query_id = qrel["query-id"]
        corpus_id = qrel["corpus-id"]
        score = qrel.get("score", 1)

        if query_id not in qrels_dict:
            qrels_dict[query_id] = {}

        qrels_dict[query_id][corpus_id] = int(score)

    return qrels_dict


def calculate_metrics(rankings, qrels_dict, top_k):
    """Calculate per-query and overall metrics."""
    import pytrec_eval

    # Prepare data for pytrec_eval
    k_values = [1, 5, 10, 100]

    # Ensure we have all queries in qrels
    all_query_ids = set(rankings.keys()) | set(qrels_dict.keys())

    # Fill in missing entries
    qrels_formatted = {}
    results_formatted = {}

    for query_id in all_query_ids:
        qrels_formatted[str(query_id)] = {
            str(corpus_id): score
            for corpus_id, score in qrels_dict.get(query_id, {}).items()
        }
        results_formatted[str(query_id)] = {
            str(corpus_id): score
            for corpus_id, score in rankings.get(query_id, {}).items()
        }

    # Calculate metrics using pytrec_eval
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_formatted, {ndcg_string, recall_string}
    )
    scores = evaluator.evaluate(results_formatted)

    # Extract per-query metrics
    per_query_metrics = {}
    for query_id, metrics in scores.items():
        per_query_metrics[query_id] = {
            "ndcg@1": metrics.get("ndcg_cut_1", 0.0),
            "ndcg@5": metrics.get("ndcg_cut_5", 0.0),
            "ndcg@10": metrics.get("ndcg_cut_10", 0.0),
            "ndcg@100": metrics.get("ndcg_cut_100", 0.0),
            "recall@5": metrics.get("recall_5", 0.0),
            "recall@10": metrics.get("recall_10", 0.0),
            "recall@100": metrics.get("recall_100", 0.0),
        }

    # Calculate overall metrics (average)
    overall_metrics = {}
    metric_keys = [
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "ndcg@100",
        "recall@5",
        "recall@10",
        "recall@100",
    ]

    for metric_key in metric_keys:
        values = [m[metric_key] for m in per_query_metrics.values()]
        overall_metrics[metric_key] = sum(values) / len(values) if values else 0.0

    return per_query_metrics, overall_metrics


def generate_local_viewer_script(output_path, dataset_name):
    """Generate a Python script for viewing visualization with local HF dataset."""
    script_content = f'''#!/usr/bin/env python3
"""Local viewer for BEIR visualization with HuggingFace dataset images.

This script starts a local web server that serves the visualization HTML
and dynamically loads images from the HuggingFace dataset.

Usage:
    python view_visualization.py
    # Then open http://localhost:8000 in your browser
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io

# Load the dataset
print("Loading HuggingFace dataset: {dataset_name}")
corpus_dataset = load_dataset("{dataset_name}", "corpus", split="test")
print(f"Loaded {{len(corpus_dataset)}} corpus documents")

# Create a mapping of corpus_id to dataset index
corpus_id_to_idx = {{}}
for idx, doc in enumerate(corpus_dataset):
    corpus_id_to_idx[doc["corpus-id"]] = idx

print(f"Built corpus ID index with {{len(corpus_id_to_idx)}} entries")


class DatasetImageHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves images from HuggingFace dataset."""
    
    def do_GET(self):
        if self.path.startswith("/dataset_image/"):
            # Extract corpus_id from URL: /dataset_image/<corpus_id>.png
            corpus_id_str = self.path.split("/")[-1].replace(".png", "")
            
            # Try to convert to int first (most datasets use int IDs)
            try:
                corpus_id = int(corpus_id_str)
            except ValueError:
                corpus_id = corpus_id_str  # Keep as string if not an int
            
            try:
                print(f"🔍 Requesting image for corpus_id: {{corpus_id}} (type: {{type(corpus_id).__name__}})")
                
                if corpus_id in corpus_id_to_idx:
                    idx = corpus_id_to_idx[corpus_id]
                    doc = corpus_dataset[idx]
                    print(f"✅ Found at index {{idx}}, serving image...")
                    
                    if "image" in doc and doc["image"] is not None:
                        img = doc["image"]
                        if isinstance(img, Image.Image):
                            # Convert PIL Image to bytes
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            
                            self.send_response(200)
                            self.send_header("Content-Type", "image/png")
                            self.end_headers()
                            self.wfile.write(img_byte_arr.read())
                            print(f"✅ Served image for corpus_id: {{corpus_id}}")
                            return
                    else:
                        print(f"⚠️  No image data for corpus_id: {{corpus_id}}")
                else:
                    print(f"❌ corpus_id {{corpus_id}} not found in index")
                    print(f"   Sample IDs from index: {{list(corpus_id_to_idx.keys())[:5]}}")
                
                # If image not found, return 404
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found")
                
            except Exception as e:
                print(f"💥 Error serving image for {{corpus_id}}: {{e}}")
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {{str(e)}}".encode())
        else:
            # Serve regular files (HTML, JSON, etc.)
            super().do_GET()


if __name__ == "__main__":
    PORT = 8000
    server = HTTPServer(("", PORT), DatasetImageHandler)
    print(f"\\n✅ Server started at http://localhost:{{PORT}}")
    print(f"📊 Open http://localhost:{{PORT}}/index.html in your browser")
    print("Press Ctrl+C to stop the server\\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")
        server.shutdown()
'''

    script_path = output_path / "view_visualization.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"  ✓ Local viewer script saved to: {script_path}")


def generate_html_visualization(
    query_data,
    corpus_data,
    rankings,
    qrels_dict,
    per_query_metrics,
    overall_metrics,
    similarities,
    output_path,
    model_name,
    dataset_name,
    similarity_fn,
    top_k,
):
    """Generate interactive HTML visualization."""
    import json

    # Prepare visualization data
    viz_data = []

    corpus_id_to_data = {c["corpus_id"]: c for c in corpus_data}

    for query in query_data:
        query_id = query["query_id"]
        query_idx = query["index"]

        # Get ground truth
        ground_truth = []
        if query_id in qrels_dict:
            for corpus_id, score in qrels_dict[query_id].items():
                if corpus_id in corpus_id_to_data:
                    corpus_info = corpus_id_to_data[corpus_id]
                    ground_truth.append(
                        {
                            "corpus_id": corpus_id,
                            "score": score,
                            "has_image": corpus_info["has_image"],
                            "metadata": corpus_info["metadata"],
                        }
                    )

        # Sort ground truth by score
        ground_truth.sort(key=lambda x: x["score"], reverse=True)

        # Get model predictions
        predictions = []
        if query_id in rankings:
            for corpus_id, sim_score in rankings[query_id].items():
                if corpus_id in corpus_id_to_data:
                    corpus_info = corpus_id_to_data[corpus_id]

                    # Check if in ground truth
                    in_gt = corpus_id in qrels_dict.get(query_id, {})
                    gt_score = qrels_dict.get(query_id, {}).get(corpus_id, 0)

                    predictions.append(
                        {
                            "corpus_id": corpus_id,
                            "similarity": float(sim_score),
                            "has_image": corpus_info["has_image"],
                            "metadata": corpus_info["metadata"],
                            "in_ground_truth": in_gt,
                            "ground_truth_score": gt_score,
                        }
                    )

        # Get similarity distribution (for heatmap)
        sim_distribution = similarities[query_idx].tolist()

        viz_data.append(
            {
                "query_id": query_id,
                "query_text": query["query_text"],
                "query_metadata": query["metadata"],
                "metrics": per_query_metrics.get(str(query_id), {}),
                "ground_truth": ground_truth,
                "predictions": predictions,
                "similarity_distribution": sim_distribution,
            }
        )

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEIR Retrieval Visualization - {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 15px;
        }}

        header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}

        h1 {{
            font-size: 28px;
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .header-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .info-card {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid #007bff;
        }}

        .info-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .info-value {{
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }}

        .controls {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .controls label {{
            font-weight: 500;
            font-size: 14px;
        }}

        .controls select, .controls input {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}

        .controls button {{
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}

        .controls button:hover {{
            background: #0056b3;
        }}

        .query-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}

        .query-header {{
            border-bottom: 2px solid #007bff;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        .query-id {{
            font-size: 13px;
            color: #666;
            margin-bottom: 8px;
        }}

        .query-text {{
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 12px;
            color: #2c3e50;
        }}

        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 12px;
        }}

        .metric-card {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .metric-value {{
            font-size: 18px;
            font-weight: 700;
        }}

        .metric-value.good {{ color: #28a745; }}
        .metric-value.medium {{ color: #ffc107; }}
        .metric-value.poor {{ color: #dc3545; }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}

        .comparison-column {{
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
        }}

        .column-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #dee2e6;
        }}

        .ground-truth {{ border-left: 3px solid #28a745; }}
        .predictions {{ border-left: 3px solid #007bff; }}

        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 12px;
        }}

        .ground-truth-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 12px;
        }}

        .doc-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
        }}

        .doc-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        .doc-card.correct {{
            border-left: 3px solid #28a745;
            background: #f0fff4;
        }}

        .doc-card.incorrect {{
            border-left: 3px solid #dc3545;
            background: #fff5f5;
        }}

        .doc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .doc-id {{
            font-size: 12px;
            color: #666;
        }}

        .doc-score {{
            font-size: 14px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
            background: #e9ecef;
        }}

        .doc-image {{
            width: 100%;
            max-width: 400px;
            height: auto;
            max-height: 300px;
            object-fit: contain;
            border-radius: 4px;
            margin-bottom: 8px;
            border: 1px solid #dee2e6;
            cursor: pointer;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}

        .no-image {{
            background: #e9ecef;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-radius: 4px;
            margin-bottom: 8px;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }}

        .badge.correct {{ background: #28a745; color: white; }}
        .badge.incorrect {{ background: #dc3545; color: white; }}
        .badge.gt-score {{ background: #ffc107; color: #333; }}

        .heatmap-container {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}

        .heatmap-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .heatmap {{
            display: flex;
            gap: 2px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}

        .heatmap-cell {{
            width: 8px;
            height: 20px;
            border-radius: 2px;
            transition: transform 0.2s;
        }}

        .heatmap-cell:hover {{
            transform: scale(1.5);
            z-index: 10;
        }}

        .metadata {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-top: auto;
        }}

        .metadata-title {{
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .no-results {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}

        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            align-items: center;
            justify-content: center;
        }}

        .modal.active {{ display: flex; }}

        .modal-content {{
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        }}

        .modal-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            background: none;
            border: none;
        }}

        @media (max-width: 1200px) {{
            .comparison-container {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 768px) {{
            .predictions-grid, .ground-truth-list {{
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            }}
        }}

        @media (max-width: 480px) {{
            .predictions-grid, .ground-truth-list {{
                grid-template-columns: 1fr;
            }}
            
            .doc-image {{
                max-height: 200px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 BEIR Retrieval Visualization</h1>
            <div class="header-info">
                <div class="info-card">
                    <div class="info-label">Model</div>
                    <div class="info-value" style="font-size: 14px;">{model_name}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Dataset</div>
                    <div class="info-value" style="font-size: 14px;">{dataset_name.split('/')[-1]}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Similarity</div>
                    <div class="info-value">{similarity_fn}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Top-K</div>
                    <div class="info-value">{top_k}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">NDCG@5</div>
                    <div class="info-value">{overall_metrics.get('ndcg@5', 0):.4f}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Recall@5</div>
                    <div class="info-value">{overall_metrics.get('recall@5', 0):.4f}</div>
                </div>
            </div>
        </header>

        <div class="controls">
            <label for="query-select">Select Query:</label>
            <select id="query-select">
                <option value="">-- Select a query --</option>
            </select>

            <button onclick="prevQuery()">← Previous</button>
            <button onclick="nextQuery()">Next →</button>

            <label for="filter-ndcg" style="margin-left: auto;">Filter by NDCG@5:</label>
            <select id="filter-ndcg" onchange="filterByNDCG()">
                <option value="all">All queries</option>
                <option value="poor">&lt; 0.3 (Poor)</option>
                <option value="medium">0.3 - 0.7 (Medium)</option>
                <option value="good">&gt; 0.7 (Good)</option>
            </select>

            <input type="text" id="search-input" placeholder="Search queries..." 
                   onkeyup="searchQueries()">
        </div>

        <div id="content">
            <div class="no-results">
                Select a query to view retrieval results
            </div>
        </div>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <button class="modal-close">&times;</button>
        <img id="modalImage" class="modal-content" alt="Full size image">
    </div>

    <script>
        const DATA = {json.dumps(viz_data, indent=2)};
        
        let currentQueryIndex = -1;
        let filteredData = DATA;

        function init() {{
            updateQuerySelect();
        }}

        function updateQuerySelect() {{
            const select = document.getElementById('query-select');
            select.innerHTML = '<option value="">-- Select a query --</option>';

            filteredData.forEach((query, index) => {{
                const option = document.createElement('option');
                option.value = index;
                const ndcg = query.metrics['ndcg@5'] || 0;
                option.textContent = `[${{ndcg.toFixed(3)}}] ${{query.query_id}}: ${{query.query_text.substring(0, 60)}}...`;
                select.appendChild(option);
            }});
        }}

        function displayQuery(queryData) {{
            const content = document.getElementById('content');
            
            const ndcg5 = queryData.metrics['ndcg@5'] || 0;
            const ndcg1 = queryData.metrics['ndcg@1'] || 0;
            const ndcg10 = queryData.metrics['ndcg@10'] || 0;
            const recall5 = queryData.metrics['recall@5'] || 0;

            const getMetricClass = (value) => {{
                if (value >= 0.7) return 'good';
                if (value >= 0.3) return 'medium';
                return 'poor';
            }};

            let html = `
                <div class="query-section">
                    <div class="query-header">
                        <div class="query-id">Query ID: ${{queryData.query_id}}</div>
                        <div class="query-text">${{queryData.query_text}}</div>
                        
                        <div class="metrics-row">
                            <div class="metric-card">
                                <div class="metric-label">NDCG@1</div>
                                <div class="metric-value ${{getMetricClass(ndcg1)}}">${{ndcg1.toFixed(3)}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">NDCG@5</div>
                                <div class="metric-value ${{getMetricClass(ndcg5)}}">${{ndcg5.toFixed(3)}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">NDCG@10</div>
                                <div class="metric-value ${{getMetricClass(ndcg10)}}">${{ndcg10.toFixed(3)}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Recall@5</div>
                                <div class="metric-value ${{getMetricClass(recall5)}}">${{recall5.toFixed(3)}}</div>
                            </div>
                        </div>

                        ${{formatMetadata('Query Metadata', queryData.query_metadata)}}
                    </div>

                    <div class="comparison-container">
                        <div class="comparison-column ground-truth">
                            <div class="column-title">📋 Ground Truth (${{queryData.ground_truth.length}})</div>
                            <div class="ground-truth-list">
                                ${{queryData.ground_truth.map(doc => formatGroundTruthDoc(doc)).join('')}}
                            </div>
                        </div>

                        <div class="comparison-column predictions">
                            <div class="column-title">🤖 Model Predictions (Top {top_k})</div>
                            <div class="predictions-grid">
                                ${{queryData.predictions.map((doc, idx) => formatPredictionDoc(doc, idx + 1)).join('')}}
                            </div>
                        </div>
                    </div>

                    <div class="heatmap-container">
                        <div class="heatmap-title">📊 Similarity Distribution (All Corpus Documents)</div>
                        <div class="heatmap">
                            ${{generateHeatmap(queryData.similarity_distribution)}}
                        </div>
                    </div>
                </div>
            `;

            content.innerHTML = html;
        }}

        function formatGroundTruthDoc(doc) {{
            const imageSrc = doc.has_image ? `/dataset_image/${{doc.corpus_id}}.png` : null;
            return `
                <div class="doc-card">
                    <div class="doc-header">
                        <div class="doc-id">${{doc.corpus_id}}</div>
                        <div class="doc-score">Score: ${{doc.score}}</div>
                    </div>
                    ${{imageSrc ? 
                        `<img src="${{imageSrc}}" class="doc-image" onclick="openModal(event, '${{imageSrc}}')" alt="Document image">` :
                        '<div class="no-image">No image</div>'
                    }}
                    ${{formatMetadata('Metadata', doc.metadata)}}
                </div>
            `;
        }}

        function formatPredictionDoc(doc, rank) {{
            const cardClass = doc.in_ground_truth ? 'correct' : 'incorrect';
            const imageSrc = doc.has_image ? `/dataset_image/${{doc.corpus_id}}.png` : null;
            return `
                <div class="doc-card ${{cardClass}}">
                    <div class="doc-header">
                        <div>
                            <span class="doc-id">#${{rank}} - ${{doc.corpus_id}}</span>
                            ${{doc.in_ground_truth ? 
                                `<span class="badge correct">✓ In GT</span>
                                 <span class="badge gt-score">GT: ${{doc.ground_truth_score}}</span>` :
                                '<span class="badge incorrect">✗ Not in GT</span>'
                            }}
                        </div>
                        <div class="doc-score">Sim: ${{doc.similarity.toFixed(4)}}</div>
                    </div>
                    ${{imageSrc ? 
                        `<img src="${{imageSrc}}" class="doc-image" onclick="openModal(event, '${{imageSrc}}')" alt="Document image">` :
                        '<div class="no-image">No image</div>'
                    }}
                    ${{formatMetadata('Metadata', doc.metadata)}}
                </div>
            `;
        }}

        function formatMetadata(title, metadata) {{
            if (!metadata || Object.keys(metadata).length === 0) return '';
            
            let html = `<div class="metadata"><div class="metadata-title">${{title}}</div>`;
            for (const [key, value] of Object.entries(metadata)) {{
                const displayValue = typeof value === 'object' ? JSON.stringify(value) : value;
                html += `<div style="margin: 2px 0; color: #555;"><strong>${{key}}:</strong> ${{displayValue}}</div>`;
            }}
            html += '</div>';
            return html;
        }}

        function generateHeatmap(similarities) {{
            const min = Math.min(...similarities);
            const max = Math.max(...similarities);
            const range = max - min || 1;
            
            return similarities.map((sim, idx) => {{
                const normalized = (sim - min) / range;
                const color = getHeatmapColor(normalized);
                return `<div class="heatmap-cell" style="background-color: ${{color}};" title="Doc ${{idx}}: ${{sim.toFixed(4)}}"></div>`;
            }}).join('');
        }}

        function getHeatmapColor(value) {{
            // Blue (low) to Red (high)
            const r = Math.round(255 * value);
            const b = Math.round(255 * (1 - value));
            return `rgb(${{r}}, 100, ${{b}})`;
        }}

        function nextQuery() {{
            if (filteredData.length === 0) return;
            currentQueryIndex = (currentQueryIndex + 1) % filteredData.length;
            document.getElementById('query-select').value = currentQueryIndex;
            displayQuery(filteredData[currentQueryIndex]);
        }}

        function prevQuery() {{
            if (filteredData.length === 0) return;
            currentQueryIndex = currentQueryIndex <= 0 ? filteredData.length - 1 : currentQueryIndex - 1;
            document.getElementById('query-select').value = currentQueryIndex;
            displayQuery(filteredData[currentQueryIndex]);
        }}

        function filterByNDCG() {{
            const filter = document.getElementById('filter-ndcg').value;
            
            if (filter === 'all') {{
                filteredData = DATA;
            }} else {{
                filteredData = DATA.filter(q => {{
                    const ndcg5 = q.metrics['ndcg@5'] || 0;
                    if (filter === 'poor') return ndcg5 < 0.3;
                    if (filter === 'medium') return ndcg5 >= 0.3 && ndcg5 <= 0.7;
                    if (filter === 'good') return ndcg5 > 0.7;
                    return true;
                }});
            }}
            
            currentQueryIndex = -1;
            updateQuerySelect();
            document.getElementById('content').innerHTML = 
                '<div class="no-results">Select a query to view retrieval results</div>';
        }}

        function searchQueries() {{
            const term = document.getElementById('search-input').value.toLowerCase();
            
            if (!term) {{
                filteredData = DATA;
            }} else {{
                filteredData = DATA.filter(q =>
                    q.query_text.toLowerCase().includes(term) ||
                    q.query_id.toString().includes(term)
                );
            }}
            
            currentQueryIndex = -1;
            updateQuerySelect();
        }}

        function openModal(event, imagePath) {{
            event.stopPropagation();
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.classList.add('active');
            modalImg.src = imagePath;
        }}

        function closeModal() {{
            const modal = document.getElementById('imageModal');
            modal.classList.remove('active');
        }}

        document.getElementById('query-select').addEventListener('change', function() {{
            const index = parseInt(this.value);
            if (!isNaN(index)) {{
                currentQueryIndex = index;
                displayQuery(filteredData[index]);
            }}
        }});

        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT') return;
            
            if (e.key === 'Escape') {{
                closeModal();
                return;
            }}
            
            if (e.key === 'ArrowRight') nextQuery();
            if (e.key === 'ArrowLeft') prevQuery();
        }});

        init();
    </script>
</body>
</html>
"""

    html_path = output_path / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ HTML saved to: {html_path}")


@app.local_entrypoint()
def main(
    model: str,
    dataset: str,
    similarity: str = "cosine",
    top_k: int = 10,
    batch_size: int = 16,
):
    r"""Local entrypoint for visualization.
    
    Examples:
        modal run visualize_beir_retrieval_modal.py --model "vidore/colpali-v1.2" \
            --dataset "Nayana-cognitivelab/nayana-beir-eval-multilang_v2" \
            --similarity "max_sim" --top-k 10
    """
    result = visualize_retrieval.remote(
        model_name=model,
        dataset_name=dataset,
        similarity=similarity,
        top_k=top_k,
        batch_size=batch_size,
    )

    print("\n" + "=" * 80)
    print("✅ Visualization Complete!")
    print("=" * 80)
    print(f"Output folder: {result['output_folder']}")
    print(f"Queries processed: {result['num_queries']}")
    print(f"Corpus size: {result['num_corpus']}")
    print("\nOverall Metrics:")
    for metric, value in result["overall_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("\n💾 To download results:")
    print(
        f"   modal volume get mteb-viz-cache {result['output_folder']} ./local_{result['output_folder']}"
    )
    print("=" * 80)
