from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import modal

app = modal.App("mteb-vidore-benchmark-local")
huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")

# CUDA configuration similar to vLLM Llama 11B
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "git", "wget", "curl")
    .uv_pip_install(
        "sentence-transformers",
        "datasets",
        "numpy",
        "scikit-learn",
        "requests",
        "tqdm",
        "pillow",
        "huggingface_hub",
        "accelerate",
    )
    .add_local_dir(
        local_path="../../mteb",
        remote_path="/mteb",
        copy=True,
        ignore=[
            "*.git*",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "env",
            "venv",
            ".env",
            ".venv",
            "pip-log.txt",
            "pip-delete-this-directory.txt",
            ".tox",
            ".coverage",
            ".coverage.*",
            ".cache",
            "nosetests.xml",
            "coverage.xml",
            "*.cover",
            "*.log",
            ".DS_Store",
            "*.egg-info",
            ".pytest_cache",
            "node_modules",
            "*.tmp",
            "*.temp",
        ],
    )
    .run_commands(
        "uv pip install --python $(command -v python) git+https://github.com/adithya-s-k/colpali.git@feat/gemma3"
    )
    .run_commands("cd /mteb && uv pip install --python $(command -v python) -e .")
    .run_commands("uv pip install --upgrade transformers --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
)


def extract_serializable_results(results: Any) -> dict[str, Any]:
    """Extract serializable data from MTEB results to avoid Modal deserialization issues."""
    if isinstance(results, list):
        serializable_results = []
        for task_result in results:
            try:
                json_str = task_result.model_dump_json()
                task_dict = json.loads(json_str)
                serializable_results.append(task_dict)
            except Exception:
                serializable_results.append(str(task_result))
        return {"task_results": serializable_results}
    else:
        try:
            json_str = results.model_dump_json()
            return json.loads(json_str)
        except Exception:
            return {"raw_results": str(results)}


@app.function(
    image=image,
    timeout=300,
)
def list_available_benchmarks() -> dict:
    """List all available benchmarks from the local mteb directory."""
    import sys

    # Add the local mteb directory to Python path
    sys.path.insert(0, "/mteb")

    import mteb

    try:
        # Get all available benchmarks
        available_benchmarks = []

        # Try to get standard ViDoRe benchmarks
        vidore_benchmarks = ["ViDoRe(v1)", "ViDoRe(v2)"]

        for benchmark_name in vidore_benchmarks:
            try:
                benchmark = mteb.get_benchmark(benchmark_name)
                available_benchmarks.append(
                    {
                        "name": benchmark_name,
                        "description": getattr(benchmark, "description", ""),
                        "tasks_count": (
                            len(benchmark.tasks) if hasattr(benchmark, "tasks") else 0
                        ),
                    }
                )
            except Exception as e:
                print(f"Could not load benchmark {benchmark_name}: {e}")

        # Try to discover custom benchmarks from the local mteb
        try:
            from mteb.benchmarks import BENCHMARK_REGISTRY

            for benchmark_name in BENCHMARK_REGISTRY.keys():
                if benchmark_name not in [b["name"] for b in available_benchmarks]:
                    try:
                        benchmark = mteb.get_benchmark(benchmark_name)
                        available_benchmarks.append(
                            {
                                "name": benchmark_name,
                                "description": getattr(
                                    benchmark, "description", "Custom benchmark"
                                ),
                                "tasks_count": (
                                    len(benchmark.tasks)
                                    if hasattr(benchmark, "tasks")
                                    else 0
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"Could not load custom benchmark {benchmark_name}: {e}")
        except ImportError:
            print("Could not import BENCHMARK_REGISTRY")

        return {
            "available_benchmarks": available_benchmarks,
            "total_count": len(available_benchmarks),
            "mteb_path": "/mteb",
        }
    except Exception as e:
        return {"error": str(e)}


@app.function(
    image=image,
    # gpu="A100-40GB",
    gpu="L40s",
    timeout=6 * 60 * 60,
    volumes={"/cache": modal.Volume.from_name("mteb-cache", create_if_missing=True)},
    secrets=[huggingface_secret],
    memory=40960,
    # concurrency_limit=1,  # Prevent Modal from spinning up multiple containers
)
def run_mteb_evaluation(
    model_name: str,
    benchmarks: list[str] | None = None,
    output_folder: str = "results",
    cache_folder: str = "/cache",
    batch_size: int | None = None,
    pooling_strategy: str | None = None,
) -> dict:
    """Run MTEB evaluation for specified model and benchmarks using local mteb directory.

    Args:
        model_name: HuggingFace model name or path
        benchmarks: List of benchmark names to evaluate (default: ViDoRe v1 and v2)
        output_folder: Folder to save results
        cache_folder: Cache folder for datasets and models
        batch_size: Batch size for model inference (optional)
        pooling_strategy: Pooling strategy for BiGemma3 models ("cls", "last", "mean"). Default: None (uses model default)

    Returns:
        Dictionary containing evaluation results
    """
    import os
    import sys

    # Add the local mteb directory to Python path
    sys.path.insert(0, "/mteb")

    import mteb
    from mteb import MTEB

    os.environ["HF_HOME"] = cache_folder
    os.environ["TRANSFORMERS_CACHE"] = cache_folder
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    print(f"Running evaluation for model: {model_name}")
    print("Using local mteb from: /mteb")

    if benchmarks is None:
        benchmarks = ["ViDoRe(v1)", "ViDoRe(v2)"]

    print(f"Benchmarks to evaluate: {benchmarks}")

    try:
        # Load model with optional pooling_strategy
        model_kwargs = {}
        if pooling_strategy is not None:
            model_kwargs["pooling_strategy"] = pooling_strategy
            print(f"Using pooling strategy: {pooling_strategy}")

        model = mteb.get_model(model_name, **model_kwargs)
        print(f"Successfully loaded model: {model_name}")

        # Get benchmarks
        benchmark_objects = []
        for benchmark_name in benchmarks:
            try:
                benchmark_obj = mteb.get_benchmark(benchmark_name)
                benchmark_objects.extend(benchmark_obj.tasks)
                print(
                    f"Added benchmark: {benchmark_name} with {len(benchmark_obj.tasks)} tasks"
                )
            except Exception as e:
                print(f"Failed to load benchmark {benchmark_name}: {e}")
                continue

        if not benchmark_objects:
            raise ValueError("No valid benchmarks found")

        evaluation = MTEB(tasks=benchmark_objects)

        # Configure evaluation parameters
        eval_kwargs = {"output_folder": output_folder, "verbosity": 2}
        if batch_size is not None:
            eval_kwargs["batch_size"] = batch_size
            print(f"Using batch size: {batch_size}")

        results = evaluation.run(model, **eval_kwargs)
        serializable_results = extract_serializable_results(results)

        print(f"Evaluation completed successfully for {model_name}")
        return {
            "model": model_name,
            "benchmarks": benchmarks,
            "status": "completed",
            "results": serializable_results,
            "timestamp": datetime.now().isoformat(),
            "mteb_source": "local",
        }

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "model": model_name,
            "benchmarks": benchmarks,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "mteb_source": "local",
        }


def save_results_locally(
    results: dict[str, Any], output_folder: str = "../../results"
) -> str:
    """Save evaluation results to a local JSON file.

    Args:
        results: The results dictionary to save
        output_folder: Local folder to save results

    Returns:
        Path to the saved file
    """
    os.makedirs(output_folder, exist_ok=True)

    model_name = results.get("model", "unknown_model").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"local_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Results saved to: {filepath}")
    return filepath


@app.local_entrypoint()
def list_benchmarks():
    """List all available benchmarks from the local mteb directory."""
    result = list_available_benchmarks.remote()
    print("Available Benchmarks (from local mteb):")
    print("=" * 50)
    print(f"Total benchmarks: {result.get('total_count', 0)}")
    print(f"MTEB source: {result.get('mteb_path', 'unknown')}")
    print()

    if "available_benchmarks" in result:
        for benchmark in result["available_benchmarks"]:
            print(f"  - {benchmark['name']}")
            if benchmark.get("description"):
                print(f"    Description: {benchmark['description']}")
            if benchmark.get("tasks_count"):
                print(f"    Tasks: {benchmark['tasks_count']}")
            print()

    if "error" in result:
        print(f"Error listing benchmarks: {result['error']}")


@app.local_entrypoint()
def main(
    model: str = "bigemma3-4b-embedding",
    benchmarks: str = "",
    output_folder: str = "../../results",
    batch_mode: bool = False,
    batch_size: int = None,
    pooling_strategy: str = None,
):
    """Main entry point for MTEB evaluation using local mteb directory.

    Args:
        model: Model name or comma-separated list of models for batch mode
               Supported models:
               - bigemma3-4b-embedding (BiGemma3 4B - custom dense embedding model)
               - vidore/colqwen2.5-v0.2 (ColQwen2.5 - late interaction)
               - vidore/colpali-v1.2 (ColPali - late interaction)
               - google/siglip-so400m-patch14-384 (SigLIP - CLIP-style)
               - Alibaba-NLP/gme-Qwen2-VL-2B-Instruct (GME-V - similar to BiGemma3)
        benchmarks: Comma-separated list of benchmark names (empty for default ViDoRe v1,v2)
        output_folder: Output folder for results
        batch_mode: Whether to run in batch mode for multiple models
        batch_size: Batch size for model inference (optional, default varies by model)
                   Recommended: 8-16 for BiGemma3, 32 for CLIP-style models
        pooling_strategy: Pooling strategy for BiGemma3 models ("cls", "last", "mean")
                         Default: None (uses model default of "last")
                         Only applicable to BiGemma3 models

    Examples:
        # Evaluate BiGemma3 on Vidore benchmark
        modal run modal_mteb_local.py --model bigemma3-4b-embedding

        # Evaluate BiGemma3 with custom batch size
        modal run modal_mteb_local.py --model bigemma3-4b-embedding --batch-size 8

        # Evaluate BiGemma3 with mean pooling strategy
        modal run modal_mteb_local.py --model bigemma3-4b-embedding --pooling-strategy mean

        # Evaluate BiGemma3 with cls pooling
        modal run modal_mteb_local.py --model bigemma3-4b-embedding --pooling-strategy cls --batch-size 8

        # Evaluate multiple models in batch mode
        modal run modal_mteb_local.py --model "bigemma3-4b-embedding,vidore/colqwen2.5-v0.2" --batch-mode

        # Evaluate on specific Vidore tasks
        modal run modal_mteb_local.py --model bigemma3-4b-embedding --benchmarks "ViDoRe(v1)"

        # List available benchmarks
        modal run modal_mteb_local.py::list_benchmarks
    """
    print("Using LOCAL mteb directory for evaluation")
    print("=" * 50)

    benchmark_list = None
    if benchmarks.strip():
        benchmark_list = [b.strip() for b in benchmarks.split(",")]

    if batch_mode:
        model_names = [m.strip() for m in model.split(",")]
        print(f"Running batch evaluation for {len(model_names)} models in parallel")
        print(f"Models: {', '.join(model_names)}")

        # Spawn parallel evaluation jobs for each model
        evaluation_calls = []
        for model_name in model_names:
            print(f"Starting evaluation job for: {model_name}")
            call = run_mteb_evaluation.spawn(
                model_name=model_name,
                benchmarks=benchmark_list,
                output_folder=output_folder,
                batch_size=batch_size,
                pooling_strategy=pooling_strategy,
            )
            evaluation_calls.append((model_name, call))

        # Collect results as they complete
        results = []
        for model_name, call in evaluation_calls:
            try:
                print(f"Waiting for results from: {model_name}")
                result = call.get()
                results.append(result)
                print(f"✓ {model_name}: {result.get('status', 'unknown')}")

                if result.get("status") == "completed":
                    local_filepath = save_results_locally(result, output_folder)
                    print(f"  Results saved to: {local_filepath}")
            except Exception as e:
                print(f"✗ {model_name}: Failed - {str(e)}")
                error_result = {
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "mteb_source": "local",
                }
                results.append(error_result)

        print(
            f"\nBatch evaluation completed: {len(results)}/{len(model_names)} models processed"
        )
        return results

    else:
        print(f"Running evaluation for single model: {model}")

        result = run_mteb_evaluation.remote(
            model_name=model,
            benchmarks=benchmark_list,
            output_folder=output_folder,
            batch_size=batch_size,
            pooling_strategy=pooling_strategy,
        )

        print(f"Evaluation completed: {result.get('status', 'unknown')}")

        if result:
            local_filepath = save_results_locally(result, output_folder)
            print(f"Results saved locally to: {local_filepath}")

        return result


if __name__ == "__main__":
    main()


# modal run modal_mteb_local.py::main \
#     --model "google/gemma-3-4b-it-bigemma3" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 8

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
#     --benchmarks "NayanaIR-Bench" \
#     --batch-size 12


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
#     --benchmarks "ViDoRe(v1)" \
#     --batch-size 12


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/Full-SFT-v1-23000" \
#     --benchmarks "NayanaIR-Bench" \
#     --batch-size 12


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/Full_SFT_v2_base_gemma_merged_1400" \
#     --benchmarks "NayanaIR-Bench" \
#     --batch-size 12


# ============================================================================
# HardNeg Models - ViDoRe v2 Evaluation (Last Token Pooling)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-252" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1950" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12


# ============================================================================
# InBatch Models - ViDoRe v2 Evaluation (Last Token Pooling)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1750" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3694" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12


# ============================================================================
# MeanPooling-HardNeg Models - ViDoRe v2 Evaluation (Mean Pooling)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1750" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-2000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# ============================================================================
# Matryoshka Models - ViDoRe v2 Evaluation (Last Token Pooling)
# ============================================================================


# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-merged-2000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# ============================================================================
# Multiple models evaluation - Batch Mode
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-Multilingual-Hi-En-Kn-merged-1386" \
#     --benchmarks "NayanaIR-Bench-v1" \
#     --batch-size 12\

# ============================================================================
# BiDocling Models - ViDoRe v2 Evaluation (Last Token Pooling)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "ibm-granite/granite-docling-258M-bidocling" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 16

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-BiDocling-MultiGPU-2500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 16

# ============================================================================
# ColGemma3 Models - ViDoRe v2 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "google/gemma-3-4b-it-colgemma3" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 8

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-2500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 8

# ============================================================================
# NayanaEmbed-ColPali Models - ViDoRe v2 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColPali-v1.3-1500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColPali-v1.3-2772" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# ============================================================================
# NayanaEmbed-ColQwen2 Models - ViDoRe v2 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColQwen2-v1.0-3000" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# ============================================================================
# NayanaEmbed-ColPali Models - NayanaIR-Bench-v12 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColPali-v1.3-1500" \
#     --benchmarks "NayanaIR-Bench-v12" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColPali-v1.3-2772" \
#     --benchmarks "NayanaIR-Bench-v12" \
#     --batch-size 12

# ============================================================================
# NayanaEmbed-ColQwen2 Models - NayanaIR-Bench-v12 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColQwen2-v1.0-3000" \
#     --benchmarks "NayanaIR-Bench-v12" \
#     --batch-size 12

# ============================================================================
# NayanaEmbed-ColSmol Models - ViDoRe v2 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColSmol-256M-2500" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColSmol-256M-5544" \
#     --benchmarks "ViDoRe(v2)" \
#     --batch-size 12

# ============================================================================
# NayanaEmbed-ColSmol Models - NayanaIR-Bench-v12 Evaluation (Multi-Vector)
# ============================================================================

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColSmol-256M-2500" \
#     --benchmarks "NayanaIR-Bench-v12" \
#     --batch-size 12

# modal run modal_mteb_local.py::main \
#     --model "Nayana-cognitivelab/NayanaEmbed-ColSmol-256M-5544" \
#     --benchmarks "NayanaIR-Bench-v12" \
#     --batch-size 12
