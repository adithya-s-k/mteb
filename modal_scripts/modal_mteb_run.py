from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import modal

app = modal.App("mteb-vidore-benchmark")

# Define the image with required dependencies using new Image Builder
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "datasets",
        "numpy",
        "scikit-learn",
        "requests",
        "tqdm",
        "pillow",
        "huggingface_hub",
        "accelerate",
        "colpali-engine",
    )
    .run_commands(
        "cd /tmp && git clone https://github.com/adithya-s-k/mteb.git",
        "cd /tmp/mteb && git checkout nayana-bench",
        "cd /tmp/mteb && uv pip install --python $(command -v python) -e .",
    )
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
    """List all available ViDoRe benchmarks."""
    import sys

    sys.path.append("/tmp/mteb")

    import mteb

    try:
        vidore_benchmarks = ["ViDoRe(v1)", "ViDoRe(v2)"]

        available_benchmarks = []
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

        return {
            "available_benchmarks": available_benchmarks,
            "total_count": len(available_benchmarks),
        }
    except Exception as e:
        return {"error": str(e)}


@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 60 * 60,
    volumes={"/cache": modal.Volume.from_name("mteb-cache", create_if_missing=True)},
)
def run_mteb_evaluation(
    model_name: str,
    benchmarks: Optional[list[str]] = None,
    output_folder: str = "results",
    cache_folder: str = "/cache",
    batch_size: Optional[int] = None,
) -> dict:
    """Run MTEB evaluation for specified model and ViDoRe benchmarks.

    Args:
        model_name: HuggingFace model name or path
        benchmarks: List of benchmark names to evaluate (default: ViDoRe v1 and v2)
        output_folder: Folder to save results
        cache_folder: Cache folder for datasets and models
        batch_size: Batch size for model inference (optional)

    Returns:
        Dictionary containing evaluation results
    """
    import os
    import sys

    sys.path.append("/tmp/mteb")

    import mteb
    from mteb import MTEB

    os.environ["HF_HOME"] = cache_folder
    os.environ["TRANSFORMERS_CACHE"] = cache_folder

    print(f"Running evaluation for model: {model_name}")

    if benchmarks is None:
        benchmarks = ["ViDoRe(v1)", "ViDoRe(v2)"]

    print(f"Benchmarks to evaluate: {benchmarks}")

    try:
        # Load model
        model = mteb.get_model(model_name)
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
        }

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "model": model_name,
            "benchmarks": benchmarks,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.function(
    image=image,
    timeout=14400,
    volumes={"/cache": modal.Volume.from_name("mteb-cache", create_if_missing=True)},
)
def batch_evaluate_models(
    model_names: list[str],
    benchmarks: Optional[list[str]] = None,
    output_folder: str = "results",
    batch_size: Optional[int] = None,
) -> list[dict]:
    """Run MTEB evaluation for multiple models.

    Args:
        model_names: List of model names to evaluate
        benchmarks: List of benchmark names (default: ViDoRe v1 and v2)
        output_folder: Output folder for results
        batch_size: Batch size for model inference (optional)

    Returns:
        List of evaluation results
    """
    results = []

    for model_name in model_names:
        print(f"Starting evaluation for {model_name}")
        try:
            result = run_mteb_evaluation.remote(
                model_name=model_name,
                benchmarks=benchmarks,
                output_folder=f"{output_folder}/{model_name.replace('/', '_')}",
                batch_size=batch_size,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to start evaluation for {model_name}: {e}")
            error_result = {
                "model": model_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(error_result)

    return results


def save_results_locally(
    results: dict[str, Any], output_folder: str = "results"
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
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Results saved to: {filepath}")
    return filepath


@app.local_entrypoint()
def list_benchmarks():
    """List all available ViDoRe benchmarks."""
    result = list_available_benchmarks.remote()
    print("Available ViDoRe Benchmarks:")
    print("=" * 50)
    print(f"Total benchmarks: {result.get('total_count', 0)}")
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
    model: str = "vidore/colqwen2.5-v0.2",
    benchmarks: str = "",
    output_folder: str = "../../results",
    batch_mode: bool = False,
    batch_size: int = None,
):
    """Main entry point for MTEB ViDoRe evaluation.

    Args:
        model: Model name or comma-separated list of models for batch mode
        benchmarks: Comma-separated list of benchmark names (empty for default ViDoRe v1,v2)
        output_folder: Output folder for results
        batch_mode: Whether to run in batch mode for multiple models
        batch_size: Batch size for model inference (optional)
    """
    benchmark_list = None
    if benchmarks.strip():
        benchmark_list = [b.strip() for b in benchmarks.split(",")]

    if batch_mode:
        model_names = [m.strip() for m in model.split(",")]
        print(f"Running batch evaluation for {len(model_names)} models")

        results = batch_evaluate_models.remote(
            model_names=model_names,
            benchmarks=benchmark_list,
            output_folder=output_folder,
            batch_size=batch_size,
        )

        print("Batch evaluation completed")

        for result in results:
            print(f"Model: {result['model']} - Status: {result['status']}")
            if result.get("status") == "completed":
                local_filepath = save_results_locally(result, output_folder)
                print(f"Results for {result['model']} saved to: {local_filepath}")

        return results

    else:
        print(f"Running evaluation for single model: {model}")

        result = run_mteb_evaluation.remote(
            model_name=model,
            benchmarks=benchmark_list,
            output_folder=output_folder,
            batch_size=batch_size,
        )

        print(f"Evaluation completed: {result.get('status', 'unknown')}")

        if result:
            local_filepath = save_results_locally(result, output_folder)
            print(f"Results saved locally to: {local_filepath}")

        return result


if __name__ == "__main__":
    main()
