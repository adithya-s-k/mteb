#!/usr/bin/env python3
"""
Auto-calibrating MTEB results visualizer with interactive charts.

Features:
- Zero configuration - extracts everything from JSON structure
- Interactive Chart.js graphs with hover tooltips
- Recursive directory scanning
- Auto-detects benchmarks, models, tasks, and metrics
- Works with any MTEB-formatted results

Usage:
    python generate_viz.py [path_to_results_dir]
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple


def extract_model_display_name(model_path: str) -> str:
    """
    Auto-extract a clean display name from model path.

    Examples:
        "vidore/colpali-v1.3" → "colpali-v1.3"
        "google/gemma-3-4b-it" → "gemma-3-4b-it"
        "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" → "HardNeg-1694"
    """
    # Take last part after /
    name = model_path.split("/")[-1]

    # Smart shortening for long names with version numbers
    if "HardNeg" in name and "merged" in name:
        # Extract version number
        match = re.search(r"merged-(\d+)", name)
        if match:
            return f"HardNeg-{match.group(1)}"
        return "HardNeg"

    # For base models
    if "gemma-3-4b-it" in name:
        return "Gemma3-Base"

    return name


def format_task_name(raw_name: str) -> str:
    """
    Auto-format CamelCase task names to readable format.

    Examples:
        "Vidore2ESGReportsRetrieval" → "Vidore2 ESG Reports Retrieval"
        "NayanaVisionRetrieval" → "Nayana Vision Retrieval"
    """
    # Insert spaces before capital letters
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", raw_name)

    # Clean up multiple spaces
    spaced = re.sub(r"\s+", " ", spaced)

    return spaced.strip()


def extract_benchmark_name(json_data: Dict[str, Any], file_path: Path) -> str:
    """
    Auto-detect benchmark name from JSON or directory structure.

    Priority:
    1. JSON 'benchmarks' field
    2. Parent directory name
    3. Filename pattern
    """
    # Try JSON field first
    benchmarks = json_data.get("benchmarks", [])
    if benchmarks and len(benchmarks) > 0:
        return benchmarks[0]

    # Try parent directory name
    parent = file_path.parent.name
    if parent and parent != "results":
        # Clean up directory name
        return parent.replace("_", " ").replace("(", " ").replace(")", "").title()

    # Fallback
    return "Unknown Benchmark"


def generate_color_palette(num_colors: int) -> List[str]:
    """
    Generate N visually distinct colors using HSL color space.
    Colors are print-friendly and work in grayscale.
    """
    if num_colors <= 0:
        return []

    # Predefined palette for common cases (better than algorithmic for small N)
    predefined = [
        "#000000",  # Black
        "#CC0000",  # Dark Red
        "#0066CC",  # Dark Blue
        "#006600",  # Dark Green
        "#660099",  # Dark Purple
        "#CC6600",  # Dark Orange
        "#006666",  # Dark Teal
        "#663300",  # Dark Brown
    ]

    if num_colors <= len(predefined):
        return predefined[:num_colors]

    # Generate additional colors if needed
    colors = predefined.copy()
    for i in range(len(predefined), num_colors):
        hue = (i * 360 / num_colors) % 360
        # Use dark, saturated colors for visibility
        colors.append(f"hsl({hue}, 70%, 35%)")

    return colors


def discover_available_metrics(results: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Auto-discover which metrics are available across all results.
    Returns list of (metric_key, display_name) tuples.
    """
    all_metrics = set()

    for result in results:
        tasks = result.get("results", {}).get("task_results", [])
        for task in tasks:
            scores = task.get("scores", {}).get("test", [{}])
            if scores:
                all_metrics.update(scores[0].keys())

    # Common metrics in preferred order
    preferred_metrics = [
        ("ndcg_at_1", "NDCG@1"),
        ("ndcg_at_5", "NDCG@5"),
        ("ndcg_at_10", "NDCG@10"),
        ("recall_at_1", "Recall@1"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("recall_at_20", "Recall@20"),
        ("recall_at_100", "Recall@100"),
        ("map_at_10", "MAP@10"),
        ("mrr_at_10", "MRR@10"),
    ]

    # Return only metrics that are available
    available = [(key, name) for key, name in preferred_metrics if key in all_metrics]

    return available


def scan_directory_recursive(path: Path) -> List[Path]:
    """Recursively scan directory for JSON files."""
    json_files = []

    if path.is_file() and path.suffix == ".json":
        return [path]

    if path.is_dir():
        json_files.extend(path.rglob("*.json"))

    return sorted(json_files)


def load_results(path: Path = None) -> List[Dict[str, Any]]:
    """Load all completed JSON results from path, with auto-enrichment."""
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    json_files = scan_directory_recursive(path)
    results = []

    print(f"📁 Scanning: {path}")
    print(f"🔍 Found {len(json_files)} JSON file(s)")

    for file in json_files:
        try:
            with open(file) as f:
                data = json.load(f)

                # Only process completed results
                if data.get("status") != "completed":
                    continue

                # Auto-enrich with extracted metadata
                data["_display_name"] = extract_model_display_name(
                    data.get("model", "Unknown")
                )
                data["_benchmark"] = extract_benchmark_name(data, file)
                data["_source_file"] = str(file)

                results.append(data)

        except Exception as e:
            print(f"⚠️  Could not load {file.name}: {e}")

    print(f"✅ Loaded {len(results)} completed result(s)")
    return results


def calculate_avg(result: Dict[str, Any], metric: str) -> float:
    """Calculate average of a metric across all tasks."""
    tasks = result.get("results", {}).get("task_results", [])
    values = [t.get("scores", {}).get("test", [{}])[0].get(metric, 0) for t in tasks]
    return sum(values) / len(values) if values else 0


def organize_by_benchmark(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by benchmark."""
    by_benchmark = {}

    for result in results:
        benchmark = result["_benchmark"]
        if benchmark not in by_benchmark:
            by_benchmark[benchmark] = []
        by_benchmark[benchmark].append(result)

    return by_benchmark


def generate_html(results: List[Dict[str, Any]]) -> str:
    """Generate interactive HTML with Chart.js visualizations."""

    if not results:
        return "<html><body><h1>No results found</h1></body></html>"

    # Organize by benchmark
    by_benchmark = organize_by_benchmark(results)

    # Generate color palette
    all_models = list(set(r["_display_name"] for r in results))
    colors = generate_color_palette(len(all_models))
    model_colors = dict(zip(all_models, colors))

    # Discover available metrics
    available_metrics = discover_available_metrics(results)

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MTEB Results - Interactive</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        @media print {
            body { margin: 0; padding: 10px; }
            .no-print { display: none; }
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        h1 {
            font-size: 28px;
            margin: 0 0 10px 0;
            font-weight: 600;
            color: #000;
        }

        h2 {
            font-size: 20px;
            margin: 30px 0 15px 0;
            font-weight: 600;
            color: #000;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }

        h3 {
            font-size: 16px;
            margin: 25px 0 12px 0;
            font-weight: 600;
            color: #333;
        }

        .timestamp {
            font-size: 13px;
            color: #666;
            margin-bottom: 20px;
        }

        .benchmark-section {
            margin-bottom: 50px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        thead {
            background: #f8f8f8;
        }

        th {
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            color: #666;
            border-bottom: 2px solid #ddd;
            cursor: pointer;
            user-select: none;
        }

        th:hover {
            background: #e8e8e8;
        }

        td {
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
        }

        .metric {
            text-align: right;
            font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
        }

        .model-name {
            font-weight: 600;
            color: #000;
        }

        .model-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
            vertical-align: middle;
        }

        .avg-row {
            background: #f0f7ff !important;
            font-weight: 600;
        }

        tbody tr:hover {
            background: #f9f9f9;
        }

        .chart-container {
            margin: 25px 0;
            padding: 20px;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .chart-wrapper {
            position: relative;
            height: 400px;
            margin-top: 15px;
        }

        .info-box {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }

        .info-box strong {
            display: block;
            margin-bottom: 5px;
            color: #1565c0;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .stat-card {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }

        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #000;
        }
    </style>
</head>
<body>
<div class="container">
"""

    # Header
    html += f"""
    <h1>📊 MTEB Evaluation Results</h1>
    <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

    <div class="info-box">
        <strong>Interactive Features:</strong>
        • Hover over chart points for detailed metrics<br>
        • Click legend items to show/hide models<br>
        • Click table headers to sort columns
    </div>
"""

    # Stats overview
    html += f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Models</div>
            <div class="stat-value">{len(all_models)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Benchmarks</div>
            <div class="stat-value">{len(by_benchmark)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Results</div>
            <div class="stat-value">{len(results)}</div>
        </div>
    </div>
"""

    # Process each benchmark
    for benchmark_name, benchmark_results in by_benchmark.items():
        html += f"""
    <div class="benchmark-section">
        <h2>📋 {benchmark_name}</h2>
"""

        # Average metrics table
        html += """
        <h3>Average Metrics Across All Tasks</h3>
        <table class="sortable">
            <thead>
                <tr>
                    <th>Model</th>
"""

        for _, display_name in available_metrics:
            html += f"                    <th>{display_name}</th>\n"

        html += """
                </tr>
            </thead>
            <tbody>
"""

        # Sort by primary metric (ndcg_at_5 or first available)
        primary_metric = (
            available_metrics[1][0]
            if len(available_metrics) > 1
            else available_metrics[0][0]
        )
        sorted_results = sorted(
            benchmark_results,
            key=lambda r: calculate_avg(r, primary_metric),
            reverse=True,
        )

        for result in sorted_results:
            model_name = result["_display_name"]
            color = model_colors[model_name]

            html += f"""
            <tr class="avg-row">
                <td class="model-name">
                    <span class="model-indicator" style="background-color: {color};"></span>
                    {model_name}
                </td>
"""

            for metric_key, _ in available_metrics:
                value = calculate_avg(result, metric_key)
                html += f'                <td class="metric">{value * 100:.2f}%</td>\n'

            html += "            </tr>\n"

        html += """
            </tbody>
        </table>
"""

        # Interactive Charts
        html += """
        <h3>Performance Charts</h3>
        <div class="chart-container">
            <strong>NDCG@k Metrics</strong>
            <div class="chart-wrapper">
                <canvas id="ndcgChart_{bench_id}"></canvas>
            </div>
        </div>
        <div class="chart-container">
            <strong>Recall@k Metrics</strong>
            <div class="chart-wrapper">
                <canvas id="recallChart_{bench_id}"></canvas>
            </div>
        </div>
""".replace(
            "{bench_id}",
            benchmark_name.replace(" ", "_").replace("(", "").replace(")", ""),
        )

        # Per-task breakdown
        html += """
        <h3>Results by Task</h3>
"""

        # Get all unique tasks for this benchmark
        all_tasks = set()
        for result in benchmark_results:
            for task in result.get("results", {}).get("task_results", []):
                all_tasks.add(task["task_name"])

        for task_name in sorted(all_tasks):
            readable_name = format_task_name(task_name)

            html += f"""
        <h4>{readable_name}</h4>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
"""

            for _, display_name in available_metrics:
                html += f"                    <th>{display_name}</th>\n"

            html += """
                </tr>
            </thead>
            <tbody>
"""

            for result in benchmark_results:
                tasks = result.get("results", {}).get("task_results", [])
                task = next((t for t in tasks if t["task_name"] == task_name), None)

                if task:
                    model_name = result["_display_name"]
                    color = model_colors[model_name]
                    scores = task.get("scores", {}).get("test", [{}])[0]

                    html += f"""
                <tr>
                    <td class="model-name">
                        <span class="model-indicator" style="background-color: {color};"></span>
                        {model_name}
                    </td>
"""

                    for metric_key, _ in available_metrics:
                        value = scores.get(metric_key, 0)
                        html += f'                    <td class="metric">{value * 100:.2f}%</td>\n'

                    html += "                </tr>\n"

            html += """
            </tbody>
        </table>
"""

        html += "    </div>\n"  # End benchmark-section

    # JavaScript for interactive charts
    html += (
        """
    <script>
        // Prepare data for Chart.js
        const benchmarkData = """
        + json.dumps(
            [
                {
                    "benchmark": result["_benchmark"],
                    "model": result["_display_name"],
                    "color": model_colors[result["_display_name"]],
                    "tasks": [
                        {
                            "name": format_task_name(t["task_name"]),
                            "raw_name": t["task_name"],
                            "scores": t.get("scores", {}).get("test", [{}])[0],
                        }
                        for t in result.get("results", {}).get("task_results", [])
                    ],
                }
                for result in results
            ]
        )
        + """;

        // Chart configuration
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'point',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: {
                                size: 12
                            }
                        },
                        onClick: (e, legendItem, legend) => {
                            const index = legendItem.datasetIndex;
                            const chart = legend.chart;
                            const meta = chart.getDatasetMeta(index);
                            meta.hidden = !meta.hidden;
                            chart.update();
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleFont: {
                            size: 13,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 12
                        },
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = (context.parsed.y * 100).toFixed(2) + '%';
                                const k = context.label;
                                return `${label}: ${value} (${k})`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Performance',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'k Value',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        };

        // Function to create chart for specific metric
        function createChart(canvasId, metric, benchmarkFilter) {
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;

            const kValues = [1, 5, 10, 20, 100];
            const labels = kValues.map(k => '@' + k);

            // Filter data for this benchmark
            const filteredData = benchmarkData.filter(d => d.benchmark === benchmarkFilter);

            const datasets = filteredData.map(result => {
                const data = kValues.map(k => {
                    const metricKey = metric + '_at_' + k;
                    const tasks = result.tasks;
                    const values = tasks.map(t => t.scores[metricKey] || 0);
                    return values.reduce((a, b) => a + b, 0) / values.length;
                });

                return {
                    label: result.model,
                    data: data,
                    borderColor: result.color,
                    backgroundColor: result.color,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.1
                };
            });

            new Chart(ctx, {
                ...chartConfig,
                data: {
                    labels: labels,
                    datasets: datasets
                }
            });
        }

        // Create charts for each benchmark
"""
        + "\n".join(
            [
                f"""
        createChart('ndcgChart_{benchmark.replace(" ", "_").replace("(", "").replace(")", "")}', 'ndcg', '{benchmark}');
        createChart('recallChart_{benchmark.replace(" ", "_").replace("(", "").replace(")", "")}', 'recall', '{benchmark}');
"""
                for benchmark in by_benchmark.keys()
            ]
        )
        + """

        // Table sorting functionality
        document.querySelectorAll('table.sortable th').forEach((th, index) => {
            if (index === 0) return; // Skip model name column

            th.addEventListener('click', function() {
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));

                const isAscending = th.classList.contains('asc');

                rows.sort((a, b) => {
                    const aVal = parseFloat(a.cells[index].textContent);
                    const bVal = parseFloat(b.cells[index].textContent);
                    return isAscending ? bVal - aVal : aVal - bVal;
                });

                rows.forEach(row => tbody.appendChild(row));

                // Toggle sort direction
                table.querySelectorAll('th').forEach(h => h.classList.remove('asc', 'desc'));
                th.classList.add(isAscending ? 'desc' : 'asc');
            });
        });
    </script>
</div>
</body>
</html>
"""
    )

    return html


def main():
    # Get path from arguments
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    print("=" * 60)
    print("📊 MTEB Results Auto-Calibrating Visualizer")
    print("=" * 60)

    results = load_results(path)

    if not results:
        print("❌ No completed results found!")
        print("Make sure JSON files with status='completed' exist in the directory.")
        sys.exit(1)

    # Generate visualization
    html = generate_html(results)

    # Save output
    output_file = (
        path / "results_visualization.html"
        if path.is_dir()
        else Path("results_visualization.html")
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Generated: {output_file}")
    print(f"\n🌐 Open with: open {output_file}")
    print("\n✨ Features:")
    print("  • Interactive Chart.js graphs with hover tooltips")
    print("  • Click legend to toggle models on/off")
    print("  • Click table headers to sort")
    print("  • Auto-detected benchmarks, models, and metrics")
    print("  • Print-friendly design")
    print("=" * 60)


if __name__ == "__main__":
    main()


# generate_viz.py

# generate_viz.py /path/to/results
