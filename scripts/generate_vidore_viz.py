#!/usr/bin/env python3
"""
Generate minimal, print-friendly HTML visualization of MTEB results.

Usage:
    python generate_viz.py [path_to_results_dir]

If no path provided, searches current directory for JSON files.
"""

import json
import glob
import sys
from pathlib import Path

# Task name mapping
TASK_NAMES = {
    'Vidore2BioMedicalLecturesRetrieval': 'MIT Biomedical Multilingual',
    'Vidore2EconomicsReportsRetrieval': 'Economics Macro Multilingual',
    'Vidore2ESGReportsHLRetrieval': 'ESG Restaurant Human English',
    'Vidore2ESGReportsRetrieval': 'ESG Restaurant Synthetic Multilingual',
}

def get_task_name(raw_name):
    """Map raw task name to readable name."""
    return TASK_NAMES.get(raw_name, raw_name)

def get_model_name(full_name):
    """Get short model name."""
    if 'HardNeg' in full_name:
        return 'HardNeg'
    if 'gemma-3-4b-it' in full_name:
        return 'Base'
    return full_name.split('/')[-1]

def load_results(path=None):
    """Load all completed JSON results from path."""
    if path:
        search_path = Path(path)
    else:
        search_path = Path.cwd()

    json_files = list(search_path.glob('*.json'))
    results = []

    for file in json_files:
        try:
            with open(file) as f:
                data = json.load(f)
                if data.get('status') == 'completed':
                    results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")

    return results

def calculate_avg(result, metric):
    """Calculate average of a metric across all tasks."""
    tasks = result.get('results', {}).get('task_results', [])
    values = [t.get('scores', {}).get('test', [{}])[0].get(metric, 0) for t in tasks]
    return sum(values) / len(values) if values else 0

def generate_html(results):
    """Generate minimal, print-friendly HTML."""

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MTEB Results</title>
    <style>
        @media print {
            body { margin: 0; padding: 10px; }
            .no-print { display: none; }
        }

        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            color: black;
            font-size: 11px;
            line-height: 1.3;
        }

        h1 {
            font-size: 18px;
            margin: 0 0 5px 0;
            font-weight: 600;
            border-bottom: 2px solid black;
            padding-bottom: 5px;
        }

        h2 {
            font-size: 13px;
            margin: 20px 0 8px 0;
            font-weight: 600;
            border-bottom: 1px solid black;
            padding-bottom: 3px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 10px;
        }

        th, td {
            padding: 4px 6px;
            text-align: left;
            border: 1px solid #000;
        }

        th {
            background: #f0f0f0;
            font-weight: 600;
            font-size: 9px;
            text-transform: uppercase;
        }

        td {
            background: white;
        }

        .metric {
            text-align: right;
            font-family: 'Courier New', monospace;
            font-size: 10px;
        }

        .model-name {
            font-weight: 600;
        }

        .avg-row {
            background: #e8e8e8 !important;
            font-weight: 600;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin: 15px 0;
        }

        .stat-box {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }

        .stat-label {
            font-size: 9px;
            text-transform: uppercase;
            margin-bottom: 3px;
        }

        .stat-value {
            font-size: 16px;
            font-weight: 600;
        }

        .chart-container {
            margin: 15px 0;
            page-break-inside: avoid;
        }

        canvas {
            border: 1px solid black;
            display: block;
            margin: 10px 0;
        }

        .timestamp {
            font-size: 9px;
            color: #666;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
"""

    # Header
    from datetime import datetime
    html += f"""
    <h1>MTEB Vidore Evaluation Results</h1>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""

    # Average metrics table
    html += """
    <h2>Average Metrics Across All Tasks</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>NDCG@1</th>
                <th>NDCG@5</th>
                <th>NDCG@10</th>
                <th>Recall@1</th>
                <th>Recall@5</th>
                <th>Recall@10</th>
                <th>Recall@20</th>
                <th>Recall@100</th>
                <th>MAP@10</th>
                <th>MRR@10</th>
            </tr>
        </thead>
        <tbody>
"""

    for result in sorted(results, key=lambda r: calculate_avg(r, 'ndcg_at_5'), reverse=True):
        model = get_model_name(result.get('model', 'Unknown'))

        metrics = {
            'ndcg_at_1': calculate_avg(result, 'ndcg_at_1'),
            'ndcg_at_5': calculate_avg(result, 'ndcg_at_5'),
            'ndcg_at_10': calculate_avg(result, 'ndcg_at_10'),
            'recall_at_1': calculate_avg(result, 'recall_at_1'),
            'recall_at_5': calculate_avg(result, 'recall_at_5'),
            'recall_at_10': calculate_avg(result, 'recall_at_10'),
            'recall_at_20': calculate_avg(result, 'recall_at_20'),
            'recall_at_100': calculate_avg(result, 'recall_at_100'),
            'map_at_10': calculate_avg(result, 'map_at_10'),
            'mrr_at_10': calculate_avg(result, 'mrr_at_10'),
        }

        html += f"""
            <tr class="avg-row">
                <td class="model-name">{model}</td>
                <td class="metric">{metrics['ndcg_at_1']*100:.2f}%</td>
                <td class="metric">{metrics['ndcg_at_5']*100:.2f}%</td>
                <td class="metric">{metrics['ndcg_at_10']*100:.2f}%</td>
                <td class="metric">{metrics['recall_at_1']*100:.2f}%</td>
                <td class="metric">{metrics['recall_at_5']*100:.2f}%</td>
                <td class="metric">{metrics['recall_at_10']*100:.2f}%</td>
                <td class="metric">{metrics['recall_at_20']*100:.2f}%</td>
                <td class="metric">{metrics['recall_at_100']*100:.2f}%</td>
                <td class="metric">{metrics['map_at_10']*100:.2f}%</td>
                <td class="metric">{metrics['mrr_at_10']*100:.2f}%</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
"""

    # Charts
    html += """
    <h2>Performance Charts</h2>
    <div class="chart-container">
        <canvas id="ndcgChart" width="1100" height="300"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="recallChart" width="1100" height="300"></canvas>
    </div>
"""

    # Per-task breakdown
    html += """
    <h2>Results by Task</h2>
"""

    # Get all unique tasks
    all_tasks = set()
    for result in results:
        for task in result.get('results', {}).get('task_results', []):
            all_tasks.add(task['task_name'])

    for task_name in sorted(all_tasks):
        readable_name = get_task_name(task_name)

        html += f"""
    <h2>{readable_name}</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>NDCG@1</th>
                <th>NDCG@5</th>
                <th>NDCG@10</th>
                <th>Recall@1</th>
                <th>Recall@5</th>
                <th>Recall@10</th>
                <th>Recall@20</th>
                <th>Recall@100</th>
                <th>MAP@10</th>
            </tr>
        </thead>
        <tbody>
"""

        for result in results:
            tasks = result.get('results', {}).get('task_results', [])
            task = next((t for t in tasks if t['task_name'] == task_name), None)

            if task:
                model = get_model_name(result.get('model', 'Unknown'))
                scores = task.get('scores', {}).get('test', [{}])[0]

                html += f"""
            <tr>
                <td class="model-name">{model}</td>
                <td class="metric">{scores.get('ndcg_at_1', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('ndcg_at_5', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('ndcg_at_10', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('recall_at_1', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('recall_at_5', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('recall_at_10', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('recall_at_20', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('recall_at_100', 0)*100:.2f}%</td>
                <td class="metric">{scores.get('map_at_10', 0)*100:.2f}%</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
"""

    # JavaScript for charts
    html += """
    <script>
        const results = """ + json.dumps([{
            'model': get_model_name(r.get('model', '')),
            'tasks': [{
                'name': get_task_name(t['task_name']),
                'scores': t.get('scores', {}).get('test', [{}])[0]
            } for t in r.get('results', {}).get('task_results', [])]
        } for r in results]) + """;

        function drawChart(canvasId, metric, title) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;

            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, w, h);

            // Title
            ctx.fillStyle = 'black';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(title, 10, 20);

            const kValues = [1, 5, 10, 20, 100];
            const padding = { top: 40, right: 30, bottom: 50, left: 60 };
            const chartW = w - padding.left - padding.right;
            const chartH = h - padding.top - padding.bottom;

            // Grid
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = padding.top + (i * chartH / 5);
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(w - padding.right, y);
                ctx.stroke();

                // Y-axis labels
                ctx.fillStyle = 'black';
                ctx.font = '10px Arial';
                ctx.textAlign = 'right';
                ctx.fillText((100 - i * 20) + '%', padding.left - 5, y + 3);
            }

            // X-axis labels
            kValues.forEach((k, i) => {
                const x = padding.left + (i * chartW / (kValues.length - 1));
                ctx.fillStyle = 'black';
                ctx.textAlign = 'center';
                ctx.fillText('@' + k, x, h - padding.bottom + 20);
            });

            // Plot lines
            const lineStyles = [
                { color: 'black', width: 2, dash: [] },
                { color: '#666', width: 2, dash: [5, 3] }
            ];

            results.forEach((result, idx) => {
                const tasks = result.tasks;
                const avgValues = kValues.map(k => {
                    const metricKey = metric + '_at_' + k;
                    const values = tasks.map(t => t.scores[metricKey] || 0);
                    return values.reduce((a, b) => a + b, 0) / values.length;
                });

                const style = lineStyles[idx % lineStyles.length];
                ctx.strokeStyle = style.color;
                ctx.lineWidth = style.width;
                ctx.setLineDash(style.dash);

                ctx.beginPath();
                avgValues.forEach((val, i) => {
                    const x = padding.left + (i * chartW / (kValues.length - 1));
                    const y = padding.top + ((1 - val) * chartH);

                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);

                    // Draw point
                    ctx.fillStyle = style.color;
                    ctx.fillRect(x - 3, y - 3, 6, 6);
                });
                ctx.stroke();
                ctx.setLineDash([]);

                // Legend
                const legendY = padding.top + idx * 15;
                ctx.strokeStyle = style.color;
                ctx.lineWidth = style.width;
                ctx.setLineDash(style.dash);
                ctx.beginPath();
                ctx.moveTo(w - padding.right - 100, legendY);
                ctx.lineTo(w - padding.right - 70, legendY);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = 'black';
                ctx.font = '10px Arial';
                ctx.textAlign = 'left';
                ctx.fillText(result.model, w - padding.right - 65, legendY + 3);
            });
        }

        drawChart('ndcgChart', 'ndcg', 'NDCG@k - Averaged Across All Tasks');
        drawChart('recallChart', 'recall', 'Recall@k - Averaged Across All Tasks');
    </script>
</body>
</html>
"""

    return html

def main():
    # Get path from arguments
    path = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Loading results from: {path or 'current directory'}")
    results = load_results(path)

    if not results:
        print("❌ No completed results found!")
        print("Make sure JSON files with status='completed' exist in the directory.")
        sys.exit(1)

    print(f"✓ Found {len(results)} completed evaluation(s)")

    html = generate_html(results)

    output_file = 'results_visualization.html'
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✓ Generated {output_file}")
    print(f"\nOpen with: open {output_file}")
    print("\nPrint-friendly features:")
    print("  - Minimal black/white design")
    print("  - Tables with all metrics")
    print("  - Line charts (NDCG@k and Recall@k)")
    print("  - Task names mapped to readable format")

if __name__ == '__main__':
    main()
