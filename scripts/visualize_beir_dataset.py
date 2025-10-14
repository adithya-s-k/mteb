#!/usr/bin/env python3
"""BEIR Dataset Visualizer.

Creates an interactive HTML visualization of BEIR format datasets from HuggingFace.
Handles flexible column structures while focusing on core BEIR columns.
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

try:
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Please install required packages:")
    print("pip install datasets pillow tqdm")
    exit(1)


def save_image_to_file(image: Image.Image, output_path: Path) -> str:
    """Save PIL Image to file and return relative path."""
    image.save(output_path)
    return str(output_path.name)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for embedding in HTML."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_beir_dataset(dataset_path: str, num_samples: int = None):
    """Load BEIR format dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_path}")

    # Load all subsets
    corpus = load_dataset(dataset_path, "corpus", split="test")
    queries = load_dataset(dataset_path, "queries", split="test")
    qrels = load_dataset(dataset_path, "qrels", split="test")

    # Sample if needed
    if num_samples:
        # Sample queries first
        queries = queries.select(range(min(num_samples, len(queries))))
        query_ids = set(queries["query-id"])

        # Filter qrels to only include sampled queries
        qrels_filtered = []
        for i, qrel in enumerate(qrels):
            if qrel["query-id"] in query_ids:
                qrels_filtered.append(i)
        qrels = qrels.select(qrels_filtered)

        # Get corpus IDs that are referenced
        corpus_ids = set(qrels["corpus-id"])
        corpus_filtered = []
        for i, doc in enumerate(corpus):
            if doc["corpus-id"] in corpus_ids:
                corpus_filtered.append(i)
        corpus = corpus.select(corpus_filtered)

    print(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels")
    return corpus, queries, qrels


def process_dataset(corpus, queries, qrels, output_dir: Path):
    """Process dataset and prepare data structures for visualization.

    Note: Multiple images per query are naturally handled through the qrels structure.
    Each query can have multiple related documents (corpus entries), and each document
    can have its own image. All related documents and their images will be displayed
    in a grid layout for each query.
    """
    print("Processing dataset...")

    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build corpus lookup
    corpus_lookup = {}
    print("Processing corpus documents...")
    for doc in tqdm(corpus, desc="Corpus"):
        corpus_id = doc["corpus-id"]

        # Save image if present
        image_path = None
        if "image" in doc and doc["image"] is not None:
            image = doc["image"]
            if isinstance(image, Image.Image):
                img_filename = f"corpus_{corpus_id}.png"
                img_path = images_dir / img_filename
                save_image_to_file(image, img_path)
                image_path = f"images/{img_filename}"

        # Extract all columns flexibly
        corpus_lookup[corpus_id] = {
            "corpus_id": corpus_id,
            "image_path": image_path,
            "metadata": {
                k: v for k, v in doc.items() if k not in ["corpus-id", "image"]
            },
        }

    # Build query lookup
    query_lookup = {}
    print("Processing queries...")
    for query in tqdm(queries, desc="Queries"):
        query_id = query["query-id"]
        query_lookup[query_id] = {
            "query_id": query_id,
            "query_text": query.get("query", ""),
            "metadata": {
                k: v for k, v in query.items() if k not in ["query-id", "query"]
            },
        }

    # Build query-document relationships
    print("Building query-document relationships...")
    query_docs = {}
    for qrel in tqdm(qrels, desc="QRels"):
        query_id = qrel["query-id"]
        corpus_id = qrel["corpus-id"]

        if query_id not in query_docs:
            query_docs[query_id] = []

        # Extract relevance info
        rel_info = {
            "corpus_id": corpus_id,
            "metadata": {
                k: v for k, v in qrel.items() if k not in ["query-id", "corpus-id"]
            },
        }
        query_docs[query_id].append(rel_info)

    return corpus_lookup, query_lookup, query_docs


def generate_html(corpus_lookup, query_lookup, query_docs, output_dir: Path):
    """Generate interactive HTML visualization."""
    print("Generating HTML visualization...")

    # Prepare data for JavaScript
    queries_data = []
    for query_id, query_info in sorted(query_lookup.items()):
        related_docs = []
        if query_id in query_docs:
            for doc_rel in query_docs[query_id]:
                corpus_id = doc_rel["corpus_id"]
                if corpus_id in corpus_lookup:
                    doc_info = corpus_lookup[corpus_id]
                    related_docs.append(
                        {
                            "corpus_id": corpus_id,
                            "image_path": doc_info["image_path"],
                            "doc_metadata": doc_info["metadata"],
                            "rel_metadata": doc_rel["metadata"],
                        }
                    )

        queries_data.append(
            {
                "query_id": query_id,
                "query_text": query_info["query_text"],
                "query_metadata": query_info["metadata"],
                "related_docs": related_docs,
            }
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEIR Dataset Visualization</title>
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
            max-width: 1600px;
            margin: 0 auto;
            padding: 15px;
        }}

        header {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}

        h1 {{
            font-size: 24px;
            margin-bottom: 8px;
        }}

        .stats {{
            color: #666;
            font-size: 14px;
        }}

        .controls {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .controls label {{
            font-weight: 500;
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
        }}

        .controls button:hover {{
            background: #0056b3;
        }}

        .query-section {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}

        .query-header {{
            border-bottom: 2px solid #007bff;
            padding-bottom: 12px;
            margin-bottom: 15px;
        }}

        .query-id {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}

        .query-text {{
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 8px;
        }}

        .metadata {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            font-size: 13px;
            margin-top: 8px;
        }}

        .metadata-title {{
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .metadata-item {{
            margin: 2px 0;
            color: #555;
        }}

        .metadata-key {{
            font-weight: 500;
            color: #333;
        }}

        .documents-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .document-card {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .document-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        .document-id {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }}

        .document-image {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 8px;
            border: 1px solid #ddd;
            cursor: pointer;
        }}

        .document-image:hover {{
            opacity: 0.95;
        }}

        .no-image {{
            background: #e9ecef;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-radius: 4px;
            margin-bottom: 8px;
        }}

        .no-results {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}

        .hidden {{
            display: none;
        }}

        /* Image modal for full-size viewing */
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

        .modal.active {{
            display: flex;
        }}

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

        .modal-close:hover {{
            color: #ccc;
        }}

        .docs-count {{
            color: #666;
            font-size: 14px;
            font-weight: normal;
        }}

        @media (max-width: 768px) {{
            .documents-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>BEIR Dataset Visualization</h1>
            <div class="stats">
                <span id="total-queries">0</span> queries |
                <span id="total-docs">0</span> documents
            </div>
        </header>

        <div class="controls">
            <label for="query-select">Select Query:</label>
            <select id="query-select">
                <option value="">-- Select a query --</option>
            </select>

            <button onclick="prevQuery()">← Previous</button>
            <button onclick="nextQuery()">Next →</button>

            <input type="text" id="search-input" placeholder="Search queries..."
                   style="flex-grow: 1; margin-left: auto;">
        </div>

        <div id="content">
            <div class="no-results">
                Select a query to view its related documents
            </div>
        </div>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="modal">
        <button class="modal-close" onclick="closeModal()">&times;</button>
        <img id="modalImage" class="modal-content" alt="Full size image">
    </div>

    <script>
        const DATA = {json.dumps(queries_data, indent=2)};

        let currentQueryIndex = -1;
        let filteredData = DATA;

        function init() {{
            document.getElementById('total-queries').textContent = DATA.length;

            const totalDocs = new Set();
            DATA.forEach(q => {{
                q.related_docs.forEach(d => totalDocs.add(d.corpus_id));
            }});
            document.getElementById('total-docs').textContent = totalDocs.size;

            updateQuerySelect();

            document.getElementById('query-select').addEventListener('change', function() {{
                const index = parseInt(this.value);
                if (!isNaN(index)) {{
                    currentQueryIndex = index;
                    displayQuery(filteredData[index]);
                }}
            }});

            document.getElementById('search-input').addEventListener('input', function() {{
                filterQueries(this.value);
            }});
        }}

        function updateQuerySelect() {{
            const select = document.getElementById('query-select');
            select.innerHTML = '<option value="">-- Select a query --</option>';

            filteredData.forEach((query, index) => {{
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Query ${{query.query_id}}: ${{query.query_text.substring(0, 80)}}...`;
                select.appendChild(option);
            }});
        }}

        function filterQueries(searchTerm) {{
            const term = searchTerm.toLowerCase();
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
            document.getElementById('content').innerHTML =
                '<div class="no-results">Select a query to view its related documents</div>';
        }}

        function displayQuery(queryData) {{
            const content = document.getElementById('content');

            const imageCount = queryData.related_docs.filter(d => d.image_path).length;
            const imageInfo = imageCount > 0 ? ` - ${{imageCount}} image${{imageCount > 1 ? 's' : ''}}` : '';

            let html = `
                <div class="query-section">
                    <div class="query-header">
                        <div class="query-id">Query ID: ${{queryData.query_id}}</div>
                        <div class="query-text">${{queryData.query_text}}</div>
                        ${{formatMetadata('Query Metadata', queryData.query_metadata)}}
                    </div>

                    <h3 style="margin-bottom: 15px;">
                        Related Documents
                        <span class="docs-count">(${{queryData.related_docs.length}} document${{queryData.related_docs.length !== 1 ? 's' : ''}}${{imageInfo}})</span>
                    </h3>

                    <div class="documents-grid">
            `;

            if (queryData.related_docs.length === 0) {{
                html += '<div class="no-results">No related documents found</div>';
            }} else {{
                queryData.related_docs.forEach(doc => {{
                    html += `
                        <div class="document-card">
                            <div class="document-id">Corpus ID: ${{doc.corpus_id}}</div>
                            ${{doc.image_path ?
                                `<img src="${{doc.image_path}}"
                                     alt="Document ${{doc.corpus_id}}"
                                     class="document-image"
                                     onclick="openModal('${{doc.image_path}}')"
                                     title="Click to enlarge">` :
                                '<div class="no-image">No image available</div>'
                            }}
                            ${{formatMetadata('Document Info', doc.doc_metadata)}}
                            ${{formatMetadata('Relevance Info', doc.rel_metadata)}}
                        </div>
                    `;
                }});
            }}

            html += '</div></div>';
            content.innerHTML = html;
        }}

        function formatMetadata(title, metadata) {{
            if (!metadata || Object.keys(metadata).length === 0) return '';

            let html = `<div class="metadata"><div class="metadata-title">${{title}}</div>`;
            for (const [key, value] of Object.entries(metadata)) {{
                const displayValue = typeof value === 'object' ? JSON.stringify(value) : value;
                html += `<div class="metadata-item"><span class="metadata-key">${{key}}:</span> ${{displayValue}}</div>`;
            }}
            html += '</div>';
            return html;
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

        // Image modal functions
        function openModal(imagePath) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.classList.add('active');
            modalImg.src = imagePath;
        }}

        function closeModal() {{
            const modal = document.getElementById('imageModal');
            modal.classList.remove('active');
        }}

        // Close modal on background click
        document.getElementById('imageModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeModal();
            }}
        }});

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT') return;

            // Close modal with Escape key
            if (e.key === 'Escape' && document.getElementById('imageModal').classList.contains('active')) {{
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

    # Write HTML file
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML visualization created at: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BEIR format datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_beir_dataset.py \\
    --dataset Nayana-cognitivelab/nayana-beir-eval-multilang_v2 \\
    --output ./viz_output \\
    --samples 50
        """,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset path (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for visualization"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of queries to sample (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    corpus, queries, qrels = load_beir_dataset(args.dataset, args.samples)

    # Process dataset
    corpus_lookup, query_lookup, query_docs = process_dataset(
        corpus, queries, qrels, output_dir
    )

    # Generate HTML
    html_path = generate_html(corpus_lookup, query_lookup, query_docs, output_dir)

    print("\n✓ Visualization complete!")
    print(f"Open {html_path} in your browser to view the visualization")


if __name__ == "__main__":
    main()


# python visualize_beir_dataset.py \
# --dataset Nayana-cognitivelab/nayana-beir-eval-multilang_v2 \
# --output ./viz_output_multilang_v2 \
# --samples 50


# python visualize_beir_dataset.py \
# --dataset Nayana-cognitivelab/nayana-beir-eval-multilang \
# --output ./viz_output_multilang_v1 \
# --samples 50
