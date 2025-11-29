from __future__ import annotations

import logging
from functools import partial

import torch

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class ColGemma3Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColGemma3 model.

    ColGemma3 is a multi-vector vision-language retrieval model based on
    Google's Gemma3 architecture. It generates late-interaction embeddings
    for efficient document and image retrieval using MaxSim scoring.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self,
            "colpali_engine",
            model_name,
            "pip install git+https://github.com/adithya-s-k/colpali.git@feat/gemma3",
        )
        from colpali_engine.models import ColGemma3, ColGemmaProcessor3

        super().__init__(
            model_name=model_name,
            model_class=ColGemma3,
            processor_class=ColGemmaProcessor3,
            revision=revision,
            device=device,
            **kwargs,
        )


# Training datasets for ColGemma3 models
COLGEMMA3_TRAINING_DATA = {
    "MSMARCO": ["train"],
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "ArxivQA": ["train"],
}

# Model metadata for ColGemma3 Base (Google Gemma3 4B IT)
colgemma3_base = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="google/gemma-3-4b-it",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="google/gemma-3-4b-it-colgemma3",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/google/gemma-3-4b-it",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets={},
)

colgemma3_colpali_merged_1848_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-ColPaliTrainSet-merged-1848-colbert",
        revision="main",
        torch_dtype=torch.float16,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-ColPaliTrainSet-merged-1848-colbert",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-ColPaliTrainSet-merged-1848-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)

# Fine-tuned ColGemma3 - Nayana IR (Document Retrieval Optimized)
colgemma3_nayana_2500 = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-2500",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-2500",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-2500",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)

# Fine-tuned ColGemma3 - Modal 750 ColBERT
colgemma3_modal_750_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)

# Fine-tuned ColGemma3 - Modal 750 ColBERT
colgemma3_modal_1848_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-1848-colbert",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-1848-colbert",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)


colgemma3_modal_merged_750_22_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-750-22-colbert",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-750-22-colbert",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)


colgemma3_modal_merged_1610_22_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-1610-22-colbert",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-1610-22-colbert",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)

# Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-1267-22-colbert-v1

colgemma3_modal_merged_1610_22_colbert = ModelMeta(
    loader=partial(
        ColGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-1267-22-colbert-v1",
        revision="main",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-ColGemma3-MultiGPU-merged-1267-22-colbert-v1",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=128,  # Multi-vector dimension after projection
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-ColGemma3-Modal-750-colbert",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLGEMMA3_TRAINING_DATA,
)
