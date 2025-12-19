from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class NetraEmbedWrapper(AbsEncoder):
    """Wrapper for NetraEmbed (BiGemma-based) models.

    NetraEmbed is a multilingual multimodal embedding model that encodes both visual
    documents and text queries into single dense vectors. It supports multiple
    languages and enables efficient similarity search at multiple embedding dimensions
    (768, 1536, 2560) through Matryoshka representation learning.

    Requires the CognitiveLab fork of colpali-engine:
    pip install git+https://github.com/adithya-s-k/colpali.git
    """

    def __init__(
        self,
        model_name: str = "Cognitive-Lab/NetraEmbed",
        revision: str | None = None,
        embedding_dim: int = 2560,
        pooling_strategy: str = "last",
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self,
            "colpali_engine",
            model_name,
            "pip install git+https://github.com/adithya-s-k/colpali.git",
        )

        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading NetraEmbed model: {model_name}")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info(f"Pooling strategy: {pooling_strategy}")

        # Load model
        self.model = BiGemma3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            pooling_strategy=pooling_strategy,
            **kwargs,
        ).eval()

        # Load processor
        self.processor = BiGemmaProcessor3.from_pretrained(
            model_name,
            use_fast=True,
        )

        logger.info(f"Model loaded on device: {self.device}")

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode texts and/or images using NetraEmbed."""
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self._encode_texts(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self._encode_images(inputs, **kwargs)

        # Handle multimodal inputs
        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No valid inputs found")

    def _encode_texts(self, dataloader: DataLoader, **kwargs) -> torch.Tensor:
        """Encode text inputs."""
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Encoding texts",
                disable=not kwargs.get("show_progress_bar", True),
            ):
                texts = batch["text"]

                # Process texts
                batch_inputs = self.processor.process_texts(texts).to(self.device)

                # Get embeddings with specified dimension
                embeddings = self.model(
                    **batch_inputs, embedding_dim=self.embedding_dim
                )
                all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # L2 normalize
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        logger.info(f"Text embeddings shape: {all_embeddings.shape}")
        return all_embeddings

    def _encode_images(self, dataloader: DataLoader, **kwargs) -> torch.Tensor:
        """Encode image inputs."""
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Encoding images",
                disable=not kwargs.get("show_progress_bar", True),
            ):
                images = batch["image"]

                # Convert to PIL if needed
                pil_images = [
                    F.to_pil_image(img) if not isinstance(img, Image.Image) else img
                    for img in images
                ]

                # Process images
                batch_inputs = self.processor.process_images(pil_images).to(self.device)

                # Get embeddings with specified dimension
                embeddings = self.model(
                    **batch_inputs, embedding_dim=self.embedding_dim
                )
                all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # L2 normalize
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        logger.info(f"Image embeddings shape: {all_embeddings.shape}")
        return all_embeddings


class ColNetraEmbedWrapper(AbsEncoder):
    """Wrapper for ColNetraEmbed (ColGemma-based) models using MaxSim scoring.

    ColNetraEmbed is a multilingual multimodal embedding model that encodes
    documents as multi-vector representations using the ColPali architecture. Each image
    patch is mapped to a contextualized embedding, enabling fine-grained matching
    between visual content and text queries through late interaction (MaxSim).

    Requires the CognitiveLab fork of colpali-engine:
    pip install git+https://github.com/adithya-s-k/colpali.git
    """

    def __init__(
        self,
        model_name: str = "Cognitive-Lab/ColNetraEmbed",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self,
            "colpali_engine",
            model_name,
            "pip install git+https://github.com/adithya-s-k/colpali.git",
        )

        from colpali_engine.models import ColGemma3, ColGemmaProcessor3

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading ColNetraEmbed model: {model_name}")

        # Load model
        self.model = ColGemma3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.model.eval()

        # Load processor
        self.processor = ColGemmaProcessor3.from_pretrained(model_name)

        logger.info(f"Model loaded on device: {self.device}")

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode using multi-vector ColNetraEmbed."""
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self._encode_texts(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self._encode_images(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No valid inputs found")

    def _encode_images(self, dataloader: DataLoader, **kwargs) -> torch.Tensor:
        """Encode images to multi-vector representations."""
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Encoding images",
                disable=not kwargs.get("show_progress_bar", True),
            ):
                imgs = [
                    F.to_pil_image(b) if not isinstance(b, Image.Image) else b
                    for b in batch["image"]
                ]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.model(**inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        logger.info(f"Image embeddings shape: {padded.shape}")
        return padded

    def _encode_texts(self, dataloader: DataLoader, **kwargs) -> torch.Tensor:
        """Encode texts with query augmentation."""
        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Encoding texts",
                disable=not kwargs.get("show_progress_bar", True),
            ):
                batch_texts = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in batch["text"]
                ]

                inputs = self.processor.process_queries(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.model(**inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        logger.info(f"Text embeddings shape: {padded.shape}")
        return padded

    def similarity(self, a, b):
        """Calculate MaxSim similarity."""
        return self.processor.score_multi_vector(a, b, device=self.device)


# Training data for NetraEmbed and ColNetraEmbed models
NETRA_TRAINING_DATA = {
    "MSMARCO",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
}

NETRA_CITATION = """
@misc{kolavi2025m3druniversalmultilingualmultimodal,
  title={M3DR: Towards Universal Multilingual Multimodal Document Retrieval}, 
  author={Adithya S Kolavi and Vyoman Jain},
  year={2025},
  eprint={2512.03514},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2512.03514}
}
"""

# 22 languages supported by NetraEmbed
NETRA_LANGUAGES = [
    "eng-Latn",  # English
    "hin-Deva",  # Hindi
    "kan-Knda",  # Kannada
    "tam-Taml",  # Tamil
    "tel-Telu",  # Telugu
    "mal-Mlym",  # Malayalam
    "mar-Deva",  # Marathi
    "ben-Beng",  # Bengali
    "guj-Gujr",  # Gujarati
    "urd-Arab",  # Urdu
    "ori-Orya",  # Odia
    "pan-Guru",  # Punjabi
    "san-Deva",  # Sanskrit
    "npi-Deva",  # Nepali
    "sin-Sinh",  # Sinhala
    "asm-Beng",  # Assamese
    "kok-Deva",  # Konkani
    "mai-Deva",  # Maithili
    "jpn-Jpan",  # Japanese
    "zho-Hans",  # Chinese (Simplified)
    "kor-Kore",  # Korean
    "tha-Thai",  # Thai
]

# NetraEmbed - Full 2560 dimension model
netraembed = ModelMeta(
    loader=NetraEmbedWrapper,
    loader_kwargs=dict(
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Cognitive-Lab/NetraEmbed",
    languages=NETRA_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8000,
    max_tokens=8192,
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/Cognitive-Lab/NetraEmbed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=NETRA_TRAINING_DATA,
    citation=NETRA_CITATION,
)

# NetraEmbed - 1536 dimension (Matryoshka)
netraembed_1536 = ModelMeta(
    loader=NetraEmbedWrapper,
    loader_kwargs=dict(
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Cognitive-Lab/NetraEmbed-1536",
    languages=NETRA_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8000,
    max_tokens=8192,
    embed_dim=1536,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/Cognitive-Lab/NetraEmbed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=NETRA_TRAINING_DATA,
    citation=NETRA_CITATION,
)

# NetraEmbed - 768 dimension (Matryoshka)
netraembed_768 = ModelMeta(
    loader=NetraEmbedWrapper,
    loader_kwargs=dict(
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Cognitive-Lab/NetraEmbed-768",
    languages=NETRA_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8000,
    max_tokens=8192,
    embed_dim=768,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/Cognitive-Lab/NetraEmbed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=NETRA_TRAINING_DATA,
    citation=NETRA_CITATION,
)

# ColNetraEmbed - Multi-vector model
colnetraembed = ModelMeta(
    loader=ColNetraEmbedWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="Cognitive-Lab/ColNetraEmbed",
    languages=NETRA_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8000,
    max_tokens=8192,
    embed_dim=128,  # Multi-vector dimension per token
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Cognitive-Lab/ColNetraEmbed",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=NETRA_TRAINING_DATA,
    citation=NETRA_CITATION,
)
