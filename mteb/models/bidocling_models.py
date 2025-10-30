from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies, requires_package

logger = logging.getLogger(__name__)


class BiDoclingWrapper:
    """Wrapper for BiDocling vision-language model for MTEB evaluation.

    BiDocling is based on IBM's Granite Docling model (Idefics3 architecture),
    adapted to generate dense embeddings for text, images, and multimodal inputs.
    It uses a custom processor (BiDoclingProcessor) and model architecture (BiDocling)
    from the colpali_engine package.

    The model produces single-vector dense embeddings extracted using a
    configurable pooling strategy (cls, last, or mean).
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-docling-258M",
        device: str | None = None,
        embedding_dim: int | None = None,
        pooling_strategy: str = "last",
        **kwargs,
    ):
        """Initialize BiDocling wrapper.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on (default: cuda if available)
            embedding_dim: Dimension of output embeddings (auto-detected if None)
            pooling_strategy: Pooling strategy to use ("cls", "last", "mean"). Default: "last"
            **kwargs: Additional arguments passed to model loading
        """
        requires_image_dependencies()
        requires_package(
            self,
            "colpali_engine",
            model_name,
            "pip install git+https://github.com/adithya-s-k/colpali.git@feat/gemma3",
        )

        from colpali_engine.models.docling.bidocling import BiDocling, BiDoclingProcessor

        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading BiDocling model: {model_name}")
        logger.info(f"Pooling strategy: {pooling_strategy}")

        # Load model
        self.model = BiDocling.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            pooling_strategy=pooling_strategy,
            **kwargs,
        ).eval()

        logger.info(f"Loading BiDoclingProcessor for: {model_name}")

        # Load processor
        self.processor = BiDoclingProcessor.from_pretrained(
            model_name,
            use_fast=True,
        )

        # Auto-detect embedding dimension if not provided
        if embedding_dim is None:
            # Idefics3Config uses text_config for the text encoder config
            self.embedding_dim = self.model.config.text_config.hidden_size
        else:
            self.embedding_dim = embedding_dim

        logger.info(f"Model loaded on device: {self.model.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode text sentences (MTEB standard interface).

        Args:
            sentences: List of text strings to encode
            task_name: Name of the task (for task-specific prompts)
            prompt_type: Type of prompt (query vs document)
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            Tensor of embeddings (n_sentences, embedding_dim)
        """
        return self.get_text_embeddings(
            texts=sentences,
            task_name=task_name,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate embeddings for text inputs.

        Args:
            texts: List of text strings
            task_name: Task name for context
            prompt_type: Query or document prompt type
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            Tensor of text embeddings (n_texts, embedding_dim)
        """
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i : i + batch_size]

                # Process texts using BiDoclingProcessor
                batch_inputs = self.processor.process_texts(batch_texts).to(
                    self.model.device
                )

                # Get embeddings from model with pooling strategy
                embeddings = self.model(
                    **batch_inputs, pooling_strategy=self.pooling_strategy
                )

                # Move to CPU and convert to float32 for compatibility
                all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Ensure L2 normalization
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        return all_embeddings

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate embeddings for image inputs.

        Args:
            images: List of PIL Images or DataLoader
            task_name: Task name for context
            prompt_type: Query or document prompt type
            batch_size: Batch size (smaller for images due to memory)
            **kwargs: Additional arguments

        Returns:
            Tensor of image embeddings (n_images, embedding_dim)
        """
        import torchvision.transforms.functional as F

        all_embeddings = []

        # Handle DataLoader input
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images, desc="Encoding images (DataLoader)"):
                    # Convert tensors to PIL Images if needed
                    pil_images = [
                        F.to_pil_image(img.to("cpu"))
                        if not isinstance(img, Image.Image)
                        else img
                        for img in batch
                    ]

                    # Process images using BiDoclingProcessor
                    batch_inputs = self.processor.process_images(pil_images).to(
                        self.model.device
                    )

                    # Get embeddings with pooling strategy
                    embeddings = self.model(
                        **batch_inputs, pooling_strategy=self.pooling_strategy
                    )

                    # Move to CPU and convert to float32
                    all_embeddings.append(embeddings.cpu().to(torch.float32))

        else:
            # Handle list of images
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(images), batch_size), desc="Encoding images"
                ):
                    batch_images = images[i : i + batch_size]

                    # Ensure all are PIL Images
                    pil_images = [
                        img if isinstance(img, Image.Image) else F.to_pil_image(img)
                        for img in batch_images
                    ]

                    # Process images
                    batch_inputs = self.processor.process_images(pil_images).to(
                        self.model.device
                    )

                    # Get embeddings with pooling strategy
                    embeddings = self.model(
                        **batch_inputs, pooling_strategy=self.pooling_strategy
                    )

                    all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Ensure L2 normalization
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        return all_embeddings

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 16,
        fusion_mode: str = "sum",
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate fused embeddings for multimodal inputs.

        For BiDocling, we support two fusion modes:
        1. "sum": Add text and image embeddings
        2. "mean": Average text and image embeddings

        Args:
            texts: List of text strings (optional)
            images: List of images or DataLoader (optional)
            task_name: Task name
            prompt_type: Prompt type
            batch_size: Batch size
            fusion_mode: How to fuse ("sum", "mean")
            **kwargs: Additional arguments

        Returns:
            Fused embeddings tensor
        """
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        # Handle single modality cases
        if texts is None:
            return self.get_image_embeddings(
                images,
                task_name=task_name,
                prompt_type=prompt_type,
                batch_size=batch_size,
            )

        if images is None:
            return self.get_text_embeddings(
                texts,
                task_name=task_name,
                prompt_type=prompt_type,
                batch_size=batch_size,
            )

        # Multimodal fusion
        if fusion_mode in ["sum", "mean"]:
            # Get separate embeddings
            text_embeddings = self.get_text_embeddings(
                texts,
                task_name=task_name,
                prompt_type=prompt_type,
                batch_size=batch_size,
            )

            # Handle DataLoader for images
            if isinstance(images, DataLoader):
                image_list = []
                for batch in images:
                    image_list.extend(batch)
                images = image_list

            image_embeddings = self.get_image_embeddings(
                images,
                task_name=task_name,
                prompt_type=prompt_type,
                batch_size=batch_size,
            )

            # Check length compatibility
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    f"Text and image counts must match for fusion. "
                    f"Got {len(text_embeddings)} texts and {len(image_embeddings)} images"
                )

            # Fuse embeddings
            if fusion_mode == "sum":
                fused = text_embeddings + image_embeddings
            else:  # mean
                fused = (text_embeddings + image_embeddings) / 2

            # Re-normalize after fusion
            fused = torch.nn.functional.normalize(fused, p=2, dim=-1)

            return fused

        else:
            raise ValueError(
                f"Unknown fusion_mode: {fusion_mode}. " f"Supported: 'sum', 'mean'"
            )

    def calculate_probs(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate similarity scores between text and image embeddings.

        Uses cosine similarity for single-vector embeddings.

        Args:
            text_embeddings: Query embeddings (n_queries, embed_dim)
            image_embeddings: Document embeddings (n_docs, embed_dim)

        Returns:
            Similarity matrix (n_queries, n_docs)
        """
        # Ensure embeddings are normalized
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        # Compute cosine similarity (dot product of normalized vectors)
        scores = torch.matmul(text_embeddings, image_embeddings.T)

        return scores


# Training datasets for BiDocling
BIDOCLING_TRAINING_DATA = {
    "MSMARCO": ["train"],
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "ArxivQA": ["train"],
}

# Model metadata for BiDocling Base (IBM Granite Docling 258M)
bidocling_base = ModelMeta(
    loader=partial(
        BiDoclingWrapper,
        model_name="ibm-granite/granite-docling-258M",
        pooling_strategy="last",
    ),
    name="ibm-granite/granite-docling-258M-bidocling",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=258_000_000,  # ~258M parameters
    memory_usage_mb=1000,
    max_tokens=8192,
    embed_dim=576,  # BiDocling embedding dimension
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/ibm-granite/granite-docling-258M",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={},
)

# Fine-tuned BiDocling - Nayana IR (Document Retrieval Optimized)
bidocling_nayana_2500 = ModelMeta(
    loader=partial(
        BiDoclingWrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiDocling-MultiGPU-2500",
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiDocling-MultiGPU-2500",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=258_000_000,  # ~258M parameters
    memory_usage_mb=1000,
    max_tokens=8192,
    embed_dim=576,  # BiDocling embedding dimension
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiDocling-MultiGPU-2500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=BIDOCLING_TRAINING_DATA,
)
