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


class BiGemma3Wrapper:
    """Wrapper for BiGemma3 vision-language model for MTEB evaluation.

    BiGemma3 is based on Google's Gemma3-4B-IT model, adapted to generate
    dense embeddings for text, images, and multimodal inputs. It uses a
    custom processor (BiGemmaProcessor3) and model architecture (BiGemma3)
    from the colpali_engine package.

    The model produces 2560-dimensional dense embeddings by default,
    extracted from the last token representation of the generative model.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str | None = None,
        embedding_dim: int = 2560,
        pooling_strategy: str = "last",
        **kwargs,
    ):
        """Initialize BiGemma3 wrapper.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on (default: cuda if available)
            embedding_dim: Dimension of output embeddings (default: 2560)
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

        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading BiGemma3 model: {model_name}")
        logger.info(f"Pooling strategy: {pooling_strategy}")

        # Load model with pooling strategy
        self.model = BiGemma3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            pooling_strategy=pooling_strategy,
            **kwargs,
        ).eval()

        logger.info(f"Loading BiGemmaProcessor3 for: {model_name}")

        # Load processor
        self.processor = BiGemmaProcessor3.from_pretrained(
            model_name,
            use_fast=True,
        )

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

                # Process texts using BiGemmaProcessor3
                batch_inputs = self.processor.process_texts(batch_texts).to(
                    self.model.device
                )

                # Get embeddings from model
                embeddings = self.model(**batch_inputs)

                # Move to CPU and convert to float32 for compatibility
                all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Truncate to desired embedding dimension if needed
        if all_embeddings.shape[1] > self.embedding_dim:
            all_embeddings = all_embeddings[:, : self.embedding_dim]

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

                    # Process images using BiGemmaProcessor3
                    batch_inputs = self.processor.process_images(pil_images).to(
                        self.model.device
                    )

                    # Get embeddings
                    embeddings = self.model(**batch_inputs)

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

                    # Get embeddings
                    embeddings = self.model(**batch_inputs)

                    all_embeddings.append(embeddings.cpu().to(torch.float32))

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Truncate to desired embedding dimension if needed
        if all_embeddings.shape[1] > self.embedding_dim:
            all_embeddings = all_embeddings[:, : self.embedding_dim]

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
        fusion_mode: str = "concat",
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate fused embeddings for multimodal inputs.

        For BiGemma3, we support three fusion modes:
        1. "concat": Process image+text together (true multimodal)
        2. "sum": Add text and image embeddings
        3. "mean": Average text and image embeddings

        Args:
            texts: List of text strings (optional)
            images: List of images or DataLoader (optional)
            task_name: Task name
            prompt_type: Prompt type
            batch_size: Batch size
            fusion_mode: How to fuse ("concat", "sum", "mean")
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
        if fusion_mode == "concat":
            # True multimodal: process image+text together
            # Note: This requires BiGemma3 to support joint processing
            # For now, we fall back to sum fusion
            logger.warning(
                "BiGemma3 concat fusion not yet implemented, using sum fusion"
            )
            fusion_mode = "sum"

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
                f"Unknown fusion_mode: {fusion_mode}. "
                f"Supported: 'concat', 'sum', 'mean'"
            )

    def calculate_probs(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate similarity scores between text and image embeddings.

        Uses the BiGemmaProcessor3 score method for consistency with
        the model's training objective.

        Args:
            text_embeddings: Query embeddings (n_queries, embed_dim)
            image_embeddings: Document embeddings (n_docs, embed_dim)

        Returns:
            Similarity matrix (n_queries, n_docs)
        """
        # Ensure embeddings are normalized
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        # Use processor's score method
        scores = self.processor.score(text_embeddings, image_embeddings)

        return scores


# Training datasets for BiGemma3
# Note: Update this based on actual training data used
BIGEMMA3_TRAINING_DATA = {
    # Add specific datasets used during BiGemma3 training
    # Example:
    # "MSMARCO": ["train"],
    # "DocVQA": ["train"],
}

# Model metadata for BiGemma3 4B Base (Instruction-tuned Gemma3)
bigemma3_base = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="google/gemma-3-4b-it",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="google/gemma-3-4b-it-bigemma3",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/google/gemma-3-4b-it",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=BIGEMMA3_TRAINING_DATA,
)

# Nayana-cognitivelab/Full_SFT_v2_base_gemma_merged_1400
# Nayana-cognitivelab/Full-SFT-v1-23000

bigemma3_ocr_sft_23000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/Full-SFT-v1-23000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/Full-SFT-v1-23000",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)

bigemma3_ocr_sft_1400 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/Full_SFT_v2_base_gemma_merged_1400",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/Full_SFT_v2_base_gemma_merged_1400",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)


bigemma3_hardneg_2300 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-2300",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)

bigemma3_hardneg_1950 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1950",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1950",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1950",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)

# Fine-tuned BiGemma3 with Hard Negatives - Nayana IR (Document Retrieval Optimized)
bigemma3_hardneg_1694 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)


# Fine-tuned BiGemma3 with Hard Negatives - Nayana IR (Document Retrieval Optimized)
bigemma3_hardneg_750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Add other datasets used in fine-tuning
    },
)

bigemma3_hardneg_500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-500",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_hardneg_252 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-252",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-252",
    languages=["eng-Latn"],
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=8000,
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-252",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

# InBatch models - use last token pooling
bigemma3_inbatch_1000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1000",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_1500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1500",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_1750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1750",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-1750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_2000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2000",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_2500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2500",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-2500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_3000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3000",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_inbatch_3694 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3694",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3694",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-InBatch-merged-3694",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

# MeanPooling-HardNeg models - use mean pooling strategy
bigemma3_meanpool_hardneg_1000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1000",
        embedding_dim=2560,
        pooling_strategy="mean",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1000",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_meanpool_hardneg_1500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1500",
        embedding_dim=2560,
        pooling_strategy="mean",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1500",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_meanpool_hardneg_1750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1750",
        embedding_dim=2560,
        pooling_strategy="mean",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1750",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-1750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_meanpool_hardneg_2000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-2000",
        embedding_dim=2560,
        pooling_strategy="mean",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-2000",
    languages=["eng-Latn"],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-MeanPooling-HardNeg-merged-2000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)
