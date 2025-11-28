from __future__ import annotations

import base64
import io
import logging
import time
from functools import partial
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
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
        print(
            f"Target embedding dimension: {embedding_dim} (Matryoshka={embedding_dim < 2560})"
        )

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
        print(f"Embedding dimension: {self.embedding_dim}")

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
        """Generate embeddings for text inputs.

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

        # Log original embedding shape
        original_dim = all_embeddings.shape[1]
        print(f"Text embeddings - Original dimension: {original_dim}")

        # Truncate to desired embedding dimension if needed (Matryoshka)
        if all_embeddings.shape[1] > self.embedding_dim:
            # Calculate norm before truncation
            norm_before = torch.norm(all_embeddings[0], p=2).item()
            print(
                f"Text embeddings - Truncating from {original_dim} to {self.embedding_dim} dims (Matryoshka)"
            )
            print(f"Text embeddings - L2 norm before truncation: {norm_before:.6f}")

            all_embeddings = all_embeddings[:, : self.embedding_dim]

            # Calculate norm after truncation (before re-normalization)
            norm_after_truncate = torch.norm(all_embeddings[0], p=2).item()
            print(
                f"Text embeddings - L2 norm after truncation (before re-norm): {norm_after_truncate:.6f}"
            )

        # Ensure L2 normalization
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        # Log final norm after re-normalization
        final_norm = torch.norm(all_embeddings[0], p=2).item()
        print(
            f"Text embeddings - Final L2 norm after re-normalization: {final_norm:.6f}"
        )
        print(f"Text embeddings - Final shape: {all_embeddings.shape}")

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
        """Generate embeddings for image inputs.

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

        # Log original embedding shape
        original_dim = all_embeddings.shape[1]
        print(f"Image embeddings - Original dimension: {original_dim}")

        # Truncate to desired embedding dimension if needed (Matryoshka)
        if all_embeddings.shape[1] > self.embedding_dim:
            # Calculate norm before truncation
            norm_before = torch.norm(all_embeddings[0], p=2).item()
            print(
                f"Image embeddings - Truncating from {original_dim} to {self.embedding_dim} dims (Matryoshka)"
            )
            print(f"Image embeddings - L2 norm before truncation: {norm_before:.6f}")

            all_embeddings = all_embeddings[:, : self.embedding_dim]

            # Calculate norm after truncation (before re-normalization)
            norm_after_truncate = torch.norm(all_embeddings[0], p=2).item()
            print(
                f"Image embeddings - L2 norm after truncation (before re-norm): {norm_after_truncate:.6f}"
            )

        # Ensure L2 normalization
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)

        # Log final norm after re-normalization
        final_norm = torch.norm(all_embeddings[0], p=2).item()
        print(
            f"Image embeddings - Final L2 norm after re-normalization: {final_norm:.6f}"
        )
        print(f"Image embeddings - Final shape: {all_embeddings.shape}")

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
        """Generate fused embeddings for multimodal inputs.

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
        """Calculate similarity scores between text and image embeddings.

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


class BiGemmaAPIWrapper(Wrapper):
    """API-based wrapper for BiGemma3 models served via vLLM endpoint.

    This wrapper connects to a deployed vLLM server (e.g., on Modal) that serves
    BiGemma3 embeddings through the OpenAI-compatible embeddings API.

    Example usage:
        wrapper = BiGemmaAPIWrapper(
            api_url="https://your-modal-endpoint.modal.run",
            model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220"
        )
        embeddings = wrapper.encode(["text1", "text2"])
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        embedding_dim: int = 2560,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 300,
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize BiGemma API wrapper.

        Args:
            api_url: Base URL of the vLLM API endpoint (e.g., https://example.modal.run)
            model_name: Model identifier to use in API requests
            embedding_dim: Expected embedding dimension (default: 2560)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Base delay in seconds between retries (exponential backoff)
            timeout: Request timeout in seconds
            batch_size: Default batch size for processing
            **kwargs: Additional arguments
        """
        requires_package(self, "requests", "HTTP requests library")
        import requests

        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.default_batch_size = batch_size
        self.session = requests.Session()

        logger.info(f"Initialized BiGemma API wrapper for {model_name}")
        logger.info(f"API endpoint: {self.api_url}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64-encoded JPEG string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _format_text_message(self, text: str) -> list[dict]:
        """Format text into vLLM message structure.

        Args:
            text: Input text string

        Returns:
            Message list in vLLM format
        """
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _format_image_message(self, image: Image.Image, text: str = "") -> list[dict]:
        """Format image (and optional text) into vLLM message structure.

        Args:
            image: PIL Image object
            text: Optional text to accompany the image

        Returns:
            Message list in vLLM format
        """
        base64_image = self._encode_image_to_base64(image)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
        if text:
            content.append({"type": "text", "text": text})

        return [{"role": "user", "content": content}]

    def _make_api_request(self, messages: list[dict]) -> list[float]:
        """Make a single API request to get embeddings.

        Args:
            messages: Message in vLLM format

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If request fails after all retries
        """
        import requests

        payload = {
            "model": self.model_name,
            "messages": messages,
            "encoding_format": "float",
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.api_url}/v1/embeddings", json=payload, timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                if "data" not in data or len(data["data"]) == 0:
                    raise ValueError("Invalid response format: missing 'data' field")

                embedding = data["data"][0]["embedding"]
                return embedding

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise Exception(
                        f"API request failed after {self.max_retries} attempts: {e}"
                    )

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode text sentences using the API.

        Args:
            sentences: List of text strings to encode
            task_name: Name of the task (for task-specific prompts)
            prompt_type: Type of prompt (query vs document)
            batch_size: Batch size for processing (default: use init value)
            **kwargs: Additional arguments

        Returns:
            Tensor of embeddings (n_sentences, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        all_embeddings = []

        # Process in batches (one request per sentence for now)
        for i in tqdm(
            range(len(sentences)),
            desc="Encoding texts via API",
            disable=kwargs.get("show_progress_bar", True) is False,
        ):
            text = sentences[i]
            messages = self._format_text_message(text)

            try:
                embedding = self._make_api_request(messages)
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode sentence {i}: {e}")
                raise

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Truncate to expected dimension if needed
        if embeddings_array.shape[1] > self.embedding_dim:
            embeddings_array = embeddings_array[:, : self.embedding_dim]

        # L2 normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / np.maximum(norms, 1e-12)

        # Convert to torch tensor
        return torch.from_numpy(embeddings_array)

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate embeddings for text inputs (alias for encode method).

        Args:
            texts: List of text strings
            task_name: Task name for context
            prompt_type: Query or document prompt type
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            Tensor of text embeddings (n_texts, embedding_dim)
        """
        return self.encode(
            texts,
            task_name=task_name,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate embeddings for image inputs via API.

        Args:
            images: List of PIL Images or DataLoader
            task_name: Task name for context
            prompt_type: Query or document prompt type
            batch_size: Batch size (default: use init value)
            **kwargs: Additional arguments

        Returns:
            Tensor of image embeddings (n_images, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        all_embeddings = []

        # Handle DataLoader input
        if isinstance(images, DataLoader):
            for batch in tqdm(
                images,
                desc="Encoding images via API (DataLoader)",
                disable=kwargs.get("show_progress_bar", True) is False,
            ):
                # Convert batch tensors to PIL Images if needed
                import torchvision.transforms.functional as F

                for img in batch:
                    # Convert tensor to PIL Image if needed
                    if not isinstance(img, Image.Image):
                        pil_image = F.to_pil_image(
                            img.cpu() if hasattr(img, "cpu") else img
                        )
                    else:
                        pil_image = img

                    messages = self._format_image_message(pil_image)

                    try:
                        embedding = self._make_api_request(messages)
                        all_embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Failed to encode image: {e}")
                        raise
        else:
            # Handle list of images
            for i in tqdm(
                range(len(images)),
                desc="Encoding images via API",
                disable=kwargs.get("show_progress_bar", True) is False,
            ):
                image = images[i]
                messages = self._format_image_message(image)

                try:
                    embedding = self._make_api_request(messages)
                    all_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to encode image {i}: {e}")
                    raise

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Truncate to expected dimension if needed
        if embeddings_array.shape[1] > self.embedding_dim:
            embeddings_array = embeddings_array[:, : self.embedding_dim]

        # L2 normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / np.maximum(norms, 1e-12)

        # Convert to torch tensor
        return torch.from_numpy(embeddings_array)

    def get_fused_embeddings(
        self,
        images: list[Image.Image],
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate embeddings for image+text pairs via API.

        Args:
            images: List of PIL Images
            texts: List of text strings (must match length of images)
            task_name: Task name for context
            prompt_type: Query or document prompt type
            batch_size: Batch size (default: use init value)
            **kwargs: Additional arguments

        Returns:
            Tensor of fused embeddings (n_pairs, embedding_dim)
        """
        if len(images) != len(texts):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of texts ({len(texts)})"
            )

        if batch_size is None:
            batch_size = self.default_batch_size

        all_embeddings = []

        for i in tqdm(
            range(len(images)),
            desc="Encoding image-text pairs via API",
            disable=kwargs.get("show_progress_bar", True) is False,
        ):
            image = images[i]
            text = texts[i]
            messages = self._format_image_message(image, text)

            try:
                embedding = self._make_api_request(messages)
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode image-text pair {i}: {e}")
                raise

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Truncate to expected dimension if needed
        if embeddings_array.shape[1] > self.embedding_dim:
            embeddings_array = embeddings_array[:, : self.embedding_dim]

        # L2 normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / np.maximum(norms, 1e-12)

        # Convert to torch tensor
        return torch.from_numpy(embeddings_array)


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


bigemma3_matryoshka_merged_2000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-merged-2000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-merged-2000",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-merged-2000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

bigemma3_matryoshka_1500_v2 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-1500-v2",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-1500-v2",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-1500-v2",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Multilingual-Hi-En-Kn-merged-1386
bigemma_multilingual_hi_en_kn_1386 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Multilingual-Hi-En-Kn-merged-1386",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Multilingual-Hi-En-Kn-merged-1386",
    languages=["eng-Latn", "hin-Deva", "kan-Knda"],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Multilingual-Hi-En-Kn-merged-1386",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2500
bigemma_6langs_2500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2500",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860
bigemma_6langs_2860 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860-vidore-3694
bigemma_6langs_2860 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860-vidore-3694",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860-vidore-3694",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-6langs-merged-2860-vidore-3694",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220
bigemma_22langs_3220 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220 (Matryoshka 768 dims)
bigemma_22langs_3220_768_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220-768dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220 (Matryoshka 1536 dims)
bigemma_22langs_3220_1536_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220-1536dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final
bigemma_matryoshka_final = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final (768 dims)
bigemma_matryoshka_final_768dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final_768dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final
bigemma_matryoshka_final_1536dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final_1536dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-Final",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

# ============================================== MODEL MERGING ===============================================

# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test (Matryoshka 768 dims)
bigemma_SLERP_merged_768_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test-768dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test (Matryoshka 1536 dims)
bigemma_SLERP_merged_1536_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test-1536dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test (Matryoshka 2560 dims)
bigemma_SLERP_merged = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLERP-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test (Matryoshka 768 dims)
bigemma_Linear_merged_768_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-768dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test (Matryoshka 1536 dims)
bigemma_Linear_merged_1536_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-1536dims",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    framework=["PyTorch", "ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test (Matryoshka 2560 dims)
bigemma_Linear_merged = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-Inbatch (Matryoshka 2560 dims)
bigemma_Linear_merged_inbatch = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-Inbatch",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-Inbatch",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Linear-Merged-Test-Inbatch",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLEPR-Merged-Test-Inbatch (Matryoshka 2560 dims)
bigemma_SLERP_merged_inbatch = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLEPR-Merged-Test-Inbatch",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLEPR-Merged-Test-Inbatch",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-SLEPR-Merged-Test-Inbatch",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-750
bigemma3_matryoshka_6langs_merged_750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-750",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-1500
bigemma3_matryoshka_6langs_merged_1500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-1500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-1500",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-1500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2250
bigemma3_matryoshka_6langs_merged_2250 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2250",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2250",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2250",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860
bigemma3_matryoshka_6langs_merged_2860 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860 (768 dims)
bigemma3_matryoshka_6langs_merged_2860_768_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
        embedding_dim=768,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860-768dims",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860 (1536 dims)
bigemma3_matryoshka_6langs_merged_2860_1536_dims = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
        embedding_dim=1536,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860-1536dims",
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-6langs-merged-2860",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-250
bigemma3_6langs_hardneg_250_v2 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-250",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-250",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-250",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-750
bigemma3_6langs_hardneg_750_v2 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-750",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1000
bigemma3_6langs_hardneg_1000_v2 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1000",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1430
bigemma3_6langs_hardneg_1430_v2 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1430",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1430",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-v2-merged-1430",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-250
bigemma3_6langs_hardneg_250 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-250",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-250",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-250",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-500
bigemma3_6langs_hardneg_500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-500",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-750
bigemma3_6langs_hardneg_750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-750",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1000
bigemma3_6langs_hardneg_1000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1000",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1250
bigemma3_6langs_hardneg_1250 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1250",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1250",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1250",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1500
bigemma3_6langs_hardneg_1500 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1500",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1500",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1500",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1750
bigemma3_6langs_hardneg_1750 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1750",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1750",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-1750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2000
bigemma3_6langs_hardneg_2000 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2000",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2000",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2000",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2145
bigemma3_6langs_hardneg_2145 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2145",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2145",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2145",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)

# Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-4500-v2
bigemma3_6langs_hardneg_2145 = ModelMeta(
    loader=partial(
        BiGemma3Wrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-4500-v2",
        embedding_dim=2560,
        pooling_strategy="last",
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-Matryoshka-4500-v2",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "mal-Mlym",
        "jpn-Jpan",
        "zho-Hans",
    ],
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
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNegs-6langs-merged-2145",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
    },
)


# ==================== API-Based Models (vLLM Endpoint) ====================
# These models use the BiGemmaAPIWrapper to connect to deployed vLLM servers


# API-based BiGemma3 22 Languages Model
bigemma3_22langs_api = ModelMeta(
    loader=partial(
        BiGemmaAPIWrapper,
        api_url="https://cognitivelab--vllm-nayanaembed-bigemma3-22langs-merged-3-6e18f1.modal.run",
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
        embedding_dim=2560,
        max_retries=3,
        retry_delay=5,
        timeout=300,
        batch_size=32,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220-api",
    revision="main",
    release_date="2025-01-01",
    languages=[
        "eng-Latn",
        "hin-Deva",
        "kan-Knda",
        "tam-Taml",
        "tel-Telu",
        "mal-Mlym",
        "mar-Deva",
        "ben-Beng",
        "guj-Gujr",
        "urd-Arab",
        "ori-Orya",
        "pan-Guru",
        "san-Deva",
        "nep-Deva",
        "sin-Sinh",
        "ara-Arab",
        "jpn-Jpan",
        "kor-Kore",
        "zho-Hans",
        "fra-Latn",
        "deu-Latn",
        "spa-Latn",
    ],
    modalities=["image", "text"],
    n_parameters=4_000_000_000,  # ~4B parameters
    memory_usage_mb=None,  # Remote API
    max_tokens=8192,  # Gemma3 context length
    embed_dim=2560,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["API"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-22langs-merged-3220",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        "MSMARCO": ["train"],
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "ArxivQA": ["train"],
        # Multilingual datasets included
    },
)
