from __future__ import annotations

import logging
from functools import partial
from typing import Any
from collections.abc import Sequence

import torch
import numpy as np
from PIL import Image
from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType
from mteb.requires_package import (
    requires_package,
    requires_image_dependencies,
)

logger = logging.getLogger(__name__)


class BiGemmaWrapper(Wrapper):
    """Wrapper for BiGemma3 models - dense single-vector embedding models based on Gemma3."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        pooling_strategy: str = "last",
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling_strategy = pooling_strategy

        # Load model - use proper arguments for HuggingFace models
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ["adapter_kwargs"]}
        if revision:
            model_kwargs["revision"] = revision

        logger.info(f"Loading BiGemma3 model: {model_name}")
        self.mdl = BiGemma3.from_pretrained(
            model_name,
            device_map=self.device,
            **model_kwargs,
        )
        self.mdl.eval()

        # Load processor
        self.processor = BiGemmaProcessor3.from_pretrained(model_name)

        # Set vector type for single-vector embeddings
        self.vector_type = "single_vector"

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray:
        """Main encode method for MTEB compatibility."""
        return self.get_text_embeddings(
            texts=list(sentences), return_numpy=True, **kwargs
        )

    def encode_input(self, inputs):
        """Encode inputs through the model with pooling strategy."""
        # BiGemma3 is based on Gemma3Model which expects standard input_ids, not doc_input_ids
        # The main_input_name = "doc_input_ids" is just for transformers compatibility, not actual input
        return self.mdl(pooling_strategy=self.pooling_strategy, **inputs)

    def get_text_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_numpy: bool = False,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        """Get single-vector embeddings for texts."""
        from tqdm import tqdm

        all_embeds = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                inputs = self.processor.process_texts(batch).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.append(outs.cpu().to(torch.float32))

        # For single-vector models, concatenate embeddings directly
        if all_embeds:
            embeddings = torch.cat(all_embeds, dim=0)
        else:
            embeddings = torch.empty(
                0, self.mdl.config.hidden_size, dtype=torch.float32
            )

        if return_numpy:
            return embeddings.numpy()
        return embeddings

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        return_numpy: bool = False,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        """Get single-vector embeddings for images."""
        import torchvision.transforms.functional as F
        from PIL import Image
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        all_embeds = []

        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(iterator):
                # Convert tensors to PIL images if needed
                imgs = [
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                    for b in batch
                ]
                inputs = self.processor.process_images(imgs).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.append(outs.cpu().to(torch.float32))

        # For single-vector models, concatenate embeddings directly
        if all_embeds:
            embeddings = torch.cat(all_embeds, dim=0)
        else:
            embeddings = torch.empty(
                0, self.mdl.config.hidden_size, dtype=torch.float32
            )

        if return_numpy:
            return embeddings.numpy()
        return embeddings

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode: str = "sum",
        **kwargs: Any,
    ):
        """Fused embeddings are not supported for BiGemma models."""
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def similarity(self, a, b):
        """Compute similarity between single-vector embeddings using cosine similarity."""
        a_torch = self._convert_to_torch_if_needed(a)
        b_torch = self._convert_to_torch_if_needed(b)
        return self.score_single_vector(a_torch, b_torch)

    @staticmethod
    def _convert_to_torch_if_needed(embeddings):
        """Convert numpy arrays to torch tensors if needed."""
        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings)
        elif isinstance(embeddings, list):
            # Handle list of numpy arrays or tensors
            converted = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    converted.append(torch.from_numpy(emb))
                else:
                    converted.append(emb)
            return converted
        return embeddings

    @staticmethod
    def score_single_vector(
        qs: torch.Tensor | list[torch.Tensor],
        ps: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the cosine similarity score for single-vector query and passage embeddings."""
        device = "cpu"

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        # Normalize inputs to 2D tensors
        def normalize_input(x):
            if isinstance(x, torch.Tensor):
                return x.unsqueeze(0) if x.ndim == 1 else x
            else:  # list
                return torch.stack(x) if len(x) > 1 else x[0].unsqueeze(0)

        qs_stacked = normalize_input(qs).to(device)
        ps_stacked = normalize_input(ps).to(device)

        # BiGemma3 already applies L2 normalization internally
        # Check if embeddings are already normalized
        qs_norms = torch.norm(qs_stacked, p=2, dim=1)
        ps_norms = torch.norm(ps_stacked, p=2, dim=1)

        # If already normalized (BiGemma3 does this internally), use dot product
        if torch.allclose(
            qs_norms, torch.ones_like(qs_norms), atol=1e-3
        ) and torch.allclose(ps_norms, torch.ones_like(ps_norms), atol=1e-3):
            scores = torch.mm(qs_stacked, ps_stacked.t()).to(torch.float32)
        else:
            # Apply normalization for cosine similarity
            qs_normalized = torch.nn.functional.normalize(qs_stacked, p=2, dim=1)
            ps_normalized = torch.nn.functional.normalize(ps_stacked, p=2, dim=1)
            scores = torch.mm(qs_normalized, ps_normalized.t()).to(torch.float32)

        # Squeeze if single query
        return scores.squeeze(0) if scores.shape[0] == 1 else scores


# Training data for BiGemma models - assuming similar to ColPali
BIGEMMA_TRAINING_DATA = COLPALI_TRAINING_DATA

bigemma3_base = ModelMeta(
    loader=partial(
        BiGemmaWrapper,
        model_name="google/gemma-3-4b-it",
        revision="0297e92d11516e25b4dc692f205a527093b2ed22",
        torch_dtype=torch.float16,
    ),
    name="google/gemma-3-4b-it",
    languages=["eng-Latn"],
    revision="0297e92d11516e25b4dc692f205a527093b2ed22",
    release_date="2025-08-11",
    modalities=["image", "text"],
    n_parameters=4_300_000_000,
    memory_usage_mb=4700,  # Update with actual memory usage
    max_tokens=128000,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali/tree/feat/gemma3",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/google/gemma-3-4b-it",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=BIGEMMA_TRAINING_DATA,
)

bigemma3_hardneg_750 = ModelMeta(
    loader=partial(
        BiGemmaWrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
        revision="3879293dad020d32df69cc48ee31dd74d52b9403",
        torch_dtype=torch.float16,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
    languages=["eng-Latn", "hin-Deva", "kan-Knda"],
    revision="3879293dad020d32df69cc48ee31dd74d52b9403",
    release_date="2025-08-11",
    modalities=["image", "text"],
    n_parameters=4_300_000_000,
    memory_usage_mb=4700,  # Update with actual memory usage
    max_tokens=128000,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali/tree/feat/gemma3",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=BIGEMMA_TRAINING_DATA,
)

bigemma3_hardneg_1694 = ModelMeta(
    loader=partial(
        BiGemmaWrapper,
        model_name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
        revision="3879293dad020d32df69cc48ee31dd74d52b9403",
        torch_dtype=torch.float16,
    ),
    name="Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
    languages=["eng-Latn", "hin-Deva", "kan-Knda"],
    revision="3879293dad020d32df69cc48ee31dd74d52b9403",
    release_date="2025-08-11",
    modalities=["image", "text"],
    n_parameters=4_300_000_000,
    memory_usage_mb=4700,  # Update with actual memory usage
    max_tokens=128000,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali/tree/feat/gemma3",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=BIGEMMA_TRAINING_DATA,
)
