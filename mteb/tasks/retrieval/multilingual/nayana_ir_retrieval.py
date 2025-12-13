from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

# Cross-lingual language mappings for NayanaIR-CrossBench (20 languages)
_CROSS_LANGS = {
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "gu": ["guj-Gujr"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "kn": ["kan-Knda"],
    "ko": ["kor-Kore"],
    "ml": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "or": ["ory-Orya"],
    "pa": ["pan-Guru"],
    "sa": ["san-Deva"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "zh": ["zho-Hans"],
}

# Monolingual language mappings for NayanaIR-Bench-Monolingual
_MONO_LANGS = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "gu": ["guj-Gujr"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "kn": ["kan-Knda"],
    "ko": ["kor-Kore"],
    "ml": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "or": ["ory-Orya"],
    "pa": ["pan-Guru"],
    "ru": ["rus-Cyrl"],
    "sa": ["san-Deva"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "zh": ["zho-Hans"],
}


def get_cross_langs(langs: list[str]) -> dict[str, list[str]]:
    return {lang: _CROSS_LANGS[lang] for lang in langs}


def get_mono_langs(langs: list[str]) -> dict[str, list[str]]:
    return {lang: _MONO_LANGS[lang] for lang in langs}


COMMON_METADATA = {
    "type": "DocumentUnderstanding",
    "category": "t2i",
    "eval_splits": ["test"],
    "main_score": "ndcg_at_5",
    "date": ("2024-12-01", "2025-05-01"),
    "task_subtypes": ["Image Text Retrieval"],
    "license": "mit",
    "annotations_creators": "derived",
    "dialect": [],
    "modalities": ["text", "image"],
    "sample_creation": "multiple",
    "bibtex_citation": r"""@misc{kolavi2025m3druniversalmultilingualmultimodal,
  title={M3DR: Towards Universal Multilingual Multimodal Document Retrieval}, 
  author={Adithya S Kolavi and Vyoman Jain},
  year={2025},
  eprint={2512.03514},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2512.03514}
}""",
    "prompt": {"query": "Find a screenshot that relevant to the user's question."},
}


def _load_data(
    path: str,
    splits: str,
    langs: list | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    if langs is None:
        corpus = {}
        queries = {}
        relevant_docs = {}
    else:
        corpus = {lang: {} for lang in langs}
        queries = {lang: {} for lang in langs}
        relevant_docs = {lang: {} for lang in langs}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query-id']}",
                "text": x["query"],
                "modality": "text",
            },
            remove_columns=["query-id", "query"],
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus-id']}",
                "modality": "image",
            },
            remove_columns=["corpus-id"],
        )

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )

        if langs is None:
            queries[split] = query_ds
            corpus[split] = corpus_ds
            relevant_docs[split] = {}
            for row in qrels_ds:
                qid = f"query-{split}-{row['query-id']}"
                did = f"corpus-{split}-{row['corpus-id']}"
                if qid not in relevant_docs[split]:
                    relevant_docs[split][qid] = {}
                relevant_docs[split][qid][did] = int(row["score"])
        else:
            for lang in langs:
                queries[lang][split] = query_ds.filter(lambda x: x["language"] == lang)

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = {}
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query-id']}"
                    did = f"corpus-{split}-{row['corpus-id']}"
                    if qid not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][qid] = {}
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


def load_data(self) -> None:
    if self.data_loaded:
        return

    # For multilingual tasks (eval_langs is a dict), pass langs for filtering
    # For monolingual tasks (eval_langs is a list), pass None to skip filtering
    langs = (
        list(self.metadata.eval_langs.keys())
        if isinstance(self.metadata.eval_langs, dict)
        else None
    )

    self.corpus, self.queries, self.relevant_docs = _load_data(
        path=self.metadata.dataset["path"],
        splits=self.metadata.eval_splits,
        langs=langs,
        revision=self.metadata.dataset["revision"],
    )

    self.data_loaded = True


class NayanaIRCrossBench(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRCrossBench",
        description="Retrieve associated pages across 20 languages according to questions using Nayana cross-lingual document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-CrossBench",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-CrossBench",
            "revision": "4b395c757c95f6481a0eb5419355b16437e54fe7",
        },
        eval_langs=get_cross_langs(list(_CROSS_LANGS.keys())),
        domains=["Web", "Written"],
        **COMMON_METADATA,
    )

    load_data = load_data


# Monolingual Task Classes for NayanaIR-Bench-Monolingual
class NayanaIRMonoBench_ar(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ar",
        description="Retrieve associated pages according to questions using Nayana Arabic document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ar",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ar",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ar"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_bn(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-bn",
        description="Retrieve associated pages according to questions using Nayana Bengali document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-bn",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-bn",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["bn"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_de(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-de",
        description="Retrieve associated pages according to questions using Nayana German document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-de",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-de",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["de"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_en(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-en",
        description="Retrieve associated pages according to questions using Nayana English document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-en",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-en",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["en"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_es(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-es",
        description="Retrieve associated pages according to questions using Nayana Spanish document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-es",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-es",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["es"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_fr(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-fr",
        description="Retrieve associated pages according to questions using Nayana French document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-fr",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-fr",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["fr"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_gu(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-gu",
        description="Retrieve associated pages according to questions using Nayana Gujarati document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-gu",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-gu",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["gu"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_hi(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-hi",
        description="Retrieve associated pages according to questions using Nayana Hindi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-hi",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-hi",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["hi"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_it(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-it",
        description="Retrieve associated pages according to questions using Nayana Italian document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-it",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-it",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["it"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_ja(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ja",
        description="Retrieve associated pages according to questions using Nayana Japanese document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ja",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ja",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ja"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_kn(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-kn",
        description="Retrieve associated pages according to questions using Nayana Kannada document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-kn",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-kn",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["kn"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_ko(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ko",
        description="Retrieve associated pages according to questions using Nayana Korean document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ko",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ko",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ko"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_ml(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ml",
        description="Retrieve associated pages according to questions using Nayana Malayalam document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ml",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ml",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ml"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_mr(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-mr",
        description="Retrieve associated pages according to questions using Nayana Marathi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-mr",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-mr",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["mr"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_or(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-or",
        description="Retrieve associated pages according to questions using Nayana Odia document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-or",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-or",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["or"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_pa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-pa",
        description="Retrieve associated pages according to questions using Nayana Punjabi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-pa",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-pa",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["pa"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_ru(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ru",
        description="Retrieve associated pages according to questions using Nayana Russian document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ru",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ru",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ru"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_sa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-sa",
        description="Retrieve associated pages according to questions using Nayana Sanskrit document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-sa",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-sa",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["sa"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_ta(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ta",
        description="Retrieve associated pages according to questions using Nayana Tamil document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-ta",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-ta",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["ta"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_te(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-te",
        description="Retrieve associated pages according to questions using Nayana Telugu document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-te",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-te",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["te"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_th(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-th",
        description="Retrieve associated pages according to questions using Nayana Thai document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-th",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-th",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["th"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data


class NayanaIRMonoBench_zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-zh",
        description="Retrieve associated pages according to questions using Nayana Chinese document retrieval dataset.",
        reference="https://huggingface.co/datasets/Cognitive-Lab/NayanaIR-MonoBench-zh",
        dataset={
            "path": "Cognitive-Lab/NayanaIR-MonoBench-zh",
            "revision": "main",
        },
        eval_langs=_MONO_LANGS["zh"],
        domains=["Academic"],
        **COMMON_METADATA,
    )

    load_data = load_data
