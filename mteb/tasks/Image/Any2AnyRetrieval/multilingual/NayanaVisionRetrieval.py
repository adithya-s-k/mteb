from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "en": ["eng-Latn"],
    "kn": ["kan-Knda"],
    "hi": ["hin-Deva"],
}

# Cross-lingual language mappings for NayanaIR-CrossBench (20 languages, excluding Arabic and Russian)
CROSSLINGUAL_LANG_MAPPING = {
    "bn": "ben-Beng",
    "de": "deu-Latn",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "fr": "fra-Latn",
    "gu": "guj-Gujr",
    "hi": "hin-Deva",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "kn": "kan-Knda",
    "ko": "kor-Kore",
    "ml": "mal-Mlym",
    "mr": "mar-Deva",
    "or": "ory-Orya",
    "pa": "pan-Guru",
    "sa": "san-Deva",
    "ta": "tam-Taml",
    "te": "tel-Telu",
    "th": "tha-Thai",
    "zh": "zho-Hans",
}

# Monolingual language mappings for NayanaIR-Bench-Monolingual
MONOLINGUAL_LANG_MAPPING = {
    "ar": "ara-Arab",
    "bn": "ben-Beng",
    "de": "deu-Latn",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "fr": "fra-Latn",
    "gu": "guj-Gujr",
    "hi": "hin-Deva",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "kn": "kan-Knda",
    "ko": "kor-Kore",
    "ml": "mal-Mlym",
    "mr": "mar-Deva",
    "or": "ory-Orya",
    "pa": "pan-Guru",
    "ru": "rus-Cyrl",
    "sa": "san-Deva",
    "ta": "tam-Taml",
    "te": "tel-Telu",
    "th": "tha-Thai",
    "zh": "zho-Hans",
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
                "image": None,
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
                "text": None,
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


class NayanaIR_v12(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaVisionRetrievalV12",
        description="Retrieve associated pages according to questions using Nayana multilingual document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/nayana-beir-eval-multilang_v12",
        dataset={
            "path": "Nayana-cognitivelab/nayana-beir-eval-multilang_v12",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIR_v1(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaVisionRetrievalV1",
        description="Retrieve associated pages according to questions using Nayana multilingual document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/nayana-beir-eval-multilang",
        dataset={
            "path": "Nayana-cognitivelab/nayana-beir-eval-multilang",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRCrossBench(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRCrossBench",
        description="Retrieve associated pages according to questions using Nayana cross-lingual document retrieval dataset with 20 languages.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-CrossBench-v3",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-CrossBench-v3",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs={
            lang: [script] for lang, script in CROSSLINGUAL_LANG_MAPPING.items()
        },
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=CROSSLINGUAL_LANG_MAPPING.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


# Monolingual Task Classes for NayanaIR-Bench-Monolingual


class NayanaIRMonoBench_ar(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ar",
        description="Retrieve associated pages according to questions using Nayana Arabic document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ar",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ar",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ar"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_bn(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-bn",
        description="Retrieve associated pages according to questions using Nayana Bengali document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-bn",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-bn",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["bn"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_de(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-de",
        description="Retrieve associated pages according to questions using Nayana German document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-de",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-de",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["de"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_en(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-en",
        description="Retrieve associated pages according to questions using Nayana English document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-en",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-en",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["en"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_es(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-es",
        description="Retrieve associated pages according to questions using Nayana Spanish document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-es",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-es",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["es"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_fr(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-fr",
        description="Retrieve associated pages according to questions using Nayana French document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-fr",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-fr",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["fr"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_gu(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-gu",
        description="Retrieve associated pages according to questions using Nayana Gujarati document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-gu",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-gu",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["gu"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_hi(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-hi",
        description="Retrieve associated pages according to questions using Nayana Hindi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-hi",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-hi",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["hi"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_it(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-it",
        description="Retrieve associated pages according to questions using Nayana Italian document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-it",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-it",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["it"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_ja(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ja",
        description="Retrieve associated pages according to questions using Nayana Japanese document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ja",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ja",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ja"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_kn(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-kn",
        description="Retrieve associated pages according to questions using Nayana Kannada document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-kn",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-kn",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["kn"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_ko(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ko",
        description="Retrieve associated pages according to questions using Nayana Korean document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ko",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ko",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ko"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_ml(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ml",
        description="Retrieve associated pages according to questions using Nayana Malayalam document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ml",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ml",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ml"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_mr(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-mr",
        description="Retrieve associated pages according to questions using Nayana Marathi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-mr",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-mr",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["mr"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_or(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-or",
        description="Retrieve associated pages according to questions using Nayana Odia document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-or",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-or",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["or"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_pa(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-pa",
        description="Retrieve associated pages according to questions using Nayana Punjabi document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-pa",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-pa",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["pa"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_ru(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ru",
        description="Retrieve associated pages according to questions using Nayana Russian document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ru",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ru",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ru"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_sa(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-sa",
        description="Retrieve associated pages according to questions using Nayana Sanskrit document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-sa",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-sa",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["sa"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_ta(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-ta",
        description="Retrieve associated pages according to questions using Nayana Tamil document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-ta",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-ta",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["ta"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_te(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-te",
        description="Retrieve associated pages according to questions using Nayana Telugu document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-te",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-te",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["te"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_th(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-th",
        description="Retrieve associated pages according to questions using Nayana Thai document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-th",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-th",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["th"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class NayanaIRMonoBench_zh(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NayanaIRMonoBench-zh",
        description="Retrieve associated pages according to questions using Nayana Chinese document retrieval dataset.",
        reference="https://huggingface.co/datasets/Nayana-cognitivelab/NayanaIR-MonoBench-zh",
        dataset={
            "path": "Nayana-cognitivelab/NayanaIR-MonoBench-zh",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=[MONOLINGUAL_LANG_MAPPING["zh"]],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{nayana2025embed,
  author = {CognitiveLab},
  title = {Nayana Embed: Multilingual Document Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 300,
                    "num_queries": 350,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
