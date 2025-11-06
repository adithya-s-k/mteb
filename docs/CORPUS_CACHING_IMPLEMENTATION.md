# Corpus Caching Implementation for Multilingual Datasets

## Summary

Implemented automatic corpus embedding caching in `Any2AnyDenseRetrievalExactSearch` to prevent re-encoding identical images across language variants in multilingual datasets like ViDoRe.

## Changes Made

### File: `mteb/evaluation/evaluators/Image/Any2AnyRetrievalEvaluator.py`

#### 1. Added Cache Infrastructure (lines 107-110)
```python
# Corpus caching for multilingual datasets
self.corpus_embeddings_cache = {}  # {corpus_id: embedding}
self.cache_hits = 0
self.cache_misses = 0
```

#### 2. Modified `search()` Method (lines 198-336)
- Added cache lookup before encoding corpus chunks
- Handles three scenarios:
  - **All cached**: Retrieves all embeddings from cache (no encoding)
  - **None cached**: Encodes all embeddings (caches for next time)
  - **Mixed**: Encodes only uncached items, combines with cached embeddings
- Automatically logs cache performance

#### 3. Added Cache Management Methods (lines 368-396)
- `clear_corpus_cache()`: Clear cache and reset statistics
- `get_cache_stats()`: Return cache hit rate and statistics

## How It Works

1. **First Language Variant**: When evaluating the first language (e.g., English), all corpus images are encoded and cached using their corpus IDs as keys.

2. **Subsequent Languages**: When evaluating additional languages (French, Spanish, German), the cache detects that corpus IDs match previously encoded items and retrieves embeddings from cache instead of re-encoding.

3. **Automatic Logging**: At the end of each evaluation, cache statistics are logged showing hit rate and performance.

## Expected Performance Impact

### Before Optimization
For ViDoRe v2 with 4 languages (English, French, Spanish, German):
- ESG Reports: 30 documents × 4 languages = **120 encodings**
- Economics Reports: 5 documents × 4 languages = **20 encodings**
- BioMedical Lectures: 27 documents × 4 languages = **108 encodings**

### After Optimization
- ESG Reports: 30 documents + 0 (cached) × 3 = **30 encodings** (75% reduction)
- Economics Reports: 5 documents + 0 (cached) × 3 = **5 encodings** (75% reduction)
- BioMedical Lectures: 27 documents + 0 (cached) × 3 = **27 encodings** (75% reduction)

**Overall: ~75% reduction in image encoding time for 4-language ViDoRe datasets**

## Verification

When running evaluations, look for log messages like:

```
INFO: Cache: 0 hits, 30 misses  # First language (English)
INFO: Cache: 30 hits, 0 misses  # Second language (French)
INFO: Cache: 30 hits, 0 misses  # Third language (Spanish)
INFO: Cache: 30 hits, 0 misses  # Fourth language (German)
INFO: Corpus cache performance: 75.0% hit rate (90 hits, 30 misses)
```

## Testing

To test the changes:

```bash
# Run a ViDoRe v2 evaluation as normal
cd mteb/modal_scripts
modal run modal_mteb_local.py::main \
    --model "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-1694" \
    --benchmarks "ViDoRe(v2)" \
    --batch-size 12
```

The cache will automatically activate and log statistics.

## Memory Considerations

- Cache stores one embedding per unique corpus item
- Memory usage: `num_corpus_items × embedding_dim × 4 bytes` (for float32)
- Example: 1000 images × 3584 dims × 4 bytes = ~14 MB (negligible)
- Cache persists across language variants within a single evaluation run
- Cache is automatically cleared between different tasks/benchmarks

## API

### Clear Cache Manually
```python
retriever = evaluation.retriever.retriever  # Get the Any2AnyDenseRetrievalExactSearch instance
retriever.clear_corpus_cache()
```

### Get Cache Statistics
```python
stats = retriever.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_percent']}%")
print(f"Cache size: {stats['cache_size']} items")
```

## Backward Compatibility

- **Fully backward compatible**: No changes to API or evaluation results
- **Automatic activation**: Caching activates automatically for image modality
- **Transparent**: Users don't need to modify their evaluation code
- **Safe**: Text modality unaffected, only optimizes image/fused embeddings

## Future Enhancements

Possible future improvements:
1. **Cross-task caching**: Share cache across different tasks with same corpus
2. **Persistent cache**: Save cache to disk for reuse across runs
3. **Smart cache eviction**: LRU or size-based cache limits for memory efficiency
4. **Language-agnostic IDs**: Normalize corpus IDs to handle different ID formats
