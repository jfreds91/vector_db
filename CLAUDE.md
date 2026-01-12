# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a homegrown vector database implementing the HNSW (Hierarchical Navigable Small World) algorithm with OpenAI CLIP embeddings for cross-modal image-text search.

## Development Commands

No formal build system exists. Development is done via Jupyter notebook.

**Install dependencies:**
```bash
pip install transformers pillow torch lmdb pydantic requests numpy
```

**Run the demo notebook:**
```bash
jupyter notebook test_module.ipynb
```

## Architecture

### Core Components

- **`jfdb/hsnw.py`** - `DataBase` class: Main API implementing HNSW insert/search operations. Loads CLIP model (ViT-Base/Patch32) for 512-dimensional embeddings.

- **`jfdb/nodes/node.py`** - `Node` dataclass: Represents vectors with `id`, `embedding` (torch tensor), `layers`, and `layer_edges` (dict mapping layer → connected node keys). Handles serialization via pickle.

- **`jfdb/backend/`** - Storage abstraction:
  - `Backend` (abstract): Defines `write_node()`, `read_node()`, `drop_backend()` interface
  - `LMDBBackend` (default): Memory-mapped file storage
  - `InMemoryBackend`: Stub for future implementation

- **`jfdb/utils/datastructures.py`** - `MaxHeap`: Priority queue using negated priorities with heapq for top-k nearest neighbor tracking.

### HNSW Algorithm

Two-phase insertion:
1. **Greedy descent** from entry layer to insertion layer (ef=1)
2. **Layer-by-layer insertion** from insertion layer to base layer 0, adding M edges per layer with pruning when edges exceed M_max threshold

Key parameters: `L` (layers), `M` (edges per insert), `M_max`/`M_max0` (max edges before pruning), `ef_construction` (candidates during insert), `m_L` (layer assignment multiplier = 1/ln(M))

Similarity uses dot product on normalized CLIP embeddings.

### Data Flow

```
Image/Text → CLIP Model → 512-dim embedding → Node → HNSW Graph → LMDB Backend
```

### Known Limitations

- Delete operation not implemented
- InMemoryBackend is a stub
- 512-dim embeddings (12KB per node) exceed LMDB 4KB page size, causing overflow pages
- No formal test suite (demo-driven via notebook)
