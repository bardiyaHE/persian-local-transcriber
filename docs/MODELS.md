# Model and runtime sources

Model weights and runtime binaries are not stored in this Git repository. Setup downloads pinned resources and validates their expected files, sizes, and checksums.

## Whisper CTranslate2 models

| Local name | Repository | Pinned revision |
|---|---|---|
| Medium | `Systran/faster-whisper-medium` | `08e178d48790749d25932bbc082711ddcfdfbc4f` |
| Large V3 Turbo | `dropbox-dash/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| Large V3 | `Systran/faster-whisper-large-v3` | `edaa852ec7e145841d8ffdb056a99866b5f0a478` |

## Local language model

- Repository: `unsloth/Qwen3.5-35B-A3B-GGUF`
- File: `Qwen3.5-35B-A3B-UD-Q4_K_L.gguf`
- Pinned revision: `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- Runtime: pinned `llama.cpp` release `b10642`

## Semantic encoder

- Repository: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Pinned revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Runtime file: `onnx/model_quint8_avx2.onnx`

## Audio enhancement

- Music/background separation: `adefossez/HTDemucs`, model `htdemucs`, pinned revision `cbc8a9b1a87023b7fd74e7b3412e6321c0eab003`.
- Speaker diarization: `pyannote/speaker-diarization-community-1` with `pyannote.audio==4.0.7`.
- The pyannote repository is gated. Users must accept its model terms and provide a Hugging Face read token during setup.
- Demucs and pyannote run in separate pinned Python environments because their Torch versions differ. Runtime inference is offline.

Upstream model cards and licenses govern model use and redistribution. Review them before commercial deployment. This repository does not redistribute the weights.
