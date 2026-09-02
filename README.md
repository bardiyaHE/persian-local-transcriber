# Persian Local Transcriber

A local-first Persian speech-to-text application with two installation profiles:

- **Lite:** Whisper Large V3 Turbo on raw audio and a locally separated main-speaker track. It produces a transcript without installing the local language model.
- **Full:** adaptive multi-model Whisper, deterministic consensus, public lexicon and local n-gram evidence, constrained Qwen reranking, and a separate local summary.

The repository contains code, documentation, and a licensed public terminology index. It contains no user recording, transcript, generated output, evaluation reference, patient record, runtime database, model weight, credential, cache, or log.

## Quick start on Windows

Clone or download the repository, open PowerShell in its directory, and choose one profile.

### Lite

Smaller download and faster setup. Produces a Turbo transcript but no Full semantic review or local summary.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Profile Lite
.\launch_ui.ps1
```

### Full

Downloads all required models and reproduces the complete local pipeline.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Profile Full
.\launch_ui.ps1
```

Running `setup.ps1` again is safe. Completed files are reused, Hugging Face downloads resume from their local cache, checksums are verified, and only missing profile components are installed. A Lite installation can be upgraded by running `setup.ps1 -Profile Full`.

Both profiles use the gated `pyannote/speaker-diarization-community-1` model. Before the first setup, accept that model's Hugging Face access terms and set `HF_TOKEN`. Setup caches all required enhancement weights; audio processing then runs offline with telemetry disabled.

The local interface opens at `http://127.0.0.1:7860`.

## What setup installs

| Component | Lite | Full | Stored in Git |
|---|---:|---:|---:|
| FFmpeg, HTDemucs, and pyannote Community-1 | yes | yes | no |
| Whisper Large V3 Turbo | yes | yes | no |
| Whisper Medium and Large V3 | no | yes | no |
| MiniLM semantic encoder | no | yes | no |
| Qwen 35B-A3B and llama.cpp | no | yes | no |
| Licensed public terminology index | no | yes | yes |
| Runtime n-gram SQLite database | no | built locally | no |

The two isolated enhancement environments add several gigabytes because Demucs and pyannote require different pinned Torch stacks. Full also requires roughly 26 GB for ASR/LLM model weights plus installation headroom. Exact usage varies with GPU runtime packages and caches.

## Command line

The installed profile is selected automatically:

```powershell
.\run_pipeline.ps1 -AudioFile 'C:\path\to\audio-file'
```

A Full installation can explicitly run the lighter path:

```powershell
.\run_pipeline.ps1 -AudioFile 'C:\path\to\audio-file' -Profile lite
```

## Privacy defaults

- Runtime audio and results are written only to ignored `inputs/` and `outputs/` directories.
- Models, generated indexes, caches, binaries, and environment files are ignored by Git.
- Google Speech fallback is **disabled by default**.
- Enabling the optional fallback can send selected audio chunks to Google. Do not enable it for confidential recordings without the required consent and policy review.

To explicitly enable it for the current PowerShell session:

```powershell
$env:PERSIAN_TRANSCRIBER_GOOGLE_FALLBACK = '1'
```

## Documentation

- [Installation and troubleshooting](docs/INSTALLATION.md)
- [Architecture and profile behavior](docs/ARCHITECTURE.md)
- [Model and runtime sources](docs/MODELS.md)
- [Lexicon sources and licenses](resources/lexicon/SOURCES.md)
- [Privacy policy](PRIVACY.md)

## Safety and limitations

Automatic transcription can be wrong, especially for names, numbers, negation, and domain-specific terms. The output is not a substitute for listening to the original recording and must not be treated as verified professional documentation without human review.
