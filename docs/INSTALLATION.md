# Installation and troubleshooting

## Prerequisites

- Windows 10 or 11
- PowerShell 7
- Python 3.12 available as `python`
- Internet access during the first setup
- NVIDIA GPU recommended; CPU mode is supported but substantially slower

The enhancement stage uses the gated `pyannote/speaker-diarization-community-1` model. Accept its access terms on Hugging Face and expose a read token before setup:

```powershell
$env:HF_TOKEN = 'hf_...'
```

The token is used only to cache the model during setup and is never written to reports. Runtime inference is forced offline and pyannote telemetry is disabled. The Full public n-gram build uses Persian Wikipedia and a public Common Voice text manifest; it does not download Common Voice audio.

## Validate before downloading

These commands validate the installer and the bundled public lexicon without starting model downloads:

```powershell
.\setup.ps1 -Profile Lite -ValidateOnly
.\setup.ps1 -Profile Full -ValidateOnly
```

## Install

```powershell
# Small transcript-only profile
.\setup.ps1 -Profile Lite

# Complete local pipeline
.\setup.ps1 -Profile Full
```

The installer creates `.venv/`, `models/`, `runtime/`, `offline-lexicon/`, `offline-corpus/`, `wheelhouse/`, `inputs/`, and `outputs/`. These directories are intentionally ignored by Git.

Large downloads are pinned to model revisions and validated. Re-running the same command resumes or reuses completed work. Download staging is kept under `%LOCALAPPDATA%\PersianLocalTranscriber\downloads`.

## Upgrade Lite to Full

```powershell
.\setup.ps1 -Profile Full
```

The existing Turbo model and Python packages are reused. Only Full-only models, runtimes, semantic resources, and the local n-gram database are added.

## Offline reinstall

`-Offline` validates and uses already cached/downloaded assets. It cannot create a first installation on a machine that has never downloaded the required files.

```powershell
.\setup.ps1 -Profile Full -Offline
```

## Health checks

Setup validates required files, model sizes and hashes, runtime executables, the public lexicon, SQLite tables, Python dependencies, and the Gradio application structure. It does not process or create a sample patient recording.

Run the health check again at any time:

```powershell
.\.venv\Scripts\python.exe .\src\healthcheck.py --root . --profile full
```

## Common failures

- **Insufficient disk space:** free additional space and rerun setup; completed downloads are reused.
- **Interrupted download:** rerun the same setup command.
- **CUDA libraries missing:** rerun setup while the NVIDIA driver and `nvidia-smi` are available.
- **pyannote access denied:** accept the Community-1 model terms, set `HF_TOKEN`, and rerun setup.
- **Full requested after Lite:** run `setup.ps1 -Profile Full` before launching Full processing.
- **Corpus source unavailable:** check access to Persian Wikipedia and Hugging Face, then rerun Full setup.
