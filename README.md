# Persian Local Transcriber

A local-first Persian speech-to-text pipeline with audio normalization, noise reduction, multi-pass Whisper transcription, deterministic consensus, constrained local-model review, and a separate summary.

The repository contains source code only. It does not include audio recordings, transcripts, generated outputs, evaluation references, domain databases, model weights, caches, credentials, or logs.

## Highlights

- Accepts any audio container and codec supported by FFmpeg; the filename extension is not trusted.
- Normalizes audio and applies DeepFilterNet before transcription.
- Uses Whisper Large V3 Turbo as the base and selectively consults additional Whisper families on uncertain regions.
- Ranks constrained alternatives with consensus, n-gram, lexical, and semantic signals.
- Uses a local Qwen model for constrained reranking and a separate summary.
- Provides a local Gradio interface and a PowerShell CLI.
- Keeps the optional online speech fallback disabled by default.

## Privacy defaults

- Uploaded audio and all generated artifacts remain under the ignored `inputs/` and `outputs/` directories.
- Model weights, locally built indexes, caches, and runtime binaries are ignored by Git.
- No sample conversation or domain database is bundled.
- Google Speech fallback is **off by default**. Enabling it can send selected audio chunks to Google and is not suitable for confidential recordings without the required consent and policy review.

To explicitly enable that optional fallback for the current PowerShell session:

```powershell
$env:PERSIAN_TRANSCRIBER_GOOGLE_FALLBACK = '1'
```

## Requirements

- Windows 10 or 11
- PowerShell 7
- Python 3.12
- NVIDIA GPU recommended; CPU execution is supported but slower
- Sufficient disk space for downloaded Whisper and Qwen model weights

The setup downloads third-party runtimes and model weights into ignored local directories. Review the corresponding upstream licenses before redistribution or commercial use.

## Quick start

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\launch_ui.ps1
```

The local interface opens at `http://127.0.0.1:7860`.

Run a file directly:

```powershell
.\run_pipeline.ps1 -AudioFile 'C:\path\to\audio-file'
```

## Pipeline

1. Detect and decode the real audio stream with FFmpeg.
2. Normalize to mono PCM and create a DeepFilterNet-enhanced copy.
3. Generate Turbo hypotheses from raw and enhanced audio.
4. Run additional Whisper models only over uncertain intervals.
5. Apply deterministic consensus and locally built lexical/semantic indexes.
6. Let the local model choose only among allowed candidates.
7. Build a separate summary and validate sensitive names, numbers, doses, and negation against source hypotheses.
8. Return the transcript, summary, confidence information, and review artifacts.

## Repository layout

- `src/pipeline.py`: audio preparation and Whisper inference
- `src/consensus_v*.py`: deterministic and semantic consensus stages
- `src/local_qwen_reranker_v10.py`: constrained local reranking
- `src/local_qwen_summarizer_v11.py`: separate evidence-checked summary
- `src/web_app.py`: local Gradio interface
- `setup.ps1`: local dependency and model setup
- `run_pipeline.ps1`: command-line entry point

## Safety and limitations

Automatic transcription can be wrong, especially for names, numbers, negation, and domain-specific terms. The output is not a substitute for listening to the original recording and must not be treated as verified professional documentation without human review.
