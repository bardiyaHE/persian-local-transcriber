# Architecture

## Shared audio path

1. FFprobe detects the actual audio stream instead of trusting the filename extension.
2. FFmpeg converts the source to normalized mono PCM.
3. DeepFilterNet creates a noise-reduced copy while the raw normalized copy is retained.

## Lite profile

1. Whisper Large V3 Turbo transcribes raw and enhanced audio.
2. The enhanced Turbo path is the transcript base.
3. Confidence and disagreement metadata are retained for review.
4. No local generative language model, semantic encoder, domain database, or external service is invoked.

Lite is a useful transcript-only application, not a placeholder installer.

## Full profile

1. Turbo processes the complete raw and enhanced audio.
2. Uncertain intervals are identified from word confidence and raw/enhanced disagreement.
3. Whisper Medium and Large V3 inspect only those intervals.
4. Deterministic consensus combines acoustic-family support, lexical evidence, n-gram context, and MiniLM similarity.
5. Qwen may choose only among candidates already admitted by the constrained lattice; its free-form text cannot enter the transcript.
6. A separate Qwen pass creates a summary.
7. Post-validation checks names, numbers, doses, negation, and unsupported additions against the transcription evidence.

## Data boundaries

- Runtime uploads never train or extend the bundled lexicon or n-gram database.
- The public terminology index contains licensed terms only.
- The n-gram SQLite file is generated locally from public sources and is not committed.
- Google Speech is an explicit opt-in fallback and is disabled in a fresh installation.

## Installation state

`runtime/install-profile.json` records the active `lite` or `full` profile. Both the CLI and web interface read this file, so they do not attempt to start components that were not installed.
