# Architecture

## Shared audio path

1. FFprobe detects the actual audio stream instead of trusting the filename extension.
2. FFmpeg converts the source to normalized mono PCM.
3. HTDemucs separates the vocal/speech stem from background music while the raw normalized copy is retained.
4. pyannote Community-1 diarizes the speech stem. If it finds multiple speakers, the speaker with the greatest exclusive speaking time is retained in a full-length, timestamp-preserving track; a 60 ms boundary pad and short fades reduce clipped word edges. A single-speaker stem is preserved without gating.
5. Whisper is explicitly forced to Persian (`fa`). A script-quality guard rejects an enhanced branch when non-Persian letters dominate while the raw branch remains valid. The raw hypothesis becomes the safe fallback, and Full mode sends the complete audio to the secondary ASR families for review.

## Lite profile

1. Whisper Large V3 Turbo transcribes raw and enhanced audio.
2. The enhanced Turbo path is the transcript base.
3. Confidence and disagreement metadata are retained for review.
4. No local generative language model, semantic encoder, domain database, or external service is invoked.

Lite is a useful transcript-only application, not a placeholder installer.

## Full profile

1. Turbo processes the complete raw and enhanced audio.
2. Uncertain intervals are identified from word confidence and raw/enhanced disagreement.
3. Whisper Large V3 reviews only the Turbo intervals.
4. Large raw/enhanced confidence and agreement are checked again. Whisper Medium runs only on the residual intervals that Large did not resolve.
5. For comparison, unheard spans are inserted from the previous cascade stage: Large views use Turbo as their backbone, while Medium views use Large plus stable Turbo spans. This reconstructs six full-text views without pretending that inserted spans are additional acoustic votes.
6. Deterministic consensus combines acoustic-family support, lexical evidence, n-gram context, and MiniLM similarity.
7. Qwen sees all six full-text views for context, but it may choose only candidates inside the uncertain intervals. Stable Turbo spans are locked, and a final coverage guard rejects any accidental deletion.
8. A separate Qwen pass creates a summary from the full reconstructed evidence and retains coverage metadata for every partial ASR source.
9. Post-validation checks names, numbers, doses, negation, speaker roles, entity subjects, and unsupported additions against the transcription evidence.

## Data boundaries

- Runtime uploads never train or extend the bundled lexicon or n-gram database.
- The public terminology index contains licensed terms only.
- The n-gram SQLite file is generated locally from public sources and is not committed.
- Google Speech is an explicit opt-in fallback and is disabled in a fresh installation.
- Demucs and pyannote weights are downloaded only during setup. Runtime sets Hugging Face offline mode, disables pyannote metrics, and does not upload audio.

## Installation state

`runtime/install-profile.json` records the active `lite` or `full` profile. Both the CLI and web interface read this file, so they do not attempt to start components that were not installed.
