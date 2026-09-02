# Privacy

This repository intentionally excludes all user recordings, transcripts, generated reports, logs, model weights, runtime indexes, databases, caches, and credentials.

It includes one licensed public terminology index under `resources/lexicon/`. That index contains dictionary terms only and contains no runtime upload, conversation, question-and-answer record, personal identifier, or patient record. Its sources and licenses are documented beside it.

Runtime data is written to directories ignored by Git. Before sharing a fork or opening a pull request, run `git status --ignored` and confirm that no user data was force-added.

The optional Google Speech fallback is disabled by default. If explicitly enabled, selected audio chunks may leave the local machine and be processed by Google. Operators are responsible for consent, applicable privacy rules, retention requirements, and provider terms.

Do not attach confidential audio or transcripts to public issues.
