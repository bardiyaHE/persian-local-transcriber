# Privacy

This repository intentionally excludes all user recordings, transcripts, generated reports, logs, model weights, local indexes, databases, caches, and credentials.

Runtime data is written to directories ignored by Git. Before sharing a fork or opening a pull request, run `git status --ignored` and confirm that no user data was force-added.

The optional Google Speech fallback is disabled by default. If explicitly enabled, selected audio chunks may leave the local machine and be processed by Google. Operators are responsible for consent, applicable privacy rules, retention requirements, and provider terms.

Do not attach confidential audio or transcripts to public issues.
