# Lexicon sources and redistribution notes

The bundled JSON index contains terminology only. It contains no audio, transcript, question-and-answer record, personal identifier, or patient record.

## PersianMedQA dictionary

- Source: [MohammadJRanjbar/PersianMedQA](https://huggingface.co/datasets/MohammadJRanjbar/PersianMedQA)
- Declared dictionary license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Attribution: Mohammad Javad Ranjbar Kalahroodi, Sepehr Karimi, Amirhossein Sheikholselami, Sepideh Ranjbar Kalahroodi, Heshaam Faili, and Azadeh Shakery, *PersianMedQA: Evaluating Large Language Models on a Persian-English Bilingual Medical Question Answering Benchmark*.
- Included material: dictionary terminology only; benchmark questions and answers are not included.

## Persian drug names

- Source: [dadashzadeh/Collection-of-drug-names-in-Persian](https://huggingface.co/datasets/dadashzadeh/Collection-of-drug-names-in-Persian)
- Declared license: MIT
- Pinned revision: `9ca2bcf9af0dce18e9e7d3ce5942c26a2f4be811`
- Included fields: English identity and Persian spelling.

## Combined index

`combined-medical-drug-index.json` is a deterministic merge of the two licensed terminology sources. It keeps source and license metadata on each imported row, excludes unsupported non-prescription product groups from the supplemental drug list, and does not add inference-time user content.

The small text overlays in this directory are project-maintained decoding rules. They contain generic vocabulary only and are not learned from runtime uploads.
