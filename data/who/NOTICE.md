# Source of this data

`codes.jsonl.gz` holds ICD-11 content extracted from WHO's ICD-API container, release 2026-01, in English and WHO's official Arabic, Chinese, French, German, Portuguese, Russian and Spanish versions. Each record keeps the ICD-11 code, title and URI.

International Classification of Diseases, Eleventh Revision (ICD-11), World Health Organization (WHO) 2019 https://icd.who.int/browse11. Licensed under the Creative Commons Attribution-NoDerivatives 3.0 IGO licence (CC BY-ND 3.0 IGO).

The index adds the metadata fields `parent_code`, `leaf`, `residual`, `depth` and `chapter` for lookup. These fields and the abbreviation list in `data/shorthand.json` are outside the WHO source content.
