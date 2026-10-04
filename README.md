# Expanding Multilingual Co-Speech Interaction: The Impact of Enhanced Gesture Units in Text-to-Gesture Synthesis for Digital Humans

**Ghazanfar Ali, Woojoo Kim, Muhammad Shahid Anwar, Jae-In Hwang, Ahyoung Choi**

**IEEE Access · 2025** · Published

[Paper / publisher](https://doi.org/10.1109/access.2025.3596328) · [Project page](https://ghazanfarali.com/research/multilingual-gesture/) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> More gesture variety supports multilingual digital-human interaction.

![Graphical abstract: GestureCLR rule construction and translation-based multilingual gesture retrieval](paper-assets/graphical-abstract.png)

*Graphical abstract diagram. GestureCLR expands a clustered gesture library for translation-based multilingual retrieval.*

## Why this research

Digital humans need varied gestures across languages. This study examines both the value of a larger motion library and whether translation can support an English gesture rule base without degrading the studied interaction.

The system extracts text and 2D pose from English monologue videos, matches those poses to captured 3D gesture units with GestureCLR, and builds a text-to-gesture rule base. Non-English input is translated into English before gesture retrieval. The study compares no gestures, a small library, and the expanded library.

## Method at a glance

**Video + captured motion** → **GestureCLR rule-map** → **Text-driven gesture retrieval**

| | Research system |
|---|---|
| Input | Text; non-English text translated into English |
| Method | Contrastive 2D-to-3D matching offline; rule-based retrieval at runtime |
| Output | Retrieved 3D co-speech gesture units for a digital human |

## Evidence and scope

51-participant study of gesture diversity and translation; reported high-noise matching improvement

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** English monologue videos; Korean-speaker motion capture; 2,035 units and 210,000 rules.

**Limitations:** Multilingual support uses translation and an English rule base. The study does not establish equivalence across all languages or cultures.

## Explore the implementation

Unit extraction, contrastive pose matching, clustered English rules and translation-first retrieval. Public motion replaces the institute's captured library.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1109/access.2025.3596328). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Please cite the research paper when using its ideas; [download the BibTeX citation](citation.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

Independent educational reimplementation of *Expanding Multilingual Co-Speech Interaction: The Impact of Enhanced Gesture Units in Text-to-Gesture Synthesis for Digital Humans* (Ali et al., IEEE Access 2025, DOI: [10.1109/ACCESS.2025.3596328](https://doi.org/10.1109/ACCESS.2025.3596328)). It follows the paper's actual multilingual design: translate input to English, then run English Sentence-BERT rule retrieval over GestureCLR-matched and clustered motion units. It does not redesign the system as a multilingual encoder.

### Setup and data contract

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the `source` line.

Verify the pipeline offline before downloading data or encoders:

```bash
python scripts/smoke.py
```

This generates every documented array plus a local 384-D SentenceTransformer fixture, then invokes the installed `extract-units`, `train`, `mine`, and multilingual `retrieve` CLI paths. It writes the result to `outputs/smoke/cli-sequence.json`. The local encoder replaces only downloadable Sentence-BERT weights; real public arrays and `all-MiniLM-L6-v2` use the same checkpoints and downstream contracts.

Prepare public data yourself. [Talking With Hands](https://github.com/facebookresearch/TalkingWithHands32M) can provide paired 3D motion for GestureCLR training; [BEAT](https://pantomatrix.github.io/BEAT/) is a public replacement for timed text/motion experiments. Follow each dataset's request process and license. Wild video data must be content you may download/process. No paper data, Korean-speaker capture, Papago credentials, model weights, or reported 2,035-unit/210,000-rule artifact is bundled.

All motion is 15 FPS, root/neck centered, fixed joint order, and flattened as `[N,F,D]`. Training `pairs.npz` has aligned `pose2d` and `motion3d`; `units.npz` has `motion3d` and string `ids`; `wild.npz` has three-second `pose2d` and aligned English `texts`. Translation JSON maps each exact source string to its English translation; generate it with a translation service you are licensed to use.

```bash
multigesture extract-units --motion data/speaker_motion.npy --variance 0.002 --closure 0.3 --output outputs/units.npz
multigesture train --pairs data/pairs.npz --output checkpoints/gestureclr.pt
multigesture mine --wild data/wild.npz --units data/units.npz --checkpoint checkpoints/gestureclr.pt --output-prefix outputs/library --clusters 100
multigesture retrieve --rules outputs/library.rules.jsonl --clusters outputs/library.clusters.npz --source-language ko --translations data/translations.json --text "안녕하세요 여러분" --output outputs/sequence.json
python -m pytest
```

Outputs keep source/English text, semantic similarity, cluster ID, chosen unit ID, and the paper's five-frame blend hint. Tune variance and closure thresholds on a development subset; `100` clusters is a paper setting, not a universal optimum.

### Limits and license

This repository starts after transcription, alignment, projection, and skeleton normalization. It does not call a commercial translation API, perform retargeting, synthesize speech, or render an avatar. Translation quality, timing, cultural appropriateness, and gesture semantics are separate failure modes; the paper's study does not prove equivalence for all languages. Code is MIT licensed; datasets, pretrained models, translations, and animations keep their original licenses.

### Citation

Machine-readable metadata is in [citation.bib](citation.bib).

```bibtex
@article{ali2025multilingual, title={Expanding Multilingual Co-Speech Interaction: The Impact of Enhanced Gesture Units in Text-to-Gesture Synthesis for Digital Humans}, author={Ali, Ghazanfar and Kim, Woojoo and Anwar, Muhammad Shahid and Hwang, Jae-In and Choi, Ahyoung}, journal={IEEE Access}, volume={13}, pages={145144--145157}, year={2025}, doi={10.1109/ACCESS.2025.3596328}}
```
