# Requirements derived from the paper

## Paper facts

- Multilingual input is translated into English and then searched in an English rule base. The contribution is not a multilingual motion/text encoder.
- Gesture units are upper-body clips of 2–3 seconds (30–45 frames at 15 FPS), selected for start/end pose proximity and motion variance.
- GestureCLR trains clean 3D and projected 2D counterparts with Gaussian noise (`0.001`, `0.01`, `0.1`) and temporal shifts, using a three-layer/five-head Transformer, 512 feed-forward width, 10D latent, and NT-Xent.
- Unit latents are grouped with Bisecting K-Means; the paper uses 100 clusters. Runtime English text uses six-word chunks, Sentence-BERT cosine lookup, and random sampling within the matched cluster.
- Five-frame transition buffers and timestamp-aware playback are downstream animation concerns.

## Reimplementation decisions

- Cluster count defaults to 100 but is adjustable and capped by sample count.
- Translation goes through a `Translator` protocol:
  - an exact-match dictionary (default);
  - an OpenAI-compatible or LibreTranslate-style HTTP client;
  - a local Hugging Face MT model directory.
  No API key or MT weights are bundled, and multilingual Sentence-BERT never replaces translation.
- Algorithm 1 greedily takes 2–3 s clips in order of minimal start/end distance and removes each from the sequence. Clips that fail the variance test are discarded.
  - Thresholds are measured after scale normalisation.
  - `--variance auto` stands in for the paper's expert elbow calibration, using the elbow of the sorted log-variance curve.
- GestureCLR inputs are divided by each sequence's body scale before augmentation, so the paper's noise variances (standard deviation √variance) are relative to body size. One augmentation condition is drawn per sample.
- Fixed-size tensors use `[N,F,D]` with optional `lengths`; padded frames are masked in attention and pooling. Paired views use identical 15 FPS skeleton conventions.
- The paper preset is 1000 epochs at batch 512, cosine annealing and best-validation checkpointing. A smaller demo preset (300 epochs, batch 64) is the default for single-take data.
- Timing follows the paper: input over 30 words is split into sentence chunks, and duration is paced per gesture and refined from TTS word timestamps when available. Units play for their true length with a five-frame blend hint.

## Acceptance criteria

The repository must extract units, train GestureCLR, mine an English rule map, and retrieve from translated text. With the dictionary translator, non-English retrieval without a supplied English translation must fail. No original motion, counts, weights, or user-study claims are packaged.


## Interactive data handoff

`scripts/start_demo.py` downloads one official BEAT BVH/TextGrid take, builds a local nine-clip bank and paired windows, refines clips to roughly 2–2.5 seconds, and fits the small pose matcher in ignored outputs. Three Korean inputs have explicit English translations in the bundled fixture; arbitrary Korean text still needs a supplied translation. The English retrieval text and selected routes remain visible. The small demo uses its own 30 FPS source clips and does not establish multilingual translation or gesture quality. The older `--example` path remains an explicitly authored offline fixture. Speech is optional; recordings and fitted weights are not bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
