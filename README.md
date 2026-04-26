# Spotify Playlist Recommender

A production-oriented deep learning recommendation system built on the **Spotify Million Playlist Dataset**, augmented with per-track audio attributes from the Spotify API.

The system is designed as an **autoregressive transformer decoder** that treats playlist generation as a sequence modeling task. Unlike standard recommenders that rely on static ID embeddings, this model is built for the realities of modern streaming:

* **Solves the Cold-Start Problem:** Uses a content-based embedding architecture (audio features + metadata) to recommend tracks never seen during training.

* **Scales to Massive Vocabularies:** Implements Sampled Softmax with Log-Q correction to efficiently handle a catalog of over 2.2 million unique tracks.

* **Intent-Driven:** Uses a frozen LLM (Gemma-300M) to project playlist names into the latent space, allowing for high-quality "Zero-Shot" generation from a title alone.

---

## Examples

Type "Trance" as a playlist name and it immediately surfaces the genre's canonical names: Armin van Buuren, ATB, Above & Beyond. Exactly what you'd expect.

![name](assets/images/recs_name.png)

Add two tracks — *Alice DJ - Better Off Alone* and *DJ Sammy - Heaven* — and the recommendations transform. The model picks up that you're not after deep prog-trance; you want the euphoric, Eurodance-inflected sound of the late 90s. Suddenly it's serving Darude, Haddaway, Vengaboys, Ian van Dahl.

![name_and_songs](assets/images/recs_name_and_songs.png)

The playlist name serves as a starting point; once you add a few tracks, the recommendations quickly start to reflect what you're actually in the mood for.

---

## Problem Statement

Three core engineering challenges drive the design of this system:

1. **Context-Aware Preference Modeling.** Standard collaborative filtering often relies on static user profiles that fail to capture shifting, session-based intent. To provide relevant recommendations, the system must treat playlisting as a sequential modeling task, where every prediction is conditioned on the immediate local context and the playlist's stated "theme".

2. **Inductive Generalization (Cold-Start).** Music catalogs are highly dynamic, with millions of tracks added annually. A "transductive" model relying on learned ID embeddings is obsolete the moment a new track is uploaded. This architecture requires an inductive approach, where track representations are computed on-the-fly from audio attributes, allowing the model to recommend items it has never seen during training.

3. **Computational Scaling to Massive Vocabularies.** The dataset contains ~2.2 million unique tracks. In a traditional multi-class classification setup, a standard cross-entropy loss produces output tensors of shape`[B, T, 2,200,000]`. This creates a memory and compute bottleneck that makes training untenable. The system must move from a "dense classification" mindset to a "contrastive ranking" framework using efficient negative sampling.

---

## Solution

### Architecture overview

The model is a **causal transformer decoder** trained on individual playlists as sequences. Each sequence is structured as:

```
[playlist_name_token]  [track_1]  [track_2]  ...  [track_n]
        ↑                                             ↑
  always present,                            predicted at each step
  never predicted
```

Because the playlist name token is always the first position and is never masked out, the model learns to condition every prediction on the playlist's intent — functioning as a persistent soft prior throughout generation.

The transformer uses **Rotary Positional Embeddings (RoPE)** rather than learned positional embeddings. Playlist lengths vary enormously in practice, and RoPE generalises more reliably to sequence lengths not seen during training.

### Track embeddings — content-based, not ID-based

Tracks are embedded from their **audio attributes** rather than a learned per-ID vector. This is the central design decision that solves cold start.

`TrackEmbedder` constructs a track's representation from three sources:

| Component | Features | Notes |
|---|---|---|
| Continuous audio features | danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo | Passed through a small MLP |
| Categorical audio features | mode, key, time signature | Per-feature learned embeddings |
| Artist embedding | artist identity | Learned embedding; unknown artists receive a zero-padding vector |

The three components are projected and fused into a single `d_model`-dimensional vector. When a new track is introduced to the catalogue, it can immediately receive a meaningful embedding from its audio attributes alone — no retraining required.

**Artist dropout** is applied to the artist embedding component during training, preventing the model from over-relying on artist identity as a shortcut. Importantly, this is applied only when embedding tracks as sequence context, and not when computing candidate item vectors in the prediction head. This ensures the model learns to look past artist identity when building context, while still scoring candidates against consistent, unperturbed item vectors.

### Playlist name embeddings

Playlist names are embedded with **[Embedding Gemma (300M)](https://huggingface.co/google/embeddinggemma-300m)**, a pre-trained, frozen text embedding model. A learned linear projection maps from the model's 768-dimensional output into the transformer's `d_model` space.

### Sampled Softmax prediction head

With ~2.2M tracks in the vocabulary, standard cross-entropy is not viable. The `SampledSoftmaxPredictionHead` reformulates next-track prediction as **contrastive learning**: for each position, the model scores the true next track against a small set of sampled negatives using dot products between the transformer's hidden state and `TrackEmbedder` output vectors. This contrastive framing is what makes dynamic catalogues possible: scoring a new track at inference time requires only its audio attributes, not a learned ID vector.

- **Popularity-based negative sampling** — negatives are drawn proportionally to how frequently each track appears in the dataset, with configurable smoothing (`smoothing_factor`) and optional uniform mixing (`uniform_mix_factor`) to prevent over-concentration on head items.
- **Log-Q correction (importance sampling)** — to correct for the bias introduced by non-uniform sampling, each logit is adjusted by subtracting $\log Q(y)$, where $Q(y)$ is the sampling probability of item $y$. This recovers an unbiased estimate of the full softmax:

$$\tilde{z}_i = \frac{z_i}{T} - \log Q(y_i)$$

- The corrected positive and negative logits are concatenated and passed to a standard cross-entropy loss, reducing each training step to an `[B, 1 + n_neg_samples]` operation rather than `[B, 2_200_000]`.
- **False negative masking** — when a sampled negative happens to collide with the true positive for a given example, it is masked out of the loss rather than penalised as a negative. This prevents contradictory gradient signals on items that are genuinely good next-track predictions.
- **Logit scaling** — dot-product logits are scaled by `1/sqrt(d_model)`, keeping the logit magnitude invariant to model width.

This setup creates **two distinct gradient paths into `TrackEmbedder`**:
1. **Through the hidden states** — `TrackEmbedder` embeds the input sequence; those embeddings flow through the full transformer stack whose output is then compared against item vectors. This path teaches the embedder to produce representations that the transformer can meaningfully contextualise.
2. **Through the item vectors** — `TrackEmbedder` also produces the candidate embeddings that the hidden state is scored against. Gradients flow directly into it from the contrastive loss, pushing the audio-feature space to be geometrically consistent with what the transformer learns to predict.

Because both paths share the same `TrackEmbedder` weights, the audio-feature embedding space is jointly shaped by what it means to *be in a context* and what it means to *be a good next item*.

### Inference optimisations

At inference time, the `TrackEmbedder` is replaced with a `FrozenTrackEmbedder`: the full forward pass is run once over the entire vocabulary and the resulting embeddings are cached as a static buffer. Subsequent lookups are then simple index operations (no MLP, no projection, no dropout) making interactive generation fast enough for real-time use in the UI.

---

## Getting Started

### Prerequisites

- Python ≥ 3.11.1
- [`uv`](https://github.com/astral-sh/uv) for environment and package management
- A GPU is strongly recommended for training

### 1. Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all dependencies declared in `pyproject.toml`.

### 2. Configure credentials

Create a `.env` file in the project root:

```env
KAGGLE_API_TOKEN=<your_kaggle_api_token>
HF_TOKEN=<your_huggingface_token>
```

- `KAGGLE_API_TOKEN` — required to download the Spotify Million Playlist Dataset via the Kaggle API.
- `HF_TOKEN` — required to download [Embedding Gemma (300M)](https://huggingface.co/google/embeddinggemma-300m), which is a gated model on Hugging Face.

### 3. Load and prepare data

```bash
data/scripts/load-and-prepare.sh
```

This script downloads the raw playlist and track datasets, converts them to Parquet, and runs the preprocessing pipeline that produces the train/test splits consumed by the model. (Takes approximately 15 minutes.)

### 4. Train

```bash
uv run python train.py
```

Training configuration (model size, learning rate schedule, batch size, checkpoint path, etc.) is set directly in [`train.py`](train.py). Checkpoints are saved to `checkpoints/model.pt` by default.

---

## Interactive UI

A web interface to generate and explore playlists using the trained model. Run it with:

```bash
./run_ui.sh
```

## License

This repository's source code is licensed under the MIT License. The Embedding Gemma model weights used by this project are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and are not covered by the MIT License.
