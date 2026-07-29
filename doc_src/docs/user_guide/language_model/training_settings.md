# Tuning Training Settings

This guide explains the training settings that most affect model quality — learning rate, early stopping, batching — and when data normalization helps or hurts. It applies to both the CNN pipeline (main tutorial) and the transformer pipeline; where they differ, both values are given.

---

## How settings work

Training is controlled entirely by `config.cfg` in your model directory. `LanguageModel` maintains an in-memory copy as `model.config`. Edit values, save, and retrain:

```python
model.config["training"]["max_steps"] = 5000
model.save_config()
model.train()
```

`save_config()` is required — `train()` re-reads `config.cfg` from disk, so unsaved edits are ignored.

---

## Learning rate

The learning rate controls how much each training step changes the model's weights. Too high and training destroys useful information faster than it learns; too low and training crawls.

| Pipeline | Default | Recommended range |
| --- | --- | --- |
| CNN (scratch or fine-tuning) | `0.001` (flat) | 0.0005 – 0.002 |
| Transformer | `0.00005` peak, warmup schedule | 0.00002 – 0.00005 |

Both defaults are spaCy's own recommended values — identical to what `spacy init config` generates — so there is nothing to change for a first run. The CNN rate applies to scratch training and `base_model=` fine-tuning alike (spaCy uses the same default for both), and its range is the practical tuning window around that default. The transformer range is the fine-tuning grid recommended by the BERT authors (Devlin et al. 2019); spaCy's default sits at its top.

**CNN** — a single flat rate at `[training.optimizer] learn_rate`.

**Transformer** — a `warmup_linear.v1` schedule at `[training.optimizer.learn_rate]`: the rate climbs from zero over `warmup_steps` (default 250), then decays linearly to zero at `total_steps`. Keep `total_steps` equal to `max_steps`.

> **Catastrophic forgetting.** When fine-tuning — sourced CNN components or a transformer — a learning rate that is too high overwrites the pre-trained weights faster than the model learns from your data. You are experiencing it if the fine-tuned model scores *below* its own base model, or if dev scores peak in the first few evaluations and then decline. The fix: lower the learning rate and retrain. Drop the CNN rate to `0.0001`–`0.0005`, or the transformer's `initial_rate` to `0.00002`.
>
> spaCy's own guidance on forgetting ([pseudo-rehearsal](https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting)) recommends **rehearsal data** rather than a different learning rate: mix general-domain text annotated by the base model into your training data, so the model keeps seeing what it already knows. Reach for that when the fine-tuned model must stay accurate on ordinary text as well as your target domain; if you only care about the target domain, the lower learning rate alone is usually enough.

---

## Early stopping

spaCy does not stop training at a fixed loss threshold — loss scales vary with corpus size and are not comparable across runs. Instead it evaluates on the **dev set** at regular intervals and stops when the score stops improving.

| Setting | Meaning | Default |
| --- | --- | --- |
| `eval_frequency` | Evaluate on dev every N steps | 200 |
| `patience` | Stop if no dev improvement for N **steps** (not evaluations) | 1600 |
| `max_steps` | Hard cap on training steps | 20000 |
| `max_epochs` | Hard cap on passes through the data (0 = unlimited) | 0 |

With the defaults, training stops after 8 evaluations (1600 / 200) without improvement.

**Small-corpus advice** (a few hundred sentences): lower `eval_frequency` to 100–200 so progress is checked often, and keep `patience` at least 8× `eval_frequency`. Very small dev sets produce noisy scores — don't trust a single evaluation's wiggle.

**How "best" is decided:** each evaluation computes a weighted average of component metrics using `[training.score_weights]`, and `model-best` is the checkpoint with the highest weighted score. `LanguageModel` **auto-computes equal weights** across all active metrics (tag_acc, pos_acc, morph_acc, lemma_acc, dep_uas, dep_las for the full pipeline) when it writes the config — including for recipes. If you want to prioritize one metric (e.g. weight `dep_las` higher because parsing matters most to you), edit `model.config["training"]["score_weights"]` *after* construction and `save_config()`.

---

## Batching and memory

**CNN** — `[training.batcher]` uses `batch_by_words.v1` with a compounding size schedule (100 → 1000 words). This rarely needs tuning; reduce the `stop` value if you run out of RAM.

**Transformer** — `batch_by_padded.v1`, where `size` is the padded-token total per batch. This is the main GPU-memory knob:

| Problem | Fix |
| --- | --- |
| CUDA out of memory | Halve `training.batcher.size` (2000 → 1000 → 500) |
| Smaller batches destabilize training | Raise `training.accumulate_gradient` so size × accumulation stays roughly constant |
| Still OOM at size 500 | Set `components.transformer.model.mixed_precision = true` |

`accumulate_gradient = 3` (the transformer recipe default) means gradients from 3 batches are summed before each weight update — the optimization behaves like a 3× larger batch without the memory cost.

**Related data settings:** `convert_assets(n_sents=10)` groups 10 sentences per training document, and `corpora.train.max_length = 2000` skips documents longer than 2000 tokens. With the defaults these never interact badly; if you raise `n_sents` a lot, check documents stay under `max_length`.

> **`n_sents` and the 100-doc validation minimum.** spaCy's preflight check counts training *docs*, not sentences, and hard-fails below 100 docs when no pipeline component is sourced from an existing model. This never triggers for the CNN tutorial (its components are sourced), but it applies to every transformer recipe — spaCy cannot see that the transformer backbone is pre-trained. With a small corpus, grouping sentences shrinks the doc count below the threshold: 300 sentences at `n_sents=10` is only 30 docs. Use `n_sents=1` for transformer training on small corpora so the doc count equals the sentence count.

---

## Data normalization

Whether normalizing historical spelling (*haue* → *have*, *vnto* → *unto*) helps depends entirely on **which architecture reads what**:

- **CNN (default UD recipe)** — the embedding layer reads the `LOWER`, `PREFIX`, `SUFFIX`, and `SHAPE` attributes of each token. Setting `Token.norm_` has **no effect** on this pipeline. What actually helps the CNN generalize across spelling variants is more training data containing the variants.
- **CNN (multilingual tagger recipe)** — this recipe's embedding uses the `NORM` attribute, so norm tables / normalization *do* reach the model.
- **Transformer** — the transformer receives the **raw document text** via its span getter, tokenized by its own subword tokenizer. `Token.norm_` is never seen by a transformer, so norm-based normalization cannot affect it.
- **MacBERTh specifically** — pre-trained on raw, unnormalized historical text. Its subword vocabulary already covers historical spelling. **Do not modernize text for MacBERTh** — normalization moves your input *away* from its training distribution and typically hurts.

The one case where text-level normalization genuinely helps: running a **modern-language transformer** (e.g. `bert-base-uncased`) on historical text, where wild spelling fragments into meaningless subwords. Even then, normalize a *copy* of the text and keep the original surface forms for your CONLL-U output, and be careful that token boundaries stay aligned.

---

## Smoke testing

Before committing to a multi-hour run, prove the plumbing works with a short one:

```python
model.config["training"]["max_steps"] = 50
model.config["training"]["eval_frequency"] = 25
model.save_config()
model.train()
```

If that completes and writes `training/{lang}/model-best`, restore the real values (or recreate the model with `force=True`) and start the full run. The transformer tutorial has this built in as a `SMOKE_TEST` flag.
