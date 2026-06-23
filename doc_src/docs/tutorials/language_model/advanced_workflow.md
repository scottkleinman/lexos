# Building a Specialized Model: Iterative Workflow

This guide covers how to build a highly accurate, domain-specific language model when no ready-made training data exists for your target — and how to improve a model you have already trained but are not satisfied with.

---

## Who this is for

This guide is for two groups:

1. **Starting from nothing** — you need a model for a domain, language variety, or historical period where no UD-annotated treebank exists and no suitable base model is available off the shelf.
2. **Improving an existing model** — you have already trained a model using the main tutorial but accuracy is not where you need it, and you want to improve it.

In both cases the solution is the same: **you get better models by getting more and better annotated training data.** The iterative workflow is the fastest way to build that data — quicker than annotating everything by hand before any training, which can take an extremely long time.

---

## The strategy

In order to get the best possible language model for your given task, you want a model that been trained on the most amount of the most relvant data. Choosing the best avaliable base model can give you a good starting point to then fine-tune with your specifc data, but if nothing related is availble starting from scratch is an option. From there to get a better language model you'll need to get data, and if you need more than is already availble, you'll need to make it. Annotating data can take a long time, this workflow guides you through using your language model to partially annotate your data, to then be corrected and fed back into the language model, further fine-tuning. You can repeat until you statified with the models accuracy.

Instead of annotating a large corpus before training anything, you start small, train a first model, use it to pre-annotate new text, correct the pre-annotations, and fine-tune. Each round, the model gets better — and better model predictions mean faster corrections.

```
[Optional: related UD data] ──► Train base model  ──► evaluate on held-out test (FINAL ONLY)
                                        │
                         ┌──────────────▼──────────────────┐
                         │  Auto-annotate new texts         │
                         │  Correct predictions by hand     │  ◄── repeat
                         │  Fine-tune model (main tutorial) │
                         └──────────────────────────────────┘
                                    ▲
                              track progress
                              on dev set only
```

The end product of this process is both a growing corpus of high-quality annotated data **and** an increasingly accurate fine-tuned model. The model is the main deliverable — and each iteration makes it better.

---

## Phase 1 — Get your first round of training data

You need some annotated sentences before you can train anything. Two approaches:

### Option A — Start with a related UD treebank (optional accelerator)

If a UD treebank exists for a related dialect, variety, or domain, it can give your first model a head start. The closer the match, the more useful it is.

**How related is related enough?** Language variants transfer best (Early Modern English → Modern English). Domain shift is next best (legal text → general English). Cross-family data (unrelated language) may hurt more than it helps.

See [Getting Training Data](training_data.md) for where to find and download treebanks.

If no related data exists — or you would rather not use it — go directly to Option B. A related treebank is a convenience, not a requirement.

### Option B — Annotate your first batch by hand

Annotate 200–500 sentences in your target domain from scratch. See [Getting Training Data](training_data.md) for annotation tools, UD guidelines, and format details.

For tool recommendations: [INCEpTION](https://inception-project.github.io/) and [Arborator Grew](https://arboratorgrew.elizia.net/) are free and support UD natively. [Prodigy](https://prodi.gy/) (commercial, from Explosion — creators of spaCy) is particularly fast once you have a model to pre-annotate with.

**Quality matters more than quantity.** Annotation quality is the ceiling on your model quality — read the [Annotation quality](#annotation-quality) section before you start.

---

## Phase 2 — Train your first base model

**For this step, follow the [main tutorial](tutorial.ipynb).**

Pass the following to the tutorial:

- **`data`** — your Phase 1 training data (related treebank, your hand annotations, or both combined)
- **`base_model`** — the best UD-trained model available for your language. If nothing suitable exists yet, omit `base_model` entirely (or omit only the components with no suitable source) — any component not specified is trained from scratch.

After training, you will have a `model-best` directory. This is your starting point, your `base_model` for bootstrapping.

---

## Phase 3 — Bootstrap: auto-annotate and correct

This is the core of the iterative workflow. Use your trained model to pre-annotate new text, then correct the output by hand. Correcting pre-annotations is significantly faster than annotating from scratch — and it gets faster each round as the model improves.

### Step 1 — Export model annotations to CONLL-U

Use the helper function below to run your model on new texts and write the predictions to a CONLL-U file:

```python
import spacy
from pathlib import Path

def export_to_conllu(model_path, texts, output_path):
    """Run model on texts and write predictions to CONLL-U format."""
    nlp = spacy.load(model_path)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts, 1):
            doc = nlp(text)
            f.write(f"# sent_id = auto-{i}\n")
            f.write(f"# text = {doc.text}\n")
            for j, token in enumerate(doc, 1):
                head = token.head.i - doc[0].i + 1 if token.head != token else 0
                feats = str(token.morph) if token.morph else "_"
                lemma = token.lemma_ if token.lemma_ else "_"
                f.write(
                    f"{j}\t{token.text}\t{lemma}\t{token.pos_}\t"
                    f"{token.tag_}\t{feats}\t{head}\t{token.dep_}\t_\t_\n"
                )
            f.write("\n")
```

Prepare your texts as a Python list of strings, one sentence per element:

```python
texts = [
    "The court shall determine the matter within thirty days.",
    "No person shall be deprived of liberty without due process.",
    # ... add as many sentences as your round batch size
]

export_to_conllu(
    model_path="path/to/model-best",
    texts=texts,
    output_path="round1_auto.conllu",
)
```

If your source is a plain text file with one sentence per line:

```python
texts = Path("new_texts.txt").read_text(encoding="utf-8").splitlines()
texts = [t.strip() for t in texts if t.strip()]
```

### Step 2 — Correct the output

Open the exported CONLL-U file in your annotation tool and review each sentence. The model's predictions are pre-filled — you are correcting, not annotating from scratch.

**In INCEpTION:**

1. Create a new project and select "Universal Dependencies" as the annotation layer
2. Import the CONLL-U file via *Documents → Import*
3. Review each sentence. The dependency tree is shown graphically — drag arcs to change the head, click labels to change the relation
4. Export via *Documents → Export → CoNLL-U*

**In Arborator Grew:**

1. Create a new project and import the CONLL-U file
2. The dependency tree is displayed as an interactive graph — click tokens and arcs to make corrections
3. Export to CONLL-U when done

**In Prodigy** (if you have a licence):

Prodigy's `dep.correct` recipe wraps the entire loop — it loads your model, shows one sentence at a time with the predicted tree, and lets you accept or correct before moving to the next. This is the fastest path for large batches.

**What to look for when correcting:**

- **HEAD** (column 7) — is the dependency arc pointing to the right governor?
- **DEPREL** (column 8) — is the relation label correct? (e.g. `nsubj`, `obj`, `case`, `det`)
- **UPOS** (column 4) — is the universal POS tag right? (`NOUN`, `VERB`, `ADP`, etc.)
- **FEATS** (column 6) — are morphological features correct? (`Number=Sing`, `Tense=Past`, etc.)

Refer to the UD guidelines for your language at [https://universaldependencies.org/guidelines.html](https://universaldependencies.org/guidelines.html) when you are unsure.

### Annotation quality

> **Annotation quality is the ceiling on your model quality.**
> Every error in your training data can become a systematic model error — and errors compound across rounds. If the model learned a wrong annotation in Round 1, its Round 2 predictions will carry that error into new sentences.

- **Better to annotate 100 sentences carefully than 300 carelessly.** Speed is a trap — a fast incorrect annotation is worse than no annotation.
- **Be consistent.** The same construction should be annotated the same way every time. Inconsistency is one of the hardest errors for a model to learn from.
- **When in doubt, look it up.** Consult the UD guidelines rather than guessing.
- **Review your own corrections.** If you repeatedly correct the same construction the same way and the model still gets it wrong, that construction may need extra examples in the next round.

### How many sentences per round

The right batch size balances two costs: too small means frequent retraining overhead; too large means correcting many inaccurate predictions before the model improves. The optimal size grows as the model improves.

| Round | Batch size | Notes |
| --- | --- | --- |
| 1 | 200–500 sentences | Predictions are rough; corrections are slow; start small |
| 2 | ~double Round 1 | Model improves; corrections get faster |
| 3+ | ~double each round | Scale with how fast corrections are going; stop doubling when time budget is reached |

**Stop iterating when:** your held-out test metrics have not improved across two consecutive fine-tune rounds.

> **Critical discipline: never use the held-out test set during iteration.** The test set is your only honest signal for when to stop. Evaluating against it during iteration leaks information and will cause you to overestimate your model's accuracy. Use the dev set to track training progress round-to-round. Reserve the test set for your final evaluation, after you have decided to stop.

---

## Phase 4 — Fine-tune

**For this step, follow the [main tutorial](tutorial.ipynb).**

Pass the following to the tutorial:

- **`data`** — all your corrected CONLL-U (All the data that matches what your fine-tuning the LM for)
- **`base_model`** — all five components sourced from the previous round's `model-best`:

```python
base_model={
    "tok2vec":              "path/to/previous-round/model-best",
    "tagger":               "path/to/previous-round/model-best",
    "morphologizer":        "path/to/previous-round/model-best",
    "trainable_lemmatizer": "path/to/previous-round/model-best",
    "parser":               "path/to/previous-round/model-best",
}
```

After training, you have a new `model-best`. This is your starting point for the next bootstrap round.

---

## Phase 5 — The iterative loop

Before fine-tuning each round, combine all corrected data from previous rounds into a single file. Training on the full accumulated corpus (not just the latest batch) produces more stable models.

```python
from pathlib import Path

def combine_conllu(round_files, output_path):
    """Concatenate multiple CONLL-U files into one training file."""
    with open(output_path, "w", encoding="utf-8") as out:
        for filepath in round_files:
            text = Path(filepath).read_text(encoding="utf-8")
            out.write(text)
            if not text.endswith("\n\n"):
                out.write("\n")

# Example: after Round 3, combine all three rounds
combine_conllu(
    round_files=["round1_corrected.conllu", "round2_corrected.conllu", "round3_corrected.conllu"],
    output_path="all_rounds_combined.conllu",
)
```


**The next auto-annotation round** uses the newly fine-tuned `model-best` — update the `model_path` argument in `export_to_conllu()` accordingly.

---

## When to stop

Track accuracy on your **dev set** after each fine-tune round. spaCy reports LAS (Labelled Attachment Score) at the end of training — this is the primary metric for dependency parsing quality.

| LAS on dev | Interpretation |
| --- | --- |
| < 60% | Weak; more data or better annotation quality needed |
| 60–75% | Reasonable for specialised text; continue if time allows |
| 75–85% | Good; typical for well-resourced domains |
| 85%+ | Strong; marginal gains per annotation hour are small |

**Stop when** any of these apply:

- Dev LAS has not improved across two consecutive fine-tune rounds
- Corrections per hour are no longer increasing (model predictions are already accurate enough that manual review is fast)
- You have reached your annotation time budget

After you have decided to stop, run the main tutorial's evaluation step against your **held-out test set** to get your final honest accuracy number.
