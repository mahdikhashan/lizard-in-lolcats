# Thesis outline

This file is the base outline of a master thesis of approximately 50 pages. It comes from all the work from the first pull request to 2026-10-11:

- the pull requests #1–#4 of `mahdikhashan/jku-thesis` (from 2026-09-24),
- the pull requests #1–#90 of `mahdikhashan/lolcats` (from 2026-09-29),
- every document in [`docs/`](../README.md) and [`docs/experiments/`](../README.md#experiments).

Each chapter has a page budget, a list of sections and the sources for each section. A source is a document of these notes. The outline gives only titles and one short line for each section. The text of the thesis must still be written.

## Title ideas

1. Linearizing Llama-3.2-1B with Lizard: a reproduction and a search for the gap
2. Lizard on a small model: why the first reproduction failed, and how RoPE in the window helps
3. From softmax to gated linear attention: a reproduction study of Lizard on Llama-3.2-1B

## Page budget

| # | Chapter | Pages |
|---|---|---|
| 1 | Introduction | 3 |
| 2 | Background | 7 |
| 3 | Related work | 3 |
| 4 | Lizard and its implementation | 7 |
| 5 | Experimental setup | 5 |
| 6 | Results of the reproduction | 5 |
| 7 | The search for the cause of the gap | 6 |
| 8 | Analysis with XAI methods | 6 |
| 9 | RoPE in the window branch | 5 |
| 10 | Discussion | 2 |
| 11 | Conclusion and future work | 1 |
| | **Total** | **50** |

The front matter and the appendices are not part of the 50 pages.

## Front matter

- Title page
- Abstract (one page, with the main numbers)
- Declaration of the use of AI tools (see appendix G)
- Table of contents, list of figures, list of tables
- List of abbreviations: GLA, SWA, AWA, RoPE, LoRA, MSE, KL, XAI, SE, TV

## Chapter 1: Introduction (3 pages)

| Section | What it says | Sources |
|---|---|---|
| 1.1 Softmax attention is expensive for long texts | The time grows with the square of the length. The memory of the cache grows with the length. | [Math formulas](../math-formula.md), section 6 |
| 1.2 Linearization: a trained model gets linear attention | A trained model keeps its weights. Only the attention changes, and a short training follows. This is cheaper than training from the start. | [Document 15](../15-attention-math-side-by-side.md) |
| 1.3 The Lizard method and its claims | Lizard uses gated linear attention and a small softmax window with sinks. The paper reports results near the teacher. | [Document 9](../09-paper-comparison.md) |
| 1.4 Research questions | RQ1: can the paper results for Llama-3.2-1B be reproduced? RQ2: which differences explain the gap? RQ3: which changes to the attention close the gap? RQ4: do XAI measurements explain the scores? | [Document 17](../17-reproduction-studies.md), section 4 |
| 1.5 Contributions | An open implementation with tests. A reproduction with a record of all runs and costs. XAI tools for distillation. RoPE in the window, which gives 98% of the teacher on PIQA and ARC-Easy. | [Docs index](../README.md) |
| 1.6 Structure of the thesis | One paragraph for each chapter | This outline |

## Chapter 2: Background (7 pages)

| Section | What it says | Sources |
|---|---|---|
| 2.1 The transformer and causal softmax attention | Queries, keys, values, the causal mask and the softmax | [Math formulas](../math-formula.md), section 6 |
| 2.2 Rotary position embedding (RoPE) | How RoPE puts the position into the queries and keys. The score depends only on the distance between two tokens. | [Document 15](../15-attention-math-side-by-side.md), [document 13](../13-lizard-attention-v2.md), P5 |
| 2.3 Linear attention and feature maps | A feature map replaces the softmax. The model can then compute the output with a fixed-size state. | [Math formulas](../math-formula.md), section 2 |
| 2.4 Hedgehog feature maps | A learned map that copies the shape of the softmax weights | [Math formulas](../math-formula.md), section 7, [hyperparameters](../hyperparameters.md) |
| 2.5 Gated linear attention (GLA) | A gate makes old tokens less important. The parallel form and the recurrent form give the same output. | [Math formulas](../math-formula.md), section 2, [document 14](../14-liger-gla.md), section 1 |
| 2.6 Sliding window attention and attention sinks | Softmax over the last tokens only, with extra logits that take part of the weight | [Math formulas](../math-formula.md), section 1 |
| 2.7 Knowledge distillation and attention transfer | A student copies a teacher. Here, each attention layer copies the output of the teacher layer. | [Math formulas](../math-formula.md), section 4 |
| 2.8 Low-rank adaptation (LoRA) | Small extra matrices train, and the base weights stay frozen | [Figure 2](../figures/README.md) |
| 2.9 How a harness scores multiple-choice tasks | The model does not write an answer. The harness compares the log-likelihood of each option. | [Temperature](../experiments/temperature.md), [document 20](../20-layer-mse-for-piqa-arc.md), section 2 |
| 2.10 The benchmarks | MMLU, PIQA, ARC-Easy, ARC-Challenge, HellaSwag and WinoGrande: what each one tests | [Document 20](../20-layer-mse-for-piqa-arc.md), [document 21](../21-mmlu-options.md), [references](../references.md) |
| 2.11 Basic terms | Fine-tuning, packing, the EOS token, padding, context size and batch | [Definitions](../appendix-definitions.md) |

## Chapter 3: Related work (3 pages)

| Section | What it says | Sources |
|---|---|---|
| 3.1 LoLCATs, Liger and Lizard | Three linearization methods. What each one keeps from softmax attention, and what each one changes. | [Document 14](../14-liger-gla.md), [document 15](../15-attention-math-side-by-side.md) |
| 3.2 Small models stay near chance on MMLU | At 1B, other methods also get MMLU near 25% | [Document 19](../19-next-steps-from-literature.md), section 1 |
| 3.3 Hybrid models with some softmax layers or heads | Models that keep a few softmax layers or heads, and how they select them | [Document 21](../21-mmlu-options.md), section 4 |
| 3.4 Reproduction studies in machine learning | Why reproductions fail, and how other studies report it | [Document 17](../17-reproduction-studies.md) |
| 3.5 Interpretability of distilled attention | Layer errors, attention plots and probes in earlier work | [XAI ideas](../experiments/xai.md), [document 22](../22-mmlu-xai-methods.md) |

## Chapter 4: Lizard and its implementation (7 pages)

| Section | What it says | Sources |
|---|---|---|
| 4.1 Lizard attention: the gated branch and the window branch | The output is the gated branch plus α times the window branch. The window has 128 keys and 4 sinks. | [Math formulas](../math-formula.md), section 3, [docs index](../README.md#glossary) |
| 4.2 The two training stages | Stage 1: each layer copies the teacher attention (MSE). Stage 2: LoRA with the next-token loss. | [Figures 1 and 2](../figures/README.md) |
| 4.3 The training algorithm in log space | How Lizard computes the gate products in a stable way | [Math formulas](../math-formula.md), section 5 |
| 4.4 The reference implementation | A dense PyTorch version, slow loop versions as the ground truth, and tests in float64 | `mahdikhashan/jku-thesis`, PR #2 |
| 4.5 The port to LoLCATs | How the Lizard layer went into the LoLCATs code, and what is different from the reference pipeline | [Document 1](../01-lizard-in-lolcats.md) |
| 4.6 Tests of the attention layer | Level 1: the formulas. Level 2: the connection into the model. What the tests show, and what they do not show. | [Document 8](../08-verification.md) |
| 4.7 The paper against the code | 17 differences between the formulas of the paper and the code. Open points in the paper. | [Math against code](../math-code-discrepancy.md) |
| 4.8 Lizard attention v2: one option for each reading | Options for each open point (D1–D3, α) and each change C1–C6. The default options give v1. | [Document 13](../13-lizard-attention-v2.md) |
| 4.9 Proofs and float64 tests | The proof audits P1–P11 and the claims about the four methods | [Document 13](../13-lizard-attention-v2.md), [document 14](../14-liger-gla.md), section 4, [document 15](../15-attention-math-side-by-side.md) |
| 4.10 Numerical stability: the NaN crash | Stage 2 stopped with NaN. The causes were an overflow in `exp` and 0 / 0. How the fix works. | [Document 5](../05-nan-crash.md) |

## Chapter 5: Experimental setup (5 pages)

| Section | What it says | Sources |
|---|---|---|
| 5.1 The teacher: Llama-3.2-1B | The size of the model and of its attention | [Hyperparameters](../hyperparameters.md) |
| 5.2 The data | Alpaca-cleaned, packed into sequences of 2048 tokens | [Document 2](../02-compute-and-cost.md), [document 4](../04-training-runs.md) |
| 5.3 The training recipe | The LoLCATs recipe of the first runs, and the recipe of the paper | [Document 4](../04-training-runs.md), [document 9](../09-paper-comparison.md), [document 11](../11-gap-analysis.md), section 12 |
| 5.4 Infrastructure | Docker image, Hugging Face Jobs, W&B and the checkpoint names | [Document 3](../03-infrastructure.md) |
| 5.5 Compute and cost | Memory, GPU choice, time and cost of each run | [Document 2](../02-compute-and-cost.md) |
| 5.6 The evaluation | The harness version, the tasks, the metrics, and the problems found during the setup | [Document 6](../06-evaluation-setup.md) |
| 5.7 Statistics | Standard error, z values, paired tests and McNemar's test | [Document 7](../07-results.md), [`window_rope`](../experiments/window-rope.md), MMLU section |
| 5.8 The teacher in this harness | PIQA 74.4, ARC-Easy 65.3 and MMLU 31.7 | [Teacher X0](../teacher-x0/summary.md), [document 7](../07-results.md) |
| 5.9 The analysis scripts | `letters.py`, `ablate.py`, `compare_stages.sh` and the XAI scripts | [Scripts](../../scripts/README.md) |

## Chapter 6: Results of the reproduction (5 pages)

| Section | What it says | Sources |
|---|---|---|
| 6.1 Runs 0, 1 and 2 | A smoke test, a full run that stopped in stage 2, and a second stage 2 | [Document 4](../04-training-runs.md) |
| 6.2 The scores against the paper | MMLU 23.3, PIQA 68.0 and ARC-Easy 54.8. The paper gives 29.8, 74.8 and 65.6. | [Document 7](../07-results.md), [document 9](../09-paper-comparison.md) |
| 6.3 The model selects "A" on MMLU | The model selects "A" for almost every question. The score of each subject follows the share of "A" answers. | [Document 7](../07-results.md), section 3, [document 11](../11-gap-analysis.md), section 13 |
| 6.4 The model uses both branches | Without the window branch, PIQA falls to 57.2. Without the gated branch, it falls to 52.3. | [Document 7](../07-results.md), section 5 |
| 6.5 The gap starts in stage 1 | PIQA is 57.6 after stage 1 and 68.0 after stage 2 | [Stage difference](../experiments/stage-difference.md) |
| 6.6 The temperature does not change the scores | The accuracy is the same at temperatures 0.1–2 | [Temperature](../experiments/temperature.md) |

## Chapter 7: The search for the cause of the gap (6 pages)

| Section | What it says | Sources |
|---|---|---|
| 7.1 Possible causes of the gap | A ranked list: recipe, data, loss scale, harness, checkpoint loading, numerical effects | [Document 11](../11-gap-analysis.md) |
| 7.2 Feature dimension 32 | A smaller feature map does not help | [Feature dimension](../experiments/feature-dimension.md) |
| 7.3 The learning rate and schedule of the paper | The gate does not saturate, but α does not move in bf16 | [Paper LR](../experiments/paper-lr.md) |
| 7.4 float32 training | α moves, and the loss falls. The accuracy does not change much. | [float32](../experiments/float32.md) |
| 7.5 β2, the minimum learning rate and gradient clipping | A lower loss, but no clear gain in accuracy | [Second round](../experiments/second-round.md), [gradient clipping](../experiments/gradient-clipping.md) |
| 7.6 Stage 2 with the recipe of the paper | A lower validation loss, but the same PIQA and ARC-Easy as Run 2 | [Stage 2 on config 1](../experiments/stage2-config1.md) |
| 7.7 A lower loss does not give a higher accuracy | Over the runs, the stage 1 loss fell from 8.16 to 3.05, but the accuracy did not increase | [Document 14](../14-liger-gla.md), section 5, [document 18](../18-gap-timeline.md), section 3 |
| 7.8 A limit for each head | All heads of a layer share the Lizard parameters. The window share has an upper limit. | [Document 12](../12-gap-analysis-2.md), section 3 |
| 7.9 Design options of v2: C1, `joint` and `hybrid` | One α for each head, and two shared denominators. C1 and R1b give a lower loss, but no gain in accuracy. R1 (`joint`) stopped. | [Document 13](../13-lizard-attention-v2.md) |
| 7.10 The LoLCATs attention as a control | The same pipeline with the LoLCATs attention gets PIQA 73.5 after stage 1. Thus the cause is in the Lizard attention. | [LoLCATs control](../experiments/lolcats-control.md) |
| 7.11 Timeline of the search | One table with each try, its result and its challenges | [Document 18](../18-gap-timeline.md) |

## Chapter 8: Analysis with XAI methods (6 pages)

| Section | What it says | Sources |
|---|---|---|
| 8.1 Layer-wise MSE | Which layers differ most from the teacher. Layer 15 and two of its heads give a large part of the loss. | [XAI: layer-wise MSE](../experiments/xai-layer-wise-mse.md) |
| 8.2 Sample attention weights | Lizard matches the local heads of the teacher worst | [XAI: sample attention weights](../experiments/xai-sample-attention-weight.md) |
| 8.3 MSE by token position | The error at the first 128 positions. These positions are important for the short prompts of PIQA and ARC-Easy. | [XAI: MSE by token position](../experiments/xai-position-mse.md) |
| 8.4 Branch decomposition | Which branch limits the fit. Inside the window, the window without RoPE explains only approximately 60% of the teacher output. | [XAI: branch decomposition](../experiments/xai-branch-fit.md) |
| 8.5 Why short prompts need a good window | 95% of the PIQA and ARC-Easy prompts are shorter than approximately 90 tokens. Thus the window does almost all the work. | [Document 20](../20-layer-mse-for-piqa-arc.md) |
| 8.6 What the XAI results show together | The window without RoPE is the most probable cause. The gated branch is the second, and the Hedgehog maps are the third. | [Document 20](../20-layer-mse-for-piqa-arc.md), section 5 |

## Chapter 9: RoPE in the window branch (5 pages)

| Section | What it says | Sources |
|---|---|---|
| 9.1 Why RoPE in the window, and the predictions | The XAI results point to the window. Six predictions before the run. | [`window_rope`](../experiments/window-rope.md), "Why this change" and "Predictions" |
| 9.2 What RoPE in the window means for the linearization | RoPE is only in the window, not in the gated branch. Inside the window, the scores are the teacher scores. | [`window_rope`](../experiments/window-rope.md), "RoPE in the window and the linearization" (PR #90) |
| 9.3 Results after stage 1 | The validation loss falls from 3.25 to 1.23. All six predictions hold. | [`window_rope`](../experiments/window-rope.md), Run 1 |
| 9.4 Results after stage 2 | PIQA 73.1 and ARC-Easy 63.8: 98.2% and 97.7% of the teacher | [`window_rope`](../experiments/window-rope.md), Run 2, [document 7](../07-results.md), section 6 |
| 9.5 The shared denominator does not help | With `hybrid`, the gate closes in 9 of 16 layers, and these layers lose the long range | [`window_rope` with `hybrid`](../experiments/window-rope-hybrid.md) |
| 9.6 Table 9 of the paper | ARC-Challenge, HellaSwag and WinoGrande against the paper | Not in the docs yet (see "Missing material") |
| 9.7 MMLU on all questions | 27.2 against 31.7 for the teacher. Above chance, but the student keeps only a third of the part of the teacher score above chance. | [`window_rope`](../experiments/window-rope.md), MMLU section, [MMLU summary](../experiments/window-rope/mmlu_full_summary.md) |
| 9.8 Letters and ties on MMLU | The teacher and the student have almost the same letter statistics. bf16 logits give ties in 20.5–24.2% of the answers. | [Document 22](../22-mmlu-xai-methods.md), [`window_rope`](../experiments/window-rope.md), findings 20 and 21 |

## Chapter 10: Discussion (2 pages)

| Section | What it says | Sources |
|---|---|---|
| 10.1 Answers to the research questions | One short answer for each of RQ1–RQ4 | Chapters 6–9 |
| 10.2 Why PIQA and ARC-Easy recover, and MMLU does not | Short prompts against long 5-shot prompts. Knowledge against local patterns. | [Document 20](../20-layer-mse-for-piqa-arc.md), [document 21](../21-mmlu-options.md), section 2 |
| 10.3 Differences from the paper that stay | The recipe, the data, the harness, and the open points of the paper | [Document 9](../09-paper-comparison.md), [math against code](../math-code-discrepancy.md) |
| 10.4 Limits of this study | One seed for each run. Some results use only 285 MMLU questions. bf16 ties. Some sources are ChatGPT answers that nobody checked. | [Document 7](../07-results.md), [document 17](../17-reproduction-studies.md), section 8 |
| 10.5 Lessons for reproduction work | Test the code first. Change one thing at a time. Measure the accuracy, not only the loss. | [Document 18](../18-gap-timeline.md), section 3 |

## Chapter 11: Conclusion and future work (1 page)

| Section | What it says | Sources |
|---|---|---|
| 11.1 Summary | The main results in one paragraph | Chapters 6–9 |
| 11.2 Future work | Softmax layers and a KL loss in stage 2. RoPE in the gated branch and a gate bias at the start. An Optuna study and the MMLU XAI scripts. | [Document 21](../21-mmlu-options.md), [`window_rope`](../experiments/window-rope.md), [document 16](../16-optuna-plan.md), [document 22](../22-mmlu-xai-methods.md), [fast feedback loop](../experiments/fast-feedback-loop.md) |

## Appendices

| Appendix | Contents | Sources |
|---|---|---|
| A. Definitions | Short definitions of the basic terms | [Definitions](../appendix-definitions.md) |
| B. Formulas and proofs | All formulas with their sources, and the proof audits | [Math formulas](../math-formula.md), [document 13](../13-lizard-attention-v2.md), [document 14](../14-liger-gla.md), [document 15](../15-attention-math-side-by-side.md) |
| C. Hyperparameters and configs | The settings of each run | [Hyperparameters](../hyperparameters.md), [document 11](../11-gap-analysis.md), section 12 |
| D. All runs and scores | One table with every run, its loss and its scores | [Docs index](../README.md#experiments), [document 18](../18-gap-timeline.md) |
| E. MMLU for each subject | The score of each subject for the teacher and the models | [Document 7](../07-results.md), appendix, [MMLU summary](../experiments/window-rope/mmlu_full_summary.md) |
| F. How to reproduce the results | The commands for training, evaluation and XAI | [Document 10](../10-open-issues-and-next-steps.md), [scripts](../../scripts/README.md) |
| G. Use of AI tools | Claude Code wrote the pull requests with the author. Some documents keep ChatGPT answers without changes. | [Docs index](../README.md), [evaluation](../evaluation.md), [document 17](../17-reproduction-studies.md), [document 22](../22-mmlu-xai-methods.md) |

## Figures and tables

### Figures that exist

| Chapter | Figure | File |
|---|---|---|
| 4 | Stage 1 and stage 2 of Lizard | [lizard-stage1.webp](../figures/lizard-stage1.webp), [lizard-stage2.webp](../figures/lizard-stage2.webp) |
| 7, 8 | Layer-wise MSE of the stage 1 checkpoints | [stage1_all_lizard.png](../experiments/xai-layer-wise-mse/stage1_all_lizard.png) |
| 7, 8 | Lizard against the LoLCATs attention, for each layer | [lolcats_vs_all_lizard.png](../experiments/xai-layer-wise-mse/lolcats_vs_all_lizard.png) |
| 8 | Attention weights of the teacher and of Lizard | [trained_heads0-3_head0.png](../experiments/xai-sample-attention-weight/trained_heads0-3_head0.png) |
| 8 | MSE by layer and position | [heatmap_all.png](../experiments/xai-position-mse/heatmap_all.png) |
| 8 | Branch decomposition | [branch_fit_all.png](../experiments/xai-branch-fit/branch_fit_all.png) |
| 9 | `window_rope` against Run 1, for each layer | [layer_mse_vs_run1.png](../experiments/window-rope/layer_mse_vs_run1.png) |
| 9 | `hybrid` against `window_rope`, for each layer | [layer_mse_vs_window_rope.png](../experiments/window-rope-hybrid/layer_mse_vs_window_rope.png) |

### Figures to make

| Chapter | Figure | Data |
|---|---|---|
| 4 | Lizard attention: the gated branch, the window and the sinks in one diagram | [Math formulas](../math-formula.md), `mahdikhashan/jku-thesis` PR #3 (open) |
| 6 | Share of each answer letter on MMLU: teacher, first model, `window_rope` | [Document 7](../07-results.md), [`window_rope`](../experiments/window-rope.md) |
| 7 | Stage 1 loss against PIQA for all runs | [Document 14](../14-liger-gla.md), section 5, [docs index](../README.md#experiments) |
| 9 | Scores of all models as a percentage of the teacher | [Document 7](../07-results.md) |

### Tables

| Chapter | Table | Source |
|---|---|---|
| 5 | The recipe of the runs against the recipe of the paper | [Document 9](../09-paper-comparison.md) |
| 6, 9 | Scores of the first model, `window_rope`, the teacher and the paper | [Document 7](../07-results.md) |
| 7 | All stage 1 runs: loss, PIQA, ARC-Easy, MMLU subset | [Document 18](../18-gap-timeline.md) |
| 9 | MMLU for each subject group | [`window_rope`](../experiments/window-rope.md), finding 18 |

## Missing material

These parts of the outline need data or text that the docs do not have yet.

| Part | What is missing | Status |
|---|---|---|
| 9.6 | HellaSwag and WinoGrande for `window_rope` after stage 2 | Running on `student06` |
| 9.6 | ARC-Challenge for `window_rope` after stage 2 | Measured on 2026-10-11, not in the docs yet |
| 9.6 | The teacher on ARC-Challenge, HellaSwag and WinoGrande in this harness | Not run |
| 1.1, 10.3 | Speed and memory of the trained model for long inputs | Not measured. The training speeds are in [document 2](../02-compute-and-cost.md). |
| 10.4 | A second seed for the best model | Not run |
| 2.10, 9.7 | The MMLU checks without training: 0-shot, rotated options, BOS | Planned in [document 21](../21-mmlu-options.md), section 3 |
