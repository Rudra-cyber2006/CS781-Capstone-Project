# CS781 Capstone Project: In-House Legal Intelligence System

Rudra Manish Jain (240889), IIT Kanpur. Codabench username: `r_240889`.

This repository holds the Kaggle notebooks behind my CS781 capstone submissions, organised by phase. Five small legal language models were pre-trained from scratch in Phase 1 on the IL-TUR corpus (dataset `Exploration-Lab/CS-781-Capstone`), and every later phase adapts those checkpoints to a downstream legal task. No pretrained language models, no LLM APIs and no outside data were used. All training ran on free Kaggle T4 GPUs (15 GB) in six-hour sessions, with fp16 mixed precision.

The facts below come from the final report, the notebooks themselves (settings cells and printed outputs) and the Codabench result files. Where a value was never saved, it says **not recorded**. A notebook's settings cell is the source for its hyperparameters; where the report states something different, both are given.

## Contents

| Folder | Phase | Task | Notebooks |
| --- | --- | --- | --- |
| [`phase1/`](phase1) | 1 | Legal LM pre-training | c-1, c-2, c-3, c-5, c-6 |
| [`phase2/`](phase2) | 2 | Legal NER (L-NER) | c-1, c-3, c-5, c-6 |
| [`phase3/`](phase3) | 3 | Rhetorical role prediction (RR) | c-1, c-3, c-5, c-6 |
| [`phase4/`](phase4) | 4 | Court judgment prediction (CJPE) | c-1 (two versions), c-3, c-6 |
| [`phase5/`](phase5) | 5 | Legal statute identification (LSI) | c-1, c-3 LoRA, c-6 LORMA, c-6 KD |
| [`phase6/`](phase6) | 6 | Abstractive summarisation | c-1, c-3 (two), c-6 LORMA, c-6 KD |
| [`phase7/`](phase7) | 7 | Prior case retrieval (PCR) | c-1, c-3 (two), c-6 (two) |

Each notebook is the copy I ran on Kaggle, with its saved outputs. To re-run one, attach the matching Phase-1 checkpoint and tokenizer as a Kaggle input, add a Kaggle secret named `HF` with a Hugging Face token (used to download the dataset), and run the cells top to bottom.

## Final results

Best valid Codabench score in each phase, with leaderboard rank on 9 November 2026 (33 valid submissions in total).

| Phase | Task | Metric | Best score | Rank | Valid submissions | Best run |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | LM pre-training | BERTScore F1 | 0.8120 | 130 | 5 | c-3 |
| 2 | L-NER | strict Macro-F1 | 0.1626 | 130 | 6 | not attributed to a run |
| 3 | RR | Macro-F1 | 0.3053 | 131 | 5 | not attributed to a run |
| 4 | CJPE | Macro-F1 | 0.5331 | 128 | 4 | not attributed to a run |
| 5 | LSI | Macro-F1 | 0.2818 | 125 | 3 | c-6 KD |
| 6 | Summarisation | BERTScore F1 | 0.8147 | 122 | 5 | c-6 KD |
| 7 | PCR | Micro-F1@7 | 0.1943 | 125 | 5 | c-6 LORMA |

## Phase 1 models

All five models use Transformer building blocks with 8 attention heads. They were trained only on the Supreme Court file of the corpus (200M training tokens, a separate 5M-token validation sample), with byte-pair-encoding tokenizers trained on 500k characters of that text.

| Model | Family | Architecture | Params | Vocab / context | Val PPL | Codabench BERTScore F1 | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c-1 | Decoder-only | 8 layers, d=1024, FFN 4096, dropout 0.1 | 110.98M | 10k / 2048 | 1673.9 | 0.7528 | [`phase1-c1-decoder.ipynb`](phase1/phase1-c1-decoder.ipynb) |
| c-5 | Decoder, grouped-query attention | 8 layers, d=1024, 8 query heads sharing 2 KV heads | 98.41M | 10k / 2048 | 1980.8 | not recorded | [`phase1-c5-gqa-decoder.ipynb`](phase1/phase1-c5-gqa-decoder.ipynb) |
| c-2 | Hourglass decoder | 3+8+2 layers, shorten x4, d=560, FFN 2000, dropout 0.15 | 51.66M | 11,022 / 2048 | 957.6 | not recorded | [`phase1-c2-hourglass-decoder.ipynb`](phase1/phase1-c2-hourglass-decoder.ipynb) |
| c-6 | Encoder-decoder | plain encoder-decoder, d=560, FFN 1000, dropout 0.15 | 55.13M | 11,022 / 1536 | 126.5 | 0.8099 | [`phase1-c6-enc-dec.ipynb`](phase1/phase1-c6-enc-dec.ipynb) |
| c-3 | Hourglass encoder-decoder | hierarchical encoder-decoder, shorten x4, d=560, FFN 1000, dropout 0.15 | 55.13M | about 11k / 1536 | 124.2 | **0.8120** | [`phase1-c3-hourglass-enc-dec.ipynb`](phase1/phase1-c3-hourglass-enc-dec.ipynb) |

Perplexities are not comparable across rows, because the models use different tokenizers and objectives. c-3 and c-6 read a 1,536-token encoder input and predict a 512-token decoder target. The hourglass models (c-2, c-3) shorten the sequence by a factor of 4 in their middle layers, following Nawrot et al. (2022).

Training: AdamW, peak learning rate 3e-4, weight decay 0.1, 500 warm-up steps, micro-batches of 4 with 32 accumulation steps, 762 scheduled steps. Gradient clipping was 1.0 for c-1 and c-5 and 5.0 for c-2, c-3 and c-6. c-1 and c-5 hit Kaggle's six-hour limit (c-1 at step 639, c-5 at step 633), so their learning-rate schedules never finished. c-2 finished in 3.40 h, c-3 in 2.16 h and c-6 in 3.26 h. The BPE tokenizer was set for 50k merges but stopped at 11,022 tokens for c-2, c-3 and c-6. Submissions were decoded with temperature 0.1 and top-p 0.95, up to 384 new tokens, cut to 128 words. The best submission (c-3) also scored ROUGE-L F1 0.1423 and BLEU-4 0.0131.

The c-2 notebook here has no saved outputs.

## How Phases 2 to 7 adapt the Phase 1 models

Every downstream model loads a Phase-1 checkpoint with the original model classes and tokenizer, then adds a task head. All adaptation methods are written directly in PyTorch, without a PEFT library.

- **LORMA** (Bihany et al., 2025): a multiplicative low-rank update W' = (I + BA) W, with B initialised to zero so the adapted model equals the Phase-1 model at step 0. This was checked numerically (maximum difference 9.5e-7 in Phase 7).
- **LoRA** (Hu et al., 2021): the additive form W' = W + (alpha / r) BA. The Phase-2 c-5 run and the Phase-3 c-6 run were labelled "LORMA" in their notebooks, but they use an additive layer, so they are LoRA. From Phase 3 onwards (c-1) and in Phases 4 to 7, LORMA is the multiplicative form.
- **QLoRA** (Dettmers et al., 2023): frozen base linears stored in 4-bit NF4 with double quantisation and fp16 compute, with LoRA adapters trained on top.
- **AdaLoRA** (Zhang et al., 2023): the update is kept in singular-value form P Lambda Q, low-importance singular values are pruned towards a target rank during training, and an orthogonality penalty is applied to P and Q.
- **BitFit** (Ben Zaken et al., 2022): only bias terms train. c-1 and c-3 have no attention biases, so only feed-forward and LayerNorm biases train (58,368 values on c-1).
- **Full / partial fine-tuning**: all weights train, or everything below a chosen stage is frozen.
- **Knowledge distillation (KD)** (Hinton et al., 2015): the frozen teacher is that phase's LORMA model (in Phase 2, the c-5 model) and the student is a shallower copy of a Phase-1 model. The student minimises the task loss plus a weighted KL divergence between temperature-softened teacher and student outputs.

Long documents are cut into overlapping windows set by `WINDOW` and `STRIDE`. A `DOC_REPR` setting keeps all windows, a fixed number spread evenly across the document (`spread`), or the first and last windows.

## Phase 2: Legal NER

Character-level spans `[start, end, label]` over 12 entity types, scored with strict Macro-F1. Training uses fold1, validation uses fold2, and the test set is fold3 (35 documents).

**Implementation (all runs).** Each BPE token gets one of 25 BIO tags (O plus B/I for each label). Spans are rebuilt from runs of B/I tags and mapped back to character offsets. Trailing `,.;:` are trimmed, because the tokenizer attaches trailing punctuation to words. Decoding the gold tags through this pipeline recovers only 0.735 to 0.930 of the gold Macro-F1, which caps what any model could score. All runs use 200-token windows with stride 10, unweighted cross-entropy, batch size 64 and 10 epochs. In the encoders (c-3, c-6) each token is predicted by the window where it sits furthest from an edge. In the causal decoders (c-1, c-5) each window after the first labels only its last 10 tokens, using the previous 190 as left context.

| Model | Method | Head and settings (from the notebook) | Local F1 (fold2) | Notebook |
| --- | --- | --- | --- | --- |
| c-1 | Full fine-tuning | linear head; LR 1e-4 | not recorded (output not saved) | [`phase2-c1-fullft-lora.ipynb`](phase2/phase2-c1-fullft-lora.ipynb) |
| c-1 | LoRA r=200 on the FFN output | 2-layer MLP head (200 hidden units, ReLU) | 0.1451 | same notebook |
| c-3 | Full fine-tuning | linear head; LR 1e-4 | 0.1431 | [`phase2-c3-fullft.ipynb`](phase2/phase2-c3-fullft.ipynb) |
| c-5 | LoRA r=20 (labelled LORMA) | LR 3e-4 | 0.1287 (0.0976 in the training log; 0.1287 when re-scored in the KD cell) | [`phase2-c5-lora-kd.ipynb`](phase2/phase2-c5-lora-kd.ipynb) |
| c-5 | KD: 8-block teacher to 4-block student (blocks 0, 2, 4, 7; 54.35M) | temperature 2.0, KD weight 0.5, student LR 1e-5, 10 epochs | 0.1338 | same notebook |
| c-6 | LoRA r=200, alpha 50, on the FFN output | 2-layer MLP head (200 hidden units, ReLU); LR 1e-4 | 0.0409 | [`phase2-c6-lora.ipynb`](phase2/phase2-c6-lora.ipynb) |

The c-3 notebook's title mentions BitFit, but no BitFit run was saved in it. Phase 2 had 6 valid Codabench submissions, with a best of 0.1626. Which run produced which Codabench score is not recorded. One early submission failed because of a duplicate span, and about a third of the spans predicted in the c-1 submission were one or two characters long.

## Phase 3: Rhetorical role prediction

Every sentence of a judgment gets one of 13 roles, across the Corporate Law (CL) and IT domains, scored with Macro-F1. One model was trained jointly on CL and IT. Local scores are on the dev split (10 documents, 2,197 sentences), averaged over the 11 roles that appear in dev. With about ten dev documents, differences of 0.02 to 0.03 are within noise.

**Implementation.** Two encodings were used. In sliding-window document mode (as in Phase 2), 200-token windows with stride 10 slide over the whole judgment. In packed document-context mode, consecutive sentences are packed into windows with read-only left context (and right context for the bidirectional c-6 encoder), each sentence is pooled from the shared window, and a sequence head (BiLSTM/BiGRU) over the sentence vectors predicts the roles. Each sentence is predicted once, and no CRF is used. c-3 needs inputs in multiples of four and has no padding mask, so sentences are split into overlapping rows and stitched back before pooling. A padding mask was added to the c-6 encoder. The loss uses `sqrt_inverse` class weights.

| Model | Method | Settings (from the notebook) | Local Macro-F1 (dev) | Notebook |
| --- | --- | --- | --- | --- |
| c-1 | LORMA (multiplicative) r=80 on q, k, v and out projections | sliding windows 200/10, LR 1e-4, batch 64, 50 epochs | 0.3185 | [`phase3-c1-lorma.ipynb`](phase3/phase3-c1-lorma.ipynb) |
| c-3 | QLoRA (NF4, double quantisation, fp16 compute) r=80 on qkv and out, in all three hourglass stages | packed context (1,024 tokens, 2 sentences of left context), LR 3e-5, head LR 3e-4, 10% warm-up, 50 epochs, batch 1 | 0.3359 | [`phase3-c3-qlora.ipynb`](phase3/phase3-c3-qlora.ipynb) |
| c-5 | AdaLoRA, initial rank 80 on q, k, v and out, target rank 80/1.5 | packed context, LR 3e-5, head LR 3e-4, orthogonality weight 0.1, 50 epochs | 0.2549 | [`phase3-c5-adalora.ipynb`](phase3/phase3-c5-adalora.ipynb) |
| c-6 | LoRA r=80 (labelled LORMA) on qkv, out, FFN-in and FFN-out | sliding windows 200/10, LR 1e-4, batch 64, 40 epochs | 0.3255 | [`phase3-c6-lora.ipynb`](phase3/phase3-c6-lora.ipynb) |

**No knowledge-distillation experiment was run in Phase 3.** The KD row (c-6 KD from LORMA, 0.33) in the earlier progress log and in the report's results table has no run behind it. Other earlier runs recorded in my log (c-6 LoRA sentence-only 0.279, c-6 LORMA document context 0.314, c-1 LORMA sentence-only about 0.31, c-1 LORMA document context about 0.20) are not in any saved notebook, so they are unverified. Phase 3 had 5 valid Codabench submissions, with a best of 0.3053. Which run produced which Codabench score is not recorded.

## Phase 4: Court judgment prediction

Binary outcome (0/1) from the whole judgment, scored with Macro-F1 over both classes. Train has 32,305 cases (18,920 class 0, 13,385 class 1) and dev has 994. Always predicting the majority class would score about 0.37.

**Implementation.** Each window is trained with its case's label, and window scores are combined per case by mean, max or vote, with a decision threshold. Class weights use `sqrt_inverse`.

| Model | Method | Settings (from the notebook) | Notebook |
| --- | --- | --- | --- |
| c-6 | LORMA r=200 on qkv, out, FFN-in and FFN-out (all 32 linear layers, 12.3M trainable) | 1,024-token windows, stride 512, all windows, mean aggregation, threshold 0.5, LR 1e-4, weight decay 0.01, batch 16, 10 epochs | [`phase4-c6-lorma-kd.ipynb`](phase4/phase4-c6-lorma-kd.ipynb) (cell E2) |
| c-6 | KD from the LORMA model: student keeps encoder blocks 0, 2, 4, 6 with LORMA r=200 | temperature 2.0, KD weight 0.5, LR 1e-4, batch 16, 10 epochs | same notebook (cell E3) |
| c-1 | BitFit (FFN and LayerNorm biases) | 1,024-token chunks, stride 512, at most 8 chunks, LR 1e-4, head LR 3e-4, batch 4 x 4 accumulation, 1 epoch capped at 60 minutes, dropout 0.1 | [`phase4-c1-bitfit-all-windows.ipynb`](phase4/phase4-c1-bitfit-all-windows.ipynb) (`DOC_REPR="all"`) and [`phase4-c1-bitfit-spread-windows.ipynb`](phase4/phase4-c1-bitfit-spread-windows.ipynb) (`DOC_REPR="spread"`, gradient checkpointing on) |
| c-3 | Full fine-tuning of the encoder (26.06M trainable) | 200-token windows, stride 100, at most 8 chunks per case (mean 7.3), LR 1e-5, batch 1 x 4 accumulation, `EPOCHS = 2` with a 60-minute training cap | [`phase4-c3-fullft.ipynb`](phase4/phase4-c3-fullft.ipynb) |

All four Phase-4 runs were done in interactive Kaggle sessions whose outputs were not saved. The notebooks here show outputs only up to the data and baseline cells, so **the dev scores are not recorded**. The four Phase-4 Codabench scores (0.5331, 0.5040, 0.4434 and 0.3323) cannot be matched to individual runs. The copy of the c-6 notebook in the report ZIP is a later revision of the same file, with a fused-attention speed-up and a per-epoch time cap added.

## Phase 5: Legal statute identification

Multi-label: each case gets a set of statute IDs out of 100. Scored with Macro-F1. Train has 42,750 cases and dev has 10,181. Gold sets average 3.9 statutes per case.

**Implementation.** Mean and max pooling over window representations, then 100 independent sigmoids trained with binary cross-entropy. Positive weights are sqrt(n_neg / n_pos), capped at 10. Window scores are combined by max, and one global threshold is tuned on dev over a grid from 0.10 to 0.80.

| Model | Method | Settings (from the notebook) | Local Macro-F1 (dev) | Codabench | Notebook |
| --- | --- | --- | --- | --- | --- |
| c-6 | LORMA r=8 on qkv, out, FFN-in and FFN-out | 512-token windows, stride 256, 8 windows spread across the document, LR 3e-4, weight decay 0.01, batch 16 (micro-batch 4), `EPOCHS = 6` in this notebook (the report's table lists 10) | 0.2679 (precision 0.23, recall 0.38) | not confirmed (see below) | [`phase5-c6-lorma.ipynb`](phase5/phase5-c6-lorma.ipynb) |
| c-6 | KD from the LORMA teacher: student keeps encoder blocks 0, 2, 4, 6, with LORMA r=8 (0.929M trainable of 17.48M) | temperature 2.0, KD weight 0.5, LR 3e-4, 9 epochs, batch 16 | 0.2784 (threshold 0.35) | **0.2818** | [`phase5-c6-kd.ipynb`](phase5/phase5-c6-kd.ipynb) |
| c-3 | LoRA r=8, alpha 16, dropout 0.05, on qkv, out, FFN-in and FFN-out from the hourglass stage upward (24 wrapped layers; 0.997M trainable of 27.06M); lower stages frozen | 512-token windows, stride 128, 8 windows spread across the document, LR 3e-4, weight decay 0.01, batch 16, 9 epochs, max aggregation, global threshold | 0.2225 (micro-F1 0.3338) | not confirmed (see below) | [`phase5-c3-lora.ipynb`](phase5/phase5-c3-lora.ipynb) |
| c-1 | BitFit (FFN and LayerNorm biases, 58,368 values; 0.161M trainable with the head) | 1,024-token windows, stride 512 (156,099 training windows), LR 1e-4, head LR 3e-4, 1 epoch | not recorded (the saved output stops at the start of epoch 1) | not recorded | [`phase5-c1-bitfit.ipynb`](phase5/phase5-c1-bitfit.ipynb) |

The c-6 LORMA model predicted 6.86 statutes per case against 3.9 in gold, with errors concentrated on the frequent statutes 39, 43, 5 and 97, and 603 test cases received an empty prediction because no minimum label count was enforced. Phase 5 had 3 valid Codabench submissions: 0.2818 (the c-6 KD file), 0.2179 (not matched to a run; the c-6 LORMA and c-3 LoRA notebooks both wrote a submission), and 0.0117 (a constant top-3 baseline, statutes 5, 34 and 97 for every case, which scores 0.0118 on dev). No c-3 partial fine-tuning run was made in Phase 5. The c-3 LoRA run predicted 7.52 statutes per test case on average, and 1,118 test cases received an empty prediction.

## Phase 6: Abstractive summarisation

Generate a summary of each judgment, scored with BERTScore F1. There is no dev split, so 703 training documents were held out for validation (6,325 train, 100 test). Documents have a median of about 3,900 c-6 tokens, and 92% exceed the 1,536-token context.

**Implementation.** The encoder-decoder models use fusion-in-decoder: up to four 1,024-token windows (stride 512, spread across the document) are encoded separately, and the decoder attends to all of them. Decoding is greedy with up to 220 new tokens (minimum 20), repetition penalty 1.1, no repeated trigrams, and a 150-word cap.

| Model | Method | Settings | Local BERTScore F1 (held-out) | Codabench | Notebook |
| --- | --- | --- | --- | --- | --- |
| c-6 | KD from the LORMA teacher: student keeps encoder blocks 0, 2, 4, 6 and decoder blocks 0, 2, 4, 6 (31.08M of the teacher's 56.39M), trained fully | KD weight 0.5, temperature 2.0, LR 1e-5, batch 4, 10 epochs, label smoothing 0.05, model chosen by validation loss | 0.8174 | **0.8147** | [`phase6-c6-kd.ipynb`](phase6/phase6-c6-kd.ipynb) |
| c-6 | LORMA r=8 on all 13 encoder and decoder linear targets (1.260M trainable of 56.39M); this is the KD teacher | label smoothing 0.05, LR 1e-4, weight decay 0.01, batch 4, 10 epochs, model chosen by validation loss; 351.9 min of training | 0.8111 | 0.8078 | [`phase6-c6-lorma.ipynb`](phase6/phase6-c6-lorma.ipynb) |
| c-3 | LoRA r=8, alpha 16, on encoder and decoder linear targets in all six hourglass stages (1.152M trainable of 56.28M) | same training and decoding settings as c-6, but no word cap on the summary (median 210 words on validation) | 0.8061 | 0.8040 | [`phase6-c3-lora.ipynb`](phase6/phase6-c3-lora.ipynb) |
| c-3 | BitFit: FFN and LayerNorm biases in every encoder and decoder stage (74 tensors, 48,480 values) | same settings as c-6 | 0.7965 | 0.7941 | [`phase6-c3-bitfit.ipynb`](phase6/phase6-c3-bitfit.ipynb) |
| c-1 | AdaLoRA on q, k, v, out, FFN-in and FFN-out, initial rank 12 pruned to target rank 8, orthogonality weight 0.1 (1.770M trainable of 112.75M). Decoder-only: the document, then an end-of-sequence separator, then the summary | 512-token windows, stride 256, 3 windows spread across the document, 256-token summary target, LR 1e-4, batch 4, 10 epochs, label smoothing 0.05 | 0.7993 | 0.7975 | [`phase6-c1-adalora.ipynb`](phase6/phase6-c1-adalora.ipynb) |

All five Phase-6 Codabench files were matched to their Kaggle outputs by checksum. A lead baseline (the first 637 words of each document) scored 0.8311 on the held-out set, higher than every trained Phase-6 model.

## Phase 7: Prior case retrieval

For each query case, rank a pool of candidate cases and return the top 7. Scored with Micro-F1@7. Queries are 817 / 118 / 237 (train / dev / test), with candidate pools of 4,320 / 1,023 / 1,727.

**Implementation (all runs).** A bi-encoder: queries and candidates go through the same Phase-1 encoder (for c-1, the decoder's hidden states with the LM head skipped). Window embeddings are mean-pooled and averaged per case, and candidates are ranked by cosine similarity. Training uses InfoNCE with temperature 0.05, in-batch negatives and 4 hard negatives drawn from the BM25 top 50. To fit a T4, gradient checkpointing and GradCache (chunked encoding with cached gradients, chunk 4) are used. All runs use 512-token windows, stride 256, at most 8 windows per case spread across the document, and batch size 16.

| Model | Method | Settings (from the notebook) | Local Micro-F1@7 (dev) | Codabench | Notebook |
| --- | --- | --- | --- | --- | --- |
| c-6 | LORMA r=80 on the 32 encoder linears (4.864M of 59.99M, 8.11%) | LR 3e-4, weight decay 0.01, 6 epochs | 0.2379 | **0.1943** | [`phase7-c6-lorma.ipynb`](phase7/phase7-c6-lorma.ipynb) |
| c-6 | KD from the LORMA teacher: student keeps encoder blocks 0, 2, 4, 6 (16.55M, 53.5% of the teacher), trained fully | listwise KD on similarity scores, weight 0.5, temperature 2.0, LR 1e-5, 10 epochs, batch 4 in the notebook (the report says 16) | 0.2346 | 0.1920 | [`phase7-c6-kd.ipynb`](phase7/phase7-c6-kd.ipynb) |
| c-3 | BitFit (encoder FFN and LayerNorm biases, all three stages) | LR 1e-4, 6 epochs | 0.2157 | 0.1683 | [`phase7-c3-bitfit.ipynb`](phase7/phase7-c3-bitfit.ipynb) |
| c-3 | QLoRA (NF4, double quantisation, fp16) with LoRA r=80, alpha 80, on encoder qkv, out, FFN-in and FFN-out | LR 3e-4, weight decay 0.01, 6 epochs | 0.0978 | 0.0638 | [`phase7-c3-qlora.ipynb`](phase7/phase7-c3-qlora.ipynb) |
| c-1 | BitFit (FFN and LayerNorm biases, all 8 blocks) | LR 1e-4, 6 epochs | 0.0700 | 0.0449 | [`phase7-c1-bitfit.ipynb`](phase7/phase7-c1-bitfit.ipynb) |

Untrained baselines on dev: BM25 0.1012, zero-shot c-6 encoder 0.1946, zero-shot c-3 encoder 0.2068, zero-shot c-1 decoder 0.0623. The c-3 QLoRA run scored below its own zero-shot encoder. All five Codabench files were matched to their Kaggle outputs by checksum.

## Submissions without code in this repository

Every run listed above now has its notebook in this repository. Codabench scores that could not be matched to a single run are marked as such in each phase's section.

## References

- Vaswani et al., 2017. Attention Is All You Need.
- Joshi et al., 2024. IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning.
- Bihany et al., 2025. LORMA.
- Hu et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models.
- Dettmers et al., 2023. QLoRA: Efficient Finetuning of Quantized LLMs.
- Zhang et al., 2023. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.
- Ben Zaken et al., 2022. BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.
- Hinton et al., 2015. Distilling the Knowledge in a Neural Network.
- Nawrot et al., 2022. Hierarchical Transformers Are More Efficient Language Models.
- Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
- Sennrich et al., 2016. Neural Machine Translation of Rare Words with Subword Units.
- Izacard and Grave, 2021. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.
- Robertson and Zaragoza, 2009. The Probabilistic Relevance Framework: BM25 and Beyond.
- van den Oord et al., 2018. Representation Learning with Contrastive Predictive Coding.
- Gao et al., 2021. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup.

The full write-up, including the analysis and the LLM usage declaration, is in the report submitted for the course.
