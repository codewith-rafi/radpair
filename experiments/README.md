# Experiment Results

Each value comes from the matching run's `metrics.json`; raw configs and logs remain in the run directories.

## Three-Way Validation

Chance R@1 ≈ 0.000659 (`n_val=1518`).

| Run | Best epoch | Val i2t R@1 | Val t2i R@1 | Shuffle i2t R@1 | Train@best i2t | Trainable | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `G1_003` | 7 | 0.01054 | 0.00922 | 0.00092 | 0.0586 | 721,408 | Frozen projections |
| `PEFT_001` | 11 | 0.01910 | 0.01910 | 0.00079 | 0.1738 | 1,016,320 | LoRA text Q/V; shared default |
| `e2e_001` | 10 | 0.02503 | 0.02503 | 0.00066 | 0.7559 | 58,803,968 | Partial e2e; train R@1 much higher than validation |

## Ablations

### Adaptation on versus off

| Setting | Run | Best epoch | Val i2t R@1 | Val t2i R@1 | Shuffle i2t R@1 |
|---|---|---:|---:|---:|---:|
| Off (`G_frozen`) | `plus_g1_freezeproj_003` | 7 | 0.01054 | 0.00922 | 0.00092 |
| On (PEFT) | `plus_g_peft_001` | 11 | 0.01910 | 0.01910 | 0.00079 |

### Findings-only versus Findings + Impression

| Setting | Run | Best epoch | Validation pairs | Chance R@1 | Val i2t R@1 | Val t2i R@1 | Versus chance |
|---|---|---:|---:|---:|---:|---:|---:|
| Findings + Impression | `plus_g_peft_001` | 11 | 1,518 | 0.00066 | 0.01910 | 0.01910 | ≈29× |
| Findings-only | `plus_g_peft_a1_findings_001` | 9 | 364 | 0.00275 | 0.0330 | 0.0165 | ≈12× |

The findings-only ablation uses a smaller validation gallery and a different sample availability rule, so its absolute Recall@1 is not directly comparable with the main validation table.

## Held-Out Test

Encoder: `plus_g_peft_001` plus disease and ITM heads. Test size: 3,189 pairs.

Chance R@1 = 1 / 3,189 ≈ 0.000314. Mean shuffled i2t R@1 ≈ 0.00044.

### Retrieval

| Direction | R@1 | R@5 | R@10 | Versus chance |
|---|---:|---:|---:|---:|
| Image→text | 0.00878 | 0.0386 | 0.0586 | ≈28× |
| Text→image | 0.01035 | 0.0426 | 0.0724 | ≈33× |

### Disease and Missing Modality

The disease head uses U-Ignore masking.

| Condition | Macro-AUROC | Macro-F1 |
|---|---:|---:|
| Image-only | 0.7117 | 0.2045 |
| Text-only | 0.8589 | 0.4818 |

Text-only performance is higher because the 14 labels are derived from report text.

### Image–Text Matching

| Metric | Value |
|---|---:|
| AUROC | 0.6942 |
| Accuracy | 0.6419 |
| Pairs | 6,378 (positive plus in-batch negative) |
