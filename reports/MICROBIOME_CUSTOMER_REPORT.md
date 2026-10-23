# Microbiome prediction: preliminary validation report

Prepared: 23 October 2026. Scope: the default Microbiome model and its supplied dataset.

**Current assessment**

The current Microbiome model has not demonstrated reliable prediction of individual genus abundances for a new participant. Its outputs should be described as exploratory estimates. Available validation does not support presenting them as accurate patient measurements or using them as a substitute for laboratory results. This assessment concerns Microbiome prediction; the Metabolite model has not been changed or re-evaluated in this investigation.

**Evidence reviewed**

The dataset contains 39 participants, ten participant inputs, and 1,638 genus outputs. Approximately 72.4% of abundance entries are zero. The saved model was evaluated using five-fold cross-validation: each participant was held out once, and a separate inner validation selected the model using only training participants. There is no independent external validation cohort.

| Measure | Observed result | Interpretation |
| --- | ---: | --- |
| Numeric error reduction versus a constant baseline | 0.80% in log1p mean squared error | Very little demonstrated improvement from participant-specific inputs on this measure |
| Average per-genus numeric R² | −0.071 | Weak average genus-level fit; performance differs between genera |
| Bacteroides held-out mean absolute error | 33,948.55 original units | Substantial error for this genus |
| Bacteroides held-out R² | −0.179 | Poor fit relative to the observed-mean reference used by R² |
| Relative-abundance profile error reduction versus baseline | 5.92% in squared Hellinger distance | Modest whole-profile improvement, not proof of accurate individual-genus predictions |

An error reduction of 0.80% is **not an accuracy rate**. R² and logarithmic error use different scales and reference calculations, so their conclusions need not coincide. Negative R² indicates a worse squared-error fit than the observed-mean reference for that metric. [R² definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

Across all numeric outputs, mean absolute error is 139.79 and RMSE is 3,269.71 original units. These aggregate measures mix genera with very different abundance levels and many zero values; they do not describe the expected error for Bacteroides or every other genus. Per-genus results should accompany aggregate scores.

**Example investigated**

For participant S1, the submitted inputs match the training row. Recorded Bacteroides abundance is **292,651**, while the saved model predicts **4,428.84**, approximately **98.49% below** the recorded value. Input matching and the Weight (Kgs) header were verified. This is a genuine model prediction error, not a demonstrated input-parsing error.

S1 is a training example, so this particular check is not an independent accuracy test. A regression model need not reproduce training records exactly. However, the discrepancy is large, and the separate held-out evaluation also shows limited predictive performance.

**Why performance is limited**

1. **Limited observations.** There are only 39 participant examples for learning and evaluating the relationships. Many genus targets have numerous zero entries. This limits the evidence available for rare or extreme values and makes performance estimates uncertain. The number of outputs alone is not proof of failure; the validation results are the relevant evidence.
2. **The numerical objective reduces the influence of extreme values.** The current model learns logarithmic abundances and uses regularization to reduce overfitting. This can pull predictions toward typical values and underpredict unusually large observations. That behavior is consistent with the S1 result, but it does not make the error acceptable.
3. **The inputs may not explain enough variation.** Age, sex, body measurements, and macronutrient intake have not demonstrated strong predictive information in the tested models. The supplied data do not establish whether additional biological, dietary, medication, or laboratory variables would resolve the problem. These remain hypotheses to investigate.
4. **The measurement scale needs confirmation.** Sample abundance totals range from 24,362 to 891,358. If these values are sequencing read counts, sequencing depth affects the totals and they should not be interpreted as absolute bacterial loads. The file alone does not establish the causes of variation between samples. [Microbiome sequencing data interpretation](https://pubmed.ncbi.nlm.nih.gov/29187837/)

**Improvements investigated**

Additional comparisons included ridge regression on original values, partial least squares, tree ensembles, nearest neighbours, kernel regression, and reversible 1–10 target scaling. Some alternatives reduced RMSE but increased average absolute error; no tested alternative demonstrated a clear overall improvement. These comparisons reused the same small cohort and are development evidence, not external confirmation.

Min/max restrictions keep outputs within observed training ranges. Scaling changes the numerical representation. Neither supplies missing predictive information or guarantees accuracy. Retrieving a known participant's recorded result would be a lookup, not improved prediction for a new participant.

**Recommended next phase and acceptance criteria**

Confirm assay units, laboratory processing, sample identifiers, and input consistency first. Agree which genera and which error measures matter for the intended use. Collect additional independent participants and assess whether relevant additional inputs improve prediction. Determine the required sample size using learning curves and the agreed performance criteria; no specific number guarantees success.

Before making a stronger accuracy claim, the model should show a consistent, meaningful improvement over simple baselines on the agreed per-genus measures, followed by evaluation on an independent cohort. More data and alternative models may help, but improvement cannot be promised in advance. Until that evidence exists, retain the exploratory status and disclose the observed limitations.

**Supporting records**

Dataset: `Genus_Abundance_Fecal Data_Combined.xlsx`.
Saved build: `run-6bc6ba95edaa4f9eac0f8c3519a49d45`, trained 23 October 2026; microbiome pipeline version 3.
The accompanying `microbiome_genus_validation.csv` contains the saved held-out numeric metrics for all genera. The full comparison is documented in `MICROBIOME_BENCHMARK.md`.
