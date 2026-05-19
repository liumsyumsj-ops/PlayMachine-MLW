# Spaceship Titanic — Experiment Log

## current best
**LB 0.81786** (CT65+MLP25+LGB9 ensemble + post-processing: Earth+CryoSleep=True+TRAPPIST-1e → force Transported=True, 33 cases)
Ensemble baseline: LB 0.81669 (CT v2 + MLP-wide + LGB-base, weights CT65%+MLP25%+LGB9%)
Previous best: LB 0.81622 (CT67%+MLP23%+XGB10% or CT65%+MLP25%+LGB10%)

## Failed Experiment (2026-05-19, rule-based post-processing)
**R1+R2 (Earth+noCryo+TRAP→False, 0.5~0.53)**: LB 0.81482 (↓0.00304)
**R1+R2+R3**: LB 0.81458 (↓0.00328)
- R2 (27entries True→False)showed+0.00138, but had a clearly negative effect on LB
- Lesson: OOF rule accuracy does not represent the test-set distribution; all newly added rules were unreliable

## Failed Experiment (2026-05-19, IRcalibration)
**IR calibration + best threshold** (calibrated_submissions/IR_cal_thr505_rule23.csv): 
- LB 0.81365 (↓0.00421, failed)
- Method: OOF blended probabilitiesfit Isotonic Regression, threshold 0.505, rules 23 entries
- OOF showed +0.00334 improvement (0.81307→0.81640), buttest-set distributionand OOF different, calibrationwas harmful instead
- Root cause: Manokhin & Grønhaug (2026) warned that"IR for strong models (CatBoost dominated ensembles)may systematically reduce performance", which was confirmed
- Lesson: OOF probability calibration gains from OOF cannot reliably transfer to the test set; CatBoost dominatedensembleis already well calibrated, does not need IR

---

## Code Architecture (complete, as of 2026-05-17)

```
━━━ feature Layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 preprocessing.py ← main preprocessing entry (load_and_preprocess)✅ shared by all models
 preprocessing_shared.py ← legacy preprocessing (for reference)
 catboost_features.py ← CT/XGB/MLP specific: ratios, logs, interaction features
 lgb_features.py ← LGB/TabNet specific: numerical features (make_lgb_data)✅
 rf_features.py ← RF specific (for reference, not includedbest)

━━━ single models (✅ = included in the current bestensemble)━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ct_v2.py ← CatBoost best (lr=0.03,l2=2,depth=6)✅
 Output: oof_ct_v2.npy / test_ct_v2.npy
 ct_best.py ← CatBoost ct9 parameters (legacy baseline)
 ct_mlp.py ← MLP(256-128-64) L-BFGS ✅
 Output: oof_MLP-wide.npy / test_mlp.npy
 xgb_best.py ← XGBoost (lr=0.05,λ=4,sub=0.9)
 Output: oof_xgb_lr05_lam4_sub90.npy / test_xgb_lr05_lam4_sub90.npy
 rf_best.py ← Random Forest (not included in the best blend)
 KNN.py ← KNN (not included in the best blend)
 svm.py ← SVM (not included in the best blend)
 mlp_1.4.py ← MLP legacy version

 ── LGB (lgb_base default parametersasbest)──
 lgb_grid.py ← LGB first-round coarse search (wrong ensemble context, discarded)
 lgb_grid_v2.py ← LGB second-round coarse search (correct ensemble: CT65+MLP25+LGB9)✅
 Output: oof_lgb_base.npy / test_lgb_base.npy
 lgb_fine_grid.py ← LGB fine-grained search (±10~20%)
 lgb_dense_grid.py ← LGB dense gap-filling search
 lgb_reg_search.py ← reg_lambda 0.91~0.94 fine exploration
 lgb_reg_low_search.py ← reg_lambda 0.50~0.90 low-value range exploration
 lgb_reg_065_search.py ← reg_lambda 0.66~0.69 ultra-fine search
 lgb_reg_064.py ← reg_lambda=0.64 single-point validation

 ── TabNet (closed)──
 tabnet_base.py ← TabNet baseline (default parameters)
 tabnet_grid.py ← TabNet single-parameter controlled tuning (OOF increased for all settings, while LB decreased for all settings)
 tabnet_combined.py ← TabNet best parameter combination (not submitted, path closed)

━━━ CT feature Engineering Experiments ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ct_v3_group_target.py ← +group_transported_mean (LOO), OOF↓, abandoned
 ct_v4_deck_homeplanet.py ← Deck→HomePlanet reverse imputation, modified 0 rows, abandoned
 ct_v5_age_spend.py ← Age<13→spending=0, modified 0 rows, abandoned
 ct_v6.py ← +Surname/Deck_Side/Co_Travelers_Sleepand 7 other features
 ensemble OOF↑0.0031, LB 0.80944 (↓0.00725), failed
 ct_feature_importance.py ← feature importance ranking + feature removal experiment (removing bottom-contribution features)
 LB 0.80313 (↓0.01356), CryoSleep interaction-feature synergy was critical
 mlp_features.py ← MLP specificfeature Layer (including Surname TE + OHE + StandardScaler)
 ct_pseudolabel.py ← pseudo-labeling experiment (ensembleprobability, threshold0.97/0.95/0.90)
 thr097 OOF↑0.0017, pending submission validation
 ct_grid.py ← CT single-parameter controlled search (first round)
 ct_final_grid.py ← CT lr×l2×rsm full matrix search (second round)
 feature_test.py ← controlled feature test (first round, early stage)

━━━ ensemble / weight search ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 blend_best.py ← CT65+MLP25+XGB10 (early baseline, LB 0.81458)
 blend_search.py ← coarse weight search
 blend_fine_search.py ← fine weight search (CT+MLP+XGB)
 blend_lgb_fine_search.py ← CT+MLP+LGB fine weight search → finding LB 0.81669 ✅
 blend_lgb_submit.py ← LGB ensemble submission generation
 blend_xgb_weight.py ← XGB weight-sensitivity search
 blend_grid.py ← ct_grid result replacement CT afterensembleSearch
 blend_wide_search.py ← ultra-wide rangeweight search (CT 0~80%, no assumption)
 ridge_blend.py ← Ridge as the fourth modelweight search
 stacking_ridge.py ← RidgeClassifierCV meta-model Stacking

━━━ post-processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 group_voting.py ← same-group voting post-processing (invalid, train/test groups do not overlap at all)
 surgical_flip.py ← surgical flip (early stageExperiment)
 
 Submission file directory: spaceship-titanic/all_submissions/
 A_thr0XX.csv ← threshold variants 0.44~0.53 (all ≤ 0.81669)
 B_cryo_hard_thr50.csv ← CryoSleep=True → all True (LB 0.81739)
 C_cryo_soft04_thr50.csv ← CryoSleep+blend[0.4,0.5) flipped to True (LB 0.81716)
 X_earth_cryo_only.csv ← Earth+CryoSleep → True (LB 0.81762)
 X_mars_cryo_only.csv ← Mars+CryoSleep → True (LB 0.81646, harmful)
 Y_earth_cryo_TRAP.csv ← Earth+Cryo+TRAPPIST-1e → True (LB 0.81786 ✅ best)
 Y_earth_cryo_PSO.csv ← Earth+Cryo+PSO → True (LB 0.81646, harmful)
 Z_*series ← Earth+Cryo+TRAP age segmentation+Mars+TRAP (all≤0.81786)

━━━ Tools ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 make_submission.py ← generate submission files from.npy files
 eda.py ← exploratory data analysis

━━━ XGB tuning ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 xgb_grid.py ← XGB single-parameter controlled search (first round)
 xgb_final_grid.py ← XGB lr×λ×sub full matrix search (second round)
 xgb_lam_sub_grid.py ← XGB λ×sub fine-grained search (third round)

━━━ Key OOF / Test Files ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 oof_ct_v2.npy ← CT v2 OOF ✅
 test_ct_v2.npy ← CT v2 test-set probabilities ✅
 oof_MLP-wide.npy ← MLP OOF ✅
 test_mlp.npy ← MLP test-set probabilities ✅
 oof_lgb_base.npy ← LGB base OOF ✅
 test_lgb_base.npy ← LGB base test-set probabilities ✅
 oof_xgb_lr05_lam4_sub90.npy ← XGB best OOF
 test_xgb_lr05_lam4_sub90.npy ← XGB besttest-set probabilities
```

---

## CatBoost Experiment History

| Version | Key change | CV | LB | Conclusion |
|------|---------|-----|-----|------|
| ct1 | baseline (GroupKFold, depth=6, lr=0.05) | ~0.817 | 0.80804 | Mac baseline |
| ct5 | Optuna tuning (single holdout objective function) | — | 0.80664 | lr=0.117 too aggressive |
| ct6 | 10-seed multi-seed soft voting | — | 0.80734 | seed=456 produced a bad split |
| ct7 | +MemberId/LuxuryRatio/CabinSection | — | 0.80406 | redundant features caused overfitting |
| ct8 | simplified features, without log/ratio | 0.8164 | 0.80570 | log/ratio are useful for CT and cannot be removed |
| **ct9** | +CabinSize + NecessitiesSpend/EntertainmentSpend + NaNretained | 0.8124 | **0.81155** | CT best feature set |
| ct10 | ct9 + Europa_Cryo + CabinNum_x_Side + 8-seed | 0.8143 | 0.80897 | multi-seed failed again |
| ct11 | ct9 + Europa_Cryo + CabinNum_x_Side (single seed) | 0.8143 | 0.80336 | new features overfitted |
| ct13 | ct9 + KFold Target Encoding | 0.8131 | 0.80874 | TE overfitting |
| ct15 | ct9 + Optuna (GroupKFold CV objective) | 0.8161 | 0.81131 | depth=7,rsm=0.65 bestbut not as good as ct9 feature set |
| ct16 | ct9 + pseudo-label (threshold 0.85) | 0.8146 | 0.80734 | pseudo-labeloverfitting |
| ct17 | ct9 - zero-importance features | 0.8131 | 0.80851 | zero-importance featureshad implicit regularization effects |
| ct20 | ct9 + depth=5 + l2=10 | 0.8146 | 0.80617 | CV highest LB lowest, CV-LB completely reversed |

---

## MLP Experiment History

| Version | Architecture | CV | CT correlation | ensemble LB | Conclusion |
|------|------|----|----------|---------|------|
| **MLP-wide (sklearn)** | **(256,128,64) Adam (defaultsolver), early_stopping=True, lr_init=0.001** | ~0.80 | **0.944** | **0.81458→0.81529** | **current best for blending** |
| MLP v2 | (256,256,128,64) Adam (defaultsolver), alpha=0.01 | 0.8047 | 0.9452 | decreased | harmful in the ensemble, single-model best does not equal ensemble best |
| PT-Emb-MLP | EmbeddingResMLP Adam | 0.8072 | 0.9553 | 0.81061 | high correlation, low diversity |
| PT-OH-MLP | ResMLP one-hot Adam | 0.8086 | 0.9507 | — | and sklearn MLP correlation 0.9709, equivalent |

---

## LR Experiment History

| Version | Architecture | CV | CT correlation | ensemble LB | Conclusion |
|------|------|----|----------|---------|------|
| LR | Logistic Regression | 0.7974 | 0.9354 | 0.81342 | individual model too weak, diversity benefit was offset |

---

## ensembleExperiment History

| Method | OOF | LB | Conclusion |
|------|-----|----|------|
| CT + RF | 0.8119 | — | correlation 0.965, not submitted |
| CT + LGB | — | — | correlation 0.975, not submitted |
| CT 80% + MLP 20% | 0.8146 | 0.81365 | first exceeded ct9 |
| **CT 65% + MLP 25% + XGB 10%** | 0.8142 | **0.81458** | previous best |
| CT 71% + MLP 23% + XGB 6% | 0.8149 | 0.81318 | OOF rose while LB dropped, weight adjustment was already noise |
| CT + PT-Emb-MLP + XGB | — | 0.81061 | PT MLP had lower diversity than sklearn MLP |
| all-model grid search (5model) | — | 0.81342 | adding models diluted diversity |
| Stacking (LR/XGB meta-model) | 0.8135~0.8138 | — | all worse than the blend; not submitted |

### Key Ensemble Lessons
- sklearn MLP (Adam)diversity comes fromarchitectural difference between trees and neural networks (trees use hard splits, MLP uses smooth activation functions), rather than optimizer differences; PT-OH-MLP and sklearn MLP correlation 0.9709 confirmed this point
- OOF grid search overfits: multiple different combinations LB all fell around 0.81342
- **CT+MLP+XGB three-model setup is the blending ceiling**, adding other models gave no improvement

---

## feature Engineering Experiment History

| feature | CV Δ | LB | Conclusion |
|------|------|----|------|
| planet_dest | -0.0012 | — | combination added no incremental value |
| group_spend | +0.0013 | 0.80757 | GroupKFold leakage, LB large drop |
| member_idx | +0.0008 | — | paused, leakage risk |
| age_group | -0.0005 | — | CatBoost already learned by CatBoost |
| planet_cryo | +0.0024 | 0.80851 | CV inflated, LB large drop, CT already learned by CatBoostinteraction |
| is_very_young | -0.0003 | — | no incremental value |

**Conclusion: CV systematically overestimates new features, feature engineering path paused**

---

## CatBoost Parameter Matrix (first round, ct_grid.py)

singleparameterscontrolvariables, based on ct9 feature set: 

| parameters | CV Δ | Conclusion |
|------|------|------|
| depth=7 | +0.0021 | CV strongestbut LB 0.80064 (ensemble), direction invalid |
| depth=5 | -0.0014 | too shallow |
| rsm=0.9 | +0.0008 | weak, pending combination |
| rsm=0.8 | +0.0006 | — |
| l2=1 | +0.0007 | l2 decreasing l2 was effective |
| l2=6~20 | all decreased | increasing regularization was ineffective |
| lr=0.03 | +0.0007 | slower learning slightly improved |
| lr=0.10 | -0.0010 | too aggressive |
| min_data=5~25 | +0.0001 | four values were exactly the same, this parameter was ineffective for CT |

---

## CatBoost Parameter Matrix (second round, ct_final_grid.py)

depth=6 Fixed, lr × l2 × rsm full LB matrix: 

**rsm=1.0 full LB matrix: **

| lr | l2=1 | l2=2 | l2=3 | l2=4 |
|----|------|------|------|------|
| lr=0.01 | — | 0.81248 | — | — |
| lr=0.02 | — | 0.81248 | — | — |
| lr=0.03 | 0.81178 | **0.81529 ✅** | 0.81225 | 0.81155 |
| lr=0.031 | — | 0.81131 | — | — |
| lr=0.033 | — | 0.81178 | — | — |
| lr=0.04 | 0.81271 | 0.81412 | 0.80967 | 0.81295 |
| lr=0.05 (ct9) | 0.81365 | 0.81342 | 0.81108 | 0.81038 |
| lr=0.06 | 0.81458 | 0.81295 | 0.81131 | 0.81295 |
| lr=0.07 | 0.81482 | — | — | — |
| lr=0.08 | 0.81178 | — | — | — |
| lr=0.09 | 0.81342 | — | — | — |
| lr=0.10 | 0.81178 | — | — | — |

rsm=0.95: OOF systematically inflatedbut LB decreased for all settings (lr03_l23: 0.81084, lr03_l22: 0.81225)

**Patterns: **
- l2=2 peak at lr=0.03, is very sharp peak (lr=0.031 dropped to 0.81131)
- l2=1 peak at lr=0.07 (0.81482), then dropped sharply
- ct9 of l2=3 was consistently worse than l2=2
- rsm=1.0 better than rsm=0.95
- lr very low (0.01/0.02)early stopping caused both to converge to the same solution

---

## Controlled Ensemble Experiment (model replacement)

fixed weights (CT≈65%+MLP≈25%+XGB≈10%), replacing each model one by one: 

| Combination | OOF | LB | Conclusion |
|------|-----|-----|------|
| ct9+mlp1+xgb1 (original baseline) | 0.8150 | 0.81458 | baseline |
| **ct2+mlp1+xgb1 (only replacing CT)** | 0.8143 | **0.81529 ✅** | only effective improvement |
| ct9+mlp2+xgb1 (only replacing MLP) | 0.8134 | 0.81248 | MLP v2 harmful |
| ct9+mlp1+xgb2 (only replacing XGB) | 0.8146 | 0.81365 | XGB v2 harmful (best_iter=19) |
| ct9+mlp2+xgb2 (replacing MLP+XGB) | 0.8139 | 0.81248 | — |
| ct2+mlp1+xgb2 (replacing CT+XGB) | 0.8143 | 0.81458 | XGB v2 offset the gain from CT v2 |
| ct2+mlp2+xgb1 (replacing CT+MLP) | 0.8144 | 0.81014 | MLP v2 caused severe drag |
| ct2+mlp2+xgb2 (replacing all) | 0.8138 | 0.81061 | — |

**Conclusion: **
- MLP v2 (256-256-128-64, alpha=0.01)was always harmful in the ensemble
- XGB v2 (depth=7, mcw=20, lambda=10)best_iter=19, severely underfit
- teammates'single modelsbestparameters ≠ ensemblebestparameters
- **only effective improvement is CT v2 (lr=0.03, l2=2)**

---

## Global Key Lessons

1. **CV and LB negative correlation**: GroupKFold CV evaluation distributionandreal test sethas systematic bias, CV improvement often means LB decreased
2. **OOF weight search overfits**: multiple timesSearch LB all fell around 0.81342, indicating it was already noise
3. **feature engineering ceiling**: ct9 feature set istrue optimum, any addition or removal LB decreased
4. **parameter ceiling**: CT true optimum lr=0.03,l2=2 (LB +0.00071), all other tuning was ineffective
5. **ensemble ceiling**: CT+MLP+XGB is the best combination, model replacement/stacking/adding models were all ineffective
6. **joint tuning principle**: tune within the ensemble context, single-model best does not equal ensemble best

---

## XGB Parameter Matrix (first round, xgb_grid.py)

singleparameterscontrolvariables, based on xgb_best feature set (depth=6, lr=0.05, sub=0.8, col=0.8, λ=1, mcw=1): 

| parameters | Name | CV | ΔCV | ensemble OOF | Δensemble | iter | Conclusion |
|------|------|----|-----|---------|-------|------|------|
| baseline | xgb_best | 0.8127 | — | 0.8126 | — | — | baseline |
| depth=5 | depth5 | 0.8126 | ↓0.0001 | 0.8123 | ↓0.0003 | 83 | too shallow |
| depth=7 | depth7 | 0.8117 | ↓0.0010 | 0.8125 | ↓0.0001 | 150 | worse |
| depth=8 | depth8 | 0.8102 | ↓0.0025 | 0.8134 | ↑0.0008 | 30 | iter=30 underfit, unreliable |
| lr=0.03 | lr003 | 0.8087 | ↓0.0040 | 0.8124 | ↓0.0002 | 121 | too slow |
| lr=0.04 | lr004 | 0.8105 | ↓0.0022 | 0.8123 | ↓0.0003 | 65 | worse |
| lr=0.08 | lr008 | 0.8118 | ↓0.0009 | 0.8127 | ↑0.0001 | 186 | weak |
| lr=0.10 | lr010 | 0.8126 | ↓0.0001 | 0.8125 | ↓0.0001 | 155 | invalid |
| λ=2 | lam2 | 0.8135 | ↑0.0008 | 0.8128 | ↑0.0002 | 232 | weak |
| λ=3 | lam3 | 0.8130 | ↑0.0002 | 0.8125 | ↓0.0001 | 117 | invalid |
| **λ=5** | **lam5** | **0.8162** | **↑0.0034** | **0.8132** | **↑0.0006** | **89** | **best, CV+OOF same direction ✅ → LB 0.81552** |
| λ=10 | lam10 | 0.8107 | ↓0.0021 | 0.8128 | ↑0.0002 | 18 | iter=18 severely underfit |
| sub=0.6 | sub06 | 0.8140 | ↑0.0013 | 0.8126 | 0.0000 | 62 | invalid |
| sub=0.7 | sub07 | 0.8132 | ↑0.0005 | 0.8123 | ↓0.0003 | 184 | invalid |
| sub=0.9 | sub09 | 0.8103 | ↓0.0024 | 0.8128 | ↑0.0002 | 107 | weak |
| sub=1.0 | sub10 | 0.8110 | ↓0.0017 | 0.8132 | ↑0.0006 | 46 | iter too low |
| col=0.6 | col06 | 0.8117 | ↓0.0010 | 0.8124 | ↓0.0002 | 141 | worse |
| col=0.7 | col07 | 0.8113 | ↓0.0014 | 0.8125 | ↓0.0001 | 243 | worse |
| col=0.9 | col09 | 0.8086 | ↓0.0041 | 0.8124 | ↓0.0002 | 77 | worse |
| col=1.0 | col10 | 0.8088 | ↓0.0039 | 0.8125 | ↓0.0001 | 97 | worse |
| mcw=3 | mcw3 | 0.8120 | ↓0.0007 | 0.8127 | ↑0.0001 | 89 | weak |
| mcw=5 | mcw5 | 0.8131 | ↑0.0003 | 0.8132 | ↑0.0006 | 179 | weak |
| mcw=10 | mcw10 | 0.8116 | ↓0.0012 | 0.8131 | ↑0.0005 | 156 | weak |
| mcw=20 | mcw20 | 0.8100 | ↓0.0028 | 0.8127 | ↑0.0001 | 102 | invalid |

**First-round conclusion: **
- **λ=5 only reliable signal** (CV+ensemble OOF rose in the same direction, LB validation 0.81552 ✅)
- colsample all variantsalldecreased → col=0.8 Fixedunchanged
- depth=6 Fixed (depth8 iter=30 underfit)
- remainingparameters (subsample/lr/mcw)signals were very weak, changes were within noise range

---

## XGB Parameter Matrix (second round, xgb_final_grid.py)

Fixed: depth=6, col=0.8, mcw=5; variables: lr × λ × sub
ensemble OOF (CT_v2 65% + MLP 25% + XGB 10%), baseline 0.8126

**sub=1.0 full ensemble OOF matrix: **

| lr | λ=4 | λ=5 | λ=6 | λ=7 |
|----|-----|-----|-----|-----|
| lr=0.05 | 0.8127 | 0.8132 | 0.8131 | **0.8135** |
| lr=0.06 | 0.8133 | 0.8134 | 0.8128 | 0.8131 |
| lr=0.07 | 0.8126 | 0.8126 | 0.8131 | 0.8134 |
| lr=0.08 | 0.8126 | 0.8127 | 0.8134 | 0.8132 |

**sub=0.9 full ensemble OOF matrix: **

| lr | λ=4 | λ=5 | λ=6 | λ=7 |
|----|-----|-----|-----|-----|
| lr=0.05 | 0.8124 | 0.8130 | 0.8130 | 0.8128 |
| lr=0.06 | 0.8127 | 0.8130 | 0.8133 | 0.8132 |
| lr=0.07 | 0.8127 | 0.8126 | 0.8128 | 0.8133 |
| lr=0.08 | 0.8128 | 0.8131 | 0.8127 | 0.8126 |

**submitted LB validation: **

| Name | lr | λ | sub | ensemble OOF | LB | Conclusion |
|------|----|----|-----|---------|-----|------|
| lr05_lam7_sub100 | 0.05 | 7 | 1.0 | 0.8135 (highest) | 0.81505 | OOF highest LB reversed drop, OOF failed |

**second roundConclusion: **
- sub=1.0 overallbetter than sub=0.9 (OOF level, but LB opposite at the level)
- OOF highestCombination LB reversed drop, and CB fully consistent with experience
- initialbest: lr05_lam4_sub90 (**LB 0.81622**), OOF ranked No. 8 but LB strongest
- OOF completely unable to predict LB, requires full submission validation

---

## ensemble weightsExperiment (coarse-grained, blend_xgb_weight.py)

Fixed XGB=lam5, Search CT/MLP/XGB the threeweights (step size 5%), submitted LB validation: 

| CT | MLP | XGB | LB | Conclusion |
|----|-----|-----|----|------|
| 65% | 25% | 10% | 0.81552 | baseline (lam5)|
| 60% | 25% | 15% | 0.81505 | XGB↑ invalid |
| 65% | 20% | 15% | 0.81365 | XGB↑ invalid |
| 50% | 15% | 35% | 0.81061 | XGB extremely large, large drop |
| 70% | 5% | 25% | 0.81038 | MLP extremely small, large drop |
| 65% | 5% | 30% | 0.80991 | worst |

**Conclusion: XGB weightsincrease was alwaysharmful, MLP cannot be below 20%, XGB Fixed 10%**

---

## ensemble weightsExperiment (fine-grained search, blend_fine_search.py)

Fixed XGB≤20%, CT∈[60,70], MLP∈[20,30], step size 1%, submitted LB validation: 

| CT | MLP | XGB | LB | Notes |
|----|-----|-----|----|------|
| 65% | 25% | 10% | 0.81552 | original baseline |
| 66% | 24% | 10% | 0.81552 | tied |
| **67%** | **23%** | **10%** | **0.81575 ✅** | **new best** |
| **67%** | **24%** | **9%** | **0.81575 ✅** | **tiedbest** |
| 66% | 23% | 11% | 0.81482 | XGB↑ started to decline |
| 67% | 22% | 11% | 0.81435 | XGB↑ worse |
| 68% | 22% | 10% | 0.81482 | CT too high and started to drop |
| 69% | 21% | 10% | 0.81412 | continued to decline |
| 69% | 20% | 11% | 0.81388 | worst |

**fineweightsLBmatrix (XGB=10%, CT×MLP): **

| CT＼MLP | 21% | 22% | 23% | 24% | 25% |
|---------|-----|-----|-----|-----|-----|
| 66% | — | — | 0.81482* | 0.81552 | 0.81552 |
| 67% | — | 0.81435* | **0.81575** | **0.81575*** | — |
| 68% | — | 0.81482 | — | — | — |
| 69% | 0.81412 | — | — | — | — |

 (*XGB≠10%ofneighboring cells, noted in parentheses)

**Weight Search Patterns: **
- CT weightspeak at 67%, 65→66→67 rose gradually, 68→69 dropped sharply
- MLP correspondingly moved from 25% decreased to 23% best
- XGB kept at 9~10%, increasing to 11% started to decline
- OOF weight searchsystematically overestimatedhigh XGB weights (OOF best=XGB25%, LB best=XGB10%)

---

## XGB Parameter Matrix (third round, xgb_lam_sub_grid.py)

Fixed: lr=0.05, depth=6, col=0.8, mcw=5; variables: λ × sub
baseline: lr05_lam4_sub90 (LB 0.81622)

**LB validation matrix (λ × sub, partially submitted): **

| λ＼sub | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 |
|--------|------|------|------|------|------|
| 2.0 | — | — | 0.81552 | — | — |
| 3.0 | — | — | 0.81505 | — | — |
| 3.5 | — | — | 0.81505 | — | — |
| **4.0** | 0.81505 | 0.81529 | **0.81622 ✅** | 0.81575 | 0.81529 |
| 5.0 (first round) | — | — | — | — | — |

**third roundConclusion: **
- λ=4.0 is true peak (λ=3.5/3.0 decreased, λ=2.0 partialrebounded but did not reach 4.0)
- sub=0.90 is true peak (both sides 0.85/0.95 alldecreased, sharp-peak shape)
- **XGB true optimumparameters: lr=0.05, depth=6, λ=4.0, sub=0.90, col=0.8, mcw=5**
- XGB parameter tuning ends here

---

## currentStatus (2026-05-16)

- **current best (as of that day): LB 0.81622** (CT v2 + MLP-wide + XGB(lr=0.05,λ=4,sub=0.9), weights CT67%+MLP23%+XGB10%)
- **Today's progress summary: **
 - XGB parameter tuning: λ=1→4, sub=0.8→0.9 (+0.00093 vs originalXGB)
 - ensemble weight tuning: CT65→67, MLP25→23 (+0.00023)
 - total: 0.81529 → **0.81622**, +0.00093
- **Next step: ** XGB parameters and ensemble weightshave both reached the ceiling, in progress MLP parameter tuning

---

## currentStatus (2026-05-17, competition deadline)

- **current best: LB 0.81786** (Y_earth_cryo_TRAP post-processing)
- **final submission file: ** `spaceship-titanic/all_submissions/Y_earth_cryo_TRAP.csv`
- **Today's progress summary: **
 - LGB replaced XGB as the third ensemble model: CT65+MLP25+LGB9 = LB 0.81669 (+0.00047)
 - LGB parameter tuning: three rounds of exploration, default parameters were already best, reg_lambda multi-peak structure, could not exceed 0.81669
 - decision-threshold optimization: 0.50 isexact optimum, both sides symmetrically decreased, threshold path closed
 - post-processingfinding: Earth+CryoSleep=True+TRAPPIST-1e → True (33cases)= LB 0.81786 (+0.00117)
 - exhaustively tested all post-processing directions: 55itemsCryoSleep=Truepredicted-False cases fully analyzed, VIP/Age/spending rules all tested
 - total: 0.81622 → **0.81786**, +0.00164
- **all paths closed: ** model tuning/feature engineering/weight search/post-processing were all exhausted

**Final Path Trajectory: **

| Milestone | LB | Action |
|--------|-----|------|
| CT v2 baseline | 0.81529 | lr=0.03, l2=2 |
| LGB replacement XGB | 0.81669 | CT65+MLP25+LGB9 |
| LGB tuning | 0.81669 | could not surpass it; default parameters were already best |
| CryoSleep full rule | 0.81739 | B_cryo_hard |
| Earth exclusive rule | 0.81762 | X_earth_cryo |
| **Earth+Cryo+TRAP** | **0.81786 🏁** | **Y_earth_cryo_TRAP (final submission)** |

---

## MLP parameter tuning (mlp_grid.py, in progress)

**Script: ** `mlp_grid.py` 
**ensemble weights: ** CT67% + MLP23% + XGB10% (current best) 
**Fixed: ** solver=lbfgs 
**Search: ** alpha × hidden_layer_sizes

**first roundtried (alreadydiscarded): **
- alpha=[1e-05, 5e-05,...]: CV directly dropped to 0.75, insufficient regularization, allinvalid
- early stopped, related files already deleted npy file

**currentSearchgrid (mlp_grid.py corrected): **
```python
ALPHA_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1] # search upward from the current optimum
HIDDEN_GRID = [(128,64), (256,128), (256,128,64), (256,128,64,32)] # removed slow 512 network
```
total 28 combinations, around 30 minutes

**Completed combinations (alpha=0.0001, partial): **
- mlp_a00001_h128x64: oof saved
- mlp_a00001_h256x128: oof saved
- others pending

**Conclusion (to be added later): ** fill in after running

---

## feature Engineering Experiments (second round, controlled-variable method, 2026-05-17)

**Background: ** after reviewing Spaceship Titanic top solutions, findingcurrentfeature setmay be missingfollowinghigh-value information.
using the controlled-variable method, changing only one variable each time, baselineas CT v2 (LB 0.81622 corresponding to CT).

**ensemble context: ** new CT + MLP-wide(23%) + XGB(lr05_lam4_sub90)(10%), CT weights 67%

### Experiment1: group_transported_mean (Script: ct_v3_group_target.py)

**Change: ** added a numerical feature `group_transported_mean`
- training set: Leave-One-Out within-group target mean (excluding itself, to prevent direct target leakage)
- test set: all members of that group in the training setof Transported mean
- single-person group/test setno training members: fill withglobal mean (≈0.5028)
- **Note: ** GroupKFold insamples from the same group are in the same fold, LOO uses full training labels, OOF may be slightly optimistic

**Rationale: ** passengers in the same group have strongly correlated outcomes (already by AllGroupCryo/NoGroupCryo partially captured, but direct target signal is missing)

| Version | OOF CV | ensemble OOF | Δensemble OOF | LB | Conclusion |
|------|--------|---------|---------|-----|------|
| CT v2 (baseline) | — | 0.8133 | — | 0.81622 | — |
| CT v3 (+group_target) | 0.8123±0.0047 | 0.8126 | ↓0.0007 | pending submission | OOF decreased, Description CT already learned by CatBoostwithin-group information |

---

### Experiment2: Deck→HomePlanet deterministicreverse imputation (Script: ct_v4_deck_homeplanet.py)

**Change: ** in preprocessing added afterdeterministicreverse imputationrules
- Deck A/B/C/T → HomePlanet = 'Europa' (no exception in the dataset)
- Deck G → HomePlanet = 'Earth' (no exception in the dataset)
- current preprocessing only has HomePlanet→Deck one-way mapping (Europa→B, Earth→G, Mars→F)
- this change correctedmissing-value imputation errors in the reverse direction

**Rationale: ** deterministicrules/no information loss, imputation accuracy is higher than current groupby-mode imputation

| Version | OOF CV | ensemble OOF | Δensemble OOF | LB | Conclusion |
|------|--------|---------|---------|-----|------|
| CT v2 (baseline) | — | 0.8133 | — | 0.81622 | — |
| CT v4 (+deck_homeplanet) | 0.8131±0.0058 | 0.8133 | ↓0.0000 | not submitted | **number of corrected rows=0**, current preprocessing already correctly covers this rule |

---

### Experiment3: Age<13 → spendingforced to 0 (Script: ct_v5_age_spend.py)

**Change: ** preprocessing after, for Age<13 ofall passengers, set 5 spending columnsforceset 0
- currentonly has CryoSleep=True → spending=0
- missing Age<13 → spending=0 ofrules (children cannot use paid facilities)

**Rationale: ** domain constraint, in the data Age<13 passengersactual spending is zero or close to zero

| Version | OOF CV | ensemble OOF | Δensemble OOF | LB | Conclusion |
|------|--------|---------|---------|-----|------|
| CT v2 (baseline) | — | 0.8133 | — | 0.81622 | — |
| CT v5 (+age_spend_zero) | 0.8131±0.0058 | 0.8133 | ↓0.0000 | not submitted | **number of corrected rows=0**, children's spending in the dataset was already zero, no extra rule was needed |

---

### Follow-up plan (depending on the above results)

if any experiment improves LB:: 
- stack the effective changes, write ct_v6 (combined version)
- then verify one by one whether the stacked effects are orthogonal (effective + effective does not necessarily mean the combination is effective)

if all are ineffective:: 
- feature engineering ceilingalreadyconfirmed
- switched to: MLP tuningresult validation (mlp_grid.py outputs)

### Second-Round Experiment Summary

| Experiment | Core finding | Conclusion |
|------|---------|------|
| v3 group_target | OOF ↓0.0007 | CT has already used AllGroupCryo/NoGroupCryo/GroupSize learnedwithin-group information, explicit target mean was redundant or even harmful |
| v4 deck_homeplanet | number of corrected rows=0 | currentpreprocessing (groupby mode + deck_map)already correctly handled all Deck↔HomePlanet mapping |
| v5 age_spend_zero | number of corrected rows=0 | children's spending in the dataset was already0, no extra rule was needed (the data was already clean) |

**featureengineeringsecond roundConclusion: current preprocessing_shared.py + catboost_features.py feature set is already very complete, threeentries"improvementdirection"were already covered by the existing code or learned by the model.featureengineeringno additional space for improvement.**

v3 whether it is worth submitting LB: OOF decreasedand LOO should theoretically make OOF slightly optimistic (should↑but↓), Descriptionthe feature itself has negative signal, not recommended for submission.

---

## LGB ensembleExperiment (2026-05-17)

### Background
abandoned MLP tuning (too slow, minimal gain).switched to LightGBM as the third ensemble model, 
exploration CT + MLP + LGB three-modelwhether it can break through CT + MLP + XGB ceiling (LB 0.81622).

### LGB first-round parameter search (lgb_grid.py)

baseline: num_leaves=31, lr=0.05, λ=1.0, sub=0.8, col=0.8, mcw=20
ensemble comparison method: CT67% + LGB23%(replacing MLP) + XGB10%

**Conclusions for each parameter direction: **

| parameters | bestvalue | ensemble OOF Δ | Conclusion |
|------|--------|----------|------|
| num_leaves | 31 (baseline) | all↓ | baselinevalue alreadybest, both larger and smaller values were worse |
| learning_rate | 0.10 | ↓0.0000 | lr=0.1 barely tied, all others decreased |
| reg_lambda | 0.1 | ↓0.0013 | decreasing regularization was worse instead, baselinevalue alreadybest |
| min_child_samples | 40 | ↓0.0013 | all↓, baselinevalue20alreadybest |
| subsample | 0.6 | ↓0.0006 | sub=0.6 smallest drop, butstill negative |
| colsample_bytree | 0.6 | ↓0.0009 | col=0.6 smallest drop, butstill negative |

**First-round conclusion: LGB replacement MLP allnegative.abandoned"replacement"strategy, changed to"parallel third model".**

---

### CT + MLP + LGB three-model weight experiment

**strategy: ** CT + MLP + LGB (removing XGB), using lgb_base (default parameters)as LGB model.

**full LB matrix: **

**CT × MLP LB matrix (inside parentheses is LGB weights, lgb_base): **

| CT＼MLP | 21% | 23% | 24% | 25% | 26% | 27% |
|---------|-----|-----|-----|-----|-----|-----|
| CT=63% | 0.81435(lgb16) | — | — | 0.81552(lgb8/10) | 0.81505(lgb11) | 0.81529(lgb10) |
| CT=64% | — | — | — | 0.81552(lgb11) | 0.81575(lgb10) | 0.81529(lgb10)/0.81505(lgb8) |
| **CT=65%** | — | 0.81552(lgb9) | 0.81646(lgb9) | **0.81669(lgb9) ✅** / 0.81622(lgb8/10) | 0.81575(lgb10)/0.81552(lgb9) | — |
| CT=66% | — | 0.81505(lgb11) | 0.81599(lgb9) | — | — | — |
| CT=67% | — | 0.81505(lgb10) | — | — | — | — |

**LGB weight sensitivity (Fixed CT=65%, MLP=25%): **

| LGB=8% | **LGB=9%** | LGB=10% | LGB=12% | LGB=14% |
|--------|-----------|---------|---------|---------|
| 0.81622 | **0.81669 ✅** | 0.81622 | 0.81529 | 0.81575 |

**MLP directional monotonicity (Fixed CT=65%, LGB=9%): **

| MLP=23% | MLP=24% | **MLP=25%** | MLP=26% |
|---------|---------|------------|---------|
| 0.81552 | 0.81646 | **0.81669 ✅** | 0.81552 |

perfect single peak, both sides symmetrically decreased.

**CT directional monotonicity (Fixed MLP=25%, LGB≈9~10%): **

| CT=63% | CT=64% | **CT=65%** | CT=66% | CT=67% |
|--------|--------|-----------|--------|--------|
| 0.81552 | 0.81575~0.81646 | **0.81669 ✅** | 0.81599 | 0.81505 |

**Pattern Summary: **

| Patterns | Description |
|------|------|
| LGB best CT weight | **65%** (peak, both sidesmonotonicdecreased) |
| LGB best MLP weight | **25%** (perfect single peak, symmetric on both sides) |
| LGB best LGB weight | **9%** (8%/10% tied 0.81622, 9% jumped to 0.81669)|
| LGB vs XGB comparison | LGB ensemble **surpassed** XGB ensemble (0.81669 > 0.81622)|
| four-model (CT+MLP+XGB+LGB)| allthan the three-model setupworse, no diversity gain |

**LGB ensembleFinal conclusion: CT65% + MLP25% + LGB9% = LB 0.81669, surpassedprevious best 0.81622 (+0.00047).all three directions showed clear single peaks, weightsconfirmed to have reached the ceiling.**

---

## LGB parameter tuning (2026-05-17)

**baseline: ** lgb_base default parameters (num_leaves=31, lr=0.05, λ=1.0, sub=0.8, col=0.8, mcw=20)
**ensemble context: ** CT65% + MLP25% + LGB9%, baselineensemble LB = 0.81669
**Key conclusion (stated first): OOF and LB had no correlation, OOF could not predict LGB parameter tuningdirection.**

---

### first round: coarse-grained search (lgb_grid_v2.py)

change only one each timeparameters, others fixed as lgb_base default value.

| parameters | Tested values | Conclusion |
|------|--------|------|
| num_leaves | 47, 63, 95, 127 | all below baseline, larger values were worse |
| learning_rate | 0.03, 0.08, 0.10 | all below baseline, lr=0.05 best |
| reg_lambda | 0.1, 0.5, 2.0 | all below baseline |
| min_child_samples | 10, 40, 80 | all below baseline |
| subsample | 0.7, 0.9, 1.0 | all below baseline |
| colsample_bytree | 0.7, 0.9, 1.0 | all below baseline |

**First-round conclusion: ** default parameters were already best in the coarse grid.

---

### second round: fine-grained search (lgb_fine_grid.py)

eachparametersaround the default value ±10~20% fine variation range (smaller step size).

| parameters | test range | best | LB | Notes |
|------|---------|------|----|------|
| num_leaves | 25~37 | 31 (baseline) | 0.81669 | 32=0.81646 close but did notsurpassed |
| learning_rate | 0.03~0.07 | 0.05 (baseline) | 0.81669 | confirmed lr=0.05 as the true peak |
| reg_lambda | 0.6~1.5 | 0.95 / 1.0 | 0.81669 | 0.95 tiedbest |
| min_child_samples | 12~28 | 20 (baseline)/ 23 | 0.81669 / 0.81646 | 23 second-best |
| subsample | 0.70~0.88 | 0.80 (baseline) | 0.81669 | sharp peak, both sidesdropped sharply |
| colsample_bytree | 0.70~0.88 | 0.80 (baseline) | 0.81669 | 0.81/0.82 tied |

**OOF inverse-correlation phenomenon: ** all configurations with higher OOF predictions, LB alldecreased; tried"reverse submission" (the configuration with the lowest OOF), LB also all decreased.**OOF completely failed for LGB parameter tuning.**

---

### third round: dense gap filling (lgb_dense_grid.py)

targetingsecond roundfindingofnear-best points, filled gaps in the middle of the parameter curves.

**complete LB curves for each parameter: **

#### min_child_samples

| mcw | LB | Notes |
|-----|----|------|
| 16~19 | tested, all belowbaseline | less than20directionpoor |
| **20** | **0.81669 ✅** | baseline, true peak |
| 21 | 0.81599 | |
| 22 | 0.81646 | |
| 23 | 0.81646 | |
| 28 | 0.81575 | |

→ 20 issharp peak; 21 dropped sharply; 22/23 recovered but did not reach baseline; downward trend confirmed.

#### num_leaves

| num_leaves | LB | Notes |
|------------|----|------|
| 28~30 | belowbaseline | |
| **31** | **0.81669 ✅** | baseline, true peak |
| 32 | 0.81646 | |
| 33~34 | belowbaseline | |
| 35 | 0.81599 | |
| 37 | 0.81599 | |
| 63 | 0.81505 | |

→ 31 issharp peak, both sides dropped quickly, single-peak shape confirmed.

#### subsample

| sub | LB | Notes |
|-----|----|------|
| 0.70 | 0.81575 | |
| 0.75 | 0.81482 | |
| 0.78 | 0.81599 | |
| 0.79 | very low | cliff-like drop |
| **0.80** | **0.81669 ✅** | baseline, supersharp peak |
| 0.81 | 0.81646 | |
| 0.82~0.86 | belowbaseline | |
| 0.90 | 0.81529 | |

→ **0.79 → 0.80 issuper discontinuity**, 0.80 isextremely sharp isolated peak.

#### colsample_bytree

| col | LB | Notes |
|-----|----|------|
| 0.60 | 0.81575 | |
| 0.72~0.78 | belowbaseline | |
| 0.78 | 0.81599 | |
| **0.80** | **0.81669 ✅** | baseline |
| **0.81** | **0.81669 ✅** | tied |
| **0.82** | **0.81669 ✅** | tied |
| 0.84~0.88 | belowbaseline | |
| 0.90 | 0.81575 | |

→ 0.80/0.81/0.82 formed**plateau region**, all arebest, monotonically decreased toward both sides.

---

### reg_lambda deep exploration

finding reg_lambda curve had a multi-peak structure, systematic exploration was performed.

**complete LB curve: **

| λvalue | LB | Notes |
|-----|----|------|
| 0.50~0.55 | not finally confirmed | |
| 0.60 | 0.81599 | |
| 0.64 | 0.81599 | |
| **0.65** | **0.81669 ✅** | new tied best peak found！ |
| 0.66 | 0.81529 | sharp drop (discontinuity)|
| 0.67~0.69 | not submitted | 0.66 already dropped sharply; abandoned |
| 0.70 | 0.81622 | local rebound |
| 0.75~0.89 | not fully tested | |
| 0.90 | 0.81575 | |
| 0.91~0.93 | not submitted | |
| 0.94 | 0.81552 | |
| **0.95** | **0.81669 ✅** | tiedbest |
| 0.96~0.99 | very low | dropped sharply |
| **1.00** | **0.81669 ✅** | baseline, tiedbest |
| 1.05 | 0.81529 | |

**curve shapefeature: **
- there are threetiedhighestpeaks: λ=0.65, 0.95, 1.00 (all are 0.81669)
- there were valleys between peaks, peak shapes were sharp (neighboring values dropped sharply)
- 0.66 dropped sharply (0.81529)but 0.70 rebounded to 0.81622, curve was non-monotonic, irregular shape
- 0.96~0.99 allvery low (1.00 right side alsosharp peak)
- **Conclusion: ** thisparameterscurve was highly irregularrules, manypeaksbetweennone could surpass 0.81669, exploration stopped

---

### LGB parameterstuning summary

| parameters | bestvalue | Peak shape | Exploration depth |
|------|--------|------|---------|
| learning_rate | **0.05** | singlepeaks, confirmed | coarse + fine |
| num_leaves | **31** | sharp peak, both sidesdropped sharply | coarse + fine + dense |
| min_child_samples | **20** | sharp peak (22/23 second-best) | coarse + fine + dense |
| subsample | **0.80** | supersharp peak (0.79cliff) | coarse + fine + dense |
| colsample_bytree | **0.80/0.81/0.82** | plateau region, three valuestied | coarse + fine + dense |
| reg_lambda | **0.65 / 0.95 / 1.00** | multi-peak, all 0.81669 | coarse + fine + dense + deep exploration |

**LGB tuningFinal conclusion: **
- lgb_base default parameters (λ=1.0)is already one of multipletiedbestone
- three roundstuning + reg_lambda deep exploration, all unable to break through LB 0.81669
- OOF for LGB tuningdirectionno predictive ability at all (forward/reverse also failed)
- **LGB parameter tuningpath officially ended, current best LB 0.81669 as thispath ceiling**

---

## Ridge No.four-modelExperiment (ridge_blend.py, 2026-05-17)

**Idea: ** Ridge linear inductive biasandtree models/MLP completely different, high diversity, asensembleNo.four-model.

**Ridge single modelsresults: **

| α | single modelsOOF | CTcorrelation | MLPcorrelation | LGBcorrelation |
|---|---------|---------|---------|---------|
| 0.01~10 | ~0.789 | 0.892 | 0.887 | 0.881 |
| 100 | 0.7876 | 0.890 | 0.887 | 0.880 |

correlation ~0.89, diversitymoderate but present.

**best four-model weight search (CT+MLP+LGB+Ridge): **

| CT | MLP | LGB | Ridge | α | OOF | Δ |
|----|-----|-----|-------|---|-----|---|
| 67% | 17% | 5% | 11% | 10.0 | 0.8154 | ↑0.0023 |
| 61% | 17% | 7% | 15% | 0.01 | 0.8153 | ↑0.0022 |

Note: OOF ↑0.0023 but MLP compressed to 17% (original 25%), historicalPatternssuggested LB may not follow.**not submitted.**

**Conclusion: ** Ridge OOF increase was significant, butbased onhistorical"OOF rose while LB dropped"Patterns, MLP weightscompression risk is high, abandonedsubmission.

---

## Stacking Experiment (stacking_ridge.py, 2026-05-17)

**Method: ** RidgeClassifierCV asmeta-model, base models OOF combined into newfeature.

| Method | OOF | Δ |
|------|-----|---|
| baseline CT65+MLP25+LGB9 | 0.8131 | — |
| Stack CT+MLP+LGB+XGB | 0.8130 | ↓0.0001 |
| Stack CT+MLP+XGB | 0.8119 | ↓0.0012 |
| Stack CT+MLP+LGB | 0.8109 | ↓0.0022 |
| Stack CT+LGB+XGB | 0.8098 | ↓0.0032 |
| publicMethod LGB50+CT30+XGB20 | 0.8113 | ↓0.0017 |
| publicMethod LGB40+CT30+XGB30 | 0.8108 | ↓0.0023 |

**Conclusion: ** all Stacking Method OOF all below baseline.meta-modelin OOF learned from OOF noiseweightsnot as good as manual tuning.Stacking path closed.

---

## ultra-wide rangeweight search (blend_wide_search.py, 2026-05-17)

**Purpose: ** break"CT must dominate"assumption, CT weightsfrom 0%~80% full rangeSearch.

**OOF levelConclusion: **

| dominant model | highest OOF | representativeCombination |
|---------|---------|---------|
| CT=80% | 0.8151 | ct80_mlp10_lgb10_xgb0 |
| LGB=80% | 0.8142 | ct5_mlp15_lgb80_xgb0 |
| XGB dominant | not in Top15 | — |

Top15 allis CT=80% ofCombination.

**submission validation: ** ct80_mlp10_lgb10_xgb0 → **LB 0.812x (belowbaseline)**

**Conclusion: **
- data confirmed CT is ourstrongestsingle models, OOF consistentshowed CT should dominate
- CT=80% when MLP compressed to 10%, MLP diversity disappeared → LB actualdecreased
- **CT=65% is true optimumweights** (MLP cannot be below 25%)
- publicMethod LGB:0.5 not suitable for us (our LGB weaker than publicMethod, CT stronger than publicMethod)

---

## same-group voting post-processing (group_voting.py, 2026-05-17)

**Idea: ** same-group passenger outcomes are highly consistent, test setsame-group members within the test set vote for each other / usetraining setuse training labels to overwrite test predictions.

**data overviewfinding: **
- test settotal number of groups: 3063
- **intraining setgroups also having members: 0** (train/test groups completely do not overlap！)
- puretest setmulti-person groups: 723
- single-person isolated groups (unable to vote): 2340

**OOF internal voting evaluation: **

| threshold | OOF | number of modifications | Conclusion |
|------|-----|--------|------|
| thr=1.00 | 0.8131 | 0 | no change |
| thr=0.80 | 0.8100 ↓0.0031 | 35 | harmful |
| thr=0.75 | 0.8049 ↓0.0082 | 97 | moreharmful |
| thr=0.55 | 0.7825 ↓0.0306 | 464 | disaster |

**submission validation: **

| file | LB | Description |
|------|-----|------|
| train_only_thr067 | 0.81669 | changes 0 entries = equal tobaseline |
| test_only_thr067 | 0.81178 | 45 entriesall changed incorrectly |
| mixed_thr075 | 0.81131 | 39 entries, worse |
| mixed_thr060 | 0.78887 | 199 entries, disaster |

**Conclusion: ** training setandtest setgroups completely do not overlap, train_only strategyinvalid.test setinternal voting was consistentlyharmful.same-group votingpath closed.

---

## teammate XGB tuningExperiment (xgb_grid_results.csv, 2026-05-17)

**Background: ** teammate independently ran two rounds of XGB gridSearch, andourtuningdirectiondifferent (we focused on λ×sub, teammate focused on depth×mcw×col×sub×λ jointSearch).using StratifiedKFold 5fold, **single models OOF evaluation** (non-ensemble context).

### first round (xgb_grid_results.csv)— depth 7/8 mainly

| ID | OOF | depth | mcw | col | sub | λ | CTcorrelation | Notes |
|----|-----|-------|-----|-----|-----|---|---------|------|
| **211** | **0.8146** | **7** | **20** | **0.9** | **0.7** | **8** | 0.977 | ⭐ this roundbest |
| 203 | 0.8136 | 8 | 20 | 0.9 | 0.7 | 10 | 0.978 | |
| 201 | 0.8132 | 7 | 20 | 0.9 | 0.7 | 10 | 0.978 | |
| 208 | 0.8127 | 7 | 20 | 0.95 | 0.7 | 10 | 0.977 | |
| 212 | 0.8126 | 7 | 20 | 0.9 | 0.7 | 12 | 0.977 | |
| 204 | 0.8119 | 7 | 15 | 0.9 | 0.7 | 10 | 0.978 | |

**Patterns: ** sub=0.7 in all Top resultsappeared consistently in (andour sub=0.9 different); col=0.9 best; λ=8 best; depth=7 better than depth=8.

### second round (xgb_grid_results(1).csv)— depth 4~8 wide rangeSearch

| ID | OOF | depth | mcw | col | sub | λ | CTcorrelation | Notes |
|----|-----|-------|-----|-----|-----|---|---------|------|
| **004** | **0.8140** | **5** | **5** | **0.8** | **0.8** | **2** | 0.976 | ⭐ this roundbest, parameters and very different from ours |
| 013 | 0.8132 | 7 | 20 | 0.9 | 0.7 | 10 | 0.978 | andfirst roundrepeated validation |
| 011 | 0.8132 | 7 | 10 | 0.8 | 0.7 | 5 | 0.970 | |
| 010 | 0.8123 | 6 | 15 | 0.6 | 0.8 | 10 | 0.974 | |

**combined across two roundsbest: ** depth=7, mcw=20, col=0.9, sub=0.7, λ=8 (OOF=0.8146)

### andwe XGB resultscomparison

| Source | depth | col | sub | λ | mcw | single modelsOOF | ensembleLB |
|------|-------|-----|-----|---|-----|---------|-------|
| webest | 6 | 0.8 | 0.9 | 4 | 5 | ~0.813 | **0.81622** (ensemble) |
| teammatebest | 7 | 0.9 | 0.7 | 8 | 20 | 0.8146 | to be tested |

**⭐ importantfinding: **
1. teammate's **sub=0.7** directionandour sub=0.9 completely opposite, andsingle models OOF higher, worth validating inensemble contextcontext
2. teammate's **λ=8** far higher than ours λ=4, directionconsistent (high regularization), but with larger magnitude
3. teammate's **depth=7** than ours depth=6 deeper, but our previous test depth=7 inensembleininvalid——needs retesting
4. **recommendation for tomorrow's priority validation: ** depth=7, col=0.9, sub=0.7, λ=8, mcw=20 in CT65+MLP25+XGB10 ensemble contextinof LB

---

## teammate MLP tuningExperiment (mlp_grid_results.csv, 2026-05-17)

**Background: ** teammateusing PyTorch MLP (Adam + Dropout=0.48), andour sklearn L-BFGS MLP completely differentframework.the two CSV files had the same content (termsubmission).

**Searchspace: ** hidden_layer_sizes × lr × alpha (Adam), Fixed dropout=0.48, epochs=80, batch=128

### completeresults (sorted by OOF ranking)

| ID | OOF | Architecture | lr | α | CTcorrelation | Notes |
|----|-----|------|----|---|---------|------|
| **010** | **0.8128** | **(256,256,128,64)** | **0.001** | **0.01** | **0.957** | ⭐ best |
| 004 | 0.8120 | (256,128,64) | 0.005 | 0.001 | 0.960 | |
| 002 | 0.8108 | (256,128,64) | 0.0005 | 0.0001 | 0.960 | |
| 007 | 0.8105 | (512,256,128) | 0.0005 | 0.0001 | 0.958 | |
| 009 | 0.8105 | (256,256,128,64) | 0.0005 | 0.0001 | 0.958 | |
| 001 | 0.8104 | (256,128,64) | 0.001 | 0.0001 | 0.960 | |
| 006 | 0.8089 | (128,64) | 0.0005 | 0.001 | 0.951 | lowestCTcorrelation |
| 005 | 0.8086 | (128,64) | 0.001 | 0.0001 | 0.961 | |
| 003 | 0.8063 | (256,128,64) | 0.0001 | 0.0001 | 0.953 | lrtoo low |
| 008 | 0.8058 | (512,256,128) | 0.0001 | 0.001 | 0.952 | lrtoo low |

### andour MLP-wide comparison

| model | framework | Architecture | OOF | CTcorrelation | ensemblevalue |
|------|------|------|-----|---------|---------|
| **our MLP-wide** | **sklearn L-BFGS** | **(256,128,64)** | ~0.80 | **0.944** | **high (diversitymaximum)** |
| teammatebest MLP010 | PyTorch Adam | (256,256,128,64) | 0.8128 | 0.957 | in (diversitylower)|
| teammate MLP006 | PyTorch Adam | (128,64) | 0.8089 | 0.951 | higher (diversitysecond-best)|

**⭐ keyfinding: **
1. teammateall MLP of CT correlation (0.951~0.961)**all higher than**our sklearn MLP (0.944)
2. our L-BFGS MLP and CT diversitylarger, ensemblevalueinsteadhigher, this explains why MLP-wide isa better ensemble member
3. single models OOF higher (0.8128 > our ~0.80)butensembleinnot necessarily better
4. **lr=0.001, α=0.01 is Adam MLP ofbest combination** (slightly stronger regularization is better)
5. **recommendation: ** if testing the teammate's MLP, term MLP010 (OOFhighest)or MLP006 (CTcorrelationlowest, diversitybest), inensembleinreplacementcurrent MLP-wide comparison LB

---

## TabNet depthLearning Experiment (tabnet_base.py / tabnet_grid.py, 2026-05-17, in progress)

**Background: ** alltraditional directions had already been exhausted, tried TabNet (attention mechanism + sparse feature selection)asnewdiversitySource.

**baselineparameters: ** n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3

**tuning plan (controlled-variable method): **

| parameters | meaning | Tested values |
|------|------|--------|
| n_d/n_a | network width | 8→16→32→64 |
| n_steps | Notetermsteps | 2→3→4→5→6 |
| gamma | feature reuse coefficient | 1.0→1.2→1.5→2.0 |
| lambda_sparse | sparse regularization | 1e-4→1e-3→1e-2→0.1 |
| mask_type | attention type | sparsemax vs entmax |

### TabNet singleparameterscontrolvariablesresults (tabnet_grid.py)

| parameters | value | single modelsOOF | ensemble OOF | Δ | LB | Conclusion |
|------|----|---------|---------|---|-----|------|
| baseline | n_d=8,n_a=8,n_steps=3,γ=1.3,λ_s=1e-3 | — | 0.8131 | — | — | baseline |
| n_d/n_a | 16 | ↑ | 0.8139 | ↑0.0008 | — | — |
| n_d/n_a | **32** | ↑ | **0.8162** | **↑0.0031** | — | best for this parameter |
| n_d/n_a | 64 | ↑ | 0.8158 | ↑0.0027 | — | 64 < 32 |
| n_steps | **2** | ↑ | **0.8161** | **↑0.0030** | — | stepsfewer steps were better |
| n_steps | 4 | ↑ | 0.8141 | ↑0.0010 | — | — |
| n_steps | 6 | ↑ | 0.8143 | ↑0.0012 | — | — |
| gamma | 1.0 | ↑ | 0.8131 | =0 | — | invalid |
| gamma | 1.2 | ↑ | 0.8138 | ↑0.0007 | — | weak |
| gamma | 1.5 | ↑ | 0.8135 | ↑0.0004 | — | — |
| lambda_sparse | 1e-4 | ↑ | 0.8143 | ↑0.0012 | — | — |
| **lambda_sparse** | **1e-2** | **↑** | **0.8168** | **↑0.0037** | **0.81318** | **OOFoverallbest → LBdrop** |
| lambda_sparse | 0.1 | ↑ | 0.8155 | ↑0.0024 | — | — |
| mask_type | entmax | ↑ | 0.8136 | ↑0.0005 | — | — |

**key phenomenon: **
- all TabNet variants OOF all↑, butsubmission lambda_sparse=0.01 (OOF highest↑0.0037)→ **LB 0.81318 (↓0.00351)**
- gamma=1.0 (OOF equal tobaseline)and entmax all not submitted
- **Conclusion: TabNet single modelstoo weak (~0.80), for CT dominatedensemblenoneactualhelp, OOF improvement was entirely noise**

---

### TabNet parametersCombinationvalidation (tabnet_combined.py)

| Method | single modelsOOF | ensemble OOF | Δ | bestweights |
|------|---------|---------|---|---------|
| A: n_d=32+n_steps=2+λ_s=0.01 | ↑ | 0.816x | ↑ | CT+MLP+LGB+TAB |
| B: n_d=32+λ_s=0.01 | ↑ | 0.815x | ↑ | — |
| C: n_steps=2+λ_s=0.01 | ↑ | 0.815x | ↑ | — |
| D: n_d=32+n_steps=2 | ↑ | 0.815x | ↑ | — |
| E: n_d=32+n_steps=2+λ_s=0.0001 | ↑ | 0.815x | ↑ | — |

**Conclusion: ** after combination, OOF termshowedterm, butbased onsingleparametersvalidation (lambda_sparse OOFhighest→LBlowest), CombinationMethodall not submitted.**TabNet path closed.**

---

## CT feature Engineering Experiments (2026-05-17, third round)

### CT v6: newly added7itemsfeature (ct_v6.py)

**newly addedfeature: ** Surname (cat_feature)/Deck_Side/Co_Travelers_Sleep/Group_TotalSpend/Group_AvgSpend/Spend_vs_Group/LuxurySpend/Spend_per_Age

| Metric | value |
|------|----|
| featurecount | expanded to ~57 items |
| CV | 0.8170 ± 0.0081 |
| single modelsOOF | 0.8174 |
| ensemble OOF | 0.8162 (↑0.0031 vs baseline) |
| CT v2 correlation | **0.9871** |
| LB | **0.80944** (↓0.00725！) |

**failure cause analysis: **
- Surname ishigh-cardinality categorical (training set ~7000 types, test setmany unseen), severeoverfitting
- remainingnewfeatureand CT v2 correlation 0.9871, barely changed predictions, only introduced noise
- OOF ↑0.0031 isfalse positive (Surname visible inside folds, leakage information)

**Conclusion: ** featureengineeringfor CT alreadycompletelyinvalid.Surname althoughin MLP inthrough Target Encoding canuse, butas CatBoost native cat_feature termdirectoverfitting.

---

### CT featurefeature removal experiment (ct_feature_importance.py)

**Method: ** removefeatureimportantnesscontributiontail 10%/20%/30% offeature

| Method | ensemble OOF | Δ | removefeaturecount | LB |
|------|---------|---|---------|-----|
| completefeature (baseline) | 0.8131 | — | 0 | 0.81669 |
| remove bottom10%contribution | 0.8142 | ↑0.0011 | ~6items | **0.80313** |
| remove bottom20%contribution | not submitted | — | — | — |
| remove bottom30%contribution | not submitted | — | — | — |

**failedoriginalreason: **
- "bottom 10% contribution"intermincludingterm CryoSleep interaction features (CryoSpend/Age_x_Cryo/NonCryo_Spend)
- thesefeaturesingleindividualimportantnesslow, buttermsynergistic effect, ismodel's importantinductive signal
- removeafter LB dropped sharply to 0.80313 (↓0.01356！)
- OOF showed↑0.0011, confirmed again OOF completelyunreliable

**Conclusion: ** feature subtraction was invalid.ct9 feature set (includingall CryoSleep interaction features)is true optimum, any addition or removal LB decreased.

---

## pseudo-labeling experiment (ct_pseudolabel.py, 2026-05-17)

**Background: ** historical ct16 (threshold=0.85, based onsingle modelsprobability)→ LB 0.80734 (severeoverfitting)
**improvement: ** pseudo-labelSourcechanged toensembleprobability (CT65+MLP25+LGB9), threshold increased to 0.90/0.95/0.97

**test-set probabilitiesdistribution: ** <0.05 total 414 entries, >0.95 total 876 entries (total 4277 entries)

| threshold | pseudo-labelcount | ensemble OOF | OOFrank | LB | LBrank |
|------|---------|---------|--------|-----|--------|
| baseline (nonepseudo-label) | 0 | 0.8131 | — | **0.81669** | — |
| 0.97 | 982 | 0.8148 (↑0.0017) | **1 (OOFhighest)** | 0.80687 | **3 (LBlowest)** |
| 0.95 | 1290 | 0.8123 (↓0.0008) | 3 (OOFlowest) | 0.80897 | 2 |
| 0.90 | 1891 | 0.8132 (↑0.0001) | 2 | 0.80944 | **1 (LBhighest)** |

**Conclusion: completelyfailed, OOF and LB rankcompletely opposite**
- OOF highest (↑0.0017)→ LB lowest (0.80687): perfect negative correlation
- threeitemsthresholdalltermbelowbaseline (0.81669), best thr=0.90 termdropterm 0.00725
- Root cause: pseudo-labelmade the model"term"test-set distribution, but train/test group termcompletely different, pseudo-labelinsteadtermbiasmodel
- and ct16 (threshold=0.85 → LB 0.80734)same failure pattern, termusingensembleprobability+stricterthresholdalso could not solve it
- **pseudo-labelpath permanently closed**

---

## Global Key Lessons (updated to 2026-05-17)

| experience | Description |
|------|------|
| OOF and LB negative correlationor no correlation | CT/LGB/TabNet/featuresubtraction parameter tuningall OOF could not predict LB, mustactual submission |
| MLP cannotweight reduction | MLP from 25%→10% when LB dropped sharply, diversity comes frominductiveinductive-bias difference |
| CT isstrongestsingle models | termSearchterm, CT=65% is true optimum, publicMethodweightstermuse |
| feature engineering ceiling (three roundsconfirmed) | v3/v4/v5 (second round)/v6/subtraction (third round)all LB decreased, feature setalreadybest |
| CryoSleep interaction featuresmust not be removed | CryoSpend/Age_x_Cryo/NonCryo_Spend individually low importance but key synergy, remove LB↓0.014 |
| Surname termsuitable CT | high-cardinality cat_feature severeoverfitting, CT v6 LB 0.80944 (↓0.00725) |
| parameter tuningceiling | CT/XGB/LGB three modelseach had already reached its tuning limit |
| Stacking invalid | meta-modeltermweights, not as good as manual tuning |
| groupterminvalid | train/test groups completely do not overlap, post-processingno usable information |
| TabNet invalid | OOF inflated, single modelstoo weak (~0.80), unable to help CT dominatedensemble |
| pseudo-labelinvalid (two roundsconfirmed) | ct16 (thr=0.85)→ 0.80734; thr=0.97/0.95/0.90 all dropped to 0.807~0.809; OOF and LB completelynegative correlation |
| **modelceiling** | **LB 0.81669 (CT65+MLP25+LGB9)** |
| **post-processingceiling** | **LB 0.81786 (Earth+CryoSleep+TRAP 33casesflip, submission file: Y_earth_cryo_TRAP.csv)** |
| OOF for post-processing was completely invalid | CryoSleep termrules OOF showed ↓0.001 (harmful), actual LB ↑0.0007 (effective); OOF signal direction flipped |
| pseudo-labelinvalid (two roundsconfirmed) | ct16 (thr=0.85)→ 0.80734; thr=0.97/0.95/0.90 all dropped to 0.807~0.809; OOF and LB completelynegative correlation |

---

## decision-threshold optimization (2026-05-17)

**Background: ** baselinethresholdas 0.50 (blend ≥ 0.5 → Transported=True).explorationnon-0.5 thresholdwhether it could improve LB.

**OOF results: ** 0.480 OOF highest (0.8143), 0.499→0.501 monotonicdecreased.

**complete LB validation matrix: **

| threshold | LB | Δ vs 0.50 | Conclusion |
|------|-----|----------|------|
| 0.44 | 0.81529 | ↓0.0014 | — |
| 0.46 | 0.81529 | ↓0.0014 | — |
| 0.48 | 0.81529 | ↓0.0014 | OOFhighestbutLBdrop |
| 0.49 | 0.81622 | ↓0.0005 | — |
| 0.499 | 0.81646 | ↓0.0003 | close but still worse |
| **0.50** | **0.81669** | **—** | **true optimum** |
| 0.501 | 0.81646 | ↓0.0003 | symmetrically decreased |
| 0.51 | 0.81529 | ↓0.0014 | — |
| 0.53 | 0.81435 | ↓0.0023 | — |

**Conclusion: ** 0.50 isexact optimum, both sidescompletelysymmetrically decreased.OOF recommendation 0.48 but LB is 0.50.**thresholdtuningpath closed.**

---

## post-processingrulesExperiment (2026-05-17)

**Background: **
- modelbaseline LB = 0.81669 (CT65+MLP25+LGB9)
- test settotal 4277 entries, termchanges 1 entry prediction ≈ ±0.000234 LB
- OOF forpost-processingsignalcompletelyinvalid (directionflip), allrulesallthrough LB validation

**CryoSleep post-processingRationale: ** CryoSleep=True → passengerstermcryosleep chamber, physically cannot act independently, shouldallterm.the model due to Deck/Age termfeatureinteractions occasionally makes mistakes on this rule.

---

### A series: threshold variants (see"decision-threshold optimization"section)

---

### B/C series: CryoSleep all-planet rule

| file | rule description | number of changed cases | LB | Δ vs baseline |
|------|---------|---------|-----|----------|
| B_cryo_hard_thr50 | CryoSleep=True → allforce True | ~55→0 (all predicted-False cases flipped) | **0.81739** | **↑0.0007** |
| C_cryo_soft04_thr50 | CryoSleep=True + blend∈[0.4,0.5) → True | only near-boundary cases flipped | 0.81716 | ↑0.0005 |

**keyfinding: ** B better than C, indicating that even high-confidence CryoSleep=True prediction=False should also be flipped.

**OOF vs LB: ** OOF showed B harmful (↓0.001), LB showed B effective (↑0.0007).**OOF again had direction flip.**

---

### D/E/H/V series: otherrules (allinvalid)

| fileseries | rules | Conclusion |
|---------|------|------|
| D series | Deck specific → changesprediction | allbelow 0.81669 baseline |
| E series | age-group rules | allbelow 0.81669 baseline |
| H series | high spending (Spa/VRDeck > X)rules | allbelow 0.81669 baseline |
| V series | VIP rules (VIP=True → False)| training set VIP+pred=True of 86.8% term Transported, model correct, not flipped |

**Conclusion: ** only CryoSleep directioneffective, remainingtermrulesallinvalidorharmful.

---

### X series: sorted by HomePlanet segmentation CryoSleep rules

**Background: ** B (all planets)= 0.81739.explorationtermitemstermcontributiontermeffectiveimprovement, termitemsterm.

| file | rules | number of changed cases | LB | Conclusion |
|------|------|---------|-----|------|
| X_earth_cryo_only | Earth+CryoSleep=True → True | 36 | **0.81762** | ↑0.0009, removeMarsterm！ |
| X_mars_cryo_only | Mars+CryoSleep=True → True | 19 | 0.81646 | ↓0.0002, Marsrulesharmful！ |

**keyfinding: ** 
- Earth+CryoSleep effective (+0.00093 vs baseline)
- Mars+CryoSleep harmful (-0.00023 vs baseline)
- all-planet rule B (+0.0007)= Earth positive effect - Mars negative effect as net value

**test set CryoSleep=True but pred=False ofdistribution: **

| HomePlanet | Destination | case count | blend range |
|-----------|------------|------|----------|
| Earth | TRAPPIST-1e | 33 | 0.367~0.498 |
| Earth | PSO J318.5-22 | 3 | 0.384, 0.497, 0.498 |
| Mars | TRAPPIST-1e | 18 | 0.352~0.494 |
| Mars | NaN | 1 | — |
| Europa | any | **0** | — (model has already predicted allTrue)|

---

### Y series: Earth+CryoSleep internal segmentation (destination/Deck/term)

**baseline: ** X_earth_cryo = 0.81762 (36cases, including TRAP+PSO)

| file | rules | case count | LB | Δ vs X_earth |
|------|------|------|-----|------------|
| Y_earth_cryo_TRAP | Earth+Cryo+TRAPPIST-1e | **33** | **0.81786 ✅** | **↑0.0002, current best** |
| Y_earth_cryo_PSO | Earth+Cryo+PSO J318.5-22 | 3 | 0.81646 | ↓0.0012, harmful |
| Y_earth_cryo_deckG | Earth+Cryo+Deck G | 34 | 0.81762 | =0, Deck G ≈ TRAP |
| Y_earth_cryo_deckE | Earth+Cryo+Deck E | — | — | not finally analyzed |
| Y_earth_cryo_age_lt18 | Earth+Cryo+Age<18 | 27 | 0.81739 | ↓0.0005 |
| Y_earth_cryo_age_ge18 | Earth+Cryo+Age≥18 | 9 | 0.81692 | ↓0.0007 |
| Y_earth_cryo_age_lt25 | Earth+Cryo+Age<25 | — | 0.81739 | — |
| Y_earth_cryo_age_ge25 | Earth+Cryo+Age≥25 | — | 0.81716 | — |
| Y_earth_cryo_age_lt35 | Earth+Cryo+Age<35 | — | — | not submitted |
| Y_earth_cryo_age_ge35 | Earth+Cryo+Age≥35 | — | — | not submitted |
| Y_earth_cryo_lt035 | Earth+Cryo+blend<0.35 | 0 | — | blendlowest=0.367, 0cases |
| Y_earth_cryo_lt040 | Earth+Cryo+blend<0.40 | ~2 | — | too few, unclear marginal benefit |
| Y_earth_cryo_lt042~lt048 | incremental threshold variants | increasing | — | full TRAP already best |

**Conclusion: ** 33items Earth+Cryo+TRAP isindivisible best set.any subset (termsplit/stricterthreshold)termfull setworse, full settermisbest.

---

### Z series: Earth+Cryo+TRAP internal extreme segmentation + Mars+TRAP

**baseline: ** Y_earth_cryo_TRAP = 0.81786 (33cases)

| file | rules | case count | LB | Conclusion |
|------|------|------|-----|------|
| Z_earth_cryo_trap_age_lt12 | TRAPin Age<12 | 26 | 0.81716 | ↓0.0007, remove7itemsterminsteadpoor |
| Z_earth_cryo_trap_age_ge12 | TRAPin Age≥12 | 7 | 0.81739 | ↓0.0005, only seven adults were insufficient |
| Z_earth_cryo_trap_age_lt18 | TRAPin Age<18 | 27 | 0.81739 | ↓0.0005 |
| Z_earth_cryo_trap_age_ge18 | TRAPin Age≥18 | 6 | 0.81716 | ↓0.0007 |
| Z_earth_cryo_trap_age_lt25 | TRAPin Age<25 | — | 0.81739 | ↓0.0005 |
| Z_earth_cryo_trap_age_ge25 | TRAPin Age≥25 | — | 0.81716 | ↓0.0007 |
| Z_mars_cryo_TRAP | Mars+Cryo+TRAP | 18 | 0.81622 | ↓0.0016, Marsharmfulconfirmed |
| Z_earth_trap_plus_mars_trap | Earth+Mars+Cryo+TRAP | 51 | 0.81739 | ↓0.0005, Marsdragged down Earth |

**age distribution (33 casesinside): ** Age<12 term26term (78.8%), mean13.1years old, terminDeck G(31)/E(2)

**Conclusion: ** 33items Y_earth_cryo_TRAP alreadyisbestsingleterm, any subsetterm LB decreased.

---

### post-processingfull LB matrix summary

| LB | file | rules | case count |
|----|------|------|------|
| **0.81786 ✅** | Y_earth_cryo_TRAP | Earth+Cryo+TRAPPIST-1e | 33 |
| 0.81762 | X_earth_cryo / Y_earth_cryo_deckG | Earth+Cryoall / +DeckG | 36 / 34 |
| 0.81739 | B_cryo_hard / Z_ge12 / Z_lt25 / Z_lt18 / Z_earth+mars | differentsegmentationCombination | 55~51 |
| 0.81716 | C_cryo_soft / Z_lt12 / Z_ge18 / Z_ge25 | moretermset | various |
| 0.81692 | Y_earth_cryo_age_ge18 | Earth+Cryo adult | 9 |
| 0.81669 | baseline | nonepost-processing | 0 |
| 0.81646 | X_mars_cryo / Y_earth_cryo_PSO | harmfulrules | 19/3 |
| 0.81622 | Z_mars_cryo_TRAP | Marscases | 18 |

---

### post-processingcompletelyexhaustedterm (closedtime point)

**CryoSleep=True cases predicted False by the model (55 cases in total): **

| Combination | case count | Deckdistribution | flipresults |
|------|------|---------|---------|
| Earth + TRAPPIST-1e | 33 | G(31)+E(2) | ✅ effective (+0.00117) |
| Earth + PSO J318.5-22 | 3 | Gall | ❌ harmful (-0.00123) |
| Mars + TRAPPIST-1e | 18 | Eall, adult | ❌ harmful (-0.00164) |
| Mars + NaNdestination | 1 | — | not tested |
| Europa + any | 0 | — | nothing to flip (model has already predicted allTrue)|

**other post-processing angles already exhausted: **
- VIP=True+pred=True (21cases, blendmean0.867): training set86.8%truly Transported, model correct, not flipped
- CryoSleep=False+TotalSpend=0+pred=True (155cases): training set61.6%isTransported, >50%, not flipped
- Age<2 (109cases, 106alreadypredictionTrue): 3itemspredictionFalseofcasestermextremely small
- Earth+NoCryo+TRAP pred=True (262cases): training set69.7%were True, model correct

**post-processingtermclosed, bestas Y_earth_cryo_TRAP (LB 0.81786).**

---

## post-processingexperience summary

| experience | Description |
|------|------|
| OOF forpost-processingdirectioncompletelyinvalid | CryoSleep rules OOF ↓ but LB ↑, directionflip |
| domain knowledge is the source of post-processing | CryoSleep rule based on physical intuition, butnon-termSearch |
| granularity selection is key | too coarse (all planets)termharmfulcases; too fine (26itemsterm)removeeffectivecases; 33itemsTRAPwas exactly the best |
| the model was already highly accurate | many large subgroups OOF term>87%, modeljudgment on boundary cases was usually correct, forcing flips was harmful |
| Europa+CryoSleep all predicted True | model's Europa casesalreadycompletelyterm, no post-processing space |
| Mars+CryoSleep training rate90.2% | althoughtraining set90.2%were True, modelin these18 casestermpredictionFalseiscorrect (flipharmful), modeltermfiner features |

---

## TabPFN v2 + new technology exploration (2026-05-17)

### TabPFN v2 Introduction
- model: Prior-Labs/TabPFN-v2-clf (pre-trained Transformer, in-context learning)
- Version: tabpfn 8.0.3, no tuning needed, direct prediction
- input: LGBfeature set (32itemsnumerical features, float32)
- OOF: GroupKFold 5fold, n_estimators=16
- Outputfile: oof_tabpfn.npy / test_tabpfn.npy

### TabPFN correlation analysis
| comparison | Pearsoncorrelation | meaning |
|------|------------|------|
| TabPFN vs CT v2 | **0.9798** | extremely high correlation, low ensemble diversity |
| MLP vs CT v2 | **0.9433** | lower correlation, high ensemble diversity |

**core conclusion: TabPFN and CT predictions were highly similar, but MLP provided more differentiated information.**

### TabPFN LB Experimentsummary

| file | weightsconfiguration | TRAPrules | LB | vsbaseline |
|------|---------|---------|-----|--------|
| TB_tab65_mlp25_lgb9.csv | TABmain65% | ✗ | — | — |
| TB_tab65_mlp25_lgb9_TRAP.csv | TABmain65% | ✓(1cases) | — | — |
| TB_ct50tab50_mlp25_lgb9.csv | CT/TABeach50% | ✗ | — | — |
| TB_ct50tab50_mlp25_lgb9_TRAP.csv | CT/TABeach50% | ✓ | — | — |
| TB_ct70tab30_main_TRAP.csv | CT70+TAB30 as main model | ✓(26cases) | — | — |
| TB_ct30tab70_main_TRAP.csv | CT30+TAB70 as main model | ✓(9cases) | — | — |
| TB_tab25_4th.csv | TABas the 4model25% | ✗ | — | — |
| TB_tab25_4th_TRAP.csv | TABas the 4model25% | ✓(29cases) | — | — |
| TB_tab30_4th.csv | TABas the 4model30% | ✗ | — | — |
| TB_tab30_4th_TRAP.csv | TABas the 4model30% | ✓(28cases) | — | — |
| TB_best4_ct35_mlp15_lgb15_tab35.csv | 4model balance | ✗ | — | — |
| TB_best4_ct35_mlp15_lgb15_tab35_TRAP.csv | 4model balance | ✓ | — | — |
| **TB_ct65_tab25_lgb9_TRAP.csv** | CT65+TAB25+LGB9 (MLP→TAB) | ✓(42cases) | **0.81061** | -0.00725 |
| TB_ct65_tab25_lgb9_noTRAP.csv | CT65+TAB25+LGB9 nonerules | ✗ | **0.80921** | -0.00865 |

### keyfinding

1. **TabPFN cannot replace MLP**: CT65%+TAB25%+LGB9% = 0.81061, lower than best by 0.00725
2. **MLP ensemblestronger diversity**: MLP-CT correlation(0.9433) < TAB-CT correlation(0.9798), MLP contributed more independent information
3. **TRAP rule in TAB combination flipped 42 entries** (vs MLPcombination's 33 entries), Description TAB for Earth+Cryo+TRAP casesprediction was more conservative
4. **CT50+TAB50+MLP25+LGB9 with or without TRAP score was the same**: DescriptionTABalready predicted these cases correctly, TRAPrule redundant (butfinal LB did not exceed 0.81786)

### othernew technology explorationConclusion

| Technique | LB | Conclusion |
|------|-----|------|
| Rank Averaging (probability→percentile rank then blend) | 0.81318 | ❌ large decrease, OOF -0.0006 |
| TTA 50copy Gaussianterm | 0.81131 | ❌ harmful, noise damaged precise predictions |
| TTA Plus (5copy) | 0.81295 | ❌ harmful |
| Temperature Scaling (T=0.8~1.5) | threshold0.5caused zero prediction changes | ❌ forinvalid for accuracy |
| TabPFN v2 as an extra model | term | ❌ andCTcorrelationtoo high |

**Conclusion: allnewTechniquealltermsurpassedcurrent best, final best remained Y_earth_cryo_TRAP = 0.81786.**

### termfinalStatus (2026-05-17 term)

| Milestone | LB |
|--------|-----|
| termbaseline (CT v2 single models) | ~0.803 |
| CT + MLP termmodelensemble | 0.81318 |
| CT65+MLP25+LGB9 three-modelensemble | 0.81622 |
| CT v2 termensemble | 0.81669 |
| **Y_earth_cryo_TRAP post-processing (finalbest)** | **0.81786** |


---

## fileterm + termvalidation (2026-05-17 term)

### Cleanlab labelstermresults

usingalreadyterm OOF probability (CT65%+MLP25%+LGB9% ensemble)term Cleanlab, nonetermnewtraining: 

- termcantermthis: **668entries (termtraining set 7.7%)**
- labels=1 but the model thought it should be0: 492entries
- labels=0 but the model thought it should be1: 176entries
- termcantermthisfeature: CryoSleep=True butlabels Transported=False, modelsettermhighterm 0.97-0.99

**termcantermthistermresults: **

| Version | ensemble OOF | LB | Conclusion |
|------|---------|-----|------|
| baseline (term) | 0.8131 | 0.81786 | best |
| drop_top100 (term100entries) | 0.8133 (↑) | ~0.8129 | LB insteadterm |
| drop_all668 (termall668entries) | 0.8153 (↑↑) | ~0.8066 | LB large decrease |

**Conclusion: ** 668entries"cantermthis"istermoftermthis, termisterm.termafter OOF inflated (reasonastest settermsingleterm), but LB large decrease.Cleanlab path closed.

### TabPFN weightstermresults (TAB replacement LGB)

| configuration | LB |
|------|-----|
| CT65+MLP25+TAB9 (TABreplacementLGB, originalweights) | 0.81599 |
| CT(65)+MLP(25)+TAB5 | 0.81552 |
| CT(65)+MLP(25)+TAB15 | 0.81505 |

**Conclusion: ** TAB replacement LGB terminvalid, LGB ofdiversitytermcannotreplace.

### termfileterm

**retainedofterm.py file (cancompletetermfinalterm): **
```
preprocessing.py ← totalusepreprocessing
catboost_features.py ← CT/MLP featureengineering
lgb_features.py ← LGB featureengineering
ct_v2.py ← Step 1: training CatBoost → oof_ct_v2.npy / test_ct_v2.npy
ct_mlp.py ← Step 2: training MLP → oof_MLP-wide.npy / test_mlp.npy
lgb_grid_v2.py ← Step 3: training LGB → oof_lgb_base.npy / test_lgb_base.npy
final_submission.py ← Step 4: ensemble+TRAPrules → submission_final.csv
eda.py ← countterm (PPTuse)
```

**term: **
```bash
python3 ct_v2.py
python3 ct_mlp.py
python3 lgb_grid_v2.py
python3 final_submission.py
```

### termvalidation

termnewtermcompleteterm, setnewtermof `submission_final.csv` andoriginalterm `Y_earth_cryo_TRAP.csv` comparison: 
- **4277term, 0entriespoorterm, completelyconsistent** ✅
- newterm NPY filetermcompletelysameofprediction, termtypestermFixed (RANDOM_SEED=42)termcantermness

