# Spaceship Titanic — Kaggle 竞赛项目说明

## 项目简介

Spaceship Titanic 是 Kaggle 上的二分类竞赛任务。目标是预测乘客是否在星际事故中被传送（Transported），训练集 8693 条，测试集 4277 条，评价指标为准确率（Accuracy）。

**最终成绩：LB 0.81786**

---

## 文件结构

```
mlw/
├── spaceship-titanic/
│   ├── train.csv                  ← 训练数据
│   ├── test.csv                   ← 测试数据
│   └── submission_final.csv       ← 最终提交文件（LB 0.81786）
│
├── preprocessing.py               ← 共用数据预处理（所有模型公用）
│
├── 单模型特征工程
│   ├── knn_features.py            ← KNN 专用：OHE + StandardScaler + PCA
│   ├── svm_features.py            ← SVM 专用：LabelEncoder + StandardScaler + SelectKBest
│   ├── rf_features.py             ← RF 专用：CabinRegion + SurnameFreq + OHE
│   ├── mlp_features.py            ← MLP 专用：LabelEncoder（Embedding）+ StandardScaler
│   ├── lgb_features.py            ← LightGBM 专用：OHE（复用 RF 风格）
│   └── catboost_features.py       ← CatBoost / 集成 MLP 专用特征工程
│
├── 单模型脚本
│   ├── KNN.py                     ← KNN 单模型训练与提交
│   ├── svm.py                     ← SVM 单模型训练与提交
│   ├── random_forest3(2).py       ← Random Forest 单模型训练与提交
│   ├── mlp_1.4.py                 ← MLP 单模型训练与提交（PyTorch，Optuna 调参）
│   └── LightGBM(1).py             ← LightGBM 单模型训练与提交
│
├── 集成流水线
│   ├── ct_v2.py                   ← Step 1：训练 CatBoost 模型（生成 OOF + 测试集概率）
│   ├── ct_mlp.py                  ← Step 2：训练 MLP 模型（生成 OOF + 测试集概率）
│   ├── lgb_grid_v2.py             ← Step 3：训练 LightGBM 模型（生成 OOF + 测试集概率）
│   └── final_submission.py        ← Step 4：集成 + 后处理 → 生成提交文件
│
├── oof_ct_v2.npy                  ← CatBoost OOF 预测概率
├── oof_MLP-wide.npy               ← MLP OOF 预测概率
├── oof_lgb_base.npy               ← LightGBM OOF 预测概率
├── test_ct_v2.npy                 ← CatBoost 测试集预测概率
├── test_mlp.npy                   ← MLP 测试集预测概率
├── test_lgb_base.npy              ← LightGBM 测试集预测概率
│
├── eda.py                         ← 探索性数据分析
├── experiments_log.md             ← 完整实验记录
└── references_IEEE.md             ← 参考文献（IEEE 格式）
```

---

## 单模型说明

### KNN（`KNN.py` + `knn_features.py`）

K 近邻分类器，对量纲敏感，需先标准化再降维。

| 项目 | 内容 |
|------|------|
| 特征编码 | OHE（类别列）+ StandardScaler |
| 降维 | PCA，保留 95% 方差（22 个主成分） |
| 超参搜索 | GridSearchCV，搜索 k、距离度量、权重 |
| 交叉验证 | StratifiedKFold 5 折 |

```bash
python3 KNN.py
```

---

### SVM（`svm.py` + `svm_features.py`）

支持向量机分类器，使用 RBF 核，对量纲敏感，需标准化。

| 项目 | 内容 |
|------|------|
| 特征编码 | LabelEncoder（类别列含 Surname）+ StandardScaler |
| 特征选择 | SelectKBest（f_classif），保留 23 个特征 |
| 超参搜索 | GridSearchCV，搜索 C、gamma |
| 交叉验证 | StratifiedKFold 5 折 |

```bash
python3 svm.py
```

---

### Random Forest（`random_forest3(2).py` + `rf_features.py`）

随机森林分类器，树模型对量纲不敏感，无需标准化。

| 项目 | 内容 |
|------|------|
| 新增特征 | CabinRegion（CabinNum // 300）、SurnameFreq（家族规模代理） |
| 特征编码 | OHE（类别列），不做 StandardScaler |
| 特征数 | 38 个 |
| 交叉验证 | GroupKFold 5 折（按 GroupId 分组，防止家庭数据泄漏） |

```bash
python3 "random_forest3(2).py"
```

---

### MLP（`mlp_1.4.py` + `mlp_features.py`）

基于 PyTorch 的多层感知机，使用 Entity Embedding 将类别特征映射为低维稠密向量，并加入残差连接。

| 项目 | 内容 |
|------|------|
| 数值特征 | 18 列，StandardScaler 标准化 |
| 类别特征 | 5 列（HomePlanet, Destination, Deck, Side, Surname），LabelEncoder → Embedding 层 |
| 模型结构 | Embedding + Pre-layer(128) + ResBlock(128×2) + Post-layer(64→1) |
| Dropout | 0.485（Optuna 调参最优值） |
| 损失函数 | BCEWithLogitsLoss |
| 交叉验证 | StratifiedKFold 5 折 |

```bash
python3 mlp_1.4.py
```

---

### LightGBM 单模型（`LightGBM(1).py` + `lgb_features.py`）

LightGBM 梯度提升树单模型，复用 RF 风格特征编码。

| 项目 | 内容 |
|------|------|
| 特征编码 | OHE（类别列），不做 StandardScaler |
| 主要参数 | n_estimators=1000, lr=0.01, max_depth=8, num_leaves=64, subsample=0.8 |
| 交叉验证 | StratifiedKFold 5 折 |

```bash
python3 "LightGBM(1).py"
```

---

## 集成模型说明

### 运行方式

按顺序执行以下脚本，每步会生成对应的 `.npy` 文件：

```bash
python3 ct_v2.py           # 约 10-15 分钟
python3 ct_mlp.py          # 约 5-10 分钟
python3 lgb_grid_v2.py     # 约 10-15 分钟
python3 final_submission.py
```

最终输出：`spaceship-titanic/submission_final.csv`

---

### 特征工程

- **缺失值处理**：数值列用中位数填充，类别列用组内众数填充；CryoSleep 与消费额相互推断
- **消费特征**：TotalSpend、NecessitiesSpend（房间 + 食物）、EntertainmentSpend（购物 + SPA + VR）
- **群组特征**：GroupSize、IsAlone、AllGroupCryo（全组冷冻）、NoGroupCryo（全组未冷冻）
- **其他**：CabinSize（同舱人数）、IsChild（年龄 ≤ 12）、Surname

### 模型与集成权重

| 模型 | 关键参数 | 集成权重 |
|------|---------|---------|
| CatBoost (ct_v2) | lr=0.03, depth=6, l2=2, iter=3000 | 65% |
| MLP (ct_mlp) | 见 MLP 单模型说明 | 25% |
| LightGBM (lgb_grid_v2) | lr=0.05, num_leaves=31, λ=1.0 | 9% |

所有模型均使用 **GroupKFold 5 折交叉验证**（按 GroupId 分组，防止同一家庭数据泄漏）。

加权平均融合：

```
blend = CT × (65/99) + MLP × (25/99) + LGB × (9/99)
```

集成 OOF 准确率：**0.8131**

### 后处理规则

训练集中 **Earth 星球 + CryoSleep=True + 目的地 TRAPPIST-1e** 的乘客几乎全部被传送，但模型在测试集上对这 33 个案例预测为 False，强制翻转后 LB 从 0.81669 提升至 **0.81786**（+0.00117）。

---

## 实验历程（关键节点）

| 方案 | LB |
|------|----|
| CatBoost 单模型基准 | ~0.803 |
| CT + MLP 双模型集成 | 0.81318 |
| CT + MLP + LGB 三模型集成 | 0.81622 |
| CT v2 参数优化 | 0.81669 |
| **+ Earth+CryoSleep+TRAPPIST-1e 后处理（最终）** | **0.81786** |

详细实验记录见 `experiments_log.md`。

---

## 依赖库

```
catboost
lightgbm
scikit-learn
pandas
numpy
torch
scipy
```

---

## 参考文献

详见 `references_IEEE.md`，涵盖 KNN、SVM、Random Forest、MLP Entity Embedding、XGBoost、LightGBM、CatBoost、Stacking 集成及概率校准相关文献共 13 篇。
