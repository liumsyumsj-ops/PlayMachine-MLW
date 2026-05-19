"""
统一预处理模块 — Spaceship Titanic
====================================
整合了四份预处理方案的共同步骤，并在各模型中保留差异化处理。

各方案来源：
  - RF / CatBoost：你的方案
  - SVM：队员方案（需要 StandardScaler）
  - LR/飞的方案（preprocessingofly）：姓氏特征、固定AgeGroup分箱
  - KNN：队员方案（需要 One-Hot + StandardScaler）

未采纳的特征工程详见文件末尾的备注说明。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ============================================================
# 0. 读取数据
# ============================================================
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


# ============================================================
# 1. 核心预处理（所有模型共用）
# ============================================================
def core_preprocess(train, test):
    """
    所有模型都需要的基础预处理步骤。
    输出：合并处理后的 train_clean 和 test_clean（含全部数值特征）。
    类别特征保留为字符串，由各模型自行编码。
    """
    train = train.copy(); test = test.copy()
    train['is_train'] = 1; test['is_train'] = 0; test['Transported'] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    # ----------------------------------------------------------
    # Step 1：结构化特征拆分
    # ----------------------------------------------------------
    # PassengerId → GroupId + GroupSize（RF/KNN/LR/CatBoost 均有）
    df['GroupId']   = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')

    # Cabin → Deck / CabinNum / Side（四份方案均有）
    df[['Deck','CabinNum','Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    # Surname（姓氏）→ 来自 LR 方案，作为家族代理特征
    df['Surname'] = df['Name'].fillna('Unknown Unknown').str.split().str[-1]

    # ----------------------------------------------------------
    # Step 2：缺失值填充（逻辑驱动，优先利用数据规律）
    # ----------------------------------------------------------

    # Step 2A：同组互补（RF/CatBoost 方案）
    # 同组乘客 HomePlanet / Destination / Cabin 几乎一致
    for col in ['HomePlanet','Destination','Cabin']:
        df[col] = df.groupby('GroupId')[col].transform(
            lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x)
    # Cabin 补全后重新拆分
    df[['Deck','CabinNum','Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    # Step 2B：CryoSleep ↔ 消费双向推断（RF/CatBoost 方案）
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    cryo_true = df['CryoSleep'] == True
    for col in spend_cols: df.loc[cryo_true & df[col].isna(), col] = 0.0
    has_spend = df[spend_cols].gt(0).any(axis=1)
    df.loc[has_spend & df['CryoSleep'].isna(), 'CryoSleep'] = False

    # Step 2C：HomePlanet → Deck 映射填充（RF/CatBoost 方案）
    deck_map = {'Europa':'B','Earth':'G','Mars':'F'}
    mask = df['Deck'].isna() & df['HomePlanet'].notna()
    df.loc[mask,'Deck'] = df.loc[mask,'HomePlanet'].map(deck_map)

    # Step 2D：剩余缺失统计兜底
    # Deck/Side：按 HomePlanet × CryoSleep 双维度分组众数
    df['CryoSleep_int'] = df['CryoSleep'].map(
        {True:1, False:0, 'True':1, 'False':0}).fillna(0).astype(int)
    for col in ['Deck','Side']:
        df[col] = df.groupby(['HomePlanet','CryoSleep_int'])[col].transform(
            lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x)
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.drop(columns=['CryoSleep_int'])

    # CabinNum：按 Deck 分组中位数
    df['CabinNum'] = df.groupby('Deck')['CabinNum'].transform(
        lambda x: x.fillna(x.median()))

    # 消费缺失填 0（SVM/KNN/LR 方案均有）
    for col in spend_cols: df[col] = df[col].fillna(0.0)

    # Age：全局中位数（四份方案均有）
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 布尔型转 int
    df['CryoSleep']   = df['CryoSleep'].fillna(False).astype(int)
    df['VIP']         = df['VIP'].fillna(False).astype(int)
    df['HomePlanet']  = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['Surname']     = df['Surname'].fillna('Unknown')

    # ----------------------------------------------------------
    # Step 3：特征工程（RF/CatBoost 方案为主，LR 方案补充）
    # ----------------------------------------------------------

    # 消费类（RF/CatBoost + LR 方案）
    df['TotalSpend']     = df[spend_cols].sum(axis=1)
    df['HasSpent']       = (df['TotalSpend'] > 0).astype(int)       # RF
    df['NoSpending']     = (df['TotalSpend'] == 0).astype(int)       # LR（与 HasSpent 互补）
    for col in spend_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpend'] + 1)       # RF
    for col in spend_cols + ['TotalSpend']:
        df[f'{col}_log']   = np.log1p(df[col])                      # RF
    df['MaxSpend']       = df[spend_cols].max(axis=1)                # RF
    df['MaxSpend_log']   = np.log1p(df['MaxSpend'])                  # RF
    df['HighSpa']        = (df['Spa'] > 1000).astype(int)            # RF
    spend_log_cols = ['RoomService_log','FoodCourt_log','ShoppingMall_log','Spa_log','VRDeck_log']
    df['SpendDiversity'] = (df[spend_log_cols] > 0).sum(axis=1)     # RF

    # 两种年龄分箱
    df['AgeGroup_qcut'] = pd.qcut(
        df['Age'], q=4, labels=['Young','MidYoung','MidOld','Senior'])  # RF（四分位）
    df['AgeGroup_cut']  = pd.cut(
        df['Age'], bins=[0,12,18,25,40,60,100], labels=False).fillna(0).astype(int)  # LR（固定分箱）

    # 组特征
    df['IsAlone']        = (df['GroupSize'] == 1).astype(int)
    df['GroupBin_solo']  = (df['GroupSize'] == 1).astype(int)
    df['GroupBin_small'] = ((df['GroupSize'] >= 2) & (df['GroupSize'] <= 4)).astype(int)
    df['GroupBin_large'] = (df['GroupSize'] >= 5).astype(int)

    # 同组冷冻比例（RF 方案）
    group_cryo = df.groupby('GroupId')['CryoSleep'].mean().rename('GroupCryoRate')
    df = df.join(group_cryo, on='GroupId')
    df['AllGroupCryo'] = (df['GroupCryoRate'] == 1.0).astype(int)
    df['NoGroupCryo']  = (df['GroupCryoRate'] == 0.0).astype(int)
    df = df.drop(columns=['GroupCryoRate'])

    # 特殊乘客标记
    df['IsChild']        = (df['Age'] <= 12).astype(int)
    df['AwakeZeroSpend'] = ((df['CryoSleep'] == 0) & (df['TotalSpend'] == 0)).astype(int)
    df['Side_bin']       = (df['Side'] == 'S').astype(int)

    # 交互特征（RF 方案）
    df['CryoSpend']     = df['CryoSleep'] * df['TotalSpend_log']
    df['Age_x_Cryo']    = df['Age'] * df['CryoSleep']
    df['VIP_x_Spend']   = df['VIP'] * df['TotalSpend_log']
    df['NonCryo_Spend'] = (1 - df['CryoSleep']) * df['TotalSpend_log']

    # 高阶交互特征（你的RF方案）
    if 'HomePlanet_Europa' not in df.columns:  # 还未 one-hot，用原始列计算
        df['Europa_x_Cryo']  = (df['HomePlanet'] == 'Europa').astype(int) * df['CryoSleep']
        df['Mars_x_Cryo']    = (df['HomePlanet'] == 'Mars').astype(int) * df['CryoSleep']
        df['VIP_x_Mars']     = df['VIP'] * (df['HomePlanet'] == 'Mars').astype(int)
        df['Child_x_Mars']   = df['IsChild'] * (df['HomePlanet'] == 'Mars').astype(int)
        df['Child_x_Europa'] = df['IsChild'] * (df['HomePlanet'] == 'Europa').astype(int)

    # Deck × Side 交互（你的RF方案，用原始列计算）
    df['DeckB_x_Side'] = (df['Deck'] == 'B').astype(int) * df['Side_bin']
    df['DeckC_x_Side'] = (df['Deck'] == 'C').astype(int) * df['Side_bin']
    df['DeckG_x_Side'] = (df['Deck'] == 'G').astype(int) * df['Side_bin']

    # MLP 方案新增：Is_Sleeping（总消费为0，含冷冻乘客）
    # 与 AwakeZeroSpend 的区别：Is_Sleeping 包含冷冻乘客，AwakeZeroSpend 只含清醒零消费
    df['Is_Sleeping'] = (df['TotalSpend'] == 0).astype(int)   # MLP 方案

    # ----------------------------------------------------------
    # Step 4：删除原始无用列
    # ----------------------------------------------------------
    df = df.drop(columns=['PassengerId','Name','Cabin','GroupId'])

    # ----------------------------------------------------------
    # Step 5：拆回 train / test
    # ----------------------------------------------------------
    train_c = df[df['is_train']==1].drop(columns=['is_train']).copy()
    test_c  = df[df['is_train']==0].drop(columns=['is_train','Transported']).copy()
    train_c['Transported'] = train_c['Transported'].astype(int)

    return train_c, test_c


# ============================================================
# 2. 模型专用编码（各模型调用）
# ============================================================

def encode_for_rf(train, test):
    """
    随机森林版编码：
    - 类别特征 one-hot
    - 使用 AgeGroup_qcut
    - 不需要标准化
    """
    train = train.copy(); test = test.copy()
    drop_cols = ['Deck_T'] if 'Deck_T' in train.columns else []
    drop_cols += ['Destination_PSO J318.5-22'] if 'Destination_PSO J318.5-22' in train.columns else []

    cat_cols = ['HomePlanet','Destination','Deck','AgeGroup_qcut']
    train = pd.get_dummies(train, columns=cat_cols, drop_first=False)
    test  = pd.get_dummies(test,  columns=cat_cols, drop_first=False)
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    # 删除 CatBoost 专用列和低价值列
    drop = ['AgeGroup_cut','Surname','Side'] + drop_cols
    train = train.drop(columns=drop, errors='ignore')
    test  = test.drop(columns=drop,  errors='ignore')

    X = train.drop(columns=['Transported'])
    y = train['Transported']
    # align(join='left') 会把 Transported 列对齐到 test，需要 drop 掉
    test = test.drop(columns=['Transported'], errors='ignore')
    return X, y, test


def encode_for_catboost(train, test):
    """
    CatBoost 版编码：
    - 类别特征保留字符串，不做 one-hot
    - 使用 AgeGroup_qcut
    """
    train = train.copy(); test = test.copy()
    cat_features = ['HomePlanet','Destination','Deck','Side']

    for col in cat_features:
        train[col] = train[col].astype(str)
        test[col]  = test[col].astype(str)

    drop = ['AgeGroup_cut','AgeGroup_qcut','Surname']
    train = train.drop(columns=drop, errors='ignore')
    test  = test.drop(columns=drop,  errors='ignore')

    X = train.drop(columns=['Transported'])
    y = train['Transported']
    return X, y, test, cat_features


def encode_for_svm(train, test):
    """
    SVM 版编码：
    - LabelEncoder 编码类别特征
    - StandardScaler 标准化
    - 使用 AgeGroup_cut（固定分箱，来自 LR 方案）
    """
    train = train.copy(); test = test.copy()
    cat_cols = ['HomePlanet','Destination','Deck','Side','Surname']

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    drop = ['AgeGroup_qcut']
    train = train.drop(columns=drop, errors='ignore')
    test  = test.drop(columns=drop,  errors='ignore')

    X = train.drop(columns=['Transported'])
    y = train['Transported']

    scaler = StandardScaler()
    X_scaled      = pd.DataFrame(scaler.fit_transform(X),      columns=X.columns)
    test_scaled   = pd.DataFrame(scaler.transform(test),        columns=test.columns)
    return X_scaled, y, test_scaled


def encode_for_knn(train, test):
    """
    KNN 版编码：
    - One-Hot 编码类别特征
    - StandardScaler 标准化
    - Surname 维度过高（2218种）不做 one-hot，用 LabelEncoder 代替
    """
    train = train.copy(); test = test.copy()
    onehot_cols = ['HomePlanet','Destination','Deck','Side']
    label_cols  = ['Surname']

    drop = ['AgeGroup_cut','AgeGroup_qcut']
    train = train.drop(columns=drop, errors='ignore')
    test  = test.drop(columns=drop,  errors='ignore')

    # One-Hot
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc_train = encoder.fit_transform(train[onehot_cols].astype(str))
    enc_test  = encoder.transform(test[onehot_cols].astype(str))
    enc_cols  = encoder.get_feature_names_out(onehot_cols)

    # LabelEncoder for Surname
    for col in label_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    train = train.drop(columns=onehot_cols).reset_index(drop=True)
    test  = test.drop(columns=onehot_cols).reset_index(drop=True)
    train = pd.concat([train, pd.DataFrame(enc_train, columns=enc_cols)], axis=1)
    test  = pd.concat([test,  pd.DataFrame(enc_test,  columns=enc_cols)], axis=1)

    X = train.drop(columns=['Transported'])
    y = train['Transported']

    scaler = StandardScaler()
    X_scaled    = pd.DataFrame(scaler.fit_transform(X),    columns=X.columns)
    test_scaled = pd.DataFrame(scaler.transform(test),     columns=test.columns)
    return X_scaled, y, test_scaled


def encode_for_mlp(train, test):
    """
    MLP 版编码 (已修复 Embedding 兼容性问题)：
    - LabelEncoder 编码类别特征（含 Surname）
    - StandardScaler 仅对数值特征标准化！绝对不能动类别特征！
    - 保留 Is_Sleeping / Total_Spend_Log
    - 使用 AgeGroup_cut（固定分箱）
    """
    train = train.copy();
    test = test.copy()
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Surname']

    # 1. 类别特征 LabelEncoding
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    drop = ['AgeGroup_qcut']
    train = train.drop(columns=drop, errors='ignore')
    test = test.drop(columns=drop, errors='ignore')

    X = train.drop(columns=['Transported'])
    y = train['Transported']

    # 2. 区分数值列和类别列
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 3. ⚠️ 核心修复：只对数值列进行 StandardScaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    test_scaled = test.copy()

    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    test_scaled[num_cols] = scaler.transform(test[num_cols])

    # 将 cat_cols 和 num_cols 一起返回，方便下游拆解双流输入
    return X_scaled, y, test_scaled, cat_cols, num_cols


# ============================================================
# 3. 使用示例
# ============================================================
if __name__ == '__main__':
    # 修改为你的路径
    train_raw, test_raw = load_data(
        "/Users/cheeseowl/mlw/spaceship-titanic/train.csv",
        "/Users/cheeseowl/mlw/spaceship-titanic/test.csv"
    )

    # 核心预处理（一次性，所有模型共用）
    train_clean, test_clean = core_preprocess(train_raw, test_raw)
    print(f"核心预处理完成：train={train_clean.shape}, test={test_clean.shape}")

    # 各模型调用对应的编码函数
    X_rf,  y_rf,  test_rf               = encode_for_rf(train_clean, test_clean)
    X_cb,  y_cb,  test_cb, cat_features = encode_for_catboost(train_clean, test_clean)
    X_svm, y_svm, test_svm              = encode_for_svm(train_clean, test_clean)
    X_knn, y_knn, test_knn              = encode_for_knn(train_clean, test_clean)
    X_mlp, y_mlp, test_mlp              = encode_for_mlp(train_clean, test_clean)

    print(f"RF特征矩阵:       {X_rf.shape}")
    print(f"CatBoost特征矩阵: {X_cb.shape}")
    print(f"SVM特征矩阵:      {X_svm.shape}")
    print(f"KNN特征矩阵:      {X_knn.shape}")
    print(f"MLP特征矩阵:      {X_mlp.shape}")

# 生成处理后的文件
    train_clean.to_csv("train_processed.csv", index=False)
    test_clean.to_csv("test_processed.csv", index=False)
    print("文件已成功导出为 train_processed.csv 和 test_processed.csv")


# ============================================================
# 备注：各模型未采纳的特征工程
# ============================================================
"""
【SVM 方案 — 未采纳的步骤】
1. Cabin缺失填充 'Z/9999/Z'：本方案改为同组互补+映射推断，逻辑更严谨。
2. 所有类别特征直接众数填充：本方案优先用同组互补和 CryoSleep↔消费互推,更准确。

【LR/飞的方案(preprocessingofly)— 未采纳的步骤】
1. 类别特征填 'Unknown' 后做 LabelEncoder:本方案改为分组众数填充，
   避免 'Unknown' 成为干扰类别。
2. train/test 分开处理(iloc 拆分）：本方案合并处理，保证统计量一致。

【KNN 方案 — 未采纳的步骤】
1. train 和 test 分开计算中位数：本方案合并后统一计算。
2. 没有 TotalSpend / 消费特征工程：本方案已补充。
3. 没有 CryoSleep↔消费互推:本方案已补充。

【MLP 方案 — 未采纳的步骤】
1. Cabin缺失填充 'Unknown/Unknown/Unknown'：本方案改为同组互补+映射推断。
2. 类别特征填 'Unknown'：本方案改为分组众数填充。
3. Cabin_Num 作为类别特征：本方案将 CabinNum 作为数值特征，更合理。

【RF/CatBoost 方案 — 已全部纳入统一预处理】
所有特征均已包含在 core_preprocess 中：
  ✓ 同组互补 / CryoSleep↔消费互推 / HomePlanet→Deck 映射
  ✓ TotalSpend / HasSpent / _ratio / _log / MaxSpend
  ✓ IsAlone / IsChild / AwakeZeroSpend / Is_Sleeping
  ✓ GroupBin / AllGroupCryo / NoGroupCryo
  ✓ CryoSpend / Age_x_Cryo / VIP_x_Spend / NonCryo_Spend
  ✓ Europa_x_Cryo / Mars_x_Cryo / VIP_x_Mars / Child_x_Mars / Child_x_Europa
  ✓ DeckB/C/G_x_Side / HighSpa / SpendDiversity
  ✓ AgeGroup_qcut(四分位)/ AgeGroup_cut(固定分箱)
  ✓ Surname(姓氏)
"""