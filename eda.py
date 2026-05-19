import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Basic Settings
# ============================================================

# blue color palette
C1 = '#eef3f8'   # very pale blue-white
C2 = '#e0eff2'   # pale ice blue
C3 = '#c0daf0'   # light blue
C4 = '#9dabd0'   # gray blue
C5 = '#b9d8f7'   # sky blue
C6 = '#90b8f1'   # medium blue
C7 = '#6182cc'   # cobalt blue
C8 = '#424d95'   # deep indigo blue

PALETTE     = [C6, C8, C5, C7, C3, C4, C2, C1]
BIN_PALETTE = [C6, C8]    # light blue=False, deep indigo=True

def cycle_colors(n):
    return (PALETTE * ((n // len(PALETTE)) + 1))[:n]

sns.set(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_safe_xlim(data, col, lower_q=0.01, upper_q=0.99):
    """
    Get a reasonable x-axis range using quantiles.
    This helps avoid extreme outliers making the useful distribution too compressed.
    """
    valid_data = data[col].dropna()

    if valid_data.empty:
        return None, None

    lower = valid_data.quantile(lower_q)
    upper = valid_data.quantile(upper_q)

    if lower == upper:
        lower = valid_data.min()
        upper = valid_data.max()

    return lower, upper
# ============================================================
# 2. Load Dataset
# ============================================================

train = pd.read_csv("spaceship-titanic/train.csv")
test  = pd.read_csv("spaceship-titanic/test.csv")

print("=" * 80)
print("Dataset Loaded Successfully")
print("=" * 80)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())

print("\nFirst 5 rows of train data:")
print(train.head())


# ============================================================
# 3. Basic Data Information
# ============================================================

print("\n" + "=" * 80)
print("Basic Information of Train Dataset")
print("=" * 80)

print(train.info())

print("\nSummary statistics of numerical columns:")
print(train.describe())

print("\nSummary statistics of categorical columns:")
print(train.describe(include="object"))


# ============================================================
# 4. Missing Value Analysis
# ============================================================

print("\n" + "=" * 80)
print("Missing Value Analysis")
print("=" * 80)

missing_train = train.isnull().sum().sort_values(ascending=False)
missing_train_percent = (train.isnull().mean() * 100).sort_values(ascending=False)

missing_summary = pd.DataFrame({
    "Missing Count": missing_train,
    "Missing Percentage (%)": missing_train_percent
})

print(missing_summary)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=missing_summary.index,
    y=missing_summary["Missing Percentage (%)"],
    palette=cycle_colors(len(missing_summary))
)
plt.xticks(rotation=45, ha="right")
plt.title("Missing Value Percentage by Column")
plt.ylabel("Missing Percentage (%)")
plt.xlabel("Columns")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_values_percentage.png", dpi=300)
plt.show()


# ============================================================
# 5. Target Variable Distribution
# ============================================================

print("\n" + "=" * 80)
print("Target Variable Distribution")
print("=" * 80)

target_counts = train["Transported"].value_counts()
target_percent = train["Transported"].value_counts(normalize=True) * 100

print(pd.DataFrame({
    "Count": target_counts,
    "Percentage (%)": target_percent
}))

plt.figure(figsize=(7, 5))
sns.countplot(data=train, x="Transported", hue="Transported", palette=BIN_PALETTE, legend=False)
plt.title("Distribution of Target Variable: Transported")
plt.xlabel("Transported")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=300)
plt.show()


# ============================================================
# 6. Feature Engineering for EDA
# ============================================================

def create_eda_features(df):
    df = df.copy()

    # Split PassengerId
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupMemberId"] = df["PassengerId"].str.split("_").str[1].astype(int)

    # Group size
    group_size = df.groupby("GroupId")["PassengerId"].transform("count")
    df["GroupSize"] = group_size
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # Split Cabin into Deck, CabinNum, Side
    df["CabinDeck"] = df["Cabin"].str.split("/").str[0]
    df["CabinNum"] = df["Cabin"].str.split("/").str[1]
    df["CabinSide"] = df["Cabin"].str.split("/").str[2]

    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    # Extract surname from Name
    df["Surname"] = df["Name"].str.split(" ").str[-1]

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    df["TotalSpend"] = df[spending_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    # Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 45, 60, 100],
        labels=["Child", "Teenager", "Young Adult", "Adult", "Middle Age", "Senior"]
    )

    return df


train_eda = create_eda_features(train)
test_eda = create_eda_features(test)

print("\n" + "=" * 80)
print("New EDA Features Created")
print("=" * 80)

new_features = [
    "GroupId", "GroupMemberId", "GroupSize", "IsAlone",
    "CabinDeck", "CabinNum", "CabinSide",
    "Surname", "TotalSpend", "NoSpend", "AgeGroup"
]

print(train_eda[new_features].head())


# ============================================================
# 7. Numerical Feature Distribution
# ============================================================

numerical_cols = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "GroupSize",
    "CabinNum"
]

print("\n" + "=" * 80)
print("Numerical Feature Distribution")
print("=" * 80)

for col in numerical_cols:
    if col in train_eda.columns:
        plt.figure(figsize=(10, 5))

        x_min, x_max = get_safe_xlim(train_eda, col)

        plot_data = train_eda[
            (train_eda[col] >= x_min) &
            (train_eda[col] <= x_max)
        ]

        sns.histplot(plot_data[col], kde=True, bins=40, color=C6, edgecolor='white', linewidth=0.3)

        plt.title(f"Distribution of {col} (1% - 99% Range)")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/distribution_{col}_trimmed.png", dpi=300)
        plt.show()


# ============================================================
# 8. Numerical Features vs Target
# ============================================================

for col in numerical_cols:
    if col in train_eda.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=train_eda, x="Transported", y=col, hue="Transported", palette=BIN_PALETTE, legend=False)
        plt.title(f"{col} vs Transported")
        plt.xlabel("Transported")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/boxplot_{col}_vs_transported.png", dpi=300)
        plt.show()


# ============================================================
# 9. Categorical Feature Distribution
# ============================================================

categorical_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "AgeGroup",
    "IsAlone",
    "NoSpend",
    "GroupSize"
]

print("\n" + "=" * 80)
print("Categorical Feature Distribution")
print("=" * 80)

for col in categorical_cols:
    if col in train_eda.columns:
        plt.figure(figsize=(10, 5))
        order = train_eda[col].value_counts().index
        sns.countplot(data=train_eda, x=col, order=order,
                      hue=col, palette=cycle_colors(len(order)), legend=False)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/countplot_{col}.png", dpi=300)
        plt.show()


# ============================================================
# 10. Categorical Features vs Target
# ============================================================

for col in categorical_cols:
    if col in train_eda.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=train_eda, x=col, hue="Transported", palette=BIN_PALETTE)
        plt.title(f"{col} vs Transported")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.legend(title="Transported")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/countplot_{col}_vs_transported.png", dpi=300)
        plt.show()


# ============================================================
# 11. Transported Rate by Categorical Features
# ============================================================

print("\n" + "=" * 80)
print("Transported Rate by Important Categorical Features")
print("=" * 80)

for col in categorical_cols:
    if col in train_eda.columns:
        transported_rate = train_eda.groupby(col)["Transported"].mean().sort_values(ascending=False)
        print(f"\nTransported rate by {col}:")
        print(transported_rate)

        plt.figure(figsize=(10, 5))
        transported_rate.plot(kind="bar", color=cycle_colors(len(transported_rate)))
        plt.title(f"Transported Rate by {col}")
        plt.xlabel(col)
        plt.ylabel("Transported Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/transported_rate_by_{col}.png", dpi=300)
        plt.show()


# ============================================================
# 12. Correlation Analysis
# ============================================================

print("\n" + "=" * 80)
print("Correlation Analysis")
print("=" * 80)

corr_df = train_eda.copy()

# Convert target to numeric
corr_df["Transported"] = corr_df["Transported"].astype(int)

# Convert boolean columns to numeric
for col in ["CryoSleep", "VIP"]:
    if col in corr_df.columns:
        corr_df[col] = corr_df[col].map({True: 1, False: 0})

selected_corr_cols = [
    "Transported",
    "Age",
    "CryoSleep",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "NoSpend",
    "GroupSize",
    "IsAlone",
    "GroupMemberId",
    "CabinNum"
]

selected_corr_cols = [col for col in selected_corr_cols if col in corr_df.columns]

correlation_matrix = corr_df[selected_corr_cols].corr()

print(correlation_matrix["Transported"].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
from matplotlib.colors import LinearSegmentedColormap
_cmap = LinearSegmentedColormap.from_list('custom_div', [C8, 'white', C6])
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=_cmap,
    square=True
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=300)
plt.show()


# ============================================================
# 13. Spending Features Analysis
# ============================================================

spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

print("\n" + "=" * 80)
print("Spending Features Analysis")
print("=" * 80)

spending_summary = train_eda[spending_cols + ["TotalSpend"]].describe()
print(spending_summary)

for col in spending_cols:
    plt.figure(figsize=(10, 5))

    x_min, x_max = get_safe_xlim(train_eda, col)

    plot_data = train_eda[
        (train_eda[col] >= x_min) &
        (train_eda[col] <= x_max)
    ]

    sns.kdeplot(
        data=plot_data,
        x=col,
        hue="Transported",
        palette=BIN_PALETTE,
        fill=True,
        common_norm=False
    )

    plt.title(f"Density Distribution of {col} by Transported (1% - 99% Range)")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kde_{col}_by_transported_trimmed.png", dpi=300)
    plt.show()


plt.figure(figsize=(10, 5))

x_min, x_max = get_safe_xlim(train_eda, "TotalSpend")

plot_data = train_eda[
    (train_eda["TotalSpend"] >= x_min) &
    (train_eda["TotalSpend"] <= x_max)
]

sns.kdeplot(
    data=plot_data,
    x="TotalSpend",
    hue="Transported",
    palette=BIN_PALETTE,
    fill=True,
    common_norm=False
)

plt.title("Density Distribution of TotalSpend by Transported (1% - 99% Range)")
plt.xlabel("TotalSpend")
plt.ylabel("Density")
plt.xlim(x_min, x_max)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/kde_TotalSpend_by_transported_trimmed.png", dpi=300)
plt.show()

# ============================================================
# Additional: Log-Scale Spending Distribution
# ============================================================

for col in spending_cols + ["TotalSpend"]:
    if col in train_eda.columns:
        temp = train_eda.copy()
        temp[f"Log_{col}"] = np.log1p(temp[col])

        plt.figure(figsize=(10, 5))
        sns.kdeplot(
            data=temp,
            x=f"Log_{col}",
            hue="Transported",
            palette=BIN_PALETTE,
            fill=True,
            common_norm=False
        )

        plt.title(f"Log-Scale Density Distribution of {col} by Transported")
        plt.xlabel(f"log1p({col})")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/log_kde_{col}_by_transported.png", dpi=300)
        plt.show()
# ============================================================
# Log Transform Before vs After Comparison (for presentation)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in zip(
    axes,
    ["TotalSpend", "Log_TotalSpend"],
    ["Before: TotalSpend (raw)", "After: TotalSpend (log1p)"]
):
    temp = train_eda.copy()
    temp["Log_TotalSpend"] = np.log1p(temp["TotalSpend"])
    plot_col = "TotalSpend" if col == "TotalSpend" else "Log_TotalSpend"
    for (label, grp), c in zip(temp.groupby("Transported"), BIN_PALETTE):
        ax.hist(grp[plot_col].dropna(), bins=50, alpha=0.7, label=str(label), color=c)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(title="Transported")

plt.suptitle("Log Transform Effect on TotalSpend Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/log_transform_comparison.png", dpi=300)
plt.show()


# ============================================================
# 14. CryoSleep and Spending Relationship
# ============================================================

print("\n" + "=" * 80)
print("CryoSleep and Spending Relationship")
print("=" * 80)

cryo_spend = train_eda.groupby("CryoSleep")[spending_cols + ["TotalSpend"]].mean()
print(cryo_spend)

plt.figure(figsize=(8, 5))
sns.boxplot(data=train_eda, x="CryoSleep", y="TotalSpend", hue="CryoSleep", palette=BIN_PALETTE, legend=False)
plt.title("TotalSpend by CryoSleep")
plt.xlabel("CryoSleep")
plt.ylabel("TotalSpend")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/totalspend_by_cryosleep.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=train_eda, x="CryoSleep", hue="Transported", palette=BIN_PALETTE)
plt.title("CryoSleep vs Transported")
plt.xlabel("CryoSleep")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cryosleep_vs_transported.png", dpi=300)
plt.show()


# ============================================================
# 15. HomePlanet, Destination, and Transported
# ============================================================

important_cat_pairs = ["HomePlanet", "Destination", "CabinDeck", "CabinSide"]

for col in important_cat_pairs:
    if col in train_eda.columns:
        cross_tab = pd.crosstab(
            train_eda[col],
            train_eda["Transported"],
            normalize="index"
        )

        print(f"\nTransported proportion by {col}:")
        print(cross_tab)

        plt.figure(figsize=(10, 5))
        cross_tab.plot(kind="bar", stacked=True, color=BIN_PALETTE)
        plt.title(f"Transported Proportion by {col}")
        plt.xlabel(col)
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/stacked_transport_by_{col}.png", dpi=300)
        plt.show()


# ============================================================
# 16. Group Size Analysis
# ============================================================

print("\n" + "=" * 80)
print("Group Size Analysis")
print("=" * 80)

group_analysis = train_eda.groupby("GroupSize")["Transported"].agg(["count", "mean"])
print(group_analysis)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=group_analysis.reset_index(),
    x="GroupSize",
    y="mean",
    hue="GroupSize",
    palette=PALETTE,
    legend=False
)
plt.title("Transported Rate by Group Size")
plt.xlabel("Group Size")
plt.ylabel("Transported Rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/transported_rate_by_group_size.png", dpi=300)
plt.show()


# ============================================================
# 17. Age Analysis
# ============================================================

print("\n" + "=" * 80)
print("Age Group Analysis")
print("=" * 80)

age_group_analysis = train_eda.groupby("AgeGroup")["Transported"].agg(["count", "mean"])
print(age_group_analysis)

plt.figure(figsize=(10, 5))
sns.boxplot(data=train_eda, x="Transported", y="Age", hue="Transported", palette=BIN_PALETTE, legend=False)
plt.title("Age Distribution by Transported")
plt.xlabel("Transported")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_by_transported.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=train_eda, x="AgeGroup", hue="Transported", palette=BIN_PALETTE)
plt.title("AgeGroup vs Transported")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/agegroup_vs_transported.png", dpi=300)
plt.show()


# ============================================================
# 18. Outlier Detection
# ============================================================

print("\n" + "=" * 80)
print("Outlier Detection using IQR Method")
print("=" * 80)

outlier_summary = {}

for col in numerical_cols:
    if col in train_eda.columns:
        Q1 = train_eda[col].quantile(0.25)
        Q3 = train_eda[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = train_eda[
            (train_eda[col] < lower_bound) |
            (train_eda[col] > upper_bound)
        ]

        outlier_summary[col] = {
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": len(outliers),
            "Outlier Percentage (%)": len(outliers) / len(train_eda) * 100
        }

outlier_summary_df = pd.DataFrame(outlier_summary).T
print(outlier_summary_df)

outlier_summary_df.to_csv(f"{OUTPUT_DIR}/outlier_summary.csv")


# ============================================================
# 19. Train vs Test Distribution Comparison
# ============================================================

print("\n" + "=" * 80)
print("Train vs Test Distribution Comparison")
print("=" * 80)

train_compare = train_eda.drop(columns=["Transported"], errors="ignore").copy()
test_compare = test_eda.copy()

train_compare["Dataset"] = "Train"
test_compare["Dataset"] = "Test"

combined = pd.concat(
    [train_compare, test_compare],
    axis=0,
    ignore_index=True
)

for col in numerical_cols:
    if col in combined.columns:
        plt.figure(figsize=(10, 5))

        x_min, x_max = get_safe_xlim(combined, col)

        plot_data = combined[
            (combined[col] >= x_min) &
            (combined[col] <= x_max)
        ].copy()

        sns.kdeplot(
            data=plot_data,
            x=col,
            hue="Dataset",
            palette=[C7, C3],
            common_norm=False
        )

        plt.title(f"Train vs Test Distribution: {col} (1% - 99% Range)")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/train_test_distribution_{col}_trimmed.png", dpi=300)
        plt.show()


for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide"]:
    if col in combined.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=combined, x=col, hue="Dataset", palette=[C7, C3])
        plt.title(f"Train vs Test Count Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/train_test_count_{col}.png", dpi=300)
        plt.show()


# ============================================================
# 20. Feature Engineering Visualizations
# ============================================================

# --- spending ratio features ---
spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

fe_df = train_eda.copy()
fe_df["TotalSpend_safe"] = fe_df["TotalSpend"].replace(0, np.nan)
for col in spending_cols:
    fe_df[f"Ratio_{col}"] = (fe_df[col] / fe_df["TotalSpend_safe"]).fillna(0)

ratio_cols = [f"Ratio_{c}" for c in spending_cols]
ratio_means = fe_df.groupby("Transported")[ratio_cols].mean()
ratio_means.columns = spending_cols
ratio_means = ratio_means.T.rename(columns={False: "Not Transported", True: "Transported"})

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ratio_means))
w = 0.35
ax.bar(x - w/2, ratio_means["Not Transported"], w, label="Not Transported", color=C6)
ax.bar(x + w/2, ratio_means["Transported"],     w, label="Transported",     color=C8)
ax.set_xticks(x)
ax.set_xticklabels(spending_cols, rotation=15)
ax.set_ylabel("Mean Spending Ratio")
ax.set_title("Spending Ratio by Category: Transported vs Not Transported")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fe_spending_ratio_by_transported.png", dpi=300)
plt.show()

# --- CryoSleep x spending interaction features ---
fe_df["CryoSleep_num"] = fe_df["CryoSleep"].map({True: 1, False: 0})
fe_df["Interaction_CryoSleep_TotalSpend"] = fe_df["CryoSleep_num"] * np.log1p(fe_df["TotalSpend"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in zip(
    axes,
    ["TotalSpend", "Interaction_CryoSleep_TotalSpend"],
    ["TotalSpend (log1p)", "CryoSleep x TotalSpend (log1p)"]
):
    plot_col = np.log1p(fe_df[col]) if col == "TotalSpend" else fe_df[col]
    for (label, grp), c in zip(fe_df.groupby("Transported"), BIN_PALETTE):
        ax.hist(plot_col[grp.index].dropna(), bins=50, alpha=0.7,
                label=str(label), color=c)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(title="Transported")

plt.suptitle("Interaction Feature: CryoSleep x TotalSpend", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fe_cryosleep_spend_interaction.png", dpi=300)
plt.show()

# ============================================================
# 21. Save EDA Summary Tables
# ============================================================

missing_summary.to_csv(f"{OUTPUT_DIR}/missing_value_summary.csv")
correlation_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
spending_summary.to_csv(f"{OUTPUT_DIR}/spending_summary.csv")
target_counts.to_csv(f"{OUTPUT_DIR}/target_distribution.csv")

print("\n" + "=" * 80)
print("EDA Completed Successfully")
print("=" * 80)
print(f"All figures and summary tables are saved in the folder: {OUTPUT_DIR}")