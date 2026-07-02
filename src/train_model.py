"""
Customer Churn Prediction - Full ML Pipeline
Author: Veda Sai Polisetty
Dataset: IBM Telco Customer Churn (7,043 customers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import shap
import pickle
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

PALETTE = {
    "primary":   "#1a56db",
    "secondary": "#e74c3c",
    "neutral":   "#374151",
    "light":     "#f9fafb",
    "accent":    "#10b981",
    "mid":       "#6b7280",
}

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
})

# ── 1. LOAD & CLEAN ──────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading and cleaning data")
print("=" * 60)

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(f"  Rows: {len(df):,}  |  Churn rate: {df['Churn'].mean()*100:.1f}%")

# ── 2. EDA CHART 1 — Churn by Contract Type ──────────────────────────────────

print("\nSTEP 2: Generating EDA charts")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Who churns? Key patterns in 7,043 customers", fontsize=14, fontweight="bold", y=1.02)

contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
bars = axes[0].bar(contract_churn.index, contract_churn.values,
                   color=[PALETTE["secondary"], PALETTE["primary"], PALETTE["accent"]],
                   width=0.5, edgecolor="white", linewidth=1.5)
axes[0].set_title("Churn rate by contract type")
axes[0].set_ylabel("Churn rate (%)")
axes[0].set_ylim(0, 55)
for bar, val in zip(bars, contract_churn.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold",
                 color=PALETTE["neutral"])
axes[0].set_xticklabels(["Month-to-\nmonth", "One\nyear", "Two\nyear"])

tenure_bins = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0–12m", "13–24m", "25–48m", "49–72m"])
tenure_churn = df.groupby(tenure_bins, observed=True)["Churn"].mean() * 100
colors_t = [PALETTE["secondary"] if v > 30 else PALETTE["primary"] if v > 15 else PALETTE["accent"]
            for v in tenure_churn.values]
bars2 = axes[1].bar(tenure_churn.index, tenure_churn.values, color=colors_t,
                    width=0.5, edgecolor="white", linewidth=1.5)
axes[1].set_title("Churn rate by customer tenure")
axes[1].set_ylabel("Churn rate (%)")
axes[1].set_ylim(0, 55)
for bar, val in zip(bars2, tenure_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold",
                 color=PALETTE["neutral"])

churned    = df[df["Churn"] == 1]["MonthlyCharges"]
not_churned = df[df["Churn"] == 0]["MonthlyCharges"]
axes[2].hist(not_churned, bins=30, alpha=0.75, color=PALETTE["primary"],  label="Stayed")
axes[2].hist(churned,     bins=30, alpha=0.75, color=PALETTE["secondary"], label="Churned")
axes[2].set_title("Monthly charges: stayed vs churned")
axes[2].set_xlabel("Monthly charges ($)")
axes[2].set_ylabel("Number of customers")
axes[2].legend(frameon=False)

plt.tight_layout()
plt.savefig("outputs/01_eda_insights.png", bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved → outputs/01_eda_insights.png")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────

print("\nSTEP 3: Feature engineering")

binary_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling","Churn"]
df_enc = df.copy()

for col in df_enc.select_dtypes("object").columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

feature_cols = [c for c in df_enc.columns if c != "Churn"]
X = df_enc[feature_cols]
y = df_enc["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. TRAIN 3 MODELS ────────────────────────────────────────────────────────

print("\nSTEP 4: Training models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=4, random_state=42,
                                         eval_metric="logloss", verbosity=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    results[name] = {"model": model, "proba": proba, "auc": auc}
    print(f"  {name:<25} AUC: {auc:.4f}")

best_name  = max(results, key=lambda n: results[n]["auc"])
best_model = results[best_name]["model"]
print(f"\n  Best model: {best_name} (AUC {results[best_name]['auc']:.4f})")

# Use XGBoost for feature importance + SHAP (tree-based, works cleanly)
xgb_model = results["XGBoost"]["model"]

# ── 5. MODEL COMPARISON CHART ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model performance comparison", fontsize=14, fontweight="bold")

names  = list(results.keys())
aucs   = [results[n]["auc"] for n in names]
colors = [PALETTE["accent"] if n == best_name else PALETTE["mid"] for n in names]
bars   = axes[0].barh(names, aucs, color=colors, height=0.45, edgecolor="white")
axes[0].set_xlim(0.5, 1.0)
axes[0].set_xlabel("ROC-AUC score")
axes[0].set_title("ROC-AUC by model")
for bar, val in zip(bars, aucs):
    axes[0].text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
                 color=PALETTE["neutral"])
winner_patch = mpatches.Patch(color=PALETTE["accent"], label=f"Best: {best_name}")
axes[0].legend(handles=[winner_patch], frameon=False)

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    lw    = 2.5 if name == best_name else 1.2
    alpha = 1.0 if name == best_name else 0.5
    axes[1].plot(fpr, tpr, lw=lw, alpha=alpha, label=f"{name} ({res['auc']:.3f})")
axes[1].plot([0,1], [0,1], "k--", lw=0.8, alpha=0.4)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curves — all models")
axes[1].legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/02_model_comparison.png", bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved → outputs/02_model_comparison.png")

# ── 6. FEATURE IMPORTANCE ────────────────────────────────────────────────────

print("\nSTEP 5: Feature importance (what drives churn)")

importances = xgb_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=True).tail(12)

readable = {
    "tenure":         "Customer tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges":   "Total charges ($)",
    "Contract":       "Contract type",
    "InternetService":"Internet service type",
    "TechSupport":    "Tech support included",
    "OnlineSecurity": "Online security addon",
    "PaymentMethod":  "Payment method",
    "PaperlessBilling":"Paperless billing",
    "Partner":        "Has partner",
    "Dependents":     "Has dependents",
    "SeniorCitizen":  "Senior citizen",
}
feat_df["Label"] = feat_df["Feature"].map(lambda x: readable.get(x, x))

fig, ax = plt.subplots(figsize=(9, 6))
colors_f = [PALETTE["secondary"] if i >= len(feat_df)-3 else PALETTE["primary"]
            for i in range(len(feat_df))]
bars = ax.barh(feat_df["Label"], feat_df["Importance"],
               color=colors_f, height=0.55, edgecolor="white")
ax.set_xlabel("Feature importance score")
ax.set_title("Top 12 churn drivers — XGBoost", fontweight="bold")

top3_patch  = mpatches.Patch(color=PALETTE["secondary"], label="Top 3 drivers")
rest_patch  = mpatches.Patch(color=PALETTE["primary"],   label="Other features")
ax.legend(handles=[top3_patch, rest_patch], frameon=False)

plt.tight_layout()
plt.savefig("outputs/03_feature_importance.png", bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved → outputs/03_feature_importance.png")

# ── 7. SHAP VALUES ───────────────────────────────────────────────────────────

print("\nSTEP 6: SHAP explainability")

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                  show=False, plot_size=None, color_bar=True)
ax = plt.gca()
ax.set_title("SHAP values — why does each feature push churn up or down?",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/04_shap_summary.png", bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved → outputs/04_shap_summary.png")

# ── 8. BUSINESS DASHBOARD CHART ──────────────────────────────────────────────

print("\nSTEP 7: Business insight summary chart")

y_pred   = best_model.predict(X_test)
cm       = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)

fig = plt.figure(figsize=(14, 5))
fig.suptitle("Customer Churn Prediction — Business Impact Summary", fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(1, 3, 1)
metrics = ["AUC", "Precision", "Recall", "F1"]
vals    = [results[best_name]["auc"], precision, recall, f1]
colors_m = [PALETTE["accent"] if v >= 0.80 else PALETTE["primary"] for v in vals]
bars = ax1.bar(metrics, vals, color=colors_m, width=0.5, edgecolor="white")
ax1.set_ylim(0, 1.1)
ax1.set_title("Model metrics")
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.2f}", ha="center", fontsize=11, fontweight="bold",
             color=PALETTE["neutral"])

ax2 = fig.add_subplot(1, 3, 2)
im = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
ax2.set_xticklabels(["Predicted\nStay","Predicted\nChurn"])
ax2.set_yticklabels(["Actually\nStayed","Actually\nChurned"])
ax2.set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i,j]), ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else PALETTE["neutral"])

ax3 = fig.add_subplot(1, 3, 3)
ax3.axis("off")
total_test   = len(y_test)
churners     = int(y_test.sum())
caught       = tp
monthly_rev  = 65
saved_sim    = caught * monthly_rev * 12

stats = [
    ("Customers scored", f"{total_test:,}"),
    ("Actual churners", f"{churners:,}"),
    ("Caught by model", f"{caught:,}  ({recall*100:.0f}%)"),
    ("Simulated ARR saved", f"${saved_sim:,.0f}"),
    ("Best model", best_name),
    ("AUC score", f"{results[best_name]['auc']:.4f}"),
]
y_pos = 0.92
for label, value in stats:
    ax3.text(0.05, y_pos, label + ":", fontsize=10, color=PALETTE["mid"],
             transform=ax3.transAxes)
    ax3.text(0.95, y_pos, value, fontsize=10, fontweight="bold",
             color=PALETTE["neutral"], ha="right", transform=ax3.transAxes)
    y_pos -= 0.14
ax3.set_title("Business impact")

plt.tight_layout()
plt.savefig("outputs/05_business_summary.png", bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved → outputs/05_business_summary.png")

# ── 9. SAVE MODEL ────────────────────────────────────────────────────────────

with open("outputs/churn_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": feature_cols}, f)
print("\nSTEP 8: Model saved → outputs/churn_model.pkl")

print("\n" + "=" * 60)
print("ALL DONE. Summary:")
print(f"  Best model  : {best_name}")
print(f"  AUC         : {results[best_name]['auc']:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  Simulated $ : ${saved_sim:,.0f} ARR saved")
print("=" * 60)
