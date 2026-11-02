import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Suppress minor library convergence/deprecation warnings
warnings.filterwarnings("ignore")

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-Learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

# Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================================================
# CONFIGURATION BLOCK
# =========================================================
INPUT_CSV_PATH = "Features/HC3_DG_merged_output.csv"   # <-- Set your CSV dataset path here
OUTPUT_FOLDER = "Result_HYLGANet_HC3DG"         # <-- Target directory for all outputs
N_SPLITS = 5
MLP_EPOCHS = 10
BATCH_SIZE = 32
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =========================================================
# MULTI-LAYER PERCEPTRON (MLP) ARCHITECTURE
# =========================================================
class GeometricMLP(nn.Module):
    def __init__(self, input_dim):
        super(GeometricMLP, self).__init__()
        hidden_dim1 = max(64, input_dim)
        hidden_dim2 = max(32, hidden_dim1 // 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# =========================================================
# LEAKAGE-FREE GRAPH METRIC EXTRACTOR
# =========================================================
def extract_graph_metrics(X_target, knn_index, reference_labels, is_train=False):
    distances, indices = knn_index.kneighbors(X_target)
    purities, avg_distances, radii = [], [], []
    
    for i in range(len(X_target)):
        start_idx = 1 if is_train else 0
        neighbor_indices = indices[i][start_idx:]
        neighbor_distances = distances[i][start_idx:]
        
        neighbor_labels = reference_labels[neighbor_indices]
        if len(neighbor_labels) > 0:
            majority_class_count = max(np.bincount(neighbor_labels))
            purity = majority_class_count / len(neighbor_labels)
        else:
            purity = 0.5
            
        avg_dist = np.mean(neighbor_distances) if len(neighbor_distances) > 0 else 0.0
        radius = neighbor_distances[-1] if len(neighbor_distances) > 0 else 0.0
        
        purities.append(purity)
        avg_distances.append(avg_dist)
        radii.append(radius)
        
    return np.column_stack((purities, avg_distances, radii))

# =========================================================
# MODEL REGISTRY
# =========================================================
def get_baseline_models(random_state=42):
    return {
        "Logistic": LogisticRegression(max_iter=1000, random_state=random_state),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, verbose=-1, random_state=random_state, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_seed=random_state)
    }

# =========================================================
# MAIN BENCHMARK PIPELINE
# =========================================================
def run_benchmark(csv_path, output_dir, n_splits=5, epochs=10, batch_size=32):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Dynamic target detection
    target_col = None
    for col in df.columns:
        if col.lower() in ['label', 'target', 'class']:
            target_col = col
            break
    if target_col is None:
        raise KeyError(f"Target column not found. Available: {list(df.columns)}")

    df = df.dropna(subset=[target_col])
    X_df = df.drop(columns=[target_col])
    original_feature_names = list(X_df.columns)

    X_raw = X_df.values
    y_raw = df[target_col].values.astype(int)
    num_samples, num_features = X_raw.shape

    generated_feature_names = [
        'Graph_Neighborhood_Purity',
        'Graph_Local_Density_AvgDist',
        'Graph_Cluster_Boundary_Radius'
    ]
    all_final_features = original_feature_names + generated_feature_names

    # Save feature inventory text document
    with open(os.path.join(output_dir, "feature_inventory_listing.txt"), "w") as f:
        f.write("=== FEATURE INVENTORY LISTING ===\n")
        f.write(f"\n[1] ORIGINAL INPUT FEATURES ({len(original_feature_names)}):\n")
        for name in original_feature_names: f.write(f"- {name}\n")
        f.write(f"\n[2] GENERATED GEOMETRIC MATRIX FEATURES ({len(generated_feature_names)}):\n")
        for name in generated_feature_names: f.write(f"- {name}\n")

    print(f"Data ready: {num_samples} samples, {num_features} original features + 3 geometric features.")

    # Model names (Renamed to MLP)
    model_names = list(get_baseline_models().keys()) + ["MLP"]
    foldwise_records = []
    
    oof_predictions = {name: {"true": [], "proba": [], "pred": []} for name in model_names}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw)):
        print(f"\n>>> Executing Fold {fold + 1}/{n_splits} <<<")
        X_train_raw, X_val_raw = X_raw[train_idx].copy(), X_raw[val_idx].copy()
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        # ----------------- LEAKAGE-FREE PREPROCESSING -----------------
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        k = max(3, int(np.sqrt(len(X_train_scaled))))
        knn_index = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
        knn_index.fit(X_train_scaled)

        X_train_geom = extract_graph_metrics(X_train_scaled, knn_index, y_train, is_train=True)
        X_val_geom = extract_graph_metrics(X_val_scaled, knn_index, y_train, is_train=False)

        X_train_final = np.hstack((X_train_scaled, X_train_geom))
        X_val_final = np.hstack((X_val_scaled, X_val_geom))

        # ----------------- TRAIN & EVALUATE BASELINES -----------------
        baselines = get_baseline_models(random_state=RANDOM_STATE + fold)
        for m_name, model in baselines.items():
            t_start = time.time()
            model.fit(X_train_final, y_train)
            train_time = time.time() - t_start

            t_test_start = time.time()
            val_preds = model.predict(X_val_final)
            test_time = time.time() - t_test_start

            if hasattr(model, "predict_proba"):
                val_proba = model.predict_proba(X_val_final)[:, 1]
            else:
                val_proba = val_preds.astype(float)

            oof_predictions[m_name]["true"].extend(y_val)
            oof_predictions[m_name]["proba"].extend(val_proba)
            oof_predictions[m_name]["pred"].extend(val_preds)

            rec = evaluate_metrics(y_val, val_preds, val_proba, m_name, fold + 1, train_time, test_time)
            foldwise_records.append(rec)

        # ----------------- TRAIN & EVALUATE MLP -----------------
        train_ds = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train).unsqueeze(1))
        val_ds = TensorDataset(torch.FloatTensor(X_val_final), torch.FloatTensor(y_val).unsqueeze(1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        mlp_model = GeometricMLP(input_dim=X_train_final.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.005)

        t_start = time.time()
        for epoch in range(epochs):
            mlp_model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = mlp_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        mlp_train_time = time.time() - t_start

        mlp_model.eval()
        t_test_start = time.time()
        with torch.no_grad():
            mlp_val_proba = mlp_model(torch.FloatTensor(X_val_final)).numpy().flatten()
            mlp_val_pred = (mlp_val_proba >= 0.5).astype(int)
        mlp_test_time = time.time() - t_test_start

        oof_predictions["MLP"]["true"].extend(y_val)
        oof_predictions["MLP"]["proba"].extend(mlp_val_proba)
        oof_predictions["MLP"]["pred"].extend(mlp_val_pred)

        rec = evaluate_metrics(y_val, mlp_val_pred, mlp_val_proba, "MLP", fold + 1, mlp_train_time, mlp_test_time)
        foldwise_records.append(rec)

    # =========================================================
    # EXPORT CSV / EXCEL SPREADSHEETS
    # =========================================================
    df_foldwise = pd.DataFrame(foldwise_records)
    # Save both .csv and .xlsx for flexibility
    df_foldwise.to_csv(os.path.join(output_dir, "foldwise_performance_metrics.csv"), index=False)
    try:
        df_foldwise.to_excel(os.path.join(output_dir, "foldwise_performance_metrics.xlsx"), index=False)
    except Exception:
        pass

    summary_metrics = []
    numeric_cols = [c for c in df_foldwise.columns if c not in ['Model', 'Fold']]
    
    for m_name in model_names:
        sub = df_foldwise[df_foldwise['Model'] == m_name]
        m_row = {'Model': m_name}
        for c in numeric_cols:
            mean_v = sub[c].mean()
            std_v = sub[c].std()
            sem_v = stats.sem(sub[c])
            ci = sem_v * stats.t.ppf((1 + 0.95) / 2., n_splits - 1) if n_splits > 1 else 0
            m_row[f"{c}_Mean"] = mean_v
            m_row[f"{c}_Std"] = std_v
            m_row[f"{c}_95CI"] = f"[{mean_v - ci:.4f}, {mean_v + ci:.4f}]"
        summary_metrics.append(m_row)

    df_summary = pd.DataFrame(summary_metrics)
    df_summary.to_csv(os.path.join(output_dir, "performance_eval_summary.csv"), index=False)
    try:
        df_summary.to_excel(os.path.join(output_dir, "performance_eval_summary.xlsx"), index=False)
    except Exception:
        pass

    error_analysis_records = []
    for m_name in model_names:
        sub = df_foldwise[df_foldwise['Model'] == m_name]
        error_analysis_records.append({
            "Model": m_name,
            "Accuracy": sub["Accuracy"].mean(),
            "FNR (False Negative Rate)": sub["FNR"].mean(),
            "FPR (False Positive Rate)": sub["FPR"].mean(),
            "Watson Error Rate": sub["Watson_Error"].mean(),
            "Total Misclassifications (Avg/Fold)": sub["Total_Errors"].mean()
        })
    df_error = pd.DataFrame(error_analysis_records).sort_values("Watson Error Rate")
    df_error.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)
    try:
        df_error.to_excel(os.path.join(output_dir, "error_analysis.xlsx"), index=False)
    except Exception:
        pass

    comp_records = []
    for m_name in model_names:
        sub = df_foldwise[df_foldwise['Model'] == m_name]
        comp_records.append({
            "Model": m_name,
            "Avg Train Time (s)": sub["Training_Time_s"].mean(),
            "Avg Test Time (s)": sub["Test_Time_s"].mean(),
            "Latency per 1k Samples (ms)": (sub["Test_Time_s"].mean() / (num_samples / n_splits)) * 1000 * 1000
        })
    df_comp = pd.DataFrame(comp_records).sort_values("Avg Train Time (s)")
    df_comp.to_csv(os.path.join(output_dir, "computational_complexity.csv"), index=False)
    try:
        df_comp.to_excel(os.path.join(output_dir, "computational_complexity.xlsx"), index=False)
    except Exception:
        pass

    # =========================================================
    # GENERATE PLOTS & CHARTS
    # =========================================================
    generate_all_plots(oof_predictions, df_summary, df_error, df_comp, df_foldwise, model_names, output_dir)
    print(f"\nAll benchmark processes complete. Check the '{output_dir}/' directory.")

# =========================================================
# HELPER: METRICS EVALUATOR
# =========================================================
def evaluate_metrics(y_true, y_pred, y_proba, model_name, fold, train_time, test_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(r_curve, p_curve)
    
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    watson_err = (fp + fn) / (tn + fp + fn + tp)
    
    return {
        "Model": model_name,
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Cohen_Kappa": kappa,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "FNR": fnr,
        "FPR": fpr,
        "Watson_Error": watson_err,
        "Total_Errors": fp + fn,
        "Training_Time_s": train_time,
        "Test_Time_s": test_time
    }

# =========================================================
# HELPER: PLOT GENERATOR
# =========================================================
def generate_all_plots(oof_preds, df_summary, df_error, df_comp, df_foldwise, model_names, output_dir):
    colors = {
        "Logistic": "#1f77b4", "NaiveBayes": "#aec7e8", "RandomForest": "#ff7f0e",
        "XGBoost": "#2ca02c", "LightGBM": "#9467bd", "CatBoost": "#8c564b",
        "MLP": "#2E7D32"
    }

    # 1. Multi-Model ROC AUC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in oof_preds.items():
        fpr, tpr, _ = roc_curve(data["true"], data["proba"])
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", lw=2, color=colors.get(name, None))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_title("Multi-Model ROC-AUC Curve Benchmark", fontsize=12, weight='bold')
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_curves.png"), dpi=300)
    plt.close()

    # 2. Multi-Model Precision-Recall (PR) Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in oof_preds.items():
        p, r, _ = precision_recall_curve(data["true"], data["proba"])
        pr_val = auc(r, p)
        ax.plot(r, p, label=f"{name} (PR-AUC = {pr_val:.4f})", lw=2, color=colors.get(name, None))
    ax.set_title("Multi-Model Precision-Recall (PR) Curve Benchmark", fontsize=12, weight='bold')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()

    # 3. FNR & FPR Bar Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df_error["Model"].tolist()
    fnrs = df_error["FNR (False Negative Rate)"].tolist()
    fprs = df_error["FPR (False Positive Rate)"].tolist()
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, fnrs, width=w, label="FNR (False Negative Rate)", color="#e57373", edgecolor="black")
    ax.bar(x + w/2, fprs, width=w, label="FPR (False Positive Rate)", color="#64b5f6", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_title("Error Rate Distribution: FNR vs FPR Comparison", fontsize=12, weight='bold')
    ax.set_ylabel("Error Rate")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fnr_fpr_comparison.png"), dpi=300)
    plt.close()

    # 4. NEW: FNR and FPR Boxplot Distribution across 5 Folds
    fig, ax = plt.subplots(figsize=(11, 5))
    fnr_data = [df_foldwise[df_foldwise["Model"] == m]["FNR"].values for m in model_names]
    fpr_data = [df_foldwise[df_foldwise["Model"] == m]["FPR"].values for m in model_names]
    
    positions = np.arange(len(model_names)) * 2.0
    bp1 = ax.boxplot(fnr_data, positions=positions - 0.4, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="#ef9a9a", color="black"),
                     medianprops=dict(color="darkred", lw=2))
    bp2 = ax.boxplot(fpr_data, positions=positions + 0.4, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="#90caf9", color="black"),
                     medianprops=dict(color="darkblue", lw=2))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9, weight="bold")
    ax.set_ylabel("Rate Distribution across Folds", fontsize=10)
    ax.set_title("Cross-Fold FNR and FPR Variance Distribution (Boxplot)", fontsize=12, weight='bold')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['FNR (False Negative Rate)', 'FPR (False Positive Rate)'], loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fnr_fpr_boxplot.png"), dpi=300)
    plt.close()

    # 5. Sorted Comparative Performance Bar Chart (Retaining requested style)
    df_sorted = df_summary.sort_values("Accuracy_Mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = df_sorted["Model"].tolist()
    accs = df_sorted["Accuracy_Mean"].tolist()
    bar_colors = ["#2E7D32" if n == "MLP" else "#85929E" for n in names]
    bars = ax.bar(names, accs, color=bar_colors, edgecolor="black", width=0.55)
    ax.set_title("Comparative Model Accuracy (Sorted 5-Fold Cross-Validation)", fontsize=12, weight='bold')
    ax.set_ylabel("Mean Accuracy Score", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, h + 0.02, f"{h:.4f}", ha='center', va='bottom', fontsize=9, weight='bold')
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_accuracy_chart.png"), dpi=300)
    plt.close()

    # 6. Computational Complexity Training Time Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_comp["Model"], df_comp["Avg Train Time (s)"], color="#ffb74d", edgecolor="black")
    ax.set_title("Computational Complexity: Average Training Time (s)", fontsize=12, weight='bold')
    ax.set_xlabel("Training Time (seconds)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(df_comp["Avg Train Time (s)"]):
        ax.text(v + 0.01, i, f" {v:.3f}s", va='center', fontsize=9, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "computational_complexity_train_time.png"), dpi=300)
    plt.close()

    # 7. Confusion Matrix for Proposed Model (Now labeled as MLP)
    mlp_true = np.array(oof_preds["MLP"]["true"])
    mlp_pred = np.array(oof_preds["MLP"]["pred"])
    cm = confusion_matrix(mlp_true, mlp_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ax.set_title("Confusion Matrix - MLP", fontsize=12, weight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    classes = ['Class 0', 'Class 1']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, weight='bold')
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    # Retaining exact target filename requested
    plt.savefig(os.path.join(output_dir, "confusion_matrix_proposed_kgem.png"), dpi=300)
    plt.close()

# =========================================================
# RUNTIME INVOCATION
# =========================================================
if __name__ == "__main__":
    if INPUT_CSV_PATH == "Features/HC3_DG_merged_output.csv" and not os.path.exists("Features/HC3_DG_merged_output.csv"):
        print("Creating mock dataset with 76 features + 'label' column for validation...")
        n_feats = 76
        n_rows = 600
        X_mock = np.random.randn(n_rows, n_feats)
        y_mock = np.random.randint(0, 2, size=n_rows)
        cols = [f"feat_{i+1}" for i in range(n_feats)] + ["label"]
        pd.DataFrame(np.column_stack((X_mock, y_mock)), columns=cols).to_csv("your_dataset.csv", index=False)

    run_benchmark(
        csv_path=INPUT_CSV_PATH,
        output_dir=OUTPUT_FOLDER,
        n_splits=N_SPLITS,
        epochs=MLP_EPOCHS,
        batch_size=BATCH_SIZE
    )
