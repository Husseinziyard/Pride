"""
Pride — Data Science Tools
Tools that the AI agent uses to perform data science tasks.
"""

import io
import sys
import json
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


# ── Global dataset store ──
_dataset_store: dict[str, pd.DataFrame] = {}


def set_dataset(df: pd.DataFrame, name: str = "main") -> None:
    _dataset_store[name] = df


def get_dataset(name: str = "main") -> pd.DataFrame | None:
    return _dataset_store.get(name)


# ═══════════════════════════════════════════
# TOOL 1: Dataset Overview / EDA
# ═══════════════════════════════════════════
@tool
def dataset_overview(dataset_name: str = "main") -> str:
    """
    Get a comprehensive overview of the loaded dataset including shape,
    columns, data types, missing values, and basic statistics.
    Use this tool FIRST after a dataset is uploaded to understand its structure.
    """
    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded. Please upload a CSV file first."

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = missing[missing > 0]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    report = f"""
**Dataset Overview**

• Shape: {df.shape[0]} rows × {df.shape[1]} columns
• Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB

**Columns & Types:**
{chr(10).join(f'  - {col}: {dtype}' for col, dtype in df.dtypes.items())}

**Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols) if numeric_cols else 'None'}
**Categorical Columns ({len(categorical_cols)}):** {', '.join(categorical_cols) if categorical_cols else 'None'}

**Missing Values:**
{chr(10).join(f'  - {col}: {count} ({missing_pct[col]}%)' for col, count in missing_report.items()) if len(missing_report) > 0 else '  No missing values found.'}

**Quick Statistics (Numeric):**
{df.describe().round(2).to_string() if numeric_cols else 'No numeric columns'}

**First 5 Rows:**
{df.head().to_string()}
"""
    return report


# ═══════════════════════════════════════════
# TOOL 2: Data Cleaning
# ═══════════════════════════════════════════
@tool
def clean_data(
    action: str,
    column: str = "",
    strategy: str = "mean",
    dataset_name: str = "main",
) -> str:
    """
    Clean the dataset. Available actions:
    - 'drop_missing': Drop rows with missing values (optionally in a specific column)
    - 'fill_missing': Fill missing values using strategy ('mean', 'median', 'mode', 'zero', 'ffill', 'bfill')
    - 'drop_duplicates': Remove duplicate rows
    - 'drop_column': Drop a specific column
    - 'convert_type': Convert column type (strategy = 'numeric', 'datetime', 'category', 'string')
    - 'remove_outliers': Remove outliers using IQR method for a numeric column

    Args:
        action: The cleaning action to perform
        column: Target column name (required for some actions)
        strategy: Strategy for fill_missing or target type for convert_type
        dataset_name: Name of the dataset
    """
    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded."

    original_shape = df.shape

    try:
        if action == "drop_missing":
            if column:
                df = df.dropna(subset=[column])
            else:
                df = df.dropna()

        elif action == "fill_missing":
            if not column:
                return "Please specify a column for fill_missing."
            if strategy == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == "median":
                df[column] = df[column].fillna(df[column].median())
            elif strategy == "mode":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif strategy == "zero":
                df[column] = df[column].fillna(0)
            elif strategy == "ffill":
                df[column] = df[column].ffill()
            elif strategy == "bfill":
                df[column] = df[column].bfill()

        elif action == "drop_duplicates":
            df = df.drop_duplicates()

        elif action == "drop_column":
            if not column:
                return "Please specify a column to drop."
            df = df.drop(columns=[column])

        elif action == "convert_type":
            if not column:
                return "Please specify a column."
            if strategy == "numeric":
                df[column] = pd.to_numeric(df[column], errors="coerce")
            elif strategy == "datetime":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            elif strategy == "category":
                df[column] = df[column].astype("category")
            elif strategy == "string":
                df[column] = df[column].astype(str)

        elif action == "remove_outliers":
            if not column:
                return "Please specify a numeric column."
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower) & (df[column] <= upper)]

        else:
            return f"Unknown action: {action}"

        set_dataset(df, dataset_name)
        return f"Done. Shape: {original_shape} → {df.shape}. Missing values remaining: {df.isnull().sum().sum()}"

    except Exception as e:
        return f"Error during cleaning: {str(e)}"


# ═══════════════════════════════════════════
# TOOL 3: Visualization
# ═══════════════════════════════════════════
@tool
def create_visualization(
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    dataset_name: str = "main",
) -> str:
    """
    Create a data visualization and save it as an image.
    Available chart types:
    - 'histogram': Distribution of a numeric column (x_column)
    - 'scatter': Scatter plot (x_column vs y_column)
    - 'bar': Bar chart of value counts (x_column)
    - 'correlation': Correlation heatmap of all numeric columns
    - 'boxplot': Box plot of a numeric column (x_column), optionally grouped by y_column
    - 'line': Line chart (x_column vs y_column)

    Args:
        chart_type: Type of chart to create
        x_column: Column for x-axis
        y_column: Column for y-axis (if needed)
        title: Chart title
        dataset_name: Name of the dataset
    """
    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded."

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use("seaborn-v0_8-darkgrid")

        if chart_type == "histogram":
            if not x_column:
                return "Specify x_column for histogram."
            df[x_column].hist(ax=ax, bins=30, edgecolor="black", alpha=0.7, color="#6C63FF")
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")

        elif chart_type == "scatter":
            if not x_column or not y_column:
                return "Specify both x_column and y_column for scatter."
            ax.scatter(df[x_column], df[y_column], alpha=0.6, c="#6C63FF", edgecolors="white", s=50)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "bar":
            if not x_column:
                return "Specify x_column for bar chart."
            counts = df[x_column].value_counts().head(20)
            counts.plot(kind="bar", ax=ax, color="#6C63FF", edgecolor="black")
            ax.set_xlabel(x_column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "correlation":
            plt.close(fig)
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "No numeric columns for correlation."
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                numeric_df.corr(), annot=True, cmap="RdYlBu_r", center=0,
                fmt=".2f", ax=ax, square=True, linewidths=0.5
            )

        elif chart_type == "boxplot":
            if not x_column:
                return "Specify x_column for boxplot."
            if y_column:
                df.boxplot(column=x_column, by=y_column, ax=ax)
                plt.suptitle("")
            else:
                df[[x_column]].boxplot(ax=ax)

        elif chart_type == "line":
            if not x_column or not y_column:
                return "Specify both x_column and y_column for line chart."
            df_sorted = df.sort_values(x_column)
            ax.plot(df_sorted[x_column], df_sorted[y_column], color="#6C63FF", linewidth=2)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        else:
            plt.close(fig)
            return f"Unknown chart type: {chart_type}"

        ax.set_title(title or f"{chart_type.title()} Chart", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filename = f"chart_{chart_type}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return f"Chart saved as '{filename}'. It is now displayed in the app."

    except Exception as e:
        plt.close("all")
        return f"Visualization error: {str(e)}"


# ═══════════════════════════════════════════
# TOOL 4: Build ML Model
# ═══════════════════════════════════════════
@tool
def build_model(
    target_column: str,
    model_type: str = "auto",
    test_size: float = 0.2,
    dataset_name: str = "main",
) -> str:
    """
    Build and evaluate a machine learning model.
    Automatically handles preprocessing (encoding, scaling, train/test split).

    Args:
        target_column: The column to predict
        model_type: 'classification', 'regression', or 'auto' (auto-detects)
        test_size: Fraction of data for testing (default 0.2)
        dataset_name: Name of the dataset
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import (
        accuracy_score, classification_report, r2_score,
        mean_squared_error, mean_absolute_error,
    )

    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded."

    if target_column not in df.columns:
        return f"Column '{target_column}' not found. Available: {list(df.columns)}"

    try:
        df_model = df.dropna(subset=[target_column]).copy()

        # Auto-detect task type
        if model_type == "auto":
            if df_model[target_column].dtype in ["object", "category"] or df_model[target_column].nunique() <= 10:
                model_type = "classification"
            else:
                model_type = "regression"

        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]

        # Encode target if needed
        label_enc = None
        if model_type == "classification" and y.dtype == "object":
            label_enc = LabelEncoder()
            y = pd.Series(label_enc.fit_transform(y), name=target_column)

        # Handle categorical features
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        X = X.select_dtypes(include=[np.number])
        if X.empty:
            return "No usable features after preprocessing."

        X = X.fillna(X.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        if model_type == "classification":
            model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            model_lr = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            model_lr = LinearRegression()

        model_rf.fit(X_train, y_train)
        model_lr.fit(X_train, y_train)

        pred_rf = model_rf.predict(X_test)
        pred_lr = model_lr.predict(X_test)

        feat_imp = pd.Series(
            model_rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(10)

        if model_type == "classification":
            acc_rf = accuracy_score(y_test, pred_rf)
            acc_lr = accuracy_score(y_test, pred_lr)
            best_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
            best_pred = pred_rf if acc_rf >= acc_lr else pred_lr

            target_names = [str(c) for c in label_enc.classes_] if label_enc else None
            report = classification_report(y_test, best_pred, target_names=target_names)

            result = f"""
**ML Model Results** — {model_type.upper()}

Split: {1-test_size:.0%} train / {test_size:.0%} test ({len(X_train)} / {len(X_test)} samples)
Features: {len(X.columns)}

**Model Comparison:**
  - Random Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.1f}%)
  - Logistic Regression Accuracy: {acc_lr:.4f} ({acc_lr*100:.1f}%)
  - Best: **{best_name}**

**Classification Report ({best_name}):**
{report}

**Top Feature Importances:**
{chr(10).join(f'  - {feat}: {imp:.4f}' for feat, imp in feat_imp.items())}
"""
        else:
            r2_rf = r2_score(y_test, pred_rf)
            r2_lr = r2_score(y_test, pred_lr)
            rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
            rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
            mae_rf = mean_absolute_error(y_test, pred_rf)
            mae_lr = mean_absolute_error(y_test, pred_lr)
            best_name = "Random Forest" if r2_rf >= r2_lr else "Linear Regression"

            result = f"""
**ML Model Results** — {model_type.upper()}

Split: {1-test_size:.0%} train / {test_size:.0%} test ({len(X_train)} / {len(X_test)} samples)
Features: {len(X.columns)}

**Random Forest:**
  - R² Score: {r2_rf:.4f} | RMSE: {rmse_rf:.4f} | MAE: {mae_rf:.4f}

**Linear Regression:**
  - R² Score: {r2_lr:.4f} | RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f}

Best: **{best_name}**

**Top Feature Importances:**
{chr(10).join(f'  - {feat}: {imp:.4f}' for feat, imp in feat_imp.items())}
"""
        return result

    except Exception as e:
        return f"Model building error: {str(e)}\n{traceback.format_exc()}"


# ═══════════════════════════════════════════
# TOOL 5: Execute Python Code
# ═══════════════════════════════════════════
@tool
def execute_python_code(code: str, dataset_name: str = "main") -> str:
    """
    Execute arbitrary Python code on the dataset.
    The dataset is available as 'df' (pandas DataFrame).
    Libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), sklearn.
    Print results to see them.

    Args:
        code: Python code to execute
        dataset_name: Dataset available as 'df'
    """
    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded."

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }

    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)

        if "df" in local_vars and isinstance(local_vars["df"], pd.DataFrame):
            if not local_vars["df"].equals(df):
                set_dataset(local_vars["df"], dataset_name)

        figs = [plt.figure(i) for i in plt.get_fignums()]
        saved_charts = []
        for idx, fig in enumerate(figs):
            fname = f"chart_custom_{idx}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
            saved_charts.append(fname)
        plt.close("all")

        output = buffer.getvalue()
        result = ""
        if output:
            result += f"**Output:**\n{output}\n"
        if saved_charts:
            result += f"**Charts saved:** {', '.join(saved_charts)}\n"
        if not result:
            result = "Code executed successfully (no output)."
        return result

    except Exception as e:
        plt.close("all")
        return f"Code execution error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout


# ═══════════════════════════════════════════
# TOOL 6: Column Analysis
# ═══════════════════════════════════════════
@tool
def analyze_column(column_name: str, dataset_name: str = "main") -> str:
    """
    Deep analysis on a specific column: distribution, unique values,
    statistics, outliers.

    Args:
        column_name: Name of the column to analyze
        dataset_name: Name of the dataset
    """
    df = get_dataset(dataset_name)
    if df is None:
        return "No dataset loaded."

    if column_name not in df.columns:
        return f"Column '{column_name}' not found. Available: {list(df.columns)}"

    col = df[column_name]
    result = f"\n**Analysis: '{column_name}'**\n"
    result += f"- Type: {col.dtype}\n"
    result += f"- Non-null: {col.count()} / {len(col)}\n"
    result += f"- Missing: {col.isnull().sum()} ({col.isnull().mean()*100:.1f}%)\n"
    result += f"- Unique: {col.nunique()}\n"

    if np.issubdtype(col.dtype, np.number):
        result += f"\n**Statistics:**\n"
        result += f"  - Mean: {col.mean():.4f}\n"
        result += f"  - Median: {col.median():.4f}\n"
        result += f"  - Std: {col.std():.4f}\n"
        result += f"  - Min: {col.min():.4f} | Max: {col.max():.4f}\n"
        result += f"  - Skewness: {col.skew():.4f}\n"
        result += f"  - Kurtosis: {col.kurtosis():.4f}\n"

        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        outliers = ((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum()
        result += f"  - IQR: {iqr:.4f}\n"
        result += f"  - Outliers (IQR): {outliers}\n"
    else:
        result += f"\n**Top Values:**\n"
        for val, count in col.value_counts().head(10).items():
            result += f"  - {val}: {count} ({count/len(col)*100:.1f}%)\n"

    return result


# ── All tools list ──
ALL_TOOLS = [
    dataset_overview,
    clean_data,
    create_visualization,
    build_model,
    execute_python_code,
    analyze_column,
]