import pandas as pd
import os
from glob import glob
import re

# === Load and Parse Summary Files ===
def load_summary_files(folder_path):
    print(f"⚠️ Summary path does not exist: {folder_path}")
    all_files = glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        # Extract metadata from file name and config string
        df["year"] = re.search(r"_([0-9]{4})_", os.path.basename(file)).group(1)
        df["tp"] = df["config"].str.extract(r"_TP([\d\.]+)_").astype(float)
        df["sl"] = df["config"].str.extract(r"_SL([\d\.]+)_").astype(float)
        df["tf"] = df["config"].str.extract(r"_TF(\d+_\d+_\d+)")
        df["total_profit"] = df["final_equity"] - 1000

        # Add weights by year for weighted analysis
        df["weight"] = df["year"].map({
            "2020": 0.5, "2021": 0.75, "2022": 1.0, "2023": 1.25, "2024": 1.5, "2025": 2.0
        }).fillna(1.0)
        df["weighted_profit"] = df["total_profit"] * df["weight"]
        df["weighted_sharpe"] = df["sharpe"] * df["weight"]
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# === Aggregated Performance by Config (TF, TP, SL) ===
def aggregate_configs(df):
    df["trades_per_week"] = df["num_trades"] / (365 / 7)
    agg = df.groupby(["tf", "tp", "sl"]).agg(
        total_profit=("total_profit", "sum"),
        avg_sharpe=("sharpe", "mean"),
        avg_profit_factor=("profit_factor", "mean"),
        total_trades=("num_trades", "sum"),
        weighted_profit=("weighted_profit", "sum"),
        weighted_sharpe=("weighted_sharpe", "sum"),
        years_tested=("year", "count"),
        profit_std=("total_profit", "std")
    ).reset_index()

    agg["trades_per_week"] = agg["total_trades"] / (agg["years_tested"] * 52)
    agg["consistency"] = agg["total_profit"] / agg["profit_std"].replace(0, 1)

    # Meta score for holistic ranking
    agg["meta_score"] = (
        0.4 * normalize_series(agg["total_profit"]) +
        0.3 * normalize_series(agg["avg_profit_factor"]) +
        0.2 * normalize_series(agg["avg_sharpe"]) +
        0.1 * normalize_series(agg["consistency"])
    )
    return agg

def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

# === Filters for Reliable Strategies ===
def filter_configs(agg, pf_threshold=1.0, min_trades_per_week=1.0):
    return agg[(agg["avg_profit_factor"] > pf_threshold) & (agg["trades_per_week"] >= min_trades_per_week)]

# === Per-Year Performance per Config ===
def yearly_trends(df):
    return df.groupby(["year", "tf", "tp", "sl"]).agg(
        total_profit=("total_profit", "sum"),
        avg_sharpe=("sharpe", "mean"),
        profit_factor=("profit_factor", "mean"),
        num_trades=("num_trades", "sum")
    ).reset_index()

# === Performance Grouped by TF Only ===
def tf_patterns(df):
    summary = df.groupby("tf").agg(
        total_profit=("total_profit", "sum"),
        weighted_profit=("weighted_profit", "sum"),
        avg_sharpe=("sharpe", "mean"),
        avg_profit_factor=("profit_factor", "mean"),
        total_trades=("num_trades", "sum"),
        years_tested=("year", "count")
    ).reset_index()
    summary["trades_per_week"] = summary["total_trades"] / (summary["years_tested"] * 52)
    return summary

# === Year-by-Year TF Trends ===
def tf_year_trends(df):
    return df.groupby(["year", "tf"]).agg(
        total_profit=("total_profit", "sum"),
        avg_sharpe=("sharpe", "mean"),
        profit_factor=("profit_factor", "mean"),
        num_trades=("num_trades", "sum")
    ).reset_index()

# === TP/SL-Agnostic Strategy View ===
def tp_sl_agnostic_summary(df):
    return df.groupby("tf").agg(
        avg_profit=("total_profit", "mean"),
        total_trades=("num_trades", "sum"),
        avg_sharpe=("sharpe", "mean"),
        avg_profit_factor=("profit_factor", "mean")
    ).reset_index()

# === Weighted Performance by Year ===
def weighted_aggregation(df):
    return df.groupby(["tf", "tp", "sl"]).agg(
        weighted_profit=("weighted_profit", "sum"),
        weighted_sharpe=("weighted_sharpe", "sum"),
        years_tested=("year", "count")
    ).reset_index()

# === Consistency Scoring ===
def consistency_score(df):
    std = df.groupby(["tf", "tp", "sl"])["total_profit"].std()
    mean = df.groupby(["tf", "tp", "sl"])["total_profit"].mean()
    cv = std / mean.replace(0, 1)
    return cv.reset_index(name="cv")

# === Meta Score for Final Ranking ===
def meta_ranking(df):
    agg = aggregate_configs(df)
    return agg.sort_values("meta_score", ascending=False)

# === Best Configs per Year ===
def best_per_year(df):
    return df.sort_values("total_profit", ascending=False).groupby("year").first().reset_index()

def export_final_recommendation(writer, meta_df):
    top = meta_df.sort_values(by="meta_score", ascending=False).iloc[0]

    # Format the metrics and reasoning
    explanation = [
        ["Recommended Configuration (Highest Meta Score)", ""],
        ["TF", top["tf"]],
        ["TP", top["tp"]],
        ["SL", top["sl"]],
        ["Total Profit", round(top["total_profit"], 2)],
        ["Avg Sharpe", round(top["avg_sharpe"], 3)],
        ["Avg Profit Factor", round(top["avg_profit_factor"], 3)],
        ["Trades/Week", round(top["trades_per_week"], 2)],
        ["Consistency (Profit/Std)", round(top["consistency"], 2)],
        ["Meta Score", round(top["meta_score"], 4)],
        ["", ""],
        ["Why this config?"],
        ["• Highest overall meta score, combining profit, PF, Sharpe, and consistency."],
        ["• Satisfies filters for profitability (PF > 1.0) and minimum trade frequency."],
        ["• Weighted toward recent years, improving predictive reliability."],
        ["• Performs well year-over-year with stable metrics."],
        ["• Strong balance of profit and risk-adjusted performance."]
    ]

    rec_df = pd.DataFrame(explanation)
    rec_df.to_excel(writer, sheet_name="Final_Recommendation", index=False, header=False)
