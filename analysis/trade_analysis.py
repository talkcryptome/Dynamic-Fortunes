import os
import pandas as pd
from glob import glob
from datetime import datetime
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

def run_trade_analysis(symbol, best_tf, best_tp, best_sl, base_trade_dir):
    # === Identify Year Dynamically ===
    pattern = os.path.join(base_trade_dir, "*", best_tf, f"trades_{symbol}_1m_*_TP{best_tp}_SL{best_sl}_Pyr1_Pct15_TF{best_tf}_*.csv")
    pattern = os.path.normpath(pattern)
    print("🔎 Searching with pattern:", pattern)
    print(f"Symbol: {symbol}")
    print(f"TF: {best_tf}")
    print(f"TP: {best_tp} ({type(best_tp)})")
    print(f"SL: {best_sl} ({type(best_sl)})")

    matched_files = glob(pattern, recursive=True)
    print("🔎 Normalized pattern:", pattern)
    print("📄 Matched files:", matched_files)

    # === LOAD & MERGE FILES ===
    dfs = []
    for file in matched_files:
        try:
            df = pd.read_csv(file)
            if not {'entry_time', 'exit_time', 'pnl'}.issubset(df.columns):
                print(f"⚠️ Skipping {file} - missing required columns")
                continue
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])
            df["year"] = df["entry_time"].dt.year
            df["month"] = df["entry_time"].dt.month
            df["hour"] = df["entry_time"].dt.hour
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    if not dfs:
        raise ValueError("❌ No valid trade files found or parsed.")

    df_all = pd.concat(dfs, ignore_index=True)

    # === INTRADAY ANALYSIS ===
    hourly = df_all.groupby("hour")["pnl"].agg(
        avg_pnl="mean",
        total_profit="sum",
        trade_count="count",
        win_rate=lambda x: (x > 0).mean()
    ).reset_index()

    profitable_hours = hourly[hourly["avg_pnl"] > 0]["hour"].tolist()

    def group_into_ranges(numbers):
        ranges = []
        for k, g in groupby(enumerate(numbers), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) == 1:
                ranges.append(f"({group[0]})")
            else:
                ranges.append(f"({group[0]}–{group[-1]})")
        return ranges

    profitable_ranges = group_into_ranges(profitable_hours)

    # === FILTERED VERSION ===
    filtered_df = df_all[df_all["hour"].isin(profitable_hours)]
    filtered_summary = {
        "final_equity": 1000 + filtered_df["pnl"].sum(),
        "total_profit": filtered_df["pnl"].sum(),
        "sharpe": filtered_df["pnl"].mean() / (filtered_df["pnl"].std() + 1e-6),
        "profit_factor": filtered_df[filtered_df["pnl"] > 0]["pnl"].sum() / abs(filtered_df[filtered_df["pnl"] < 0]["pnl"].sum() + 1e-6),
        "total_trades": len(filtered_df)
    }

    # === MONTHLY ANALYSIS ===
    monthly_agg = df_all.groupby("month")["pnl"].agg(
        avg_pnl="mean",
        total_profit="sum",
        trade_count="count",
        win_rate=lambda x: (x > 0).mean()
    ).reset_index()

    monthly_yoy = df_all.groupby(["year", "month"])["pnl"].agg(
        avg_pnl="mean",
        total_profit="sum",
        trade_count="count",
        win_rate=lambda x: (x > 0).mean()
    ).reset_index()

    # === RECOMMENDATION TEXT ===
    rec_text = f"""
    Recommendation Based on Time Filter Analysis

    ✅ Config: TF={best_tf}, TP={best_tp}, SL={best_sl}
    ✅ Profitable hours: {profitable_hours}
    ✅ Hour ranges: {', '.join(profitable_ranges)}

    Filtering trades by these hours improves strategy performance significantly:

    - Final Equity: ${filtered_summary['final_equity']:.2f}
    - Total Profit: ${filtered_summary['total_profit']:.2f}
    - Sharpe Ratio: {filtered_summary['sharpe']:.3f}
    - Profit Factor: {filtered_summary['profit_factor']:.3f}
    - Total Trades: {filtered_summary['total_trades']}

    We recommend limiting trade execution to the profitable hours above to improve risk-adjusted returns and consistency.
    """.strip()

    # === EXPORT TO EXCEL ===
    output_path = os.path.join("results", "Trades", symbol, f"trade_analysis_{symbol}_TF{best_tf}_TP{best_tp}_SL{best_sl}.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, index=False, sheet_name="All_Trades")
        hourly.to_excel(writer, index=False, sheet_name="Intraday_Performance")
        pd.DataFrame({"profitable_hours": profitable_hours}).to_excel(writer, index=False, sheet_name="Profitable_Hours")
        pd.DataFrame({"hour_ranges": profitable_ranges}).to_excel(writer, index=False, sheet_name="Hour_Ranges")
        pd.DataFrame([filtered_summary]).to_excel(writer, index=False, sheet_name="Filtered_Summary")
        monthly_agg.to_excel(writer, index=False, sheet_name="Monthly_Aggregate")
        monthly_yoy.to_excel(writer, index=False, sheet_name="Monthly_YearOverYear")
        pd.DataFrame([{"Recommendation": rec_text}]).to_excel(writer, index=False, sheet_name="Final_Recommendation")

    print("✅ Trade analysis written to:", output_path)
if __name__ == "__main__":
    symbol = "XLMUSDT"
    best_tf = "5_15_30"
    best_tp = 0.5
    best_sl = 1.0
    base_trade_dir = "../results/Trades/XLMUSDT"

    run_trade_analysis(symbol, best_tf, best_tp, best_sl, base_trade_dir)
