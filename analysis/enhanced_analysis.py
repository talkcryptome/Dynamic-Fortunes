from .enhanced_analysis_script import *
import pandas as pd

def run_enhanced_analysis(symbol):
# === Accept arguments ===

    summary_path = f"results/summary/{symbol}"
# === Load and Process ===
    df = load_summary_files(summary_path)
    agg = aggregate_configs(df)
    agg_sorted = aggregate_configs(df).sort_values(by="weighted_profit", ascending=False)

# === Filters and Sorting ===
    filtered = filter_configs(agg)
    filtered_sorted = filtered.sort_values(by="meta_score", ascending=False)
    yearly = yearly_trends(df).sort_values(by=["total_profit", "year"], ascending=[False, True])
    tf_perf = tf_patterns(df).sort_values(by="weighted_profit", ascending=False)
    tf_yearly = tf_year_trends(df).sort_values(by="total_profit", ascending=False)
    tp_sl_agnostic = tp_sl_agnostic_summary(df).sort_values(by="avg_profit", ascending=False).head(10)

# === Additional Analysis ===
    weighted_df = weighted_aggregation(df)
    consistency = consistency_score(df)
    top_yearly = best_per_year(df)
    meta = meta_ranking(df)

# === Display Results ===
    #print("🔍 Profit Factor > 1.0 Trades per week ≥ 1.0 (sorted by meta_score)")
    #print(filtered_sorted.head(10))

    #print("\n📅 Year-by-year breakdown of each config")
    #print(yearly.head(10))


    #print("\n📊 Aggregate performance grouped by TF (ignoring TP/SL)")
    #print(tf_perf.head(10))

    #print("\n🕒 Year-by-year TF performance")
    #print(tf_yearly.head(10))

    #print("\n🧠 TP/SL-Agnostic Summary (best average profit by TF)")
    #print(tp_sl_agnostic_summary(df).sort_values(by="avg_profit", ascending=False).head(10))

    print("\n⭐ Meta Ranked Configs")
    print(meta.head(10))

    #print("\n🏆 Top Configs per Year")
    #print(top_yearly)

    #print("\n📈 Consistency Scores (lowest CV is best)")
    #print(consistency.sort_values("cv").head(10))

    output_path = "results/enhanced_analysis_results.xlsx"
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        export_final_recommendation(writer, meta)
        agg.to_excel(writer, sheet_name="All Aggregated", index=False)
        filtered_sorted.to_excel(writer, sheet_name="Filtered Sorted", index=False)
        yearly.to_excel(writer, sheet_name="Yearly Trends", index=False)
        tf_perf.to_excel(writer, sheet_name="TF Summary", index=False)
        tf_yearly.to_excel(writer, sheet_name="TF Yearly", index=False)
        tp_sl_agnostic.to_excel(writer, sheet_name="TP_SL_Agnostic", index=False)
        meta.to_excel(writer, sheet_name="Meta Ranking", index=False)
        top_yearly.to_excel(writer, sheet_name="Top Per Year", index=False)
        consistency.to_excel(writer, sheet_name="Consistency", index=False)
        
    print(f"\n✅ Results exported to {output_path}")
    # === Extract best config (first row of meta if not empty) ===
    if not meta.empty:
        best_row = meta.iloc[0]
        tf_parts = best_row["tf"].split("_")
        best_config = {
            "symbol": symbol,
            "tf": best_row["tf"],
            "tp": best_row["tp"],
            "sl": best_row["sl"]
        }
        return best_config
    else:
        return None

# ✅ Run if executed directly
if __name__ == "__main__":
    run_enhanced_analysis("XLMUSDT")
