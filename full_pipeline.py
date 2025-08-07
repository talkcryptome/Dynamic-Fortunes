
import os
from backtest import run_backtests
from file_organizer import organize_files
from analysis.enhanced_analysis import run_enhanced_analysis
from analysis.trade_analysis import run_trade_analysis
from itertools import product

INDICATOR_GROUPS = {
    #"GroupA": {"ema", "macd", "rsi", "atr"},
    "GroupB": {"stoch_rsi","cci"},
    #"GroupC": {"adx", "ichimoku", "obv", "williamsr"},
    #"GroupD": {"ema", "macd", "rsi", "bollinger", "atr", "zscore"}
}

def full_pipeline(symbol, tp_list, sl_list, pyramid_list, equity_pct_list, tf_configs, data_dir, output_dir,comp_threshold,align_threshold):
    print("🔄 Step 1: Running Backtests...")
    for group_name, indicators in INDICATOR_GROUPS.items():
        print(f"\n🚀 Testing {group_name} with indicators: {indicators}")
        #run_backtests(symbol, tp_list, sl_list, pyramid_list, equity_pct_list, tf_configs, data_dir, indicators, group_name,comp_threshold,align_threshold)
        
        for comp_threshold, align_threshold in product(comp_threshold_list, align_threshold_list):
            print(f"\n🚀 Running: COMP={comp_threshold}, ALIGN={align_threshold}")
            run_backtests(symbol, tp_list, sl_list, pyramid_list, equity_pct_list,
                        tf_configs, data_dir, enabled_indicators={"stoch_rsi"},  # or full set
                        group_name=f"CT{comp_threshold}_AT{align_threshold}",
                        comp_threshold=comp_threshold,
                        align_threshold=align_threshold)


    #run_backtests(symbol, tp_list, sl_list, pyramid_list, equity_pct_list, tf_configs, data_dir,enabled_indicators)

    print("📁 Step 2: Organizing Files...")
    organize_files()

    print("📊 Step 3: Running Summary Analysis...")
    top_config = run_enhanced_analysis(symbol)

    print("📈 Step 4: Running Trade Analysis on Best Config...")
    if top_config:
        run_trade_analysis(
            symbol=top_config["symbol"],
            best_tf=top_config["tf"],
            best_tp=top_config["tp"],
            best_sl=top_config["sl"],
            base_trade_dir=os.path.join(output_dir, "Trades", symbol)
        )
    else:
        print("⚠️ No valid top configuration found from summary analysis.")

# Example execution
if __name__ == "__main__":
    symbol="XLMUSDT"
    tp_list=[1.0,1.5]
    sl_list=[0.5,1.0,1.5]
    pyramid_list=[1]
    equity_pct_list=[0.15]
    tf_configs=[{"base": 1, "mid": 5, "high": 30}]
    data_dir="data/1m data"
    output_dir = "results"
    comp_threshold_list = [0.5,1]
    align_threshold_list = [1,2]
    #enabled_indicators = {"ema", "macd", "rsi", "atr", "stoch_rsi", "bollinger", "adx", "cci", "ichimoku", "obv", "williamsr", "zscore"}


    full_pipeline(symbol, tp_list, sl_list, pyramid_list, equity_pct_list, tf_configs, data_dir, output_dir,comp_threshold_list,align_threshold_list)
