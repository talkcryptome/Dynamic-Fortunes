import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
import hashlib
import pickle
from datetime import datetime
from filelock import FileLock
from typing import Set
from itertools import product


# === User Parameters ===
MAX_PARALLEL = 60
INITIAL_EQUITY = 1000.0
ALLOW_LONG = True
ALLOW_SHORT = False
ATR_LEN = 14
ATR_AVG_LEN = 20
COMP_THRESHOLD = 1.0
ALIGN_THRESHOLD = 2

# === Indicator Toggles ===
#ENABLED_INDICATORS = {
#    "ema", "macd", "rsi", "stoch_rsi", "bollinger",
#    "adx", "cci", "ichimoku", "obv", "williamsr", "zscore", "atr"
#}



# === Caching ===
def get_cache_key(tf_minutes, symbol):
    return hashlib.md5((str(tf_minutes) + symbol).encode()).hexdigest()

def load_cache(tf_minutes, symbol):
    key = get_cache_key(tf_minutes, symbol)
    path = f".cache/merged_{key}.pkl"
    lock = FileLock(f"{path}.lock", timeout=10)
    if os.path.exists(path):
        try:
            with lock, open(path, "rb") as f:
                return pickle.load(f)
        except:
            os.remove(path)
    return None

def save_cache(tf_minutes, merged, symbol):
    os.makedirs(".cache", exist_ok=True)
    key = get_cache_key(tf_minutes, symbol)
    path = f".cache/merged_{key}.pkl"
    lock = FileLock(f"{path}.lock", timeout=10)
    with lock, open(path, "wb") as f:
        pickle.dump(merged, f)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-9)

# === Indicator Calculation ===
def compute_indicators(df, enabled_indicators):
    df = df.copy()
    if "ema" in enabled_indicators:
        df["ema_fast"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["ema_med"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["price_vs_ema"] = df["Close"] - df["ema_slow"]
        df["ema_diff"] = df["ema_fast"] - df["ema_med"]
    if "macd" in enabled_indicators:
        ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_sig"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd_line"] - df["macd_sig"]
    if "rsi" in enabled_indicators or "stoch_rsi" in enabled_indicators:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
    if "stoch_rsi" in enabled_indicators:
        print("compute stoch rsi")
        rsi = df["rsi"]
        df["stoch_rsi"] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-9)
    if "bollinger" in enabled_indicators:
        ma = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["bb_upper"] = ma + 2 * std
        df["bb_lower"] = ma - 2 * std
        df["bb_percent"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    if "adx" in enabled_indicators:
        up_move = df["High"].diff()
        down_move = df["Low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), -down_move, 0)
        tr = pd.DataFrame({
            "tr1": df["High"] - df["Low"],
            "tr2": abs(df["High"] - df["Close"].shift()),
            "tr3": abs(df["Low"] - df["Close"].shift())
        }).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
        df["adx"] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    if "cci" in enabled_indicators:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        ma = tp.rolling(20).mean()
        md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df["cci"] = (tp - ma) / (0.015 * md + 1e-9)
    if "ichimoku" in enabled_indicators:
        nine_high = df["High"].rolling(9).max()
        nine_low = df["Low"].rolling(9).min()
        df["tenkan_sen"] = (nine_high + nine_low) / 2
        high26 = df["High"].rolling(26).max()
        low26 = df["Low"].rolling(26).min()
        df["kijun_sen"] = (high26 + low26) / 2
    if "obv" in enabled_indicators:
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
    if "williamsr" in enabled_indicators:
        high = df["High"].rolling(14).max()
        low = df["Low"].rolling(14).min()
        df["williamsr"] = -100 * (high - df["Close"]) / (high - low + 1e-9)
    if "zscore" in enabled_indicators:
        mean = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["zscore"] = (df["Close"] - mean) / (std + 1e-9)
    if "atr" in enabled_indicators:
        tr = pd.DataFrame({
            "hl": df["High"] - df["Low"],
            "hc": (df["High"] - df["Close"].shift()).abs(),
            "lc": (df["Low"] - df["Close"].shift()).abs()
        }).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()
        df["atr_avg"] = df["atr"].rolling(ATR_AVG_LEN).mean()
    return df

# === Resampling ===
def resample_htf(df, tf_minutes, enabled_indicators):
    rule = f"{tf_minutes}min"
    agg = df.resample(rule, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    return compute_indicators(agg, enabled_indicators).shift(1)

def merge_htf_signals(base_df, tf_minutes, symbol, enabled_indicators):
    cached = load_cache(tf_minutes, symbol)
    if cached is not None:
        return cached
    htf_data = {}
    for name, minutes in tf_minutes.items():
        htf = resample_htf(base_df, minutes, enabled_indicators)
        htf.columns = [f"{col}_{name}" for col in htf.columns]
        htf_data[name] = htf
    merged = base_df.copy()
    for name in tf_minutes.keys():
        merged = merged.merge(htf_data[name], left_index=True, right_index=True, how="left")
    merged = merged.ffill()
    save_cache(tf_minutes, merged, symbol)
    return merged


# === Strategy Function ===

def run_strategy(df: pd.DataFrame, tp_mult, sl_mult, pyramiding, percent_of_equity, comp_threshold, align_threshold,progress_desc="Running Strategy", enabled_indicators: Set[str] = None) -> tuple:
    if enabled_indicators is None:
        enabled_indicators = {
            "ema", "macd", "rsi", "atr", "stoch_rsi", "bollinger", "adx",
            "cci", "ichimoku", "obv", "williamsr", "zscore"
        }

    equity = INITIAL_EQUITY
    open_trades = []
    closed_trades = []
    equity_curve = []
    daily_pnl = {}
    suffixes = ["base", "mid", "high"]
    debug_enabled = "bollinger" in enabled_indicators or "stoch_rsi" in enabled_indicators or "cci" in enabled_indicators
    for i in tqdm(range(500, len(df)), desc=progress_desc):
        row = df.iloc[i]
        ts = row.name
        open_trades = [t for t in open_trades if t.get("active", False)]

        # === Score Calculation ===
        comp_long, comp_short = 0, 0
        align_long, align_short = 0, 0

        if "ema" in enabled_indicators:
            ema_scores = [normalize(row[f"ema_diff_{suf}"], -1, 1) for suf in suffixes]
            ema_scores_s = [normalize(-row[f"ema_diff_{suf}"], -1, 1) for suf in suffixes]
            comp_long += np.mean(ema_scores)
            comp_short += np.mean(ema_scores_s)
            align_long += all(row[f"ema_diff_{suf}"] > 0 for suf in suffixes)
            align_short += all(row[f"ema_diff_{suf}"] < 0 for suf in suffixes)

        if "macd" in enabled_indicators:
            macd_scores = [normalize(row[f"macd_diff_{suf}"], -1, 1) for suf in suffixes]
            macd_scores_s = [normalize(-row[f"macd_diff_{suf}"], -1, 1) for suf in suffixes]
            comp_long += np.mean(macd_scores)
            comp_short += np.mean(macd_scores_s)
            align_long += all(row[f"macd_diff_{suf}"] > 0 for suf in suffixes)
            align_short += all(row[f"macd_diff_{suf}"] < 0 for suf in suffixes)

        if "rsi" in enabled_indicators:
            rsi_scores = [normalize(row[f"rsi_{suf}"], 30, 70) for suf in suffixes]
            comp_long += np.mean(rsi_scores)
            comp_short += np.mean(rsi_scores)

        if "price_vs_ema" in df.columns:
            price_scores = [normalize(row[f"price_vs_ema_{suf}"], -1, 1) for suf in suffixes]
            price_scores_s = [normalize(-row[f"price_vs_ema_{suf}"], -1, 1) for suf in suffixes]
            comp_long += np.mean(price_scores)
            comp_short += np.mean(price_scores_s)
            align_long += all(row[f"price_vs_ema_{suf}"] > 0 for suf in suffixes)
            align_short += all(row[f"price_vs_ema_{suf}"] < 0 for suf in suffixes)

        atr_boost = 0.0
        atr_base = row["atr_base"] if "atr" in enabled_indicators else row["Close"] * 0.005  # Fallback: 0.5% range
        atr_avg_base = row["atr_avg_base"] if "atr" in enabled_indicators else atr_base

        if "atr" in enabled_indicators and atr_base > atr_avg_base:
            atr_boost = 0.2
            comp_long += atr_boost
            comp_short += atr_boost
        
        if "stoch_rsi" in enabled_indicators:
            stoch_raws = [row.get(f"stoch_rsi_{suf}", 0.5) for suf in suffixes]
            
            # Long bias: reward low Stoch RSI (oversold)
            long_scores = [1 - s for s in stoch_raws]  # 1.0 at 0.0, 0.0 at 1.0
            comp_long += np.mean(long_scores)
            
            # Short bias: reward high Stoch RSI (overbought)
            short_scores = stoch_raws  # 1.0 at 1.0, 0.0 at 0.0
            comp_short += np.mean(short_scores)

            # Optional alignment condition: all 3 are oversold for long or overbought for short
            align_long += all(s < 0.2 for s in stoch_raws)
            align_short += all(s > 0.8 for s in stoch_raws)

        if "bollinger" in enabled_indicators:
            bb_scores = [1 - abs(row.get(f"bb_percent_{suf}", 0.5) - 0.5) * 2 for suf in suffixes]
            comp_long += np.mean(bb_scores)
            comp_short += np.mean([1 - s for s in bb_scores])
            align_long += all(row.get(f"bb_percent_{suf}", 0.5) < 0.3 for suf in suffixes)  # Below lower band
            align_short += all(row.get(f"bb_percent_{suf}", 0.5) > 0.7 for suf in suffixes)  # Above upper band

        if "cci" in enabled_indicators:
            cci_raws = [row.get(f"cci_{suf}", 0) for suf in suffixes]
            cci_scores = [min(abs(x) / 200, 1.0) for x in cci_raws]
            comp_long += np.mean(cci_scores)
            comp_short += np.mean([1 - s for s in cci_scores])

        if "adx" in enabled_indicators:
            adx_scores = [normalize(row.get(f"adx_{suf}", 20), 10, 50) for suf in suffixes]
            comp_long += np.mean(adx_scores)
            comp_short += np.mean(adx_scores)  # ADX indicates trend strength, not direction

        if "ichimoku" in enabled_indicators:
            tenkan_above_kijun = [row.get(f"tenkan_sen_{suf}", 0) > row.get(f"kijun_sen_{suf}", 0) for suf in suffixes]
            tenkan_below_kijun = [row.get(f"tenkan_sen_{suf}", 0) < row.get(f"kijun_sen_{suf}", 0) for suf in suffixes]
            comp_long += sum(tenkan_above_kijun) / len(suffixes)
            comp_short += sum(tenkan_below_kijun) / len(suffixes)
            align_long += all(tenkan_above_kijun)
            align_short += all(tenkan_below_kijun)

        if "obv" in enabled_indicators:
            obv_scores = [normalize(row.get(f"obv_{suf}", 0), -1e9, 1e9) for suf in suffixes]  # Rescaled
            comp_long += np.mean(obv_scores)
            comp_short += np.mean([1 - s for s in obv_scores])

        if "williamsr" in enabled_indicators:
            willr_scores = [normalize(row.get(f"williamsr_{suf}", -50), -100, 0) for suf in suffixes]
            comp_long += np.mean([1 - s for s in willr_scores])  # Closer to -100 is more oversold
            comp_short += np.mean(willr_scores)

        if "zscore" in enabled_indicators:
            z_scores = [normalize(row.get(f"zscore_{suf}", 0), -2, 2) for suf in suffixes]
            comp_long += np.mean([1 - abs(s - 0.25) for s in z_scores])  # Favor slight undervaluation
            comp_short += np.mean([1 - abs(s + 0.25) for s in z_scores])  # Favor slight overvaluation


        current_price = row["Close"]

        # === Entry Conditions ===
        if ALLOW_LONG and comp_long >= comp_threshold and align_long >= align_threshold and len(open_trades) < pyramiding:
            size = equity * percent_of_equity / current_price
            tp_price = current_price + atr_base * tp_mult
            sl_price = current_price - atr_base * sl_mult
            open_trades.append({
                "entry": current_price, "size": size, "tp": tp_price, "sl": sl_price,
                "type": "long", "active": True, "entry_time": ts, "atr": atr_base, "tpmult": tp_mult, "slmult": sl_mult
            })

        if ALLOW_SHORT and comp_short >= comp_threshold and align_short >= align_threshold and len(open_trades) < pyramiding:
            size = equity * percent_of_equity / current_price
            tp_price = current_price - atr_base * tp_mult
            sl_price = current_price + atr_base * sl_mult
            open_trades.append({
                "entry": current_price, "size": size, "tp": tp_price, "sl": sl_price,
                "type": "short", "active": True, "entry_time": ts, "atr": atr_base, "tpmult": tp_mult, "slmult": sl_mult
            })
        if debug_enabled and i % 50000 == 0 and 0.5 <= comp_long <= 1.5:
            print(f"\n🔍 Row {i} | Time: {ts}")
            print(f"comp_long: {comp_long:.3f}, align_long: {align_long}")
            #print(f"Scores – BB: {bb_scores if 'bollinger' in enabled_indicators else 'N/A'}")
            print(f"StochRSI: {stoch_raws if 'stoch_rsi' in enabled_indicators else 'N/A'}")
            print(f"StochRSI Long Scores: {long_scores if 'stoch_rsi' in enabled_indicators else 'N/A'}")
            #print(f"         CCI: {cci_scores if 'cci' in enabled_indicators else 'N/A'}")
        # === Exit Conditions ===
        for trade in open_trades:
            if not trade["active"]: continue
            if trade["type"] == "long":
                if row["Low"] <= trade["sl"]:
                    loss = (trade["entry"] - trade["sl"]) * trade["size"]
                    equity -= loss
                    daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0) - loss
                    trade["active"] = False
                    closed_trades.append({
                        "type": trade["type"], "entry_time": trade["entry_time"], "exit_time": ts,
                        "entry_price": trade["entry"], "exit_price": trade["sl"],
                        "size": trade["size"], "pnl": -loss, "reason": "sl",
                        "duration": (ts - trade["entry_time"]).total_seconds() / 60, "atr":trade["atr"], "tp_mult":trade["tpmult"], "sl_mult":trade["slmult"]
                    })
                elif row["High"] >= trade["tp"]:
                    gain = (trade["tp"] - trade["entry"]) * trade["size"]
                    equity += gain
                    daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0) + gain
                    trade["active"] = False
                    closed_trades.append({
                        "type": trade["type"], "entry_time": trade["entry_time"], "exit_time": ts,
                        "entry_price": trade["entry"], "exit_price": trade["tp"],
                        "size": trade["size"], "pnl": gain, "reason": "tp",
                        "duration": (ts - trade["entry_time"]).total_seconds() / 60, "atr":trade["atr"], "tp_mult":trade["tpmult"], "sl_mult":trade["slmult"]
                    })
            else:
                if row["High"] >= trade["sl"]:
                    loss = (trade["sl"] - trade["entry"]) * trade["size"]
                    equity -= loss
                    daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0) - loss
                    trade["active"] = False
                    closed_trades.append({
                        "type": trade["type"], "entry_time": trade["entry_time"], "exit_time": ts,
                        "entry_price": trade["entry"], "exit_price": trade["sl"],
                        "size": trade["size"], "pnl": -loss, "reason": "sl",
                        "duration": (ts - trade["entry_time"]).total_seconds() / 60, "atr":trade["atr"], "tp_mult":trade["tpmult"], "sl_mult":trade["slmult"]
                    })
                elif row["Low"] <= trade["tp"]:
                    gain = (trade["entry"] - trade["tp"]) * trade["size"]
                    equity += gain
                    daily_pnl[ts.date()] = daily_pnl.get(ts.date(), 0) + gain
                    trade["active"] = False
                    closed_trades.append({
                        "type": trade["type"], "entry_time": trade["entry_time"], "exit_time": ts,
                        "entry_price": trade["entry"], "exit_price": trade["tp"],
                        "size": trade["size"], "pnl": gain, "reason": "tp",
                        "duration": (ts - trade["entry_time"]).total_seconds() / 60, "atr":trade["atr"], "tp_mult":trade["tpmult"], "sl_mult":trade["slmult"]
                    })

        equity_curve.append((ts, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["Time", "Equity"]).set_index("Time")
    drawdown = (equity_df["Equity"].cummax() - equity_df["Equity"]).max()
    returns = equity_df["Equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if not returns.empty and returns.std() != 0 else 0
    pnl_df = pd.DataFrame(list(daily_pnl.items()), columns=["Date", "PnL"]).set_index("Date")
    return equity, len(equity_curve), drawdown, sharpe, equity_df, pnl_df, closed_trades


# === Backtest Worker ===
def run_config(params):
    index, total, tp_mult, sl_mult, pyramiding, equity_pct, tf_minutes, df, symbol, enabled_indicators, group_name,comp_threshold,align_threshold = params

    print(f"\n🧪 Running config {index}/{total}: TP={tp_mult}, SL={sl_mult}, Pyr={pyramiding}, "
          f"Eq%={equity_pct}, TF={tf_minutes}, Indicators={enabled_indicators}")

    merged = merge_htf_signals(df, tf_minutes, symbol, enabled_indicators)
    result = run_strategy(merged, tp_mult, sl_mult, pyramiding, equity_pct, enabled_indicators=enabled_indicators,comp_threshold=comp_threshold,align_threshold=align_threshold)
    final_equity, bars, dd, sharpe, eq_curve, pnl_df, trades = result

    config_id = (
        f"{symbol}_TP{tp_mult}_SL{sl_mult}_Pyr{pyramiding}_Pct{int(equity_pct*100)}_"
        f"TF{tf_minutes['base']}_{tf_minutes['mid']}_{tf_minutes['high']}_"
        f"{group_name}"
    )

    os.makedirs("results", exist_ok=True)
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(f"results/trades_{config_id}.csv", index=False)

    profitable_trades = sum(1 for t in trades if t["pnl"] > 0)
    total_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    profit_factor = total_profit / total_loss if total_loss != 0 else float("inf")
    total_profit_absolute = final_equity - INITIAL_EQUITY

    summary = {
        "config": config_id,
        "final_equity": final_equity,
        "drawdown": dd,
        "sharpe": sharpe,
        "total_profit": total_profit_absolute,
        "num_trades": len(trades),
        "profitable_trades": profitable_trades,
        "profit_factor": profit_factor
    }

    return summary


# === Main Entry ===
def run_backtests(symbol, TP_MULT_LIST, SL_MULT_LIST, PYRAMIDING_LIST, EQUITY_PCT_LIST, TF_CONFIGS, data_dir, enabled_indicators,group_name,comp_threshold,align_threshold):
    data_folder = "data/1m data"
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    all_results = []

    for file in all_files:
        file_path = os.path.join(data_folder, file)
        symbol = file.replace("_1m.csv", "").replace(".csv", "")
        print(f"\n📂 Running backtests for {symbol}...\n")

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Open time"] if "Open time" in df.columns else df.columns[0])
        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

        combos = list(product(TP_MULT_LIST, SL_MULT_LIST, PYRAMIDING_LIST, EQUITY_PCT_LIST, TF_CONFIGS))
        inputs = [
            (i + 1, len(combos), tp, sl, pyr, pct, tf, df, symbol, enabled_indicators,group_name,comp_threshold,align_threshold)
            for i, (tp, sl, pyr, pct, tf) in enumerate(combos)
        ]

        print("\n🔧 Precomputing unique cache entries...\n")
        seen_keys = set()
        for input_tuple in inputs:
            tf = input_tuple[6]
            tf_key = frozenset(tf.items())
            if tf_key not in seen_keys:
                seen_keys.add(tf_key)
                print(f"📦 Caching TF config: {tf}")
                print(f"📦 Caching indicators config: {enabled_indicators}")
                merge_htf_signals(df, tf, symbol, enabled_indicators)

        results = []
        try:
            with Pool(processes=MAX_PARALLEL) as pool:
                for result in tqdm(pool.imap_unordered(run_config, inputs), total=len(inputs)):
                    results.append(result)
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.")
            pool.terminate()
            pool.join()
            return

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(f"results/summary_{symbol}_{now}.csv", index=False)

        all_results.extend(results)

    if all_results:
        final_summary = pd.DataFrame(all_results)
        final_summary.to_csv(f"results/full_summary_{now}.csv", index=False)
        print("\n✅ All backtests complete. Master summary saved.")



if __name__ == "__main__":
    run_backtests()
