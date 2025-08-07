import os
import shutil
import re

def organize_files():
    SOURCE_DIR = "results"
    DEST_TRADES = os.path.join(SOURCE_DIR, "Trades")
    DEST_SUMMARY = os.path.join(SOURCE_DIR, "Summary")

    # Pattern for trade files
    trade_pattern = re.compile(
        r"trades_(?P<coin>[A-Z]+USDT)_1m_(?P<year>\d{4})_TP[\d.]+_SL[\d.]+_Pyr\d+_Pct\d+_TF(?P<tf>\d+_\d+_\d+)(?:_Group[A-Z])?\.csv"
    )

    # Pattern for summary files
    summary_pattern = re.compile(
        r"summary_(?P<coin>[A-Z]+USDT)_1m_(?P<year>\d{4})_(?P<datetime>\d{8}_\d{6})\.csv"
    )
    # Pattern for full summary files
    full_summary_pattern = re.compile(
        r"full_summary_(?P<datetime>\d{8}_\d{6})\.csv"
    )

    moved = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in files:
            # === TRADE FILES ===
            if filename.startswith("trades_") and filename.endswith(".csv"):
                trade_match = trade_pattern.match(filename)
                if trade_match:
                    coin = trade_match.group("coin")
                    year = trade_match.group("year")
                    tf = trade_match.group("tf")

                    dest_dir = os.path.join(DEST_TRADES, coin, year, tf)
                    os.makedirs(dest_dir, exist_ok=True)

                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(dest_dir, filename)

                    if os.path.abspath(src_path) != os.path.abspath(dest_path):
                        shutil.move(src_path, dest_path)
                        moved += 1

            # === SUMMARY FILES ===
            elif filename.startswith("summary_") and filename.endswith(".csv"):
                summary_match = summary_pattern.match(filename)
                if summary_match:
                    coin = summary_match.group("coin")
                    dest_dir = os.path.join(DEST_SUMMARY, coin)
                    os.makedirs(dest_dir, exist_ok=True)

                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(dest_dir, filename)

                    if os.path.abspath(src_path) != os.path.abspath(dest_path):
                        shutil.move(src_path, dest_path)
                        moved += 1
            # === full SUMMARY FILES ===
            elif filename.startswith("full_summary_") and filename.endswith(".csv"):
                summary_match = full_summary_pattern.match(filename)
                if summary_match:
                    #coin = summary_match.group("coin")
                    #dest_dir = os.path.join(DEST_SUMMARY, coin)
                    os.makedirs(DEST_SUMMARY, exist_ok=True)

                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(DEST_SUMMARY, filename)

                    if os.path.abspath(src_path) != os.path.abspath(dest_path):
                        shutil.move(src_path, dest_path)
                        moved += 1            

    print(f"✅ Trade & summary files organized. {moved} files moved.")

# ✅ Run if executed directly
if __name__ == "__main__":
    organize_files()
