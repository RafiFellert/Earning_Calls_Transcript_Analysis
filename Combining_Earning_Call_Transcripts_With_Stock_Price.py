import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
import inspect

# ==========================================
# CONFIGURATION / INPUTS
# ==========================================
root_folder = Path(
    "./transcripts"
)  # Root folder containing all ticker subfolders
output_csv = "transcripts_with_returns.csv"

# Regex Patterns:
# Matches both YYYY-MM-DD and YYYY-Mmm-DD formats (e.g., 2016-04-16 or 2016-Apr-16)
date_pattern = re.compile(r"(\d{4}-(?:\d{2}|[A-Za-z]{3})-\d{2})")

# Extracts the ticker directly before .txt (e.g., "2016-04-16-AAPL.txt" -> "AAPL")
ticker_pattern = re.compile(r"-([A-Za-z0-9]+)\.txt$", re.IGNORECASE)

# ==========================================
# 1. DISCOVER & READ ALL TXT FILES
# ==========================================
print(f"Scanning for .txt files in: {root_folder.resolve()}...")

# rglob("*.txt") recursively searches root_folder and all subdirectories
txt_files = list(root_folder.rglob("*.txt"))
print(f"Found {len(txt_files)} text files across all subfolders.")

raw_records = []

for file_path in txt_files:
    filename = file_path.name

    # Extract Date
    date_match = date_pattern.search(filename)
    dt_obj = None
    formatted_date_str = None
    print(f"Line {inspect.currentframe().f_lineno}:  date_match = {date_match}")
    if date_match:
        raw_date_str = date_match.group(1)
        # Handle numeric month vs abbreviated text month
        for fmt in ("%Y-%m-%d", "%Y-%b-%d"):
            try:
                dt_obj = datetime.strptime(raw_date_str, fmt)
                formatted_date_str = dt_obj.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    # Extract Ticker from filename
    ticker_match = ticker_pattern.search(filename)
    ticker = ticker_match.group(1).upper() if ticker_match else None

    # Read File Content
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        file_content = ""

    raw_records.append(
        {
            "Ticker": ticker,
            "Date_Obj": dt_obj,
            "Date": formatted_date_str,
            "TXT": file_content,
            "File_Path": str(file_path),
        }
    )

df_all = pd.DataFrame(raw_records)

# ==========================================
# 2. BATCH FETCH STOCK DATA VIA YFINANCE
# ==========================================
valid_mask = df_all["Ticker"].notna() & df_all["Date_Obj"].notna()
valid_records = df_all[valid_mask].copy()

price_movement_map = {}

if not valid_records.empty:
    unique_tickers = valid_records["Ticker"].unique().tolist()
    min_date = valid_records["Date_Obj"].min() - timedelta(days=2)
    max_date = valid_records["Date_Obj"].max() + timedelta(days=5)

    print(
        f"\nBatch fetching stock prices for {len(unique_tickers)} tickers "
        f"between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}..."
    )

    # threads=False & ignore_tz=True prevents SQLite database locks in yfinance
    stock_data = yf.download(
        tickers=unique_tickers,
        start=min_date.strftime("%Y-%m-%d"),
        end=max_date.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
        ignore_tz=True,
    )
    
    if len(unique_tickers) == 1 and isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(0)
        
    # Process price evaluations per transcript record
    for _, row in valid_records.iterrows():
        t = row["Ticker"]
        dt = row["Date_Obj"]
        date_str = row["Date"]

        start_str = dt.strftime("%Y-%m-%d")
        end_str = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            if len(unique_tickers) > 1:
                df_ticker = stock_data[t].dropna(how="all")
            else:
                df_ticker = stock_data.dropna(how="all")

            df_slice = df_ticker.loc[start_str:end_str]

            print(f"Line {inspect.currentframe().f_lineno}:  df_slice.empty = {df_slice.empty}, len(df_slice) = {len(df_slice)}")
            if not df_slice.empty and len(df_slice) >= 1:
                if isinstance(df_slice.columns, pd.MultiIndex):
                    start_open = float(
                        df_slice[("Open", ticker)].iloc[0]
                    )
                else:
                    start_open = float(
                        df_slice["Open"].iloc[0]
                    )
                end_close = float(df_slice["Close"].iloc[-1])
                is_up = "YES" if end_close > start_open else "NO"
                print(f"Line {inspect.currentframe().f_lineno}:  Ris_up = {is_up}")
            else:
                is_up = "NO DATA"
                print(f"Line {inspect.currentframe().f_lineno}:  Xis_up = {is_up}")
        except Exception as e:
            is_up = "NO DATA"
            print(
                f"Line {inspect.currentframe().f_lineno}: "
                f"is_up = {is_up}, "
                f"Exception Type = {type(e).__name__}, "
                f"Message = {e}"
            )

        print(f"Line {inspect.currentframe().f_lineno}:  Zis_up = {is_up}")
        price_movement_map[(t, date_str)] = is_up

# Map results back to main dataframe
df_all["Is Price UP"] = df_all.apply(
    lambda r: price_movement_map.get((r["Ticker"], r["Date"]), "N/A"), axis=1
)

# ==========================================
# 3. EXPORT FINAL OUTPUT
# ==========================================
final_df = df_all[["Date", "Ticker", "TXT", "Is Price UP"]].copy()

final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\nFinished! Processed {len(final_df)} files.")
print(f"Excel successfully saved to: {output_csv}")
