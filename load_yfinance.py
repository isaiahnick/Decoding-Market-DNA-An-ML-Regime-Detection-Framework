# load_yfinance.py
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

def get_yfinance_instruments():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    query = """
    SELECT instrument_id, proxy
    FROM instruments
    WHERE source_hint LIKE '%yfinance%'
      AND (data_type IS NULL OR data_type NOT IN ('DIY'))
    ORDER BY instrument_id
    """
    return pd.read_sql(query, engine)

def save_ohlc_data(instrument_id, df, currency="USD"):
    if df is None or df.empty:
        return 0
    
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    # Clean and prepare data
    data = df.reset_index().rename(columns=str.lower)
    data.rename(columns={"adj close": "adj_close"}, inplace=True)
    
    # Ensure all OHLC columns exist
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col not in data.columns:
            data[col] = None
    
    # Select required columns
    data = data[["date", "open", "high", "low", "close", "adj_close"]].copy()
    data["instrument_id"] = instrument_id
    data["currency"] = currency
    
    # Save via temp table
    data.to_sql("_yf_temp", engine, if_exists="replace", index=False)
    
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO prices 
                (instrument_id, date, open, high, low, close, adj_close, currency)
            SELECT instrument_id, date, open, high, low, close, adj_close, currency 
            FROM _yf_temp
        """))
        conn.execute(text("DROP TABLE _yf_temp"))
    
    return len(data)

def load_yfinance_data(start_date="1970-01-01", end_date=None):
    instruments = get_yfinance_instruments()
    
    if instruments.empty:
        print("No Yahoo Finance instruments found in database")
        return
    
    total_loaded = 0
    for _, row in instruments.iterrows():
        instrument_id = int(row["instrument_id"])
        symbol = row["proxy"]
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
            count = save_ohlc_data(instrument_id, hist)
            print(f"Loaded {symbol}: {count:,} records")
            total_loaded += count
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
    
    print(f"Total records loaded: {total_loaded:,}")

if __name__ == "__main__":
    load_yfinance_data()