# load_fred.py
import os
import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

def get_fred_instruments():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    query = """
    SELECT instrument_id, proxy
    FROM instruments
    WHERE data_type = 'FRED'
    ORDER BY instrument_id
    """
    return pd.read_sql(query, engine)

def save_series(instrument_id, series, currency="USD"):
    if series is None or series.empty:
        return 0
    
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    # Prepare data
    df = series.dropna().reset_index()
    df.columns = ["date", "value"]
    df["instrument_id"] = instrument_id
    df["currency"] = currency
    
    # Save via temp table
    df.to_sql("_fred_temp", engine, if_exists="replace", index=False)
    
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO prices (instrument_id, date, value, currency)
            SELECT instrument_id, date, value, currency FROM _fred_temp
        """))
        conn.execute(text("DROP TABLE _fred_temp"))
    
    return len(df)

def load_fred_data():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable not set")
    
    fred = Fred(api_key=api_key)
    instruments = get_fred_instruments()
    
    if instruments.empty:
        print("No FRED instruments found in database")
        return
    
    total_loaded = 0
    for _, row in instruments.iterrows():
        instrument_id = int(row["instrument_id"])
        series_id = row["proxy"]
        
        try:
            series = fred.get_series(series_id)
            count = save_series(instrument_id, series)
            print(f"Loaded {series_id}: {count:,} records")
            total_loaded += count
        except Exception as e:
            print(f"Failed to load {series_id}: {e}")
    
    print(f"Total records loaded: {total_loaded:,}")

if __name__ == "__main__":
    load_fred_data()