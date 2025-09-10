# seed_instruments.py
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"
CSV_PATH = "proxies.csv"

def seed_instruments():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    # Create table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id INTEGER PRIMARY KEY,
        proxy         TEXT NOT NULL,
        name          TEXT,
        category      TEXT,
        data_type     TEXT,
        source_hint   TEXT,
        UNIQUE (proxy, data_type)
    );
    """
    
    upsert_sql = """
    INSERT INTO instruments (proxy, name, category, data_type, source_hint)
    VALUES (:proxy, :name, :category, :data_type, :source_hint)
    ON CONFLICT(proxy, data_type) DO UPDATE SET
        name=excluded.name,
        category=excluded.category,
        source_hint=excluded.source_hint;
    """
    
    # Read and validate CSV
    try:
        df = pd.read_csv(CSV_PATH, encoding='cp1252')
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH)
    
    required_columns = {"proxy", "name", "category", "data_type", "source_hint"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")
    
    # Insert data
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        
        for _, row in df.iterrows():
            conn.execute(text(upsert_sql), {
                "proxy": row["proxy"],
                "name": row["name"],
                "category": row["category"],
                "data_type": row["data_type"],
                "source_hint": row["source_hint"],
            })
    
    print(f"Loaded {len(df)} instruments into database")

if __name__ == "__main__":
    seed_instruments()