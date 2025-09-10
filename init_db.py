# init_db.py
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

def create_tables():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    schema_sql = """
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id INTEGER PRIMARY KEY,
        proxy         TEXT NOT NULL,
        name          TEXT,
        category      TEXT,
        data_type     TEXT,
        source_hint   TEXT,
        UNIQUE (proxy, data_type)
    );

    CREATE TABLE IF NOT EXISTS prices (
        instrument_id INTEGER NOT NULL,
        date          DATE NOT NULL,
        open          REAL, 
        high          REAL, 
        low           REAL, 
        close         REAL, 
        adj_close     REAL,
        value         REAL, 
        currency      TEXT,
        PRIMARY KEY (instrument_id, date),
        FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
    );

    CREATE TABLE IF NOT EXISTS loads (
        instrument_id INTEGER NOT NULL,
        loaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        start_date    DATE,
        end_date      DATE,
        notes         TEXT
    );
    """
    
    with engine.begin() as conn:
        statements = [stmt.strip() for stmt in schema_sql.strip().split(';') if stmt.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
    
    print(f"Database initialized at: {DB_PATH}")

if __name__ == "__main__":
    create_tables()