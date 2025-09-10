# data_summary.py
import pandas as pd
from sqlalchemy import create_engine

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

def get_data_availability():
    """Get data availability summary for all instruments"""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    query = """
    SELECT 
        i.proxy, i.name, i.category, i.data_type,
        MIN(p.date) as first_date,
        COUNT(p.date) as total_records
    FROM instruments i
    LEFT JOIN prices p ON i.instrument_id = p.instrument_id
    GROUP BY i.instrument_id, i.proxy, i.name, i.category, i.data_type
    ORDER BY i.category, i.proxy
    """
    
    df = pd.read_sql(query, engine)
    df['first_date'] = pd.to_datetime(df['first_date'])
    return df

def print_availability_report(df):
    """Print formatted data availability report"""
    print("Data Availability Report")
    print("=" * 100)
    print(f"{'Proxy':<15} {'Name':<45} {'First Date':<12} {'Records':<10}")
    print("-" * 100)
    
    current_category = None
    for _, row in df.iterrows():
        if row['category'] != current_category:
            if current_category is not None:
                print()
            current_category = row['category']
            print(f"\n{row['category'].upper()}:")
            print("-" * 50)
        
        proxy = row['proxy'][:14]
        name = row['name'][:44] if row['name'] else "N/A"
        first_date = row['first_date'].strftime('%Y-%m-%d') if pd.notna(row['first_date']) else "NO DATA"
        records = f"{row['total_records']:,}" if row['total_records'] > 0 else "0"
        
        print(f"  {proxy:<13} {name:<44} {first_date:<12} {records:<10}")

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    has_data = df[df['total_records'] > 0]
    no_data = df[df['total_records'] == 0]
    
    print(f"Instruments with data: {len(has_data)}")
    print(f"Instruments without data: {len(no_data)}")
    print(f"Total instruments: {len(df)}")
    
    if len(has_data) > 0:
        earliest = has_data['first_date'].min()
        latest = has_data['first_date'].max()
        print(f"\nEarliest data starts: {earliest.strftime('%Y-%m-%d')}")
        print(f"Latest data starts: {latest.strftime('%Y-%m-%d')}")
        
        if len(no_data) > 0:
            print(f"\nInstruments missing data:")
            for _, row in no_data.iterrows():
                print(f"  - {row['proxy']} ({row['name']})")

def check_instrument(proxy_name, limit=10):
    """Show detailed data for a specific instrument"""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    query = """
    SELECT i.proxy, i.name, i.category, i.data_type,
           p.date, p.value, p.close, p.adj_close
    FROM instruments i
    LEFT JOIN prices p ON i.instrument_id = p.instrument_id
    WHERE i.proxy = :proxy
    ORDER BY p.date
    LIMIT :limit
    """
    
    df = pd.read_sql(query, engine, params={"proxy": proxy_name, "limit": limit})
    
    if df.empty:
        print(f"No data found for: {proxy_name}")
        return None
    
    print(f"\nData sample for {proxy_name}:")
    print("-" * 50)
    print(df.to_string(index=False))
    return df

def run_data_summary():
    """Generate complete data availability report"""
    df = get_data_availability()
    print_availability_report(df)
    print_summary_stats(df)
    return df

if __name__ == "__main__":
    run_data_summary()