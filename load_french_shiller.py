# load_french_shiller.py
import io
import zipfile
import re
import requests
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"
SHILLER_CSV = "/Users/isaiahnick/Market Regime/data/shiller_data.csv"

FRENCH_5F_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FRENCH_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

def save_factor_series(proxy, series, currency="USD"):
    """Save a factor series to the database"""
    if series is None or series.empty:
        return 0
    
    clean_series = series.dropna()
    if clean_series.empty:
        return 0
    
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    # Get instrument ID
    instrument_query = "SELECT instrument_id FROM instruments WHERE proxy = :proxy"
    result = pd.read_sql(instrument_query, engine, params={"proxy": proxy})
    if result.empty:
        raise RuntimeError(f"Instrument '{proxy}' not found in database")
    
    instrument_id = int(result.iloc[0, 0])
    
    # Prepare data
    df = clean_series.reset_index()
    df.columns = ["date", "value"]
    df["instrument_id"] = instrument_id
    df["currency"] = currency
    
    # Save via temp table
    df.to_sql("_factor_temp", engine, if_exists="replace", index=False)
    
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO prices (instrument_id, date, value, currency)
            SELECT instrument_id, date, value, currency FROM _factor_temp
        """))
        conn.execute(text("DROP TABLE _factor_temp"))
    
    return len(df)

def download_french_csv(url):
    """Download and extract CSV from French data library zip file"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        with zf.open(csv_files[0]) as f:
            return f.read().decode("utf-8", errors="ignore")

def parse_french_monthly_data(csv_text):
    """Parse monthly data from French CSV text"""
    lines = csv_text.splitlines()
    
    # Find first data line (YYYYMM format)
    data_pattern = re.compile(r"^\s*\d{6}\s*,")
    data_start = None
    header_line = None
    
    for i, line in enumerate(lines):
        if data_pattern.match(line):
            data_start = i
            # Find header line (skip blank lines going backwards)
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            header_line = j if j >= 0 else None
            break
    
    if data_start is None or header_line is None:
        raise RuntimeError("Could not parse French data format")
    
    # Parse header
    header = [col.strip() for col in lines[header_line].split(",")]
    if len(header) <= 1:
        header = lines[header_line].strip().split()
    
    if not header[0] or header[0].lower() not in ("", " ", "date", "yyyymm"):
        header[0] = "Date"
    
    # Collect data lines
    data_lines = []
    for k in range(data_start, len(lines)):
        line = lines[k]
        if not line.strip() or not data_pattern.match(line):
            if "Annual" in line:
                break
        data_lines.append(line)
    
    # Create DataFrame
    csv_data = ",".join(header) + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Filter valid date rows and convert dates
    df = df[df.iloc[:,0].astype(str).str.match(r"^\d{6}$")].copy()
    
    date_col = df.iloc[:,0].astype(str)
    years = date_col.str.slice(0,4).astype(int)
    months = date_col.str.slice(4,6).astype(int)
    df["date"] = pd.to_datetime(dict(year=years, month=months, day=1)) + MonthEnd(0)
    
    df = df.drop(columns=[df.columns[0]])
    
    # Convert percentages to decimals
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    
    return df.set_index("date").sort_index()

def load_french_factors():
    """Load French factor data"""
    # Load 5-factor data
    print("Downloading French 5-factor data...")
    factor5_csv = download_french_csv(FRENCH_5F_URL)
    factor5_data = parse_french_monthly_data(factor5_csv)
    
    # Load momentum data
    print("Downloading French momentum data...")
    momentum_csv = download_french_csv(FRENCH_MOM_URL)
    momentum_data = parse_french_monthly_data(momentum_csv)
    
    # Map factor columns
    factor_cols = {col.strip().upper(): col for col in factor5_data.columns}
    
    smb_col = factor_cols.get("SMB")
    hml_col = factor_cols.get("HML") 
    rmw_col = factor_cols.get("RMW")
    cma_col = factor_cols.get("CMA")
    
    if not all([smb_col, hml_col, rmw_col, cma_col]):
        raise RuntimeError(f"Missing French factors in data: {list(factor5_data.columns)}")
    
    # Find momentum column
    mom_cols = [col for col in momentum_data.columns if col.strip().upper().startswith("MOM")]
    if not mom_cols:
        raise RuntimeError(f"No momentum column found: {list(momentum_data.columns)}")
    mom_col = mom_cols[0]
    
    # Save factors
    factors = [
        ("FF_SMB", factor5_data[smb_col]),
        ("FF_HML", factor5_data[hml_col]),
        ("FF_RMW", factor5_data[rmw_col]),
        ("FF_CMA", factor5_data[cma_col]),
        ("FF_UMD", momentum_data[mom_col])
    ]
    
    total_records = 0
    for factor_name, factor_series in factors:
        count = save_factor_series(factor_name, factor_series)
        total_records += count
        start_date = factor_series.dropna().index.min()
        end_date = factor_series.dropna().index.max()
        print(f"Loaded {factor_name}: {count} records")
    
    print(f"Total French factor records: {total_records:,}")

def parse_shiller_dates(date_column):
    """Parse various Shiller date formats to month-end dates"""
    dates = date_column.astype(str).str.strip()
    
    # Try standard YYYY-MM format first
    parsed_dates = pd.to_datetime(dates + "-01", errors="coerce")
    
    # Handle other formats
    missing_dates = parsed_dates.isna()
    if missing_dates.any():
        # Try normalized formats
        normalized = dates.str.replace(r"[./]", "-", regex=True)
        normalized = normalized.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-\2-01", regex=True)
        parsed_normalized = pd.to_datetime(normalized, errors="coerce")
        parsed_dates.loc[missing_dates] = parsed_normalized.loc[missing_dates]
        
        # Try float format (YYYY.MM)
        still_missing = parsed_dates.isna()
        if still_missing.any():
            float_dates = pd.to_numeric(dates[still_missing], errors="coerce")
            valid_floats = float_dates.notna()
            
            if valid_floats.any():
                float_values = float_dates[valid_floats]
                years = np.floor(float_values).astype(int)
                months = np.rint((float_values - years) * 100).astype(int).clip(1, 12)
                date_df = pd.DataFrame({'year': years, 'month': months, 'day': 1})
                float_parsed = pd.to_datetime(date_df, errors="coerce")
                parsed_dates.loc[still_missing & valid_floats] = float_parsed
    
    return parsed_dates + MonthEnd(0)

def load_shiller_data():
    """Load Shiller CAPE and dividend yield data"""
    df = pd.read_csv(SHILLER_CSV)
    
    # Find required columns (case insensitive)
    col_map = {col.lower(): col for col in df.columns}
    date_col = col_map.get("date")
    p_col = col_map.get("p")
    d_col = col_map.get("d") 
    cape_col = col_map.get("cape")
    
    if not all([date_col, p_col, d_col, cape_col]):
        raise RuntimeError(f"Missing required columns. Found: {list(df.columns)}")
    
    # Parse dates
    df["parsed_date"] = parse_shiller_dates(df[date_col])
    df = df.dropna(subset=["parsed_date"]).sort_values("parsed_date")
    
    # Clean CAPE data
    cape_data = df[cape_col].replace("NA", np.nan)
    cape_data = pd.to_numeric(cape_data, errors="coerce")
    cape_series = cape_data.copy()
    cape_series.index = df["parsed_date"]
    
    # Calculate dividend yield
    p_data = pd.to_numeric(df[p_col], errors="coerce")
    d_data = pd.to_numeric(df[d_col], errors="coerce") 
    dy_data = (d_data / p_data).replace([np.inf, -np.inf], np.nan)
    dy_series = dy_data.copy()
    dy_series.index = df["parsed_date"]
    
    # Save data
    cape_count = save_factor_series("CAPE", cape_series)
    dy_count = save_factor_series("SPX_DY", dy_series)
    
    print(f"Loaded CAPE: {cape_count} records")
    print(f"Loaded SPX_DY: {dy_count} records")

def load_all_data():
    """Load both French factors and Shiller data"""
    print("Loading French factors...")
    load_french_factors()
    
    print("\nLoading Shiller data...")
    load_shiller_data()

if __name__ == "__main__":
    load_all_data()