# compute_returns.py
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

# Transform rules for different data types
PASS_THROUGH = {"FF_MKT_RF", "FF_SMB", "FF_HML", "FF_RMW", "FF_CMA", "FF_UMD"}
PCT_CHANGE = {
    "CPIAUCSL", "CPILFESL", "PPIACO", "PPIIDC", "INDPRO", "M2SL",
    "PAYEMS", "HOUST", "PERMIT", "DEXJPUS", "DEXUSUK", "DEXCAUS", 
    "DTWEXM", "DCOILWTICO"
}
DIFF = {"DFF", "DGS10", "TB3MS", "T10Y2Y", "T10YIE", "UNRATE", "CAPE", "SPX_DY"}

def clean_data(series):
    """Remove infinite values and convert to float"""
    return series.astype(float).replace([np.inf, -np.inf], np.nan)

def log_return(series):
    """Calculate log returns where both current and previous values are positive"""
    clean_series = clean_data(series)
    mask = (clean_series > 0) & (clean_series.shift(1) > 0)
    returns = np.where(mask, np.log(clean_series / clean_series.shift(1)), np.nan)
    return pd.Series(returns, index=series.index)

def pct_change(series):
    """Calculate percentage change, treating zeros as missing"""
    clean_series = clean_data(series).replace(0.0, np.nan)
    return clean_series.pct_change(fill_method=None)

def diff(series):
    """Calculate first difference for levels that can be negative"""
    clean_series = clean_data(series)
    return clean_series.diff()

def winsorize(df, lower=0.01, upper=0.99):
    """Cap extreme values at specified quantiles"""
    result = df.copy()
    for col in result.columns:
        series = result[col]
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        result[col] = series.clip(lower=lower_bound, upper=upper_bound)
    return result

def zscore(df):
    """Standardize each column to zero mean and unit variance"""
    return (df - df.mean(skipna=True)) / df.std(skipna=True, ddof=0)

def transform_series(series, proxy):
    """Apply appropriate transformation based on data type"""
    if proxy in PASS_THROUGH:
        return series
    elif proxy in PCT_CHANGE or (series.dropna() > 0).all():
        return pct_change(series)
    elif proxy in DIFF or not (series.dropna() > 0).all():
        return diff(series)
    else:
        return pct_change(series)

def load_raw_data():
    """Load all price data from database"""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    query = """
    SELECT i.instrument_id, i.proxy, p.date,
           p.open, p.high, p.low, p.close, p.adj_close,
           p.value, p.currency
    FROM prices p
    JOIN instruments i ON p.instrument_id = i.instrument_id
    """
    
    df = pd.read_sql(query, engine, parse_dates=["date"])
    
    if df.empty:
        raise SystemExit("No data found. Run data loading scripts first.")
    
    return df, engine

def compute_monthly_returns():
    """Transform daily data to monthly returns"""
    df, engine = load_raw_data()
    
    monthly_data = []
    
    for (instrument_id, proxy), group in df.groupby(["instrument_id", "proxy"]):
        group = group.sort_values("date")
        
        # Use adjusted close for equity/ETF data
        if group["adj_close"].notna().any():
            series = group.set_index("date")["adj_close"].resample("ME").last()
            returns = log_return(series)
        else:
            # Use value column for macro/factor data
            series = group.set_index("date")["value"].resample("ME").last()
            returns = transform_series(series, proxy)
        
        # Store results
        monthly_data.append(pd.DataFrame({
            "instrument_id": instrument_id,
            "proxy": proxy,
            "date": returns.index,
            "value": returns.values
        }))
    
    return pd.concat(monthly_data, ignore_index=True).sort_values(["date", "proxy"])

def save_monthly_data(monthly_data, engine):
    """Save monthly returns in both long and wide formats"""
    # Save long format
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS prices_monthly"))
    monthly_data.to_sql("prices_monthly", engine, index=False)
    print("Saved monthly returns (long format)")
    
    # Create wide format
    wide = monthly_data.pivot_table(index="date", columns="proxy", values="value", aggfunc="last")
    wide = wide.sort_index()
    
    # Save raw wide format
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS factors_monthly_raw"))
    wide.reset_index().to_sql("factors_monthly_raw", engine, index=False)
    print("Saved raw factors (wide format)")
    
    # Save winsorized and standardized version
    processed = winsorize(wide, lower=0.01, upper=0.99)
    processed = zscore(processed)
    
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS factors_monthly_z"))
    processed.reset_index().to_sql("factors_monthly_z", engine, index=False)
    print("Saved standardized factors (wide format)")

def compute_all_returns():
    """Main function to compute and save all monthly returns"""
    print("Computing monthly returns...")
    monthly_data = compute_monthly_returns()
    
    _, engine = load_raw_data()
    save_monthly_data(monthly_data, engine)
    
    print(f"Processed {len(monthly_data)} monthly observations")

if __name__ == "__main__":
    compute_all_returns()