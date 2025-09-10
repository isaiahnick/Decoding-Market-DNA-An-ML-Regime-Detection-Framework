# Setup Instructions: Adding New Factors and Running Regime Analysis

This guide walks you through adding custom proxy factors to the regime detection model and getting it running successfully.

## Prerequisites

- Python 3.8+ with required packages:
  ```bash
  pip install pandas numpy sqlalchemy yfinance fredapi requests scikit-learn matplotlib seaborn
  ```

## Step 1: Get FRED API Key

1. Go to [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
2. Create a free account
3. Navigate to "My Account" → "API Keys"
4. Request an API key (usually approved instantly)
5. Set the environment variable:
   ```bash
   export FRED_API_KEY=your_api_key_here
   ```
   Or add to your shell profile:
   ```bash
   echo 'export FRED_API_KEY=your_api_key_here' >> ~/.bashrc
   source ~/.bashrc
   ```

## Step 2: Download and Prepare Shiller Data

1. **Download the data:**
   - Go to [http://www.econ.yale.edu/~shiller/data.htm](http://www.econ.yale.edu/~shiller/data.htm)
   - Download "U.S. Stock Markets 1871-Present and CAPE Ratio" (Excel file)

2. **Clean the data:**
   - Open the Excel file and locate the monthly data sheet
   - Export columns for: Date, Price (P), Dividend (D), CAPE
   - Save as `shiller_data.csv` in your `/data` folder
   - Ensure the format looks like:
     ```csv
     Date,P,D,CAPE
     1871-01,4.44,0.26,NA
     1871-02,4.50,0.26,NA
     ```
   - Date format should be YYYY-MM

3. **Verify the file:**
   - Check that CAPE values start appearing (usually around 1881)
   - Ensure no extra header rows or footnotes

## Step 3: Adding New Proxy Factors

### Edit the proxies.csv file:

The file requires these columns: `category,proxy,name,source_hint,data_type`

**For FRED data:**
```csv
INTEREST RATES,UNRATE,Unemployment Rate,FRED,FRED
COMMODITIES,GOLDAMGBD228NLBM,Gold Price,FRED,FRED
```

**For Yahoo Finance data:**
```csv
EQUITY,SPY,SPDR S&P 500 ETF,yfinance,YF
CREDIT,AGG,iShares Core Aggregate Bond ETF,yfinance,YF
```

**For Bloomberg data:**
```csv
COMMODITIES,BCOM Index,Bloomberg Commodity Index,bloomberg,BBG
CREDIT,IG1 Index,CDX IG 5Y Spread,bloomberg,BBG
```

### Categories to organize by:
- `EQUITY`, `CREDIT`, `COMMODITIES`, `FOREIGN CURRENCY`
- `INTEREST RATES`, `LOCAL INFLATION`, `EQUITY VOLATILITY`
- `MOMENTUM`, `VALUE`, `QUALITY`, `LOW RISK`, `SMALL CAP`
- `TREND FOLLOWING`, `FX CARRY`, `CREDIT SPREADS`

### Finding FRED Series IDs:
1. Search FRED website for your desired series
2. The series ID is in the URL: `https://fred.stlouisfed.org/series/UNRATE`
3. Use the ID (UNRATE) as your proxy

### Yahoo Finance Tickers:
- Use standard ticker symbols (SPY, QQQ, VTI, etc.)
- For futures, use format like GC=F for gold

## Step 4: Database Setup and Data Loading

### Initialize the system:
```bash
# Create database structure
python init_db.py

# Load your proxy definitions
python seed_instruments.py

# Load data from all sources
python load_fred.py          # FRED economic data
python load_yfinance.py       # Yahoo Finance market data
python load_french_shiller.py # Academic factors and Shiller data

# If using Bloomberg data:
python load_bloomberg.py     # Bloomberg terminal data (optional)

# Process into monthly returns
python compute_returns.py

# Verify data availability
python data_summary.py
```

### Check your data:
```bash
python data_summary.py
```
Look for:
- Instruments with good historical coverage
- Date ranges that make sense
- Missing data that might need attention