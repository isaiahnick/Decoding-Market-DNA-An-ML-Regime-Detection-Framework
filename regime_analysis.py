# regime_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "/Users/isaiahnick/Market Regime/factor_lens.db"

def load_factor_data():
    """Load factor data starting from 1990"""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    
    query = """
    SELECT * FROM factors_monthly_z 
    WHERE date >= '1990-01-01' 
    ORDER BY date
    """
    df = pd.read_sql(query, engine, parse_dates=['date'])
    df = df.set_index('date')
    
    # Remove DTWEXM which stops in 2019 and cuts off recent data
    if 'DTWEXM' in df.columns:
        df = df.drop(columns=['DTWEXM'])
        print("Excluded DTWEXM (stops in 2019)")
    
    print(f"Loading data from 1990-01-01...")
    print(f"Raw data: {df.shape[0]} months, {df.shape[1]} factors")
    print(f"Available factors: {list(df.columns)}")
    
    # Remove factors with too many missing values (need 80% coverage)
    min_observations = len(df) * 0.8
    factor_coverage = df.count()
    good_factors = factor_coverage >= min_observations
    
    print(f"\nFactor coverage analysis:")
    for factor in df.columns:
        coverage_pct = factor_coverage[factor] / len(df)
        status = "✓" if good_factors[factor] else "✗"
        print(f"  {status} {factor}: {coverage_pct:.1%} coverage")
    
    # Keep only factors with good coverage
    df = df.loc[:, good_factors]
    
    # Remove rows with missing data
    original_rows = len(df)
    df = df.dropna()
    dropped_rows = original_rows - len(df)
    
    if dropped_rows > 0:
        print(f"\nDropped {dropped_rows} incomplete months")
    
    print(f"\nFinal dataset: {df.shape[0]} months, {df.shape[1]} factors")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Factors: {list(df.columns)}")
    
    if df.shape[1] < 3:
        raise RuntimeError("Insufficient factors for regime analysis")
    
    return df

def select_optimal_clusters(data, max_clusters=6):
    """Find optimal number of clusters using BIC"""
    n_clusters_range = range(2, min(max_clusters + 1, data.shape[1] + 1))
    bic_scores = []
    
    print(f"\nTesting {len(n_clusters_range)} different cluster counts...")
    for n in n_clusters_range:
        gmm = GaussianMixture(n_components=n, random_state=42, max_iter=200)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
        print(f"  {n} clusters: BIC = {gmm.bic(data):.1f}")
    
    optimal_idx = np.argmin(bic_scores)
    optimal_clusters = n_clusters_range[optimal_idx]
    print(f"\nOptimal clusters: {optimal_clusters} (lowest BIC)")
    
    return optimal_clusters

def fit_regime_model(data, n_clusters):
    """Fit Gaussian Mixture Model"""
    print(f"\nFitting GMM with {n_clusters} clusters...")
    
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        max_iter=300,
        covariance_type='full'
    )
    
    gmm.fit(data)
    regime_probs = gmm.predict_proba(data)
    regime_labels = gmm.predict(data)
    
    print(f"Model converged: {gmm.converged_}")
    print(f"Log-likelihood: {gmm.score(data) * len(data):.1f}")
    
    return gmm, regime_probs, regime_labels

def analyze_regime_characteristics(data, gmm, regime_labels):
    """Analyze each regime's characteristics"""
    n_regimes = gmm.n_components
    regime_stats = {}
    
    print(f"\n{'='*60}")
    print("REGIME ANALYSIS")
    print(f"{'='*60}")
    
    for regime in range(n_regimes):
        mask = regime_labels == regime
        if not mask.any():
            continue
            
        regime_data = data[mask]
        mean_returns = regime_data.mean()
        volatilities = regime_data.std()
        frequency = mask.sum() / len(data)
        
        regime_stats[regime] = {
            'frequency': frequency,
            'mean_returns': mean_returns,
            'volatilities': volatilities,
            'periods': mask.sum()
        }
        
        print(f"\nRegime {regime + 1}:")
        print(f"  Frequency: {frequency:.1%} ({mask.sum()} months)")
        print(f"  Average volatility: {volatilities.mean():.3f}")
        
        # Show top factors
        if len(mean_returns) > 0:
            top_positive = mean_returns.nlargest(3)
            top_negative = mean_returns.nsmallest(3)
            
            print(f"  Strongest positive factors:")
            for factor, value in top_positive.items():
                print(f"    {factor}: {value:.3f}")
            
            print(f"  Strongest negative factors:")
            for factor, value in top_negative.items():
                print(f"    {factor}: {value:.3f}")
    
    return regime_stats

def create_regime_timeline(data, regime_probs, regime_labels):
    """Create timeline of regime probabilities"""
    n_regimes = regime_probs.shape[1]
    
    regime_df = pd.DataFrame(
        regime_probs,
        index=data.index,
        columns=[f'Regime_{i+1}' for i in range(n_regimes)]
    )
    regime_df['Most_Likely'] = regime_labels
    
    return regime_df

def identify_regime_periods(regime_df, min_duration=3):
    """Find sustained regime periods"""
    most_likely = regime_df['Most_Likely']
    
    print(f"\n{'='*60}")
    print(f"SUSTAINED REGIME PERIODS (min {min_duration} months)")
    print(f"{'='*60}")
    
    periods = []
    current_regime = most_likely.iloc[0]
    start_date = most_likely.index[0]
    start_idx = 0
    
    for i in range(1, len(most_likely)):
        if most_likely.iloc[i] != current_regime:
            duration = i - start_idx
            end_date = most_likely.index[i-1]
            
            if duration >= min_duration:
                periods.append({
                    'regime': current_regime + 1,
                    'start': start_date,
                    'end': end_date,
                    'duration': duration
                })
                print(f"Regime {current_regime + 1}: {start_date.date()} to {end_date.date()} ({duration} months)")
            
            current_regime = most_likely.iloc[i]
            start_date = most_likely.index[i]
            start_idx = i
    
    # Handle final period
    duration = len(most_likely) - start_idx
    if duration >= min_duration:
        end_date = most_likely.index[-1]
        periods.append({
            'regime': current_regime + 1,
            'start': start_date,
            'end': end_date,
            'duration': duration
        })
        print(f"Regime {current_regime + 1}: {start_date.date()} to {end_date.date()} ({duration} months)")
    
    return periods

def plot_regime_analysis(data, regime_df, regime_stats):
    """Create visualizations"""
    n_regimes = len(regime_stats)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Regime probability timeline
    ax1 = axes[0, 0]
    for i in range(n_regimes):
        if f'Regime_{i+1}' in regime_df.columns:
            ax1.plot(regime_df.index, regime_df[f'Regime_{i+1}'], 
                    label=f'Regime {i+1}', alpha=0.7)
    
    ax1.set_title('Regime Probabilities Over Time')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Regime frequency
    ax2 = axes[0, 1]
    frequencies = [regime_stats[i]['frequency'] for i in range(n_regimes)]
    bars = ax2.bar(range(1, n_regimes + 1), frequencies)
    ax2.set_title('Regime Frequencies')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Regime')
    
    for bar, freq in zip(bars, frequencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{freq:.1%}', ha='center', va='bottom')
    
    # 3. Average volatility by regime
    ax3 = axes[1, 0]
    avg_vols = [regime_stats[i]['volatilities'].mean() for i in range(n_regimes)]
    ax3.bar(range(1, n_regimes + 1), avg_vols)
    ax3.set_title('Average Factor Volatility by Regime')
    ax3.set_ylabel('Average Volatility')
    ax3.set_xlabel('Regime')
    
    # 4. Recent regime evolution
    ax4 = axes[1, 1]
    recent_data = regime_df.tail(60)  # Last 5 years
    
    regime_cols = [f'Regime_{i+1}' for i in range(n_regimes)]
    available_cols = [col for col in regime_cols if col in recent_data.columns]
    
    if available_cols:
        ax4.stackplot(recent_data.index, 
                     *[recent_data[col] for col in available_cols],
                     labels=available_cols, alpha=0.7)
        ax4.set_title('Recent Regime Evolution (Last 5 Years)')
        ax4.set_ylabel('Probability')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()

def run_regime_analysis():
    """Main analysis function"""
    print("MARKET REGIME ANALYSIS")
    print("="*60)
    
    # Load data with optimal start date
    data = load_factor_data()
    
    if data.shape[1] < 3:
        raise RuntimeError("Need at least 3 factors for meaningful analysis")
    
    # Find optimal clusters
    optimal_k = select_optimal_clusters(data.values)
    
    # Fit model
    gmm, regime_probs, regime_labels = fit_regime_model(data.values, optimal_k)
    
    # Analyze results
    regime_stats = analyze_regime_characteristics(data, gmm, regime_labels)
    regime_df = create_regime_timeline(data, regime_probs, regime_labels)
    regime_periods = identify_regime_periods(regime_df)
    
    # Create plots
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    plot_regime_analysis(data, regime_df, regime_stats)
    
    # Save results
    output_file = '/Users/isaiahnick/Market Regime/regime_results.csv'
    regime_df.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")
    
    return regime_df, regime_stats, gmm

if __name__ == "__main__":
    regime_df, regime_stats, model = run_regime_analysis()