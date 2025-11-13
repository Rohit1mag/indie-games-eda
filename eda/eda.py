#!/usr/bin/env python3
"""
Initial Exploratory Data Analysis (EDA) for Indie Game Success Prediction

This script performs initial exploration of the indie games dataset to:
1. Understand data structure and quality
2. Identify key variables and potential signals
3. Visualize distributions, correlations, and trends
4. Identify feature engineering opportunities and challenges
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import ast
from collections import Counter
warnings.filterwarnings('ignore')

# Set style for publication-quality visuals
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Create output directory for visualizations (current directory)
output_dir = Path('.')
output_dir.mkdir(exist_ok=True)

print("="*70)
print("EXPLORATORY DATA ANALYSIS: INDIE GAME SUCCESS PREDICTION")
print("="*70)
print()

print("1. Loading dataset...")
df = pd.read_csv('../games_prepared.csv')
print(f"   ✓ Loaded {len(df):,} games")
print(f"   ✓ Dataset has {len(df.columns)} columns")
print()
print("   IMPORTANT: All visualizations use the FULL dataset (no train/test split)")
print(f"   Total games in analysis: {len(df):,}")
print()


print("2. Data Quality Assessment...")
print()

# Basic statistics
print("   Dataset Overview:")
print(f"   - Total games: {len(df):,}")
# Extract year from release_date if needed
if 'year' not in df.columns and 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
print(f"   - Date range: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"   - Success rate: {df['success'].mean()*100:.1f}% ({df['success'].sum():,} successful games)")
print()

# Missing values analysis
print("   Missing Values Analysis:")
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)
significant_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)

if len(significant_missing) > 0:
    print(f"   - Features with >5% missing values: {len(significant_missing)}")
    for feat, pct in significant_missing.head(10).items():
        print(f"     • {feat}: {pct:.1f}% missing")
else:
    print("   - No features with >5% missing values")
print()

# Key variables summary
print("   Key Variables:")
key_vars = ['price', 'required_age', 'achievements', 'dlc_count', 'success', 'year']
for var in key_vars:
    if var in df.columns:
        if df[var].dtype in ['int64', 'float64']:
            print(f"   - {var}: mean={df[var].mean():.2f}, median={df[var].median():.2f}, "
                  f"min={df[var].min():.0f}, max={df[var].max():.0f}")
        else:
            print(f"   - {var}: {df[var].dtype}, {df[var].nunique()} unique values")
print()


print("3. Creating Visualizations...")
print("   Visual 1: Success Rate Trend Over Time (Distribution/Trend)")
print(f"   Using FULL dataset: {len(df):,} games")
print(f"   (Note: 2025 data only includes games released through March 2025)")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Temporal Analysis: Success Rate and Game Releases Over Time\n(Note: 2025 contains only partial data through March)', 
             fontsize=16, fontweight='bold', y=0.98)

# Success rate by year (using FULL dataset)
yearly_stats = df.groupby('year').agg({
    'success': ['mean', 'count']
}).reset_index()
yearly_stats.columns = ['year', 'success_rate', 'total_games']

axes[0].plot(yearly_stats['year'], yearly_stats['success_rate'] * 100, 
             marker='o', linewidth=3, markersize=8, color='#2ecc71', label='Success Rate')
axes[0].fill_between(yearly_stats['year'], yearly_stats['success_rate'] * 100, 
                     alpha=0.3, color='#2ecc71')
axes[0].set_xlabel('Release Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Success Rate Declining Over Time', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=10)

# Add annotation for 2025 partial data
axes[0].text(0.98, 0.05, '⚠️ 2025: Partial data\n(through March only)', 
            transform=axes[0].transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='right', va='bottom')

# Number of games released by year
axes[1].bar(yearly_stats['year'], yearly_stats['total_games'], 
            color='#3498db', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Release Year', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Games Released', fontsize=12, fontweight='bold')
axes[1].set_title('Releases Over Time', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for year, count in zip(yearly_stats['year'], yearly_stats['total_games']):
    axes[1].text(year, count + 500, f'{count:,}', ha='center', va='bottom', fontsize=9)

# Add annotation for 2025 partial data on bottom subplot
axes[1].text(0.98, 0.95, '⚠️ 2025: Partial data\n(through March only)', 
            transform=axes[1].transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='right', va='top')

plt.tight_layout()
plt.savefig(output_dir / '01_temporal_success_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_temporal_success_trend.png")


print("   Visual 2: Success Rate by Price Groups (Grouping)")
print(f"   Using FULL dataset: {len(df):,} games")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Feature Exploration: Price and Success Relationship', 
             fontsize=16, fontweight='bold', y=0.98)

# Price distribution (using FULL dataset)
axes[0].hist(df['price'], bins=50, color='#3498db', alpha=0.7, edgecolor='black', range=(0, 50))

# Calculate mean for paid games only
mean_paid = df[df['price'] > 0]['price'].mean()

# Show mean for paid games
axes[0].axvline(mean_paid, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${mean_paid:.2f}')

axes[0].set_xlabel('Price (USD)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title(f'Price Distribution (Full Dataset: {len(df):,} games)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Note: Showing prices up to $50 for clarity, but all games included in analysis
if (df['price'] > 50).sum() > 0:
    axes[0].text(0.98, 0.95, f'({(df["price"] > 50).sum():,} games >$50 not shown)', 
                transform=axes[0].transAxes, fontsize=9, ha='right', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Success rate by price bins (using FULL dataset)
# Note: Using -0.01 as lower bound to include price == 0 (free games)
price_bins = pd.cut(df['price'], bins=[-0.01, 5, 10, 15, 20, 30, 100], 
                    labels=['$0-5', '$5-10', '$10-15', '$15-20', '$20-30', '$30+'])
price_success = df.groupby(price_bins)['success'].agg(['mean', 'count']).reset_index()
price_success.columns = ['price_range', 'success_rate', 'count']

# Verify we're using the full dataset
total_in_bins = price_success['count'].sum()
if total_in_bins < len(df):
    print(f"   ⚠️  Warning: Only {total_in_bins:,} games assigned to bins (expected {len(df):,})")
    print(f"      Missing: {len(df) - total_in_bins:,} games (likely price outliers)")
else:
    print(f"   ✓ All {len(df):,} games included in price bins")

# Filter out bins with too few games (less than 100) for display only
price_success_display = price_success[price_success['count'] >= 100]

bars = axes[1].bar(range(len(price_success_display)), price_success_display['success_rate'] * 100, 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(price_success_display)))
axes[1].set_xticklabels(price_success_display['price_range'], fontsize=10)
axes[1].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Price Range', fontsize=11, fontweight='bold')
axes[1].set_title(f'Success Rate by Price Range\n(Full Dataset: {len(df):,} games)', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, rate, count) in enumerate(zip(bars, price_success_display['success_rate'], price_success_display['count'])):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate*100:.1f}%\n(n={count:,})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '02_price_success_relationship.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_price_success_relationship.png")

print("   Visual 3: Pre-Launch Feature Correlations (Correlation)")
print(f"   Using FULL dataset: {len(df):,} games")

# Extract language count for correlation analysis
if 'supported_languages' in df.columns:
    df['language_count'] = df['supported_languages'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

# Calculate platform count (sum of windows, mac, linux flags)
platform_cols = ['windows', 'mac', 'linux']
available_platforms = [col for col in platform_cols if col in df.columns]
if available_platforms:
    df['platform_count'] = df[available_platforms].sum(axis=1)

# Select pre-launch numerical features only (exclude release_year - temporal leakage)
numerical_features = ['price', 'required_age', 'achievements', 'dlc_count', 
                      'language_count', 'success']
available_features = [f for f in numerical_features if f in df.columns]

corr_matrix = df[available_features].corr()

# Create a more informative correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Show full matrix (not just lower triangle) for better readability
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8, "label": "Correlation"},
            vmin=-0.5, vmax=0.5, ax=ax)

# Highlight the success row/column
ax.set_title('Pre-Launch Feature Correlations with Success\n(Actionable Features Only)', 
             fontsize=14, fontweight='bold', pad=15)

# Make success row/column labels bold
labels = ax.get_xticklabels()
for i, label in enumerate(labels):
    if 'success' in label.get_text().lower():
        label.set_weight('bold')
        label.set_color('darkred')
labels = ax.get_yticklabels()
for i, label in enumerate(labels):
    if 'success' in label.get_text().lower():
        label.set_weight('bold')
        label.set_color('darkred')

plt.tight_layout()
plt.savefig(output_dir / '03_feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_feature_correlations.png")


print("   Visual 4: Feature Distributions by Success Class (Boxplots)")
print(f"   Using FULL dataset: {len(df):,} games")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions by Success Class\n(Pre-Launch Features Predicting Success)', 
             fontsize=16, fontweight='bold', y=0.98)

# Flatten axes for easier iteration
axes = axes.flatten()

# Define features to plot and their labels
features_to_plot = [
    ('price', 'Price (USD)'),
    ('achievements', 'Achievement Count'),
    ('dlc_count', 'DLC Count'),
    ('required_age', 'Required Age'),
    ('language_count', 'Language Count'),
    ('platform_count', 'Platform Count')
]

# Create boxplots
for idx, (feature, label) in enumerate(features_to_plot):
    if feature in df.columns:
        # Prepare data
        unsuccessful = df[df['success'] == 0][feature].dropna()
        successful = df[df['success'] == 1][feature].dropna()
        
        # Create boxplot
        bp = axes[idx].boxplot([unsuccessful, successful], 
                               labels=['Unsuccessful', 'Successful'],
                               patch_artist=True,
                               widths=0.6,
                               showmeans=True,
                               meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        colors = ['#e74c3c', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[idx].set_ylabel(label, fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_title(f'{label}', fontsize=12, fontweight='bold')
        
        # Add sample sizes
        axes[idx].text(0.5, 0.98, f'n={len(unsuccessful):,} | n={len(successful):,}',
                      transform=axes[idx].transAxes, fontsize=9,
                      ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '04_feature_distributions_by_success.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_feature_distributions_by_success.png")


print()
print("   Detailed Feature Statistics by Success Class:")
print()

for feature, label in features_to_plot:
    if feature in df.columns:
        unsuccessful = df[df['success'] == 0][feature].dropna()
        successful = df[df['success'] == 1][feature].dropna()
        
        print(f"   {label.upper()}")
        print(f"   {'-' * 70}")
        print(f"   {'Metric':<20} {'Unsuccessful':<20} {'Successful':<20} {'Difference':<10}")
        print(f"   {'-' * 70}")
        
        # Calculate statistics
        metrics = {
            'Count': (len(unsuccessful), len(successful)),
            'Mean': (unsuccessful.mean(), successful.mean()),
            'Median': (unsuccessful.median(), successful.median()),
            'Std Dev': (unsuccessful.std(), successful.std()),
            'Min': (unsuccessful.min(), successful.min()),
            'Q1 (25%)': (unsuccessful.quantile(0.25), successful.quantile(0.25)),
            'Q3 (75%)': (unsuccessful.quantile(0.75), successful.quantile(0.75)),
            'Max': (unsuccessful.max(), successful.max()),
            'IQR': (unsuccessful.quantile(0.75) - unsuccessful.quantile(0.25), 
                   successful.quantile(0.75) - successful.quantile(0.25)),
        }
        
        for metric_name, (unsc_val, sc_val) in metrics.items():
            if metric_name == 'Count':
                diff = f"{sc_val - unsc_val:+.0f}"
            else:
                if sc_val != 0:
                    pct_diff = ((sc_val - unsc_val) / abs(unsc_val) * 100) if unsc_val != 0 else 0
                    diff = f"{pct_diff:+.1f}%"
                else:
                    diff = "N/A"
            
            print(f"   {metric_name:<20} {unsc_val:<20.2f} {sc_val:<20.2f} {diff:<10}")
        
        # Calculate separation score (Cohen's d effect size)
        pooled_std = np.sqrt(((len(unsuccessful)-1)*unsuccessful.std()**2 + 
                             (len(successful)-1)*successful.std()**2) / 
                            (len(unsuccessful) + len(successful) - 2))
        if pooled_std > 0:
            cohens_d = (successful.mean() - unsuccessful.mean()) / pooled_std
            print(f"   {'Cohen\'s d':<20} {'':<20} {'':<20} {cohens_d:>6.3f}")
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "LARGE"
            print(f"   {'Effect Size':<20} {'':<20} {'':<20} {effect}")
        
        print()


print("   Visual 5: Platform Distribution Analysis (Bar Charts)")
print(f"   Using FULL dataset: {len(df):,} games")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Platform Support Analysis', fontsize=16, fontweight='bold', y=0.98)

# Left plot: Individual platform support counts
if all(col in df.columns for col in ['windows', 'mac', 'linux']):
    platform_cols = ['windows', 'mac', 'linux']
    platform_counts = {col: df[col].sum() for col in platform_cols}
    platform_pcts = {col: (df[col].sum() / len(df) * 100) for col in platform_cols}
    
    platforms = list(platform_counts.keys())
    counts = list(platform_counts.values())
    
    bars = axes[0].bar(platforms, counts, color=['#3498db', '#e67e22', '#1abc9c'], 
                       alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Number of Games', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Platform', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Games Supporting Each Platform\n(Total: {len(df):,} games)', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, platform_pcts.values()):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Right plot: Platform combination distribution
    # Create platform combination categories
    df['platform_combo'] = (df['windows'].astype(int).astype(str) + 
                            df['mac'].astype(int).astype(str) + 
                            df['linux'].astype(int).astype(str))
    
    # Map combinations to labels
    combo_map = {
        '100': 'Windows Only',
        '110': 'Windows + Mac',
        '101': 'Windows + Linux',
        '111': 'All 3 Platforms',
        '010': 'Mac Only',
        '001': 'Linux Only',
        '011': 'Mac + Linux',
        '000': 'None'
    }
    
    df['platform_combo_label'] = df['platform_combo'].map(combo_map)
    combo_counts = df['platform_combo_label'].value_counts().sort_values(ascending=True)
    
    # Create color gradient
    colors_combo = plt.cm.Set3(np.linspace(0, 1, len(combo_counts)))
    
    bars2 = axes[1].barh(range(len(combo_counts)), combo_counts.values, 
                         color=colors_combo, alpha=0.8, edgecolor='black')
    axes[1].set_yticks(range(len(combo_counts)))
    axes[1].set_yticklabels(combo_counts.index, fontsize=10)
    axes[1].set_xlabel('Number of Games', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Platform Combination Distribution\n({len(df):,} games)', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars2, combo_counts.values)):
        width = bar.get_width()
        axes[1].text(width + 200, bar.get_y() + bar.get_height()/2.,
                    f' {count:,} ({count/len(df)*100:.1f}%)', 
                    ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '05_platform_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 05_platform_distribution.png")

# Print platform distribution statistics
print()
print("   Platform Distribution Statistics:")
print()
if all(col in df.columns for col in ['windows', 'mac', 'linux']):
    print(f"   {'Platform':<20} {'Count':<15} {'Percentage':<15}")
    print(f"   {'-' * 50}")
    for platform in ['windows', 'mac', 'linux']:
        count = df[platform].sum()
        pct = (count / len(df)) * 100
        print(f"   {platform.capitalize():<20} {count:<15,} {pct:<14.1f}%")
    
    print()
    print(f"   {'Platform Combination':<30} {'Count':<15} {'Percentage':<15}")
    print(f"   {'-' * 60}")
    for combo, count in combo_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {combo:<30} {count:<15,} {pct:<14.1f}%")
    
    # Success rates by platform combination
    print()
    print(f"   Success Rate by Platform Combination:")
    print(f"   {'-' * 60}")
    print(f"   {'Platform Combination':<30} {'Success Rate':<20}")
    print(f"   {'-' * 60}")
    for combo in df['platform_combo_label'].unique():
        if pd.notna(combo):
            success_rate = df[df['platform_combo_label'] == combo]['success'].mean() * 100
            print(f"   {combo:<30} {success_rate:<19.1f}%")
    print()

print("   Visual 6: Class Distribution Pie Chart ")
print(f"   Using FULL dataset: {len(df):,} games")

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Class Distribution: Success vs Failure', 
             fontsize=16, fontweight='bold', y=0.98)

# Class distribution (using FULL dataset)
class_counts = df['success'].value_counts()
colors_class = ['#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax.pie([class_counts[0], class_counts[1]], 
                                    labels=['Unsuccessful\n(<20k owners)', 'Successful\n(≥20k owners)'],
                                    autopct='%1.1f%%', colors=colors_class, startangle=90,
                                    textprops={'fontsize': 13, 'weight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')

# Add title with dataset info
ax.set_title(f'Class Distribution\n({len(df):,} games)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / '04_class_imbalance_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_class_imbalance_features.png")


print("   Visual 7: Top 20 Genres by Frequency (Grouped Bar Chart)")
print(f"   Using FULL dataset: {len(df):,} games")

# Extract and parse genres
if 'genres' in df.columns:
    # Flatten genres - parse Python list strings using ast.literal_eval
    all_genres_unsuccessful = []
    all_genres_successful = []
    
    # Define non-genre tags to exclude
    non_genre_tags = {'Indie', 'Early Access', 'Free To Play'}
    
    for idx, row in df.iterrows():
        if pd.notna(row['genres']):
            try:
                # Safely parse Python list string representation
                genres_list = ast.literal_eval(str(row['genres']))
                # Ensure it's a list
                if isinstance(genres_list, list):
                    # Clean, strip whitespace, and filter out non-genre tags
                    genres_list = [g.strip() for g in genres_list 
                                  if g and g.strip() not in non_genre_tags]
                else:
                    genres_list = []
            except (ValueError, SyntaxError):
                # Fallback: if parsing fails, skip this entry
                genres_list = []
            
            if genres_list:  # Only process if we have valid genres
                if row['success'] == 0:
                    all_genres_unsuccessful.extend(genres_list)
                else:
                    all_genres_successful.extend(genres_list)
    
    # Get counts for each class
    unsuccessful_counts = Counter(all_genres_unsuccessful)
    successful_counts = Counter(all_genres_successful)
    
    # Get all unique genres and top 20 by total count
    all_genres_combined = Counter(all_genres_unsuccessful) + Counter(all_genres_successful)
    top_20_genres = dict(all_genres_combined.most_common(20))
    top_genres_list = list(top_20_genres.keys())
    
    # Prepare data for grouped bar chart
    unsuccessful_vals = [unsuccessful_counts.get(g, 0) for g in top_genres_list]
    successful_vals = [successful_counts.get(g, 0) for g in top_genres_list]
    
    # Calculate success rates for each genre
    success_rates = []
    for i, genre in enumerate(top_genres_list):
        total = unsuccessful_vals[i] + successful_vals[i]
        if total > 0:
            rate = (successful_vals[i] / total) * 100
        else:
            rate = 0
        success_rates.append(rate)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_genres_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, unsuccessful_vals, width, label='Unsuccessful Games', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, successful_vals, width, label='Successful Games', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Genres: Successful vs Unsuccessful Games\n(Grouped by Success Class - with Success Rate %)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_genres_list, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars and success rate above bars
    for i, (bars, vals) in enumerate([(bars1, unsuccessful_vals), (bars2, successful_vals)]):
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # Add count label
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=7)
    
    # Add success rate percentage above each genre group (between the two bars)
    for i, rate in enumerate(success_rates):
        x_pos = i
        # Get the maximum height for proper positioning
        max_height = max(unsuccessful_vals[i], successful_vals[i])
        # Place success rate label above the bars
        ax.text(x_pos, max_height * 1.05, f'{rate:.1f}%', 
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_top_20_genres_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 06_top_20_genres_comparison.png")
    
    # Print genre statistics
    print()
    print("   Top 20 Genres Statistics:")
    print()
    print(f"   {'Genre':<30} {'Unsuccessful':<15} {'Successful':<15} {'Success Rate':<15}")
    print(f"   {'-' * 75}")
    
    genre_stats = []
    for genre in top_genres_list:
        unsc = unsuccessful_counts.get(genre, 0)
        sc = successful_counts.get(genre, 0)
        total = unsc + sc
        success_rate = (sc / total * 100) if total > 0 else 0
        genre_stats.append((genre, unsc, sc, success_rate))
    
    # Sort by success rate (descending)
    genre_stats.sort(key=lambda x: x[3], reverse=True)
    
    for genre, unsc, sc, success_rate in genre_stats:
        print(f"   {genre:<30} {unsc:<15} {sc:<15} {success_rate:<14.1f}%")
    
    print()
    print(f"   Total genres in dataset: {len(all_genres_combined):,}")
    print(f"   Total genre instances (unsuccessful): {len(all_genres_unsuccessful):,}")
    print(f"   Total genre instances (successful): {len(all_genres_successful):,}")
    print()


# Save summary statistics
# Calculate decline for summary stats
if len(yearly_stats) > 1:
    temporal_decline = ((yearly_stats['success_rate'].iloc[-1] - yearly_stats['success_rate'].iloc[0]) 
                       / yearly_stats['success_rate'].iloc[0]) * 100
else:
    temporal_decline = None

summary_stats = {
    'total_games': len(df),
    'success_rate': float(df['success'].mean()),
    'date_range': f"{df['year'].min():.0f}-{df['year'].max():.0f}",
    'class_imbalance_ratio': float(class_counts[0] / class_counts[1]),
    'features_with_missing_data': len(significant_missing),
    'temporal_decline': float(temporal_decline) if temporal_decline is not None else None
}

import json
with open(output_dir / 'eda_summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)


