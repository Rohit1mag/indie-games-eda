#!/usr/bin/env python3
"""
Complete Pipeline for Final Project: Predicting Indie Game Commercial Success
"""

import pandas as pd
import numpy as np
import json
import pickle
import ast
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    precision_recall_curve, roc_curve, auc,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_list_column(val):
    """Parse string representation of list"""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(parsed, list):
            return parsed
        return []
    except:
        return []

def extract_language_count(lang_str):
    """Count number of supported languages"""
    langs = parse_list_column(lang_str)
    return len(langs) if isinstance(langs, list) else 0

def has_multiplayer(categories_str):
    """Check if game has multiplayer/co-op"""
    categories = parse_list_column(categories_str)
    if not isinstance(categories, list):
        return 0
    
    mp_keywords = ['multiplayer', 'multi-player', 'co-op', 'pvp', 'online']
    categories_lower = [str(c).lower() for c in categories]
    
    for keyword in mp_keywords:
        if any(keyword in cat for cat in categories_lower):
            return 1
    return 0

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a model and return metrics"""
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    results = {
        'model_name': model_name,
        'train': {
            'precision': float(train_precision),
            'recall': float(train_recall),
            'f1': float(train_f1)
        },
        'test': {
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': cm.tolist(),
        'y_test_proba': y_test_proba.tolist(),
        'y_test_pred': y_test_pred.tolist(),
        'y_test_true': y_test.tolist()
    }
    
    print(f"\n   {model_name}:")
    print(f"     Test Precision: {test_precision:.3f}")
    print(f"     Test Recall:    {test_recall:.3f}")
    print(f"     Test F1:        {test_f1:.3f}")
    print(f"     Test PR-AUC:    {pr_auc:.3f}")
    print(f"     Test ROC-AUC:   {roc_auc:.3f}")
    
    return model, results


def feature_engineering(input_file='games_prepared.csv', output_file='games_with_features.csv'):
    """Create 68 pre-launch features from raw data"""
    print("="*70)
    print("PART 1: FEATURE ENGINEERING")
    print("="*70)
    
    # Load prepared data
    print("\n[1/4] Loading games_prepared.csv...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df):,} games")
    
    # Create copy for feature engineering
    df_features = df.copy()
    
    # Parse release date
    df_features['release_date'] = pd.to_datetime(df_features['release_date'], errors='coerce')
    df_features['release_year'] = df_features['release_date'].dt.year
    df_features['release_month'] = df_features['release_date'].dt.month
    df_features['release_quarter'] = df_features['release_date'].dt.quarter
    
    # =========================================================================
    # NUMERICAL FEATURES
    # =========================================================================
    print("\n[2/4] Engineering numerical features...")
    
    numerical_features = []
    
    # Basic numerical features
    for feat in ['price', 'dlc_count', 'achievements', 'required_age']:
        if feat in df_features.columns:
            df_features[feat] = df_features[feat].fillna(0)
            
            # Apply log scaling to dlc_count and achievements
            if feat in ['dlc_count', 'achievements']:
                df_features[feat] = np.log1p(df_features[feat])
                print(f"   {feat} (log-scaled)")
            else:
                print(f"   {feat}")
            
            numerical_features.append(feat)
    
    # Language count
    if 'supported_languages' in df_features.columns:
        df_features['language_count'] = df_features['supported_languages'].apply(extract_language_count)
        numerical_features.append('language_count')
        print(f"  language_count (avg: {df_features['language_count'].mean():.1f})")
    
    # Platform count
    platform_cols = ['windows', 'mac', 'linux']
    available_platforms = [p for p in platform_cols if p in df_features.columns]
    if available_platforms:
        df_features['platform_count'] = df_features[available_platforms].fillna(False).astype(int).sum(axis=1)
        numerical_features.append('platform_count')
        print(f"  platform_count (avg: {df_features['platform_count'].mean():.2f})")
    
    # Description features
    if 'short_description' in df_features.columns:
        df_features['desc_length'] = df_features['short_description'].fillna('').apply(len)
        df_features['desc_word_count'] = df_features['short_description'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        numerical_features.extend(['desc_length', 'desc_word_count'])
        print(f"  desc_length, desc_word_count")
    
    # Detailed features text features
    if 'detailed_features' in df_features.columns:
        df_features['detailed_features_length'] = df_features['detailed_features'].fillna('').apply(len)
        df_features['detailed_features_word_count'] = df_features['detailed_features'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        numerical_features.extend(['detailed_features_length', 'detailed_features_word_count'])
        print(f"  detailed_features_length, detailed_features_word_count")
    
    # TF-IDF features
    if 'short_description' in df_features.columns:
        print("\n   Creating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=50,
            min_df=10,
            max_df=0.7,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Combine short_description and detailed_features
        descriptions = df_features['short_description'].fillna('')
        if 'detailed_features' in df_features.columns:
            detailed = df_features['detailed_features'].fillna('')
            descriptions = descriptions + ' ' + detailed
        
        tfidf_matrix = tfidf.fit_transform(descriptions)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{word.replace(" ", "_")}' for word in tfidf.get_feature_names_out()],
            index=df_features.index
        )
        
        df_features = pd.concat([df_features, tfidf_df], axis=1)
        numerical_features.extend(tfidf_df.columns.tolist())
        
        # Save TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        
        print(f"  Created {len(tfidf_df.columns)} TF-IDF features")
    
    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================
    print("\n[3/4] Engineering categorical features...")
    
    categorical_features = []
    
    # Multi-hot encode all genres
    if 'genres' in df_features.columns:
        df_features['genres_list'] = df_features['genres'].apply(parse_list_column)
        
        # Multi-label binarizer
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(df_features['genres_list'])
        
        # Convert to dataframe and add to features
        genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f'genre_{g.replace(" ", "_")}' for g in mlb.classes_],
            index=df_features.index
        )
        df_features = pd.concat([df_features, genres_df], axis=1)
        
        # Add all new columns to categorical_features
        genre_cols = genres_df.columns.tolist()
        categorical_features.extend(genre_cols)
        print(f"  Multi-hot encoded {len(mlb.classes_)} genres ({len(genre_cols)} features)")
    
    # Platform flags
    for platform in available_platforms:
        df_features[platform] = df_features[platform].fillna(False).astype(int)
        categorical_features.append(platform)
    print(f"  Platform flags: {available_platforms}")
    
    # Multi-hot encode all categories
    if 'categories' in df_features.columns:
        df_features['categories_list'] = df_features['categories'].apply(parse_list_column)
        
        # Multi-label binarizer
        mlb_cat = MultiLabelBinarizer()
        categories_encoded = mlb_cat.fit_transform(df_features['categories_list'])
        
        # Convert to dataframe and add to features
        categories_df = pd.DataFrame(
            categories_encoded,
            columns=[f'category_{c.replace(" ", "_")}' for c in mlb_cat.classes_],
            index=df_features.index
        )
        df_features = pd.concat([df_features, categories_df], axis=1)
        
        # Add all new columns to categorical_features
        category_cols = categories_df.columns.tolist()
        categorical_features.extend(category_cols)
        print(f"  Multi-hot encoded {len(mlb_cat.classes_)} categories ({len(category_cols)} features)")
    
    # Developer experience
    if 'developers' in df_features.columns:
        dev_counts = df_features['developers'].value_counts()
        df_features['dev_game_count'] = df_features['developers'].map(dev_counts)
        categorical_features.append('dev_game_count')
        print(f"  dev_game_count")
    
    # Free flag
    df_features['is_free'] = (df_features['price'] == 0).astype(int)
    categorical_features.append('is_free')
    print(f"  is_free")
    
    # Temporal features
    categorical_features.extend(['release_month', 'release_quarter'])
    df_features['released_q4'] = (df_features['release_quarter'] == 4).astype(int)
    categorical_features.append('released_q4')
    print(f"  Temporal features")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n[4/4] Saving engineered dataset...")
    
    # Save feature list
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'target': 'success',
        'temporal_column': 'release_year'
    }
    
    with open('feature_config.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save dataset
    df_features.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("FEATURE SUMMARY")
    print(f"{'='*70}")
    print(f"Total numerical features:   {len(numerical_features)}")
    print(f"Total categorical features: {len(categorical_features)}")
    print(f"Total features:              {len(numerical_features) + len(categorical_features)}")
    print(f"Target variable:             success")
    print(f"Output file:                 {output_file}")
    print(f"{'='*70}\n")
    
    return df_features, numerical_features, categorical_features

# ============================================================================
# PART 2: MODEL TRAINING
# ============================================================================

def train_models(df_features, numerical_features, categorical_features):
    """Train and evaluate multiple models"""
    
    # Prepare features
    print("\nPreparing features...")
    X = df_features[numerical_features + categorical_features].copy()
    y = df_features['success'].copy()
    
    # Remove rows with missing target
    valid_idx = ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"   Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    print(f"   Valid games: {len(X):,}")
    print(f"   Success rate: {y.mean():.1%}")
    
    # Temporal split (train on 2020-2022, test on 2024+)
    if 'release_year' in df_features.columns:
        years = df_features.loc[valid_idx, 'release_year']
        train_mask = (years >= 2020) & (years <= 2023)
        test_mask = years >= 2024
        
        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        
        print(f"\n   Temporal split:")
        print(f"   Train (2020-2023): {len(X_train):,} games ({y_train.mean():.1%} success)")
        print(f"   Test (2024+):      {len(X_test):,} games ({y_test.mean():.1%} success)")
    else:
        # Fallback to random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"\n   Random split:")
        print(f"   Train: {len(X_train):,} games ({y_train.mean():.1%} success)")
        print(f"   Test:  {len(X_test):,} games ({y_test.mean():.1%} success)")
    
    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    num_feature_names = numerical_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    print(f"   Processed features: {X_train_processed.shape[1]}")
    
    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # =========================================================================
    # BASELINE MODEL
    # =========================================================================
    print("\nTraining models...")
    
    # 1. Baseline: Stratified Random
    print("\n   1. Baseline (Stratified Random)...")
    np.random.seed(42)  # For reproducibility
    baseline_pred = np.random.binomial(1, y_train.mean(), size=len(y_test))
    baseline_proba = np.full(len(y_test), y_train.mean())
    
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, baseline_proba)
    baseline_pr_auc = auc(recall_curve, precision_curve)
    
    fpr, tpr, _ = roc_curve(y_test, baseline_proba)
    baseline_roc_auc = auc(fpr, tpr)
    
    baseline_results = {
        'model_name': 'Baseline (Stratified Random)',
        'train': {'precision': float(y_train.mean()), 'recall': 1.0, 'f1': float(2*y_train.mean()/(1+y_train.mean()))},
        'test': {
            'precision': float(baseline_precision),
            'recall': float(baseline_recall),
            'f1': float(baseline_f1),
            'pr_auc': float(baseline_pr_auc),
            'roc_auc': float(baseline_roc_auc)
        },
        'confusion_matrix': confusion_matrix(y_test, baseline_pred).tolist(),
        'y_test_proba': baseline_proba.tolist(),
        'y_test_pred': baseline_pred.tolist(),
        'y_test_true': y_test.tolist()
    }
    
    print(f"     Test Precision: {baseline_precision:.3f}")
    print(f"     Test Recall:    {baseline_recall:.3f}")
    print(f"     Test F1:        {baseline_f1:.3f}")
    print(f"     Test PR-AUC:    {baseline_pr_auc:.3f}")
    
    # 2. Logistic Regression
    print("\n   2. Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model, lr_results = evaluate_model(
        lr_model, X_train_processed, y_train, X_test_processed, y_test, "Logistic Regression"
    )
    
    # 3. Random Forest
    print("\n   3. Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model, rf_results = evaluate_model(
        rf_model, X_train_processed, y_train, X_test_processed, y_test, "Random Forest"
    )
    
    # 4. XGBoost (if available)
    xgb_model = None
    xgb_results = None
    print("\n   4. XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model, xgb_results = evaluate_model(
        xgb_model, X_train_processed, y_train, X_test_processed, y_test, "XGBoost"
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print("\nExtracting feature importance...")
    
    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'feature': all_feature_names[:len(rf_model.feature_importances_)],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = rf_importance.head(15).to_dict('records')
    print(f"\n   Top 10 Random Forest Features:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"     {i:2d}. {feat['feature']:30s} {feat['importance']*100:5.2f}%")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\nSaving results...")
    
    all_results = {
        'baseline': baseline_results,
        'logistic_regression': lr_results,
        'random_forest': rf_results,
        'feature_importance': top_features,
        'data_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_success_rate': float(y_train.mean()),
            'test_success_rate': float(y_test.mean()),
            'num_features': len(numerical_features),
            'cat_features': len(categorical_features),
            'total_features': len(numerical_features) + len(categorical_features)
        }
    }
    
    if xgb_results:
        all_results['xgboost'] = xgb_results
    
    with open('model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save best model (Random Forest)
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    print("\n[6/6] Model training complete!")
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - model_results.json")
    print("  - best_model.pkl")
    print("  - preprocessor.pkl")
    print("\n" + "="*70)
    
    return all_results

# ============================================================================
# PART 3: VISUALIZATIONS
# ============================================================================

def create_visualizations(results_file='model_results.json'):
    """Create all visualizations from model results"""
    print("\n" + "="*70)
    print("PART 3: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Load results
    print("\nLoading model results...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\nCreating confusion matrices...")
    
    models_to_plot = ['baseline', 'logistic_regression', 'random_forest']
    if 'xgboost' in results:
        models_to_plot.append('xgboost')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models_to_plot):
        if model_name not in results:
            continue
        
        model_results = results[model_name]
        cm = np.array(model_results['confusion_matrix'])
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Failed', 'Success'],
                    yticklabels=['Failed', 'Success'])
        ax.set_title(f'{model_results["model_name"]}\n'
                     f'Precision: {model_results["test"]["precision"]:.3f} | '
                     f'Recall: {model_results["test"]["recall"]:.3f} | '
                     f'F1: {model_results["test"]["f1"]:.3f}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Remove empty subplot
    if len(models_to_plot) < 4:
        axes[3].remove()
    
    plt.tight_layout()
    plt.savefig('01_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("  Saved: 01_confusion_matrices.png")
    plt.close()
    
    # =========================================================================
    # 2. ROC CURVES
    # =========================================================================
    print("\nCreating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name in models_to_plot:
        if model_name not in results:
            continue
        
        model_results = results[model_name]
        y_true = np.array(model_results['y_test_true'])
        y_proba = np.array(model_results['y_test_proba'])
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2,
                label=f'{model_results["model_name"]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_roc_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: 02_roc_curves.png")
    plt.close()
    
    # =========================================================================
    # 3. PRECISION-RECALL CURVES
    # =========================================================================
    print("\nCreating Precision-Recall curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name in models_to_plot:
        if model_name not in results:
            continue
        
        model_results = results[model_name]
        y_true = np.array(model_results['y_test_true'])
        y_proba = np.array(model_results['y_test_proba'])
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, linewidth=2,
                label=f'{model_results["model_name"]} (PR-AUC = {pr_auc:.3f})')
    
    # Baseline (random)
    baseline_pr = results['baseline']['test']['pr_auc']
    ax.axhline(y=baseline_pr, color='k', linestyle='--', linewidth=1,
               label=f'Baseline (PR-AUC = {baseline_pr:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_pr_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: 03_pr_curves.png")
    plt.close()
    
    # =========================================================================
    # 4. FEATURE IMPORTANCE
    # =========================================================================
    print("\nCreating feature importance plot...")
    
    feature_importance = results['feature_importance']
    top_15 = feature_importance[:15]
    
    features = [f['feature'] for f in top_15]
    importances = [f['importance'] * 100 for f in top_15]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(features)), importances, color='steelblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp + 0.1, i, f'{imp:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('04_feature_importance.png', dpi=300, bbox_inches='tight')
    print("  Saved: 04_feature_importance.png")
    plt.close()
    
    # =========================================================================
    # 5. MODEL COMPARISON BAR CHART
    # =========================================================================
    print("\nCreating model comparison chart...")
    
    metrics = ['precision', 'recall', 'f1', 'pr_auc']
    model_names = []
    for model_name in models_to_plot:
        if model_name in results:
            model_names.append(results[model_name]['model_name'])
    
    comparison_data = []
    for model_name in models_to_plot:
        if model_name not in results:
            continue
        model_results = results[model_name]
        row = {'Model': model_results['model_name']}
        for metric in metrics:
            row[metric.replace('_', ' ').title()] = model_results['test'][metric]
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_title = metric.replace('_', ' ').title()
        bars = ax.bar(df_comparison['Model'], df_comparison[metric_title], color='steelblue')
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, max(df_comparison[metric_title]) * 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('05_model_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: 05_model_comparison.png")
    plt.close()
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete pipeline"""

    try:
        # Step 1: Feature Engineering
        df_features, numerical_features, categorical_features = feature_engineering()
        
        # Step 2: Model Training
        results = train_models(df_features, numerical_features, categorical_features)
        
        # Step 3: Visualizations
        create_visualizations()

        return 0
        
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

