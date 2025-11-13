# Exploratory Data Analysis Summary

  

## Indie Game Commercial Success Prediction

  
**Dataset:** Steam Games Data (March 2025)

**Author:** Rohit Kota

**Date:** November 2025

  
---

  

## What is Your Dataset and Why Did You Choose It?

  

### Dataset Description

  
I'm using a game dataset I found on Kaggle.  I narrowed it down to **46,957 indie games** released between 2020 and 2025. This started as the full Steam catalog with 88,899 games, filtered down using three main criteria:

- **Price range:** $0-$30, which is the typical indie pricing range (AAA games are usually $60+)
- **Studio size:** Only developers that have 3 or fewer games in the dataset - this filters out big publishers and focuses on true indie studios
- **Time period:** Games released from 2020 onwards, which captures the COVID-era gaming boom and the post-COVID market

I think this gives a good representation of what "indie games" actually are in the Steam marketplace.

  

### Data Source and Structure

The data source is in the form of CSV at [Stream's 2025 game dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/). The data was scraped from Steam in March 2025.

It includes basically everything you'd see on a Steam game page:

- **Basic info:** Game name, release date, price, age rating
- **Content features:** Number of achievements, DLC count, game descriptions, genres and categories
- **Platform support:** Whether it runs on Windows, Mac, or Linux
- **Metadata:** Developer name, publisher, supported languages
- **Performance metrics:** Estimated owners (which I use to define success), reviews, etc.
- **Feedback data:**  Reviews, recommendations 

There are 50 columns total, covering pretty much all the metadata Steam makes available. The most important one for my project is `estimated_owners`, which I use to determine if a game is "successful" or not.
  

### Why This Dataset?

  

I chose this dataset for several reasons:

  

1.  **It's actually useful:** Predicting indie game success is a real problem that developers, publishers, and investors care about. The indie game market is really unpredictable, so if I can build a model that helps, that would be pretty cool.

  

2.  **Perfect for classification:** This is clearly a binary classification problem - either a game succeeds (≥20,000 owners) or it doesn't. That fits perfectly with the course requirements.

  

3.  **Pre-launch prediction is the key:** The interesting part is that I'm only using features available BEFORE a game launches. This makes the predictions actually actionable - a developer could use this before spending years making a game. Most other approaches use post-launch data like reviews, which defeats the purpose.

  

4.  **Lots of different feature types:** The dataset has numerical features (price, achievements), categorical features (genres, platforms), text (descriptions), and temporal data (release dates). This gives me a lot to work with for feature engineering.

  

5.  **Interesting time period:** 2020-2025 covers the COVID gaming boom and the market saturation that followed. There are some really interesting temporal patterns I can explore.

  

6.  **Real impact potential:** If this works, developers could use it to make better decisions about pricing, features, marketing, etc. before investing years of their life into a project.

  

---

  

## What Did You Learn from Your EDA?

  

### Key Findings

  

#### 1. **The Class Imbalance is Really Bad**

  

Only **17.0%** of indie games actually succeed (reach ≥20K owners). That means **83.0%** fail. In numbers, that's **7,989 successful games** vs **38,968 unsuccessful games**, which is a **4.88:1** imbalance ratio.

  
 If I just predicted "failure" for everything, I'd get 83% accuracy, which sounds good but is completely useless. I'll need to use class weighting, focus on PR-AUC instead of accuracy, and probably do some threshold optimization.

  
#### 2. **Success Rates Are Declining Over Time**

  
This was probably the most surprising finding. The success rate has dropped dramatically:

  
-  **2020:** 29.2% success rate (6,694 games released)

-  **2021:** 28.0% success rate (6,275 games)

-  **2022:** 19.8% success rate (6,793 games)

-  **2023:** 15.0% success rate (9,532 games)

-  **2024:** 9.9% success rate (14,762 games)

-  **2025:** 1.4% success rate (2,901 games - **partial data through March only**)

  
From 2020-2024, that's a **66% decline**. Meanwhile, the number of games released per year has more than doubled. So we're seeing massive market saturation - way more games are coming out, but way fewer are succeeding.

This has interesting implications for modeling. I have to chose my train/test data splits in a way that it captures this trend and also validates it. 



#### 3. **Price Matters, But Not Linearly**

The average price is **$6.31** and the median is **$4.99**, so most indie games are pretty cheap. But the relationship with success isn't straightforward - games in the **$10-15 range** tend to do better, while both free games and expensive ones (>$30) have lower success rates. This makes sense - free games need different monetization strategies, and expensive indie games might struggle to justify the price.

Pricing strategy is definitely going to be an important feature.

  

#### 4. **Some Features Correlate with Success**

  

I did a deep statistical analysis comparing feature distributions between successful and unsuccessful games:

 
Looking at pre-launch numerical features:

- **Achievements** are positively correlated - games with more achievements tend to succeed more
- **Genres** Some genres like RPG, Massively Multiplayer and Strategy are both popular and tend to succeed more
- **Platform support** Windows is a must have, which is both intuitive and clear from the data
- **DLC count** shows mild positive correlation - games that plan DLC content tend to be more successful (maybe shows commitment/planning?)
- **Required age** doesn't really matter - most games are rated for all ages anyway. Lots of games have zero as the required age.

I'm excluding release year from correlation analysis because it's not something developers can control - it's just temporal leakage. I want edactionable features.

  

#### 5. **Genre Patterns - Actionable Insight**

  
The genre analysis showed some interesting observations:

**Top Genres by Success Rate (out of 27 unique genres after filtering):**


1.  **RPG - 23.7% success rate** (7,145 unsuccessful, 2,224 successful)

2.  **Massively Multiplayer - 22.3% success rate** (945 unsuccessful, 271 successful)

3.  **Strategy - 20.2% success rate** (7,532 unsuccessful, 1,910 successful)

4.  **Adventure - 20.1% success rate** (15,319 unsuccessful, 3,852 successful)


**Low Performers:**

- Casual - 12.8% (despite being most abundant with 19,603 total games!)

- Racing - 14.1%

- Sports - 15.3%

**Key Insight:** Casual games are the most common (17,085 unsuccessful) but have the lowest success rate. This is the opposite of what you'd expect - you'd think "casual" games would do well. This suggests market saturation in casual: lots of competitors, fewer winners. Meanwhile, RPG and Strategy games have clearer paths to success, suggesting these genres are either underserved or they attract more dedicated audiences willing to buy.

  
#### 6. **Pre-Launch vs Post-Launch Features**

  

This is crucial for my project. Many useful features ARE available pre-launch:

- Price, achievements, DLC plans (if announced)

- Platform support, genres, descriptions

- Release date (planned)

- Language support

  
But some features are definitely post-launch only and I need to exclude them:
  

- Reviews and review scores

- Discount status

- User-generated tags

- Days since release

 
I needed to be careful here to avoid data leakage.

  

#### 7. **Data Quality is Pretty Good**


Most of the important numerical features have low missingness (<5%). There are 7 features with >5% missing values, but most aren't critical:

- `score_rank`: 100% missing (not important)
- `metacritic_url`: 98.4% missing (not relevant for pre-launch)
- `reviews`: 92.3% missing (post-launch anyway)
- `notes`: 78.4% missing (not important)
- `website`: 62.2% missing (nice to have but not critical)
- `support_url`: 57.0% missing (nice to have)
- `support_email`: 10.4% missing (nice to have)

The critical features like price, achievements, DLC count, and descriptions are mostly complete, so I'm in good shape for machine learning.

 
### Visualizations Created

  

I created 7 comprehensive visualizations:

 
1.  **Temporal Success Trend:** Shows how success rates dropped from 29.2% in 2020 to 9.9% in 2024, plus the increasing number of releases over time. Really shows the market saturation problem.
  
2.  **Price-Success Relationship:** Shows the optimal pricing range ($10-15) and the overall price distribution. Mean is $6.31, which is pretty low. Most games are clustered in the $0-5 range.
  

3.  **Feature Correlations:** Heatmap showing which pre-launch numerical features correlate most with success. 


4.  **Feature Distributions by Success Class (Boxplots):** Side-by-side comparison of 6 key features showing how they differ between successful and unsuccessful games. The boxplots also helped me understand the features like 'Achievement Count', 'DLC Count' and 'Required Age' data fields better. While they have a good range, lot of them are also '0' compressing the box plot. I may consider using logarithmic scaling for some of this data that has a long tail of range.
  

5.  **Platform Distribution Analysis:** Bar charts showing platform support. Windows dominates at 100%, with Mac at 15.4% and Linux at 11.5%. Most games (80.4%) are Windows-only. Cross-platform games (especially Windows+Mac) show 5-6% higher success rates than Windows-only.

  

6.  **Top 20 Genres Comparison:** Grouped bar chart comparing successful vs unsuccessful games by genre. Key finding: RPG has highest success rate at 23.7%, followed by Massively Multiplayer (22.3%) and Strategy (20.2%). Casual games have lowest success rate at 12.8% despite being most abundant.

  

7.  **Class Imbalance Analysis:** Pie chart showing the 4.88:1 imbalance ratio. Makes it really clear how skewed the data is (83% fail, 17% succeed).

  

### Potential Feature Engineering Options

 
I'm planning to create several types of features:


**Numerical Features:**

- Price (already have it)

- Achievement count (already have it). Consider logarithmic scaling

- DLC count (already have it). Consider logarithmic scaling

- Language count (need to extract from `supported_languages` - count the commas)

- Description length (character and word count from `short_description`)

- Platform count (just sum the windows/mac/linux flags)

- Required age (already have it)

  
**Categorical Features:**

  
- Primary genre (extract the first genre from the `genres` list)

- Platform flags (windows, mac, linux - already have these)

- Release month/quarter (extract from `release_date`)

- Multiplayer indicator (check if `categories` contains multiplayer/co-op keywords)

- Free-to-play indicator (just check if price == 0)


**Text Features:**

  
- TF-IDF vectors from `short_description` (probably 50 features)

- Maybe keyword extraction for common game terms (action, adventure, RPG, horror, etc.)

- Could do sentiment analysis later if I have time

  

### Modeling Challenges

  

1.  **Class Imbalance**

  

The **4.88:1 imbalance ratio** is pretty severe. If I just predicted "failure" for everything, I'd get 83% accuracy, which sounds great but is useless. I'll need to:

  
- Use `class_weight='balanced'` in my models

- Focus on PR-AUC instead of ROC-AUC (better for imbalanced data)

- Use precision, recall, and F1-score for evaluation

- Maybe do threshold optimization

  

2.  **Temporal Bias**

 
The success rate dropped 66% from 2020 to 2024 (95% if including partial 2025 data), which means the market fundamentally changed. Have to pay attention to this in my train/test split to capture this trend and also validate


3.  **Feature Selection**

  
I'll have 50+ potential features including TF-IDF text features, which could lead to overfitting. I'll need to do feature importance analysis, use regularization, and maybe do some dimensionality reduction.


### Open Questions

  

1.  **Success Threshold**

  

I'm using ≥20,000 owners as the threshold for "commercial success." Is this right? I could explore different thresholds (10K, 50K) to see how sensitive the results are.

  

2.  **Feature Importance**

  

Which features actually matter? Are the TF-IDF text features useful or just noise? I won't know until I train some models.

  

3.  **Temporal Patterns**

  

Is the success rate decline linear or accelerating? Are there seasonal patterns (like holiday releases doing better)? This could inform feature engineering, like adding a "release quarter" feature.

  

4.  **Genre-Specific Patterns**

  

Do different genres have different success patterns? Should I train genre-specific models? That could be a future enhancement.

  

5.  **External Factors**

 
There are things I don't have data for that probably matter:

- Marketing budget

- Developer reputation/history (I have the developer name but not their track record)

- Competition at release time

The model will only capture what's in the data, so this is a limitation.

  

## Conclusion

  
I think the dataset is **suitable for classification** and has real potential for meaningful predictions. The exploratory data analysis revealed some patterns

**The Big Picture:** Success prediction informs strategic decisions (what genre, what price point, what platforms?) rather than post-launch performance signals. This makes it actionable for developers.



