# Predicting Indie Game Success: Can We Beat the Odds?

  

## What We're Trying to Solve

The indie game market is brutal. Most developers spend years creating a game, pour their heart into it, and then watch it fail commercially. I wanted to see if machine learning could help—specifically, could we predict which games would succeed *before* they even launch?  

I analyzed nearly 47,000 indie games released on Steam between 2020 and 2025. The goal was simple: predict whether a game would sell at least 20,000 copies using only information available during development. No cheating with post-launch reviews or sales data — just features developers can actually control.

  

## The Data 

  

### What We're Working With

I started with Steam's entire game catalog (about 90,000 games) and filtered it down to true indie games. My criteria were:

-  **Small studios**: Developers with 3 or fewer games (filtering out big publishers)

-  **Indie pricing**: $0-30 range (AAA games typically cost $60+)

-  **Recent releases**: 2020-2025 (the COVID era and beyond)

This gave me 46,957 indie games to analyze.


### Sobering Reality

Here is a sobering truth: only **17% of indie games** reached 20,000 ownerships. That means 83% failed to hit even this modest commercial threshold. This imbalance makes prediction really challenging—a naive model could just predict "failure" for everything and be right 83% of the time, which isn't helpful.
  
Even worse, the market is getting harder. Success rates dropped from 29% in 2020 to just 8.5% in 2024. More games are launching every year, but fewer are succeeding. Market saturation is real.


##  Features

  
The key to this project was using _only_ pre-launch features—things a developer knows before hitting that "publish" button. This makes the predictions actually useful in the real world.

  
Built **136 features** total, split into multiple categories:

  
**Basic Game Info (10 features):**

- Price, number of achievements, DLC count, age rating

- Language support count (how many languages the game supports)

- Platform count (Windows, Mac, Linux)

- Description length and word count

- Detailed features length and word count

  
**Game Description Text (50 features):**

- Used TF-IDF to extract the 50 most important words and phrases from combined short descriptions and detailed features

- This captures things like genre keywords ("horror", "rpg", "adventure") and game concepts ("world", "story", "multiplayer")

- Turns out, _how_ you describe your game matters a lot

- The features combine both short_description and detailed_features columns for richer text analysis
  

**Genre Features (30 features):**

- Multi-hot encoded all 30 unique genres (1 if game has that genre, 0 otherwise)

- Captures the full breadth of genre tags rather than just the primary genre

- Examples: Action, RPG, Casual, Adventure, Simulation, Strategy, Indie, Casual, Puzzle, etc.
 
 
**Category Features (40 features):**
 
- Multi-hot encoded all 40 unique Steam categories

- Examples: Single-player, Multiplayer, Steam Trading Cards, Steam Cloud, Achievements, etc.

  
**Platform & Metadata Features (6 features):**

- Platform flags (Windows, Mac, Linux)

- Whether it has multiplayer

- Free-to-play or paid flag

- Release timing (month, quarter, Q4 holiday release)

- Developer experience (how many games they've made before)

  

### What I Didn't Use

  
This is important—I explicitly _excluded_ anything that happens after launch:


-  Review scores and counts (obviously post-launch)

- Discount percentages (happens later)

-  Community tags (players add these after release)

- Time on market metrics

  
Some features were borderline. For example, achievements are designed pre-launch but only become visible after release. I kept these because developers genuinely plan them beforehand—they're part of the pre-launch design process.

  

## Training 

  

### Avoiding leakage


I couldn't just randomly split the data into training and test sets. That would cause data leakage. 
  
Instead, I used a **temporal split**:

 
-  **Training**: 2020-2023 games (29,294 games, 22.1% success rate)

-  **Test**: 2024+ games (17,663 games, 8.5% success rate)

  
This simulates real deployment—training on the past, predicting the future. It also makes the problem harder because the market got tougher over time.
  

### Four Approaches

  
I tried four different models to see what works best:

  
**1. Baseline (Stratified Random)**

Just randomly predicting success based on the training set's success rate (25.6%). This gives us a sanity check—any real model should beat this.

  

**2. Logistic Regression**

The classic, interpretable option. I used balanced class weights to handle the imbalance. It's simple, fast, and gives us interpretable coefficients.

  

**3. Random Forest**

An ensemble of 100 decision trees. Good at handling non-linear patterns and provides feature importance. Also used balanced class weights.

  

**4. XGBoost**

Advanced gradient boosting. Automatically handles the class imbalance with scale_pos_weight. Usually the best performer on structured data.

  

### Handling the Imbalance

  
With 83% of games failing (looking at the full dataset), I had to make sure I handled the class imbalance. For each model, I used:


-  `class_weight='balanced'` for Logistic Regression and Random Forest

-  `scale_pos_weight` for XGBoost (automatically adjusts for imbalance)

- Focus on PR-AUC instead of accuracy (more meaningful for imbalanced data) and 'precision' to make sure we prefer model with higher precision even if it comes at moderate expense of recall.

  

## Results

  
Here's how the models performed on 2024+ games (the test set):

  

| Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |

| -------------------- | --------- | --------- | --------- | --------- | --------- |

| Baseline (Random) | 8.3% | 21.4% | 11.9% | 0.543 | N/A |

| Logistic Regression | 16.6% | 66.1% | 26.5% | 0.298 | 0.745 |

| **Random Forest** | **49.4%** | **23.6%** | **32.0%** | **0.319** | **0.744** |

| XGBoost | 19.3% | 60.5% | 29.3% | 0.324 | 0.744 |

  
### Comparison of models

  
**Random Forest wins** with the best F1-score (32.0%) and by far the best precision (49.4%). But there's an interesting trade-off here:


-  **Random Forest** is conservative: When it says a game will succeed, it's right 49% of the time—nearly 6× better than the 8.3% baseline. But it only catches 24% of successful games (low recall).

-  **Logistic Regression** is aggressive: It catches 66% of successful games (high recall) but flags way too many false positives—only 17% of its predictions are correct.

-  **XGBoost** sits in between: Catches 61% of successes with 19% precision. It has the best PR-AUC though (0.324).

For this problem, I'd choose **Random Forest** as the winner. When you're trying to guide developers on multi-year projects, you want to be confident in your predictions. Random Forest's 49% precision is nearly 6× better than randomly guessing, and that's valuable signal. 


### Visualizing the Results
  
Generated five comprehensive visualizations

**1. Confusion Matrices** Shows Random Forest's strength in precision (fewer false positives) versus Logistic Regression's tendency to predict more games as successful.

**2. ROC Curves**  All three trained models cluster around ROC-AUC of 0.74-0.75, significantly better than the baseline.

**3. Precision-Recall Curves** - More informative for imbalanced datasets. Random Forest's curve shows better precision at high thresholds, making it the most reliable for conservative predictions.

**4. Feature Importance** - A bar chart that shows which features drive Random Forest's predictions. 

**5. Model Comparison** - A comprehensive comparison chart plotting all models across multiple metrics (Precision, Recall, F1-Score, PR-AUC, ROC-AUC), making it easy to see Random Forest's better performance


### Why I think these numbers are good?

  
At first glance, 49% precision might seem low. But consider:

- The baseline is 8.3%—we're **6× better**

- We're predicting creative product success years in advance

- We're only using pre-launch features (no reviews, no marketing data)

- The market is incredibly competitive and unpredictable

  
In reality, improving a developer's odds from 8.3% to 49% is huge. That's the difference between most games failing and having a real shot at success.



## What increases the chances of success?

  
Looking at Random Forest's feature importance, here are the top 10 factors:

1.  **Description length** (5.99%) - Detailed short_descriptions correlate with success

2.  **Description word count** (5.35%) - Quality of short_descriptions matter

3.  **Price** (5.03%) - Pricing strategy is critical

4.  **Achievements** (4.51%) - Shows game depth

5.  **Language support** (4.25%) - More languages = bigger market

6.  **TF-IDF: "game"** (3.12%) - Description keywords matter

7.  **DLC count** (2.25%) - Shows long-term commitment

8.  **Steam Trading Cards** (2.17%) - Platform integrations that unlock additional incentives matter

9.  **Steam Cloud** (1.55%) - Cloud streaming adds value

10.  **TF-IDF: "world"** (1.55%) - World-building games do better


### Key Takeaways for Developers

If you're making an indie game, this research suggests:

1.  **Write a good description** - It's the #1 and #2 predictor. Don't just slap together a few sentences. Explain your game well.

2.  **Think about pricing** - The $10-15 range tends to work best for indies. Don't underprice thinking it'll help—free games actually have _lower_ success rates unless you have a monetization strategy.

3.  **Design for depth** - Achievements signal engagement and replay value. Plan for 20-30 achievements.

4.  **Go multilingual** - Supporting multiple languages (especially major markets like Chinese, Spanish, German) significantly improves your odds.

5.  **Plan your DLC** - Even announcing future content shows commitment and correlates with success.

  

## Potential sources of error and/or bias

  

### 1. Rapid evolution of the market

The biggest concern for error is probably temporal drift. The market in 2024 is fundamentally different from 2020-2023:

- Way more competition (games per year more than doubled)
- Success rates dropped 71%
- Player preferences may have shifted

The model learned from the "old market" but has to predict in the "new market" thats ever changing

### 2. Selection/Availability Bias in the Data

  We only see games that actually launched on Steam.  We don't have data on  games that were 
  - cancelled during development
  - launched on other platforms
  - never got past the prototype stage

This means the "true" failure rate might be even higher than 83%.


### 3. Feature Measurement Issues

 
Some features might have measurement errors:
- DLC count could include planned DLC that was cancelled
- Achievement counts might change post-launch
- Descriptions might get edited after release which could impact the performance


## Improvements and ideas for the final report


This is just the first draft—a solid baseline. Here's what I'd explore for the final version:


-  **Sentiment analysis** on descriptions (are positive descriptions correlated with success?)

-  **Developer history** tracked properly (how did their previous games perform?)

-  **Market saturation metrics** (how many similar games launched recently?)

-  **Test overfitting** With 136 features and ~29,000 training examples, there's a risk of overfitting. Try to remove less important features

-  **Hyperparameter tuning** with GridSearchCV or RandomizedSearchCV

-  **Threshold optimization** for different business scenarios

 -  **Genre-specific models** (maybe RPGs and Casual games need different models?)

-  **Price tier models** (free-to-play vs budget vs premium games)

---