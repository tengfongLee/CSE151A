# CSE151A
ML_project

## Dataset
Dataset is stored on Google Drive:  (Since data is bigger than 100MB so I have to do this)

https://drive.google.com/drive/folders/1-FgLo4-60RSu90v8pcpO4wi5B3rW-qAM?dmr=1&ec=wgc-drive-globalnav-goto

Here is the oringal source :
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?resource=download&select=recommendations.csv

## Environment Set Up (Google Colab)

1. Open the notebook in Google Colab. https://colab.google/
2. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')



For questions 6 and 4, I first checked for duplicate records and missing values. Since the data was clean, I then manually inspected the features to identify any irrelevant ones, such as steam_deck, which I decided to drop.
Finally, I normalized some features to ensure they are on a common scale. I also dropped features such as review_id, since it is randomly generated and does not carry any meaningful information for the analysis.
In this case, I applied a log transform because some of the numeric features were highly skewed.


You can run the project notebook directly in Google Colab here:  



## Introduction

In this project, the goal was to predict whether a user would **recommend a game on Steam** based on review metadata and additional information from users and games datasets. Instead of using libraries like `sklearn`, we built a **Decision Tree Classifier from scratch** (CART with Gini impurity). The dataset was quite large, so a lot of effort went into preprocessing and memory management to make sure it could run smoothly on Google Colab.

---

## Data Preprocessing (<h1 style="color:red;">⚠️ From Here is what I have added for MS3</h1>)

### 1. Cleaning and Typing

* Converted `helpful`, `funny`, and `hours` into numeric types.
* Converted `date` and `date_release` into datetime objects.
* Converted `is_recommended` into binary values (1 = recommended, 0 = not recommended).

### 2. Enrichment

* Merged in `users.csv` to include fields like `country`, `account_age`, and `num_friends`.
* Merged in `games.csv` to include `price_final`, `positive_ratio`, `rating`, `win/mac/linux` support, `date_release`, and `genres`.

### 3. Feature Expansion

To give the model more useful signals, we engineered extra features:

* **Log transforms:** `log_hours`, `log_helpful`, `log_funny` (to reduce skew from heavy-tailed distributions).
* **Ratios:** `helpful_per_hour`, `funny_per_hour`.
* **Interaction:** `hf_interact = helpful * funny`.
* **Temporal features:**

  * `recency_days` = days since the review was written.
  * Extracted `year`, `month`, `day of week` from review date.
  * `game_age_days` = days since the game’s release.

### 4. Encoding and Imputation

* One-hot encoding for small categorical columns (low number of unique values).
* Frequency encoding for high-cardinality columns like `genres` and `country` to save memory.
* Median imputation for missing numeric values.
* All numeric features were downcast to `float32` or `int8` to reduce memory usage.

---

## Model: Decision Tree (From Scratch)

We implemented a **CART decision tree** that:

* Uses **Gini impurity** as the splitting criterion.
* Stops splitting if:

  * maximum depth is reached,
  * a node has fewer than `min_leaf` samples,
  * or the node is already pure.
* Regularization parameters we used:

  * `max_depth = 10`
  * `min_leaf = 100`
* To make training efficient on large data:

  * We limited candidate thresholds (`max_candidates = 32`).
  * For very large columns, thresholds are sampled from quantiles on a subsample.

---

## Evaluation

We performed an **80/20 train/test split**.

* Training accuracy: **(printed in output)**
* Test accuracy: **(printed in output)**
* Confusion matrix: printed to show counts of TP, TN, FP, FN.

### Observations

* If training accuracy is much higher than test accuracy → the tree is **overfitting**.

  * Fix: decrease `max_depth` or increase `min_leaf`.
* If both training and test accuracy are low → the tree is **underfitting**.

  * Fix: increase depth, lower `min_leaf`, or add more features.

In our case, after tuning, the train and test performance were reasonably close, which suggests the tree was not severely overfitting.

---

## Memory Management

Because the dataset is very large, we had to optimize carefully:

* Avoided making full `.copy()` calls on big DataFrames.
* Used `float32` and `int8` types instead of the default `float64`.
* Used frequency encoding instead of full one-hot for high-cardinality columns.
* Limited candidate thresholds per feature and sampled quantiles instead of checking every possible split.

This allowed the model to train without running out of RAM on Colab.

