# US Accidents: Big Data Severity Analysis

Analysing 4.7 million US traffic accident records (2020–2023) to find when, where and in what conditions accidents happen, and predicting accident severity with a PySpark machine learning pipeline. Built for the *Big Data Analytics & Visualisation* course at Al Hussein Technical University (Jun 2026).

## Problem

A data consultancy needs to show how a large, messy dataset can be turned into decisions, for example where and when road-safety resources should be focused. The challenge is to process a multi-GB dataset efficiently on a single machine, extract insights, and build a model that predicts how severe an accident will be (levels 1–4).

## Data

- Source: [US Accidents (2016–2023), Kaggle, by Sobhan Moosavi](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), file `US_Accidents_March23.csv` (about 3 GB)
- Filtered to 2020–2023: 4,751,751 records, 46 columns (time, location, weather, and road features such as junctions, crossings and traffic signals)
- The dataset is not included in this repo because of its size. The notebook downloads it automatically with `kagglehub`.

## Pipeline

1. **Efficient loading:** Polars lazy scanning (`scan_csv`) filters by year before loading data into memory.
2. **Cleaning:**
   - Dropped 11 redundant, mostly empty or zero-variance columns.
   - Removed rows with missing City or day/night values (23,139 rows).
   - Filled missing weather values with the median, missing precipitation with 0, and missing text values with "Unknown".
   - Removed physically impossible readings.
   - Result: 4,728,586 clean rows with no nulls.
3. **Exploratory analysis:** interactive Plotly charts of accidents by severity, hour, month, day vs night, top 10 states, weather condition vs severity, and a correlation heatmap.
4. **Automated ML pipeline (PySpark):** reusable `load → feature_engineering → explore → train_evaluate` functions.
   - **Time features:** hour, month, day of week, rush hour, weekend.
   - **Weather and road features:** visibility and temperature bands, heavy rain flag, and a *Road Complexity Score* (the count of road features near the accident).
   - **Model:** StringIndexer → VectorAssembler → StandardScaler → Random Forest (100 trees, depth 10), trained on an 80/20 split (3.78M train / 0.95M test rows).

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.889 |
| Weighted precision | 0.879 |
| Weighted F1 | 0.854 |

**Important context:** severity level 2 makes up 87.6% of all accidents, so a model that always predicts "2" already scores about 0.876 accuracy. The Random Forest only slightly beats this baseline, which means it mostly learns the majority class. Accuracy alone is misleading for this dataset.

Top features by importance: `Distance(mi)`, `Year`, `State`, `Month`, `Road_Complexity_Score`, `Hour`.

## Limitations and next steps

- **Class imbalance:** add class weights or resampling, and report per-class recall and macro F1, especially for the rare, most serious levels 3 and 4.
- **Possible leakage:** `Distance(mi)` is the length of road affected by the accident, which is only known afterwards. A model that predicts severity at the moment of the accident should exclude it.
- **Data quality:** the data is aggregated from several reporting sources, and 2023 only covers January to March.

## Tools

Python, Polars, PySpark (MLlib), Plotly, Google Colab, kagglehub

## How to run

Open the notebook in Google Colab and run all cells. Colab already includes PySpark. Otherwise:

```bash
pip install polars pyspark plotly kagglehub pandas
```
