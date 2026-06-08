# Sales Forecasting Using Machine Learning

## Project Overview

This project demonstrates a complete workflow for forecasting sales using historical data. The goal was to predict sales and identify key factors driving sales performance.

## Dataset

- 2,924 sales records
- Includes `Date`, `Region`, `Product`, `Sales`
- Seasonal patterns, regional differences, and product-level variability
- Cleaned for missing values, duplicates, and inconsistent categories

## Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Categorical Encoding
4. Train/Test Split
5. Linear Regression Model Training
6. Model Evaluation
7. Feature Importance Analysis
8. Visualizations saved in `images/`

## Model Performance

| Metric | Value  |
| ------ | ------ |
| MAE    | 195.09 |
| RMSE   | 248.79 |
| R²     | 0.668  |

## Key Findings

- North region generated the highest sales.
- Laptop products outperformed others.
- Seasonal trends influenced sales significantly.
- Product and region were strongest predictors.

## Visualizations

- **Monthly Sales Trend**: `images/monthly_sales_trend.png`
- **Sales by Region**: `images/sales_by_region.png`
- **Product Performance**: `images/product_performance.png`
- **Actual vs Predicted Sales**: `images/actual_vs_predicted.png`
- **Feature Importance**: `images/feature_importance.png`
- **Combined Metrics & Feature Importance**: `images/metrics_and_feature_importance.png`

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn

## Author

Senet Sasu

Project completed as part of Future Interns program.
