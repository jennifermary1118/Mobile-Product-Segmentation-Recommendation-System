# 📱 Mobile Product Segmentation & Recommendation System

## Project Overview
A Streamlit-based machine learning application for analyzing mobile products, grouping them into meaningful clusters, and recommending similar products.

## Features

### 📊 Dashboard
- Total number of records
- Total number of columns
- Brand distribution
- Brand-wise bar chart

### 🔍 Cluster Analysis
- Cluster distribution
- Cluster characteristics
- Average price and ratings
- Battery, camera, performance, design and display ratings
- Helpful votes
- Cluster insights
- PCA-based 2D visualization

### 🤖 Product Recommendations
Users can select a mobile product and find similar products using a saved Nearest Neighbors model.

Recommendations use:
- Price
- Rating
- Battery life rating
- Camera rating
- Performance rating
- Design rating
- Display rating

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- K-Means Clustering
- Nearest Neighbors
- PCA

## Project Structure

```text
Mobile-Product-Segmentation-Recommendation/
├── app.py
├── README.md
├── requirements.txt
├── mobile_processed.csv
├── kmeans.pkl
├── nn_model.pkl
├── scaler.pkl
└── recommendation_scaler.pkl
```

## Required Files

The application expects these files in the same folder as `app.py`:

- `mobile_processed.csv` – processed mobile product dataset
- `kmeans.pkl` – saved K-Means model
- `nn_model.pkl` – saved Nearest Neighbors model
- `scaler.pkl` – scaler for cluster analysis
- `recommendation_scaler.pkl` – scaler for recommendations

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Keep all required files together

Make sure the CSV file and all `.pkl` files are available in the project directory.

### 3. Run the application

```bash
streamlit run app.py
```

## requirements.txt

```text
streamlit
pandas
scikit-learn
joblib
```

## Recommendation Features

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
```

## Cluster Analysis Features

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
helpful_votes
```

## Cluster Insights

| Cluster | Description |
|---|---|
| Cluster 0 | Mid-range / Average Products |
| Cluster 1 | Premium Products |
| Cluster 2 | Low-rated / Poor-performing Products |
| Cluster 3 | High-performing / Highly Rated Products |

## Machine Learning Workflow

```text
Processed Mobile Dataset
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
Mobile Product Segments
        ↓
Cluster Analysis
        ↓
PCA Visualization
```

### Recommendation Workflow

```text
Selected Mobile Product
        ↓
Recommendation Features
        ↓
Recommendation Scaling
        ↓
Nearest Neighbors Model
        ↓
Similar Mobile Products
```

## Application Navigation

The sidebar contains:

1. Dashboard
2. Cluster Analysis
3. Recommendations

## Project Objective

The objective of this project is to use machine learning techniques to segment mobile products into meaningful groups and recommend similar products based on price, ratings, and specifications.

## Project Details

**Project Name:** Mobile Product Segmentation & Recommendation System  
**Framework:** Streamlit  
**Language:** Python  
**Machine Learning:** K-Means, Nearest Neighbors, PCA  
**Data Processing:** Pandas  
**Model Serialization:** Joblib

## Conclusion

This project combines data analysis, clustering, dimensionality reduction, and recommendation techniques into an interactive Streamlit application for exploring and recommending mobile products.
