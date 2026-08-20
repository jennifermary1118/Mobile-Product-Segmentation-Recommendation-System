import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA

# Load processed dataset
mobile_df = pd.read_csv('mobile_processed.csv')

# Load saved models
kmeans = joblib.load('kmeans.pkl')
nn_model = joblib.load('nn_model.pkl')

# Load saved scalers
scaler = joblib.load('scaler.pkl')
recommendation_scaler = joblib.load('recommendation_scaler.pkl')

recommendation_features = [
    'price_usd',
    'rating',
    'battery_life_rating',
    'camera_rating',
    'performance_rating',
    'design_rating',
    'display_rating'
]

X_recommend = mobile_df[recommendation_features]

X_recommend_scaled = recommendation_scaler.transform(
    X_recommend
)

st.title("📱 Mobile Product Segmentation & Recommendation System")

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Cluster Analysis",
        "Recommendations"
    ]
)

# Dashboard
if page == "Dashboard":

    st.subheader("Dataset Overview")

    st.write(f"Total Records: {len(mobile_df)}")
    st.write(f"Total Columns: {mobile_df.shape[1]}")

    st.subheader("Brand Distribution")

    brand_counts = mobile_df['brand'].value_counts()

    st.bar_chart(brand_counts)


# Cluster Analysis
elif page == "Cluster Analysis":

    st.title("CLUSTER ANALYSIS")

    

    cluster_counts = mobile_df['cluster'].value_counts().sort_index()

    st.write("### Cluster Distribution")

    st.write(cluster_counts)

    st.bar_chart(cluster_counts)

    st.write("### Cluster Characteristics")

    cluster_summary = (
        mobile_df
        .groupby('cluster')[
            [
                'price_usd',
                'rating',
                'battery_life_rating',
                'camera_rating',
                'performance_rating',
                'design_rating',
                'display_rating',
                'helpful_votes'
            ]
        ]
        .mean()
        .round(2)
    )

    st.dataframe(cluster_summary)
    st.write("### Cluster Insights")

    cluster_insights = {
        0: "Mid-range / Average Products",
        1: "Premium Products",
        2: "Low-rated / Poor-performing Products",
        3: "High-performing / Highly Rated Products"
    }

    for cluster, description in cluster_insights.items():
        count = cluster_counts.get(cluster, 0)

        st.write(
            f"**Cluster {cluster} — {description}** "
            f"({count:,} records)"
        )

    st.write("### Cluster Visualization")

    cluster_features = [
        'price_usd',
        'rating',
        'battery_life_rating',
        'camera_rating',
        'performance_rating',
        'design_rating',
        'display_rating',
        'helpful_votes'
    ]

    X_cluster = mobile_df[cluster_features]

    X_cluster_scaled = scaler.transform(X_cluster)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_cluster_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=['PCA1', 'PCA2']
    )

    pca_df['cluster'] = mobile_df['cluster']

    st.scatter_chart(
        pca_df,
        x='PCA1',
        y='PCA2',
        color='cluster'
    )
# Recommendations
elif page == "Recommendations":

    st.subheader("📱 Product Recommendation")

    st.write(
        "Select a mobile product to find similar products "
        "based on price, ratings, and specifications."
    )
    selected_index = st.selectbox(
        "Select a mobile product",
        mobile_df.index,
        format_func=lambda x: (
            f"{mobile_df.loc[x, 'brand']} - "
            f"{mobile_df.loc[x, 'model']}"
        )
    )

    if st.button("Get Recommendations"):

        st.write("### Selected Product")

        selected_product = mobile_df.loc[selected_index]

        st.write(
            f"**{selected_product['brand']} {selected_product['model']}**"
        )

        st.write(
            f"Price: ${selected_product['price_usd']:.2f} | "
            f"Rating: {selected_product['rating']}"
        )

        distances, indices = nn_model.kneighbors(
            X_recommend_scaled[selected_index].reshape(1, -1)
        )

        recommended_indices = indices[0][1:]

        recommendations = mobile_df.iloc[recommended_indices][
            [
                'brand',
                'model',
                'price_usd',
                'rating',
                'battery_life_rating',
                'camera_rating',
                'performance_rating',
                'design_rating',
                'display_rating'
            ]
        ]

        st.write("### Recommended Products")

        st.dataframe(recommendations)