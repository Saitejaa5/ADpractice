import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------
# 1️⃣ Load Data (Upload CSV)
# ---------------------------------------
st.title("📊 Customer Segmentation Dashboard")

uploaded_file = st.file_uploader("Upload your preprocessed CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------------------------------
    # 2️⃣ PCA for Visualization
    # ---------------------------------------
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

    # ---------------------------------------
    # 3️⃣ K-Means Clustering
    # ---------------------------------------
    k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)

    # ---------------------------------------
    # 4️⃣ Visualizations
    # ---------------------------------------
    st.subheader("📍 PCA Clusters Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100, ax=ax)
    ax.set_title("PCA Scatter Plot with K-Means Clusters")
    st.pyplot(fig)

    # ---------------------------------------
    # 5️⃣ Cluster Insights
    # ---------------------------------------
    st.subheader("📈 Average Spending by Cluster")
    spending_cols = [col for col in df.columns if 'Mnt' in col]
    spending_summary = df.groupby('Cluster')[spending_cols].mean().round(2)
    st.dataframe(spending_summary)

    st.subheader("👥 Demographics Overview")
    demographic_cols = ['Age', 'Income', 'Total_Children'] if set(['Age', 'Income', 'Total_Children']).issubset(df.columns) else []
    if demographic_cols:
        demographic_summary = df.groupby('Cluster')[demographic_cols].mean().round(2)
        st.dataframe(demographic_summary)
    else:
        st.write("Demographic columns not found.")

    # ---------------------------------------
    # 6️⃣ Download Results
    # ---------------------------------------
    st.subheader("⬇️ Download Clustered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "clustered_data.csv", "text/csv")

else:
    st.info("Please upload a preprocessed CSV file to begin.")
