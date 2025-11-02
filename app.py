
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="ECO 423 Big Data Dashboard", layout="wide")


st.title("📊 Big Data Analysis Dashboard")
st.markdown("""
### ECO 423 — Big Data Analysis Project 2  
**Students:** *Aizere Muratbek, Burakanova Dilnaz*  
**University:** *Suleyman Demirel University (SDU)*
""")


# Load data

@st.cache_data
def load_data(path='cleaned_data.csv'):
    df_local = pd.read_csv(path)
    return df_local

try:
    df = load_data('cleaned_data.csv')
except FileNotFoundError:
    st.error("File 'cleaned_data.csv' not found in current folder. Please place the CSV in the same folder as app.py.")
    st.stop()

# Preview
st.subheader("Dataset preview")
st.dataframe(df.head(8))

# Diagnostics
st.markdown("### Quick diagnostics")
st.write("Number of records:", len(df))
st.write("Columns in dataset:", df.columns.tolist())
st.write(df.dtypes)


# KPI Row 

st.markdown("## Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Total records
col1.metric('Total Records', f"{len(df):,}")

# KPI 2: Avg Calories Burned (if present)
if 'calories_burned' in df.columns:
    col2.metric('Avg Calories Burned', f"{df['calories_burned'].mean():.2f}")
else:
    col2.metric('Avg Calories Burned', 'N/A')

# KPI 3: % Active (≥3 days/week) using workout_frequency_daysweek if present
if 'workout_frequency_daysweek' in df.columns:
    pct_active = (df['workout_frequency_daysweek'] >= 3).mean() * 100
    col3.metric('% Active (≥3 days/week)', f"{pct_active:.1f}%")
else:
    col3.metric('% Active (≥3 days/week)', 'N/A')

# KPI 4: Avg BMI (if present)
if 'bmi' in df.columns:
    col4.metric('Avg BMI', f"{df['bmi'].mean():.2f}")
else:
    col4.metric('Avg BMI', 'N/A')


# Sidebar: Filters

st.sidebar.header("Filters / Controls (use to slice the data)")

# Gender filter
if 'gender' in df.columns:
    genders = df['gender'].dropna().unique().tolist()
    sel_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
    gender_mask = df['gender'].isin(sel_genders)
else:
    gender_mask = pd.Series(True, index=df.index)

# Workout type filter (useful alternative to 'category')
if 'workout_type' in df.columns:
    types = df['workout_type'].dropna().unique().tolist()
    sel_types = st.sidebar.multiselect("Workout Type", options=types, default=types[:3] if len(types)>3 else types)
    type_mask = df['workout_type'].isin(sel_types)
else:
    type_mask = pd.Series(True, index=df.index)

# Experience level
if 'experience_level' in df.columns:
    levels = df['experience_level'].dropna().unique().tolist()
    sel_levels = st.sidebar.multiselect("Experience Level", options=levels, default=levels)
    level_mask = df['experience_level'].isin(sel_levels)
else:
    level_mask = pd.Series(True, index=df.index)

# Age group filter
if 'age_group' in df.columns:
    ages = df['age_group'].dropna().unique().tolist()
    sel_ages = st.sidebar.multiselect("Age Group", options=ages, default=ages)
    age_mask = df['age_group'].isin(sel_ages)
else:
    age_mask = pd.Series(True, index=df.index)

# Combine masks to filter dataframe
combined_mask = gender_mask & type_mask & level_mask & age_mask
df_filtered = df[combined_mask].copy()

st.markdown("### Filters applied")
st.write("Records after filter:", len(df_filtered))

# Middle: Visualizations (Inverted Pyramid: trends & comparisons)

st.markdown("## Dashboard Visualizations (Trends & Comparisons)")

# 1) Comparison by workout_type (if present) - bar chart
st.markdown("### 1. Category / Workout Type Comparison (Composition & Comparison)")
if 'workout_type' in df_filtered.columns:
    metric_for_comp = 'calories_burned' if 'calories_burned' in df_filtered.columns else ('rating' if 'rating' in df_filtered.columns else None)
    if metric_for_comp:
        comp = df_filtered.groupby('workout_type')[metric_for_comp].mean().reset_index().sort_values(metric_for_comp, ascending=False)
        fig_comp = px.bar(comp, x='workout_type', y=metric_for_comp, title=f'Average {metric_for_comp} by Workout Type')
        st.plotly_chart(fig_comp, use_container_width=True)
        st.write("**Meaning:** shows which workout types (e.g., cardio, strength) lead to higher average", metric_for_comp)
    else:
        st.warning("No suitable numeric metric (calories_burned or rating) to compare by workout_type.")
else:
    st.info("No 'workout_type' column found — skipping workout_type comparison.")


# 2) Distribution of main metric
st.markdown("### 2. Distribution (Range & Grouping)")
main_metric = 'calories_burned' if 'calories_burned' in df_filtered.columns else ('rating' if 'rating' in df_filtered.columns else None)
if main_metric:
    fig_hist = px.histogram(df_filtered, x=main_metric, nbins=40, title=f'{main_metric} Distribution')
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write("**Meaning:** distribution shows central tendency, spread, and outliers for", main_metric)
else:
    st.warning("No main numeric metric (calories_burned or rating) found for distribution plot.")


# 3) Relationship: scatter of two numeric features
st.markdown("### 3. Relationship: Scatter Plot (Correlation / Relationship)")
numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) >= 2:
    x_col = st.selectbox('Choose X axis (numeric)', options=numeric_cols, index=0, key='x_axis')
    y_col = st.selectbox('Choose Y axis (numeric)', options=numeric_cols, index=1, key='y_axis')
    color_by = 'workout_type' if 'workout_type' in df_filtered.columns else None
    fig_scat = px.scatter(df_filtered, x=x_col, y=y_col, color=color_by, hover_data=['age', 'gender'] if 'age' in df_filtered.columns else None,
                          title=f'{y_col} vs {x_col}')
    st.plotly_chart(fig_scat, use_container_width=True)
    st.write("**Meaning:** the scatter plot reveals relationships between two numeric variables (e.g., BMI vs calories burned).")
else:
    st.warning("Not enough numeric columns to create a scatter plot.")


# Bottom: Granular details & Data Table

st.markdown("## Detailed Data / Raw Table (Bottom of the dashboard)")
st.write("This section provides the raw filtered data for deeper inspection (granular details).")
st.dataframe(df_filtered)

# CSV download
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.download_button("Download filtered data as CSV", csv, "filtered_data.csv", "text/csv")

# ---------------------------
# Advanced: KMeans clustering (unsupervised)
# ---------------------------
st.markdown("## 🤖 Advanced Analytics — KMeans Clustering (Segmentation)")
st.write("This section segments users into groups with similar numeric profiles. It can show 'types' of users (e.g., high-burn, low-burn).")

# choose numeric columns automatically but avoid columns that are clearly counts or weird ID columns
numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
# Remove obviously coarse/binary flags or columns that won't help kmeans if needed (optional)
if 'burns_calories_bin' in numeric_cols:
    numeric_cols.remove('burns_calories_bin')

if len(numeric_cols) >= 2:
    st.write("Numeric columns used for clustering:", numeric_cols)
    X = df_filtered[numeric_cols].dropna()
    if len(X) < 2:
        st.warning("Not enough non-null rows for clustering.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider('Select number of clusters (k)', 2, 8, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df_filtered = df_filtered.copy()
        df_filtered['cluster'] = clusters

        # Centroids (scaled)
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
        st.subheader("Cluster centroids (scaled values)")
        st.dataframe(centroids)

        # Visualize cluster sizes
        st.subheader("Cluster sizes")
        st.bar_chart(df_filtered['cluster'].value_counts().sort_index())

        # 2D visualization using first two numeric cols
        fig_cluster = px.scatter(df_filtered, x=numeric_cols[0], y=numeric_cols[1], color='cluster',
                                 title=f'KMeans clusters (k={k})', hover_data=['workout_type'] if 'workout_type' in df_filtered.columns else None)
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.write("**Meaning:** clusters group similar records. Centroids summarize the typical profile of each cluster (in scaled units).")
else:
    st.warning("Not enough numeric columns for clustering — need at least 2 numeric features.")
