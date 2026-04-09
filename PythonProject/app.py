import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

# ----------------------------
# LOAD DATA (NU HARDCODAT)
# ----------------------------
st.title("Analiza Supermarket")

uploaded_file = st.file_uploader("Încarcă dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview date")
    st.dataframe(df.head())

    # ----------------------------
    # PREPROCESARE
    # ----------------------------
    st.subheader("Preprocesare date")

    # Conversie dată
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature engineering
    df['Month'] = df['Date'].dt.month
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour

    # Verificare valori lipsă
    missing = df.isnull().sum()
    st.write("Valori lipsă:", missing)

    # ----------------------------
    # AGREGĂRI
    # ----------------------------
    st.subheader("Analiza vânzărilor")

    sales_by_city = df.groupby('City')['Sales'].sum().reset_index()
    st.write(sales_by_city)

    fig, ax = plt.subplots()
    sns.barplot(x='City', y='Sales', data=sales_by_city, ax=ax)
    st.pyplot(fig)

    # ----------------------------
    # ANALIZĂ PRODUSE
    # ----------------------------
    product_sales = df.groupby('Product line')['Sales'].sum().reset_index()

    fig2, ax2 = plt.subplots()
    sns.barplot(y='Product line', x='Sales', data=product_sales, ax=ax2)
    st.pyplot(fig2)

    # ----------------------------
    # ENCODING
    # ----------------------------
    st.subheader("Encoding")

    df_encoded = pd.get_dummies(df, drop_first=True)
    st.write("Dimensiune după encoding:", df_encoded.shape)

    # ----------------------------
    # SCALING
    # ----------------------------
    scaler = StandardScaler()

    numeric_cols = ['Unit price', 'Quantity', 'Sales', 'Rating']
    df_scaled = df[numeric_cols]

    df_scaled = scaler.fit_transform(df_scaled)

    # ----------------------------
    # CLUSTERING (KMeans)
    # ----------------------------
    st.subheader("Clustering")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    df['Cluster'] = clusters

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df['Sales'], y=df['Rating'], hue=df['Cluster'], ax=ax3)
    st.pyplot(fig3)

    # ----------------------------
    # REGRESIE LOGISTICĂ
    # ----------------------------
    st.subheader("Regresie Logistică")

    # Target: rating mare (>7)
    df['High_Rating'] = (df['Rating'] > 7).astype(int)

    X = df[['Sales', 'Quantity', 'Unit price']]
    y = df['High_Rating']

    model = LogisticRegression()
    model.fit(X, y)

    st.write("Coeficienți logistic regression:", model.coef_)

    # ----------------------------
    # REGRESIE MULTIPLĂ (STATSMODELS)
    # ----------------------------
    st.subheader("Regresie multiplă")

    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(df['Sales'], X_sm).fit()

    st.text(model_sm.summary())

    # ----------------------------
    # ANALIZĂ TIMP
    # ----------------------------
    st.subheader("Analiză temporală")

    sales_by_month = df.groupby('Month')['Sales'].sum()

    fig4, ax4 = plt.subplots()
    sales_by_month.plot(kind='line', ax=ax4)
    st.pyplot(fig4)

else:
    st.warning("Încarcă un fișier pentru a începe analiza")