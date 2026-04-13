import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURARE APLICATIE
# =========================================================
st.set_page_config(
    page_title="Proiect - Marketing Bancar",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Analiza activității unei bănci și posibilități de extindere")

DATA_PATH = "Bank_Marketing_Dataset.csv"
TARGET = "TermDepositSubscribed"
ID_COL = "ClientID"


# =========================================================
# FUNCTII
# =========================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def replace_unknown_with_nan(df):
    df = df.copy()
    values_to_replace = ["unknown", "Unknown", "UNKNOWN", "NA", "N/A", "", " "]
    return df.replace(values_to_replace, np.nan)


def convert_yes_no_columns(df):
    df = df.copy()
    yes_no_map = {
        "Yes": 1, "No": 0,
        "YES": 1, "NO": 0,
        "yes": 1, "no": 0
    }

    for col in df.columns:
        if df[col].dtype == "object":
            unique_values = set(df[col].dropna().astype(str).unique())
            if len(unique_values) > 0 and unique_values.issubset(set(yes_no_map.keys())):
                df[col] = df[col].map(yes_no_map)

    return df


def fill_missing_values(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].mode(dropna=True).empty:
            df[col] = df[col].fillna("Necunoscut")
        else:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    return df


def detect_outliers_iqr(df, col):
    s = pd.to_numeric(df[col], errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((s < lower) | (s > upper)).sum()

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Prag inferior": lower,
        "Prag superior": upper,
        "Număr outlieri": int(count)
    }


def cap_outliers_iqr(df, columns):
    df = df.copy()

    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = s.clip(lower, upper)

    return df


def prepare_data(df):
    df = df.copy()
    df = replace_unknown_with_nan(df)
    df = convert_yes_no_columns(df)
    df = fill_missing_values(df)
    return df

# INCARCARE DATE
try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("Fișierul Bank_Marketing_Dataset.csv trebuie să fie în același folder cu app.py.")
    st.stop()

df = prepare_data(raw_df)

if TARGET not in df.columns:
    st.error(f"Coloana target {TARGET} nu există în fișier.")
    st.stop()

# SIDEBAR
st.sidebar.header("Navigare")
section = st.sidebar.radio(
    "Alege secțiunea:",
    [
        "1. Prezentarea datelor",
        "2. Curățarea datelor",
        "3. Analiză exploratorie",
        "4. Grupare și agregare",
        "5. Clusterizare KMeans",
        "6. Regresie logistică"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtre")
region_options = ["Toate"] + sorted(df["Region"].astype(str).unique().tolist())
segment_options = ["Toate"] + sorted(df["CustomerSegment"].astype(str).unique().tolist())

selected_region = st.sidebar.selectbox("Regiune", region_options)
selected_segment = st.sidebar.selectbox("Segment client", segment_options)

filtered_df = df.copy()

if selected_region != "Toate":
    filtered_df = filtered_df[filtered_df["Region"].astype(str) == selected_region]

if selected_segment != "Toate":
    filtered_df = filtered_df[filtered_df["CustomerSegment"].astype(str) == selected_segment]


# =========================================================
# 1. PREZENTAREA DATELOR
# =========================================================
if section == "1. Prezentarea datelor":
    st.header("1. Prezentarea setului de date")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Număr rânduri", filtered_df.shape[0])
    c2.metric("Număr coloane", filtered_df.shape[1])
    c3.metric("Variabile numerice", filtered_df.select_dtypes(include=[np.number]).shape[1])
    c4.metric("Variabile categorice", filtered_df.select_dtypes(exclude=[np.number]).shape[1])

    st.subheader("Primele 10 înregistrări")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.subheader("Tipurile de date")
    info_df = pd.DataFrame({
        "Coloană": filtered_df.columns,
        "Tip date": filtered_df.dtypes.astype(str).values,
        "Valori lipsă": filtered_df.isna().sum().values,
        "Valori distincte": [filtered_df[col].nunique() for col in filtered_df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistici descriptive")
    st.dataframe(filtered_df.describe().T, use_container_width=True)

# 2. CURATAREA DATELOR
elif section == "2. Curățarea datelor":
    st.header("2. Curățarea datelor")

    st.subheader("Valori lipsă")
    missing_df = pd.DataFrame({
        "Coloană": filtered_df.columns,
        "Număr valori lipsă": filtered_df.isna().sum().values,
        "Procent (%)": (filtered_df.isna().mean() * 100).round(2).values
    }).sort_values("Număr valori lipsă", ascending=False)

    st.dataframe(missing_df, use_container_width=True)

    numeric_cols = [col for col in filtered_df.select_dtypes(include=[np.number]).columns if col not in [TARGET, ID_COL]]

    st.subheader("Detectarea outlierilor")
    selected_numeric = st.selectbox("Alege o variabilă numerică:", numeric_cols)

    outlier_info = detect_outliers_iqr(filtered_df, selected_numeric)
    outlier_df = pd.DataFrame([outlier_info])
    st.dataframe(outlier_df, use_container_width=True)

    fig_box = px.box(
        filtered_df,
        y=selected_numeric,
        points="outliers",
        title=f"Boxplot pentru {selected_numeric}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Exemplu de tratare a outlierilor")
    default_cols = ["AnnualIncome", "AccountBalance", "CreditScore", "MarketingScore"]
    default_cols = [col for col in default_cols if col in numeric_cols]

    selected_clip_cols = st.multiselect(
        "Selectează coloanele pentru limitarea outlierilor:",
        numeric_cols,
        default=default_cols
    )

    if selected_clip_cols:
        clipped_df = cap_outliers_iqr(filtered_df, selected_clip_cols)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Înainte de tratare**")
            st.dataframe(filtered_df[selected_clip_cols].describe().T, use_container_width=True)

        with col2:
            st.markdown("**După tratare**")
            st.dataframe(clipped_df[selected_clip_cols].describe().T, use_container_width=True)

# 3. ANALIZA EXPLORATORIE
elif section == "3. Analiză exploratorie":
    st.header("3. Analiză exploratorie")

    numeric_cols = [col for col in filtered_df.select_dtypes(include=[np.number]).columns if col != TARGET]

    selected_hist = st.selectbox("Alege variabila pentru histogramă:", numeric_cols)

    fig_hist = px.histogram(
        filtered_df,
        x=selected_hist,
        nbins=40,
        title=f"Distribuția variabilei {selected_hist}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    if "Region" in filtered_df.columns:
        temp_region = filtered_df.groupby("Region")[TARGET].mean().reset_index()
        temp_region[TARGET] = (temp_region[TARGET] * 100).round(2)

        fig_region = px.bar(
            temp_region.sort_values(TARGET, ascending=False),
            x="Region",
            y=TARGET,
            title="Rata de subscriere pe regiuni (%)"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    if "CustomerSegment" in filtered_df.columns:
        temp_segment = filtered_df.groupby("CustomerSegment")[TARGET].mean().reset_index()
        temp_segment[TARGET] = (temp_segment[TARGET] * 100).round(2)

        fig_segment = px.bar(
            temp_segment.sort_values(TARGET, ascending=False),
            x="CustomerSegment",
            y=TARGET,
            title="Rata de subscriere pe segmente de clienți (%)"
        )
        st.plotly_chart(fig_segment, use_container_width=True)

    st.subheader("Matrice de corelație")
    corr_cols = [
        "Age", "AnnualIncome", "CreditScore", "AccountBalance",
        "TotalTransactions", "MarketingScore", "ResponsePropensity",
        TARGET
    ]
    corr_cols = [col for col in corr_cols if col in filtered_df.columns]

    corr = filtered_df[corr_cols].corr(numeric_only=True)

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Corelațiile dintre variabile"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# 4. GRUPARE SI AGREGARE
elif section == "4. Grupare și agregare":
    st.header("4. Grupare și agregare în pandas")

    st.subheader("Agregare pe regiune")
    region_summary = filtered_df.groupby("Region").agg(
        NumarClienti=(TARGET, "count"),
        RataSubscriere=(TARGET, "mean"),
        VenitMediu=("AnnualIncome", "mean"),
        SoldMediu=("AccountBalance", "mean")
    ).reset_index()

    region_summary["RataSubscriere"] = (region_summary["RataSubscriere"] * 100).round(2)
    st.dataframe(region_summary, use_container_width=True)

    fig_region_summary = px.bar(
        region_summary.sort_values("RataSubscriere", ascending=False),
        x="Region",
        y="RataSubscriere",
        title="Rata de subscriere pe regiuni (%)"
    )
    st.plotly_chart(fig_region_summary, use_container_width=True)

    st.subheader("Agregare pe segment de client")
    segment_summary = filtered_df.groupby("CustomerSegment").agg(
        NumarClienti=(TARGET, "count"),
        RataSubscriere=(TARGET, "mean"),
        ScorMarketingMediu=("MarketingScore", "mean"),
        PropensitateMedie=("ResponsePropensity", "mean")
    ).reset_index()

    segment_summary["RataSubscriere"] = (segment_summary["RataSubscriere"] * 100).round(2)
    st.dataframe(segment_summary, use_container_width=True)

    fig_segment_summary = px.bar(
        segment_summary.sort_values("RataSubscriere", ascending=False),
        x="CustomerSegment",
        y="RataSubscriere",
        title="Rata de subscriere pe segmente (%)"
    )
    st.plotly_chart(fig_segment_summary, use_container_width=True)

    st.subheader("Agregare personalizată")
    group_col = st.selectbox(
        "Alege coloana de grupare:",
        ["Region", "CustomerSegment", "EducationLevel", "LastContactMonth"]
    )

    custom_summary = filtered_df.groupby(group_col).agg(
        VenitMediu=("AnnualIncome", "mean"),
        SoldMediu=("AccountBalance", "mean"),
        ScorMarketingMediu=("MarketingScore", "mean"),
        RataSubscriere=(TARGET, "mean")
    ).reset_index()

    custom_summary["RataSubscriere"] = (custom_summary["RataSubscriere"] * 100).round(2)
    st.dataframe(custom_summary, use_container_width=True)

# 5. CLUSTERIZARE KMEANS
elif section == "5. Clusterizare KMeans":
    st.header("5. Clusterizare KMeans")

    st.write("Clusterizarea grupează clienții în segmente similare.")

    cluster_features = [
        "Age",
        "AnnualIncome",
        "AccountBalance",
        "CreditScore",
        "MarketingScore",
        "ResponsePropensity"
    ]

    X_cluster = filtered_df[cluster_features].copy()
    X_cluster = fill_missing_values(X_cluster)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Metoda Elbow
    inertias = []
    k_values = range(2, 9)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    elbow_df = pd.DataFrame({"k": list(k_values), "Inertia": inertias})

    fig_elbow = px.line(
        elbow_df,
        x="k",
        y="Inertia",
        markers=True,
        title="Metoda Elbow"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    k = st.slider("Alege numărul de clustere:", 2, 8, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    cluster_df = filtered_df.copy()
    cluster_df["Cluster"] = clusters

    st.subheader("Numărul de clienți în fiecare cluster")
    cluster_sizes = cluster_df["Cluster"].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["Cluster", "Număr clienți"]
    st.dataframe(cluster_sizes, use_container_width=True)

    fig_cluster_size = px.bar(
        cluster_sizes,
        x="Cluster",
        y="Număr clienți",
        title="Distribuția clienților pe clustere"
    )
    st.plotly_chart(fig_cluster_size, use_container_width=True)

    st.subheader("Profilul mediu al clusterelor")
    cluster_profile = cluster_df.groupby("Cluster")[cluster_features + [TARGET]].mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

    fig_scatter = px.scatter(
        cluster_df,
        x="AnnualIncome",
        y="AccountBalance",
        color=cluster_df["Cluster"].astype(str),
        title="Vizualizarea clusterelor",
        hover_data=["Age", "CreditScore", "MarketingScore", TARGET]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# 6. REGRESIE LOGISTICA
elif section == "6. Regresie logistică":
    st.header("6. Regresie logistică")

    st.write(
        """
        Modelul estimează dacă un client va subscrie sau nu la depozitul la termen.
        """
    )

    model_features = [
        "Age",
        "AnnualIncome",
        "AccountBalance",
        "CreditScore",
        "MarketingScore",
        "ResponsePropensity"
    ]

    model_df = filtered_df[model_features + [TARGET]].copy()
    model_df = fill_missing_values(model_df)

    # Tratare outlieri
    model_df = cap_outliers_iqr(model_df, model_features)

    X = model_df[model_features]
    y = model_df[TARGET]

    # Scalare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Împărțire train-test
    test_size = st.slider("Procent pentru setul de test:", 0.10, 0.40, 0.20, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("Rezultatele modelului")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Număr variabile", len(model_features))

    st.subheader("Matrice de confuzie")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicție 0", "Predicție 1"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Raport de clasificare")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Coeficienții modelului")
    coef_df = pd.DataFrame({
        "Variabilă": model_features,
        "Coeficient": model.coef_[0]
    }).sort_values("Coeficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True)

    fig_coef = px.bar(
        coef_df,
        x="Variabilă",
        y="Coeficient",
        title="Importanța variabilelor în regresia logistică"
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Predicție pentru un client nou")
    with st.form("predictie_client"):
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        income = st.number_input("AnnualIncome", min_value=0.0, value=50000.0)
        balance = st.number_input("AccountBalance", value=10000.0)
        credit = st.number_input("CreditScore", min_value=300, max_value=900, value=650)
        marketing = st.number_input("MarketingScore", min_value=0.0, max_value=1.0, value=0.50)
        response = st.number_input("ResponsePropensity", min_value=0.0, max_value=1.0, value=0.50)

        submitted = st.form_submit_button("Calculează predicția")

    if submitted:
        new_client = pd.DataFrame([{
            "Age": age,
            "AnnualIncome": income,
            "AccountBalance": balance,
            "CreditScore": credit,
            "MarketingScore": marketing,
            "ResponsePropensity": response
        }])

        new_client_scaled = scaler.transform(new_client)
        predicted_class = model.predict(new_client_scaled)[0]
        predicted_prob = model.predict_proba(new_client_scaled)[0][1]

        st.success(
            f"Probabilitatea estimată de subscriere este {predicted_prob:.2%}. "
            f"Clasa prezisă este: {int(predicted_class)}."
        )
