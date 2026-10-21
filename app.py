import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import kagglehub
import faiss
from openai import OpenAI
import pickle

# -----------------------------
# Global styling
# -----------------------------
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("ticks")

st.set_page_config(
    page_title="Sustainable Fertilizer Optimization",
    layout="wide"
)

# -----------------------------
# Load full notebook template
# -----------------------------
def load_template(path="full_template_text.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Template file missing."

TEMPLATE_TEXT = load_template()

# -----------------------------
# Sidebar: crop selection + API key + chatbot
# -----------------------------
st.sidebar.subheader("✏️ Crop Selection")

crop = st.sidebar.text_input("Main crop name", value="Wheat")
crop2 = st.sidebar.text_input("Second crop name (optional)", value="Barley")

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI API Key")

api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password").strip()

_client = None
if api_key:
    try:
        _client = OpenAI(api_key=api_key)
    except Exception:
        st.sidebar.error("Invalid API key.")
        _client = None

# -----------------------------
# RAG setup — SAFE CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=50, overlap=10):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

RAG_DOCS = chunk_text(TEMPLATE_TEXT, chunk_size=50, overlap=10)

@st.cache_resource
def build_faiss_index(docs, client):
    if client is None:
        return None, None

    texts = [d.strip() for d in docs]
    all_vectors = []

    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_vectors.extend([e.embedding for e in resp.data])

    vectors = np.array(all_vectors, dtype="float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, texts

faiss_index, faiss_texts = build_faiss_index(RAG_DOCS, _client)

# -----------------------------
# Chatbot (RAG)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Fertilizer Assistant")

chat_query = st.sidebar.text_input("Ask a question:", key="chat_input")
st.session_state["chat_input"] = chat_query


if _client and chat_query:
    if faiss_index is None:
        st.sidebar.error("RAG index unavailable.")
    else:
        q_emb = _client.embeddings.create(
            model="text-embedding-3-small",
            input=[chat_query.strip()[:500]]
        ).data[0].embedding


        q_vec = np.array([q_emb], dtype="float32")
        D, I = faiss_index.search(q_vec, k=5)

        context = "\n\n---\n\n".join([faiss_texts[i] for i in I[0]])

        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": chat_query}
            ],
            temperature=0
        )
        st.sidebar.write(resp.choices[0].message.content.strip())

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_fertilizer_data():
    path = kagglehub.dataset_download("raaad3000/faostat-fertilizers-indicators")
    return pd.read_csv(
        os.path.join(
            path,
            "Environment_Fertilizers_E_All_Data_(Normalized)",
            "Environment_Fertilizers_E_All_Data_(Normalized).csv"
        ),
        encoding="latin1"
    )

@st.cache_data
def load_crop_yield_data():
    crop_yield = kagglehub.dataset_download("vijayveersingh/faostat-crops-and-livestock-data")
    return pd.read_csv(crop_yield + "/Production_Crops_Livestock_E_All_Data_NOFLAG.csv")

fert = load_fertilizer_data()
df_yield = load_crop_yield_data()

# -----------------------------
# Prepare fertilizer data
# -----------------------------
fert_crop = fert[fert["Element"] == "Use per area of cropland"]

fert_npk = fert_crop[
    fert_crop["Item"].isin([
        "Nutrient nitrogen N (total)",
        "Nutrient phosphate P2O5 (total)",
        "Nutrient potash K2O (total)"
    ])
]

fert_npk["Nutrient"] = fert_npk["Item"].map({
    "Nutrient nitrogen N (total)": "Nitrogen (N)",
    "Nutrient phosphate P2O5 (total)": "Phosphorus (P₂O₅)",
    "Nutrient potash K2O (total)": "Potassium (K₂O)"
})

fert_npk = fert_npk.dropna(subset=["Value"])

aggregates = [
    "World",
    "Western Europe",
    "Western Asia",
    "Eastern Africa",
    "Least Developed Countries",
    "Low Income Food Deficit Countries",
    "Northern America",
    "Europe",
    "Africa",
    "Asia",
    "Oceania"
]

fert_npk_clean = fert_npk[~fert_npk["Area"].isin(aggregates)]

df_fert_global = fert.groupby("Year")["Value"].mean().reset_index()

# -----------------------------
# Prepare crop yield data
# -----------------------------
df_long = df_yield.melt(
    id_vars=[
        "Area Code", "Area Code (M49)", "Area",
        "Item Code", "Item Code (CPC)", "Item",
        "Element Code", "Element", "Unit"
    ],
    var_name="Year",
    value_name="Value"
)

df_long["Year"] = df_long["Year"].str.replace("Y", "").astype(int)

crop_yield_only = df_long[df_long["Element"] == "Yield"]

global_yield = crop_yield_only.groupby(["Year", "Item"])["Value"].sum().reset_index()

# -----------------------------
# Build crop datasets
# -----------------------------
def build_crop_df(crop_name: str):
    selected_crop_df = crop_yield_only[crop_yield_only["Item"] == crop_name]
    if selected_crop_df.empty:
        return None

    global_crop = (
        selected_crop_df.groupby("Year")["Value"]
        .mean()
        .reset_index()
    )
    global_crop["Crop"] = crop_name

    global_fert = fert.groupby("Year")["Value"].mean().reset_index()

    df_crop = pd.merge(
        global_crop,
        global_fert,
        on="Year",
        how="inner",
        suffixes=("_yield", "_fert")
    )
    return df_crop

df_crop_main = build_crop_df(crop)
df_crop_second = build_crop_df(crop2) if crop2 else None

# -----------------------------
# NOTEBOOK STARTS HERE
# -----------------------------
st.title("🌍 Sustainable Fertilizer Optimization")

st.markdown("### 📘 Full Notebook (Editable)")
st.text_area("Notebook Text", value=TEMPLATE_TEXT, height=1200)

st.markdown("---")

# -----------------------------
# CODE BLOCKS (Colab style)
# -----------------------------
st.markdown("### 🧪 Code Used in This Notebook")

st.code("""
# Example: Melt crop yield data
df_long = df_yield.melt(
    id_vars=[
        "Area Code", "Area Code (M49)", "Area",
        "Item Code", "Item Code (CPC)", "Item",
        "Element Code", "Element", "Unit"
    ],
    var_name="Year",
    value_name="Value"
)
""")

# -----------------------------
# EDA SECTION
# -----------------------------
st.header("📊 Data & Exploratory Analysis")

# Global NPK Fertilizer Trends
st.subheader("Global NPK Fertilizer Trends")

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    data=fert_npk_clean,
    x="Year",
    y="Value",
    hue="Nutrient",
    palette={
        "Nitrogen (N)": "#2E7D32",
        "Phosphorus (P₂O₅)": "#8D6E63",
        "Potassium (K₂O)": "#F9A825"
    },
    ax=ax
)
ax.grid(True)
st.pyplot(fig)

# Distribution of NPK Fertilizer Use
st.subheader("Distribution of NPK Fertilizer Use")

fig2 = plt.figure(figsize=(10, 6))
sns.boxplot(
    data=fert_npk_clean,
    x="Nutrient",
    y="Value",
    hue="Nutrient",
    palette={
        "Nitrogen (N)": "#2E7D32",
        "Phosphorus (P₂O₅)": "#8D6E63",
        "Potassium (K₂O)": "#F9A825"
    },
    legend=False
)
st.pyplot(fig2)

# Global Fertilizer Use Over Time
st.subheader("Global Fertilizer Use Over Time")

fig3 = plt.figure(figsize=(10, 5))
plt.plot(
    df_fert_global["Year"],
    df_fert_global["Value"],
    color="#4A7C59",
    linewidth=2.8,
    marker="o"
)
st.pyplot(fig3)

# -----------------------------
# CROP MODELING
# -----------------------------
st.header(f"🌾 Modeling: {crop} Yield vs Fertilizer")

if df_crop_main is None:
    st.error(f"Could not find crop `{crop}` in FAOSTAT yield data.")
else:
    # Yield over time
    st.subheader(f"Global {crop} Yield Over Time")
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df_crop_main["Year"], df_crop_main["Value_yield"], linewidth=2.5)
    plt.title(f"Global {crop} Yield Over Time", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.grid(True)
    st.pyplot(fig)

    # Yield vs Fertilizer
    st.subheader(f"{crop} Yield vs Fertilizer Use")
    fig2 = plt.figure(figsize=(10, 6))
    plt.scatter(
        df_crop_main["Value_fert"],
        df_crop_main["Value_yield"],
        color="#B58A4A",
        edgecolor="black",
        s=85
    )
    plt.title(f"{crop} Yield vs Fertilizer Use", fontweight="bold")
    plt.xlabel("Fertilizer Use (kg/ha)")
    plt.ylabel("Yield (kg/ha)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Correlation
    st.subheader("Correlation Between Fertilizer Use and Yield")
    st.dataframe(df_crop_main[["Value_fert", "Value_yield"]].corr())

    # Linear Regression
    st.subheader("Model 1: Linear Regression")
    X = df_crop_main[["Value_fert"]]
    y = df_crop_main["Value_yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"- **Slope:** {model.coef_[0]:.2f}")
    st.markdown(f"- **Intercept:** {model.intercept_:.2f}")
    st.markdown(f"- **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.markdown(f"- **R²:** {r2_score(y_test, y_pred):.3f}")

    fig3 = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="#B58A4A", edgecolor="black", s=85)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="#7C6A4A",
        linewidth=2.2
    )
    plt.title(f"Model 1: Actual vs Predicted {crop} Yield", fontweight="bold")
    plt.xlabel("Actual Yield (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig3)

# -----------------------------
# OPTIMAL FERTILIZER
# -----------------------------
st.header(f"🎯 Optimal Fertilizer Level for {crop}")

if df_crop_main is not None:
    X = df_crop_main[["Value_fert"]]
    y = df_crop_main["Value_yield"]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    model2 = LinearRegression()
    model2.fit(X_poly, y)

    a = model2.coef_[1]
    b = model2.coef_[0]
    x_opt = -b / (2 * a)

    x_opt_df = pd.DataFrame({"Value_fert": [x_opt]})
    x_opt_poly = poly.transform(x_opt_df)
    y_opt = model2.predict(x_opt_poly)[0]

    st.markdown(f"**Optimal Fertilizer (kg/ha):** `{x_opt:.2f}`")
    st.markdown(f"**Predicted Yield at Optimum (kg/ha):** `{y_opt:.2f}`")

    # Curve plot
    x_range = np.linspace(
        df_crop_main["Value_fert"].min(),
        df_crop_main["Value_fert"].max(),
        200
    )
    x_range_df = pd.DataFrame({"Value_fert": x_range})
    X_curve = poly.transform(x_range_df)
    y_curve = model2.predict(X_curve)

    fig_opt = plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_curve, color="#7C6A4A", linewidth=2.5)
    plt.scatter(
        df_crop_main["Value_fert"],
        df_crop_main["Value_yield"],
        color="#B58A4A",
        edgecolor="black",
        s=85,
        alpha=0.8
    )
    plt.scatter(
        x_opt,
        y_opt,
        color="#4A7C59",
        s=120,
        edgecolor="black",
        zorder=5
    )
    plt.title(f"Model 3: Optimal Fertilizer Level for Maximum {crop} Yield", fontweight="bold")
    plt.xlabel("Fertilizer Use (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_opt)

# -----------------------------
# SECOND CROP ANALYSIS
# -----------------------------
st.header(f"🌾 Second Crop Analysis: {crop2}")

if df_crop_second is None:
    st.info("Enter a valid second crop name to enable this section.")
else:
    st.subheader(f"Global {crop2} Yield Over Time")

    fig_sc = plt.figure(figsize=(10, 5))
    plt.plot(
        df_crop_second["Year"],
        df_crop_second["Value_yield"],
        color="#d5b37e",
        linewidth=2.5
    )
    plt.title(f"Global {crop2} Yield Over Time", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.grid(True)
    st.pyplot(fig_sc)

    st.subheader(f"{crop2} Yield vs Fertilizer Use")

    fig_sc2 = plt.figure(figsize=(10, 6))
    plt.scatter(
        df_crop_second["Value_fert"],
        df_crop_second["Value_yield"],
        color="#B58A4A",
        edgecolor="black",
        s=85,
        alpha=0.85
    )
    plt.title(f"{crop2} Yield vs Fertilizer Use", fontweight="bold")
    plt.xlabel("Fertilizer Use (kg/ha)")
    plt.ylabel("Yield (kg/ha)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_sc2)

# -----------------------------
# SUMMARY
# -----------------------------
st.header("📜 Summary & Export")

summary_text = f"""
Sustainable Fertilizer Optimization Summary

Main crop: {crop}
Second crop: {crop2}

This project uses FAOSTAT fertilizer and yield data to model how fertilizer application
affects crop performance. Linear and polynomial regression models quantify the relationship
between fertilizer use (kg/ha) and yield (kg/ha), and identify an optimal fertilizer level
that maximizes predicted yield while avoiding unnecessary input use.
"""

st.download_button(
    label="📥 Download summary as .txt",
    data=summary_text,
    file_name="fertilizer_optimization_summary.txt",
    mime="text/plain"
)


