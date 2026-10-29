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
# Load entire repository for RAG
# -----------------------------
def load_repo_files():
    docs = []

    allowed_extensions = (
        ".py",
        ".txt",
        ".md"
    )

    ignored_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".streamlit"
    }

    for root, dirs, files in os.walk("."):

        dirs[:] = [
            d for d in dirs
            if d not in ignored_dirs
        ]

        for file in files:

            if file.endswith(allowed_extensions):

                try:
                    path = os.path.join(root, file)

                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                        errors="ignore"
                    ) as f:

                        content = f.read()

                    docs.append(
                        f"FILE: {path}\n\n{content}"
                    )

                except Exception:
                    pass

    return docs

REPO_FILES = load_repo_files()

ALL_REPO_TEXT = "\n\n".join(REPO_FILES)

MAX_TEMPLATE_CHARS = 50000

ALL_REPO_TEXT = ALL_REPO_TEXT[:MAX_TEMPLATE_CHARS]

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
# RAG setup — SAFE & SIMPLE
# -----------------------------
def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """
    Hard-split the template into fixed-size character chunks.
    No overlap, no sliding window, no surprises.
    """
    text = text or ""
    text = text[:30000]  # hard cap the entire template to avoid huge inputs
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

RAG_DOCS = []

for doc in REPO_FILES:
    RAG_DOCS.extend(
        chunk_text(doc, max_chars=1000)
    )

@st.cache_resource
def build_faiss_index(docs: list[str], _client: OpenAI):

    if _client is None or not docs:
        return None, None

    texts = [d.strip() for d in docs if d.strip()]
    all_vectors = []

    try:
        for t in texts:
            safe_t = t[:2000]

            resp = _client.embeddings.create(
                model="text-embedding-3-small",
                input=[safe_t]
            )

            all_vectors.append(resp.data[0].embedding)

    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None, None

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

if _client and chat_query:
    if faiss_index is None:
        st.sidebar.error("RAG index unavailable.")
    else:
        # embed the (short) user query
        q_text = chat_query.strip()[:500]
        q_emb = _client.embeddings.create(
            model="text-embedding-3-small",
            input=[q_text]
        ).data[0].embedding

        q_vec = np.array([q_emb], dtype="float32")
        D, I = faiss_index.search(q_vec, k=5)

        # build a compact context from the top chunks
        retrieved_chunks = [faiss_texts[i] for i in I[0]]
        context = "\n\n---\n\n".join(retrieved_chunks)
        context = context[:6000]  # final safety cap

        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a project assistant for the Sustainable Fertilizer Optimization project. "
                        "Use the retrieved repository context when it is relevant and available. "
                        "If the answer cannot be found in the repository context, use your general knowledge to provide a helpful answer. "
                        "When discussing this specific project, prioritize information found in the repository. "
                        f"\n\nRepository Context:\n{context}"
                    ),
                },
                {"role": "user", "content": chat_query},
            ],
            temperature=0,
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

fert_npk = fert_npk.copy()

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

df_crop_main = build_crop_df(crop)
df_crop_second = build_crop_df(crop2) if crop2 else None

# -----------------------------
# NOTEBOOK STARTS HERE
# -----------------------------
st.title("🌾Sustainable Fertilizer Optimization🌱")

st.header("🌍 Problem & Impact")

problem_impact = st.text_area(
    "Problem & Impact",
    value="""
Fertilizer plays a critical role in global food production, but excessive application can increase costs and create environmental risks.

This project explores global fertilizer trends, crop yield relationships, and optimal fertilizer levels using FAOSTAT data.
""",
    height=150,
    key="problem_impact"
)

st.header("📦 Imports")

imports_text = st.text_area(
    "Imports Notes",
    value="""
Core libraries are used for data manipulation, visualization, and machine learning.

The imports below support data loading, exploratory analysis, regression modeling, optimization, and the AI assistant.
""",
    height=150,
    key="imports_notes"
)

st.code("""
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
""")
# -----------------------------
# EDA SECTION
# -----------------------------
st.header("📊 Data & Exploratory Analysis")

eda_intro = st.text_area(
    "EDA Overview",
    value=f"""
This project uses two global, annual datasets:

• Fertilizer use (kg/ha)

• Crop yield (kg/ha) for {crop} and {crop2}

The fertilizer dataset provides global fertilizer application per year, while the yield dataset provides crop-specific yields.

Exploratory Analysis

The EDA examines:

• long-term fertilizer use trends

• yield trends for {crop} and {crop2}

• the relationship between fertilizer and yield

• where diminishing returns begin as fertilizer increases

This analysis builds the foundation for the regression and polynomial models that follow.
""",
    height=500,
    key="eda_intro"
)

st.subheader("Load Fertilizer Dataset")
fert_load_notes = st.text_area(
    "Dataset Notes",
    value="""
This dataset contains global fertilizer application data from FAOSTAT. I load it here to prepare the data for cleaning, filtering, and merging with crop yield information in later steps.
""",
    height=120,
    key="fert_load_notes"
)
st.code("""
path = kagglehub.dataset_download(
    "raaad3000/faostat-fertilizers-indicators"
)

fert = pd.read_csv(
    os.path.join(
        path,
        "Environment_Fertilizers_E_All_Data_(Normalized)",
        "Environment_Fertilizers_E_All_Data_(Normalized).csv"
    ),
    encoding="latin1"
)
""")
# Global NPK Fertilizer Trends
st.subheader("Global NPK Fertilizer Trends")

st.code("""
fig, ax = plt.subplots(figsize=(12, 6))

fig.patch.set_facecolor("#F2F2F0")
ax.set_facecolor("#FAFAF7")

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

ax.grid(
    True,
    which="both",
    axis="both",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

leg = ax.legend()
leg.get_frame().set_facecolor("#FAFAF7")
leg.get_frame().set_edgecolor("#D9D9D6")

ax.set_ylabel("Nutrient Application Rate (kg/ha)")
ax.set_title(
    "Global NPK Fertilizer Use Over Time (kg/ha)",
    fontweight="bold"
""")
fig, ax = plt.subplots(figsize=(12, 6))

fig.patch.set_facecolor("#F2F2F0")
ax.set_facecolor("#FAFAF7")

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

ax.grid(
    True,
    which="both",
    axis="both",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

leg = ax.legend()
leg.get_frame().set_facecolor("#FAFAF7")
leg.get_frame().set_edgecolor("#D9D9D6")

ax.set_ylabel("Nutrient Application Rate (kg/ha)")
ax.set_title(
    "Global NPK Fertilizer Use Over Time (kg/ha)",
    fontweight="bold"
)
st.pyplot(fig)

npk_interpretation = st.text_area(
    "Interpretation",
    value="""
This chart shows that global fertilizer use has become increasingly nitrogen-heavy over the past 60 years. Nitrogen application rises sharply, while phosphorus and potassium remain relatively stable.
""",
    height=150,
    key="npk_interpretation"
)

# Distribution of Fertilizer Use by Nutrient
st.subheader("Distribution of Fertilizer Use by Nutrient (kg/ha)")

distribution_intro = st.text_area(
    "Distribution Commentary",
    value="""
This boxplot shows how fertilizer application rates vary across countries for nitrogen, phosphorus, and potassium. Nitrogen displays the widest spread and the highest outliers, reflecting large differences in how intensively countries apply it. Phosphorus and potassium have lower, tighter distributions, indicating more consistent use across countries.
""",
    height=220,
    key="distribution_intro"
)

st.code("""
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("#F2F2F0")

ax = plt.gca()
ax.set_facecolor("#FAFAF7")

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
    legend=False,
    ax=ax
)

ax.grid(
    True,
    which="both",
    axis="y",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

ax.set_axisbelow(True)

plt.xlabel("Nutrient Type")
plt.ylabel("Fertilizer Use (kg/ha)")
plt.title(
    "Distribution of NPK Fertilizer Use (kg/ha)",
    fontweight="bold"
)
""")

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("#F2F2F0")

ax = plt.gca()
ax.set_facecolor("#FAFAF7")

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
    legend=False,
    ax=ax
)

ax.grid(
    True,
    which="both",
    axis="y",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

ax.set_axisbelow(True)

plt.xlabel("Nutrient Type")
plt.ylabel("Fertilizer Use (kg/ha)")
plt.title(
    "Distribution of NPK Fertilizer Use (kg/ha)",
    fontweight="bold"
)

st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("#F2F2F0")

ax = plt.gca()
ax.set_facecolor("#FAFAF7")

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
    legend=False,
    ax=ax
)

ax.grid(
    True,
    which="both",
    axis="y",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

ax.set_axisbelow(True)

plt.xlabel("Nutrient Type")
plt.ylabel("Fertilizer Use (kg/ha)")
plt.title(
    "Distribution of NPK Fertilizer Use (kg/ha)",
    fontweight="bold"
)

st.pyplot(fig)

distribution_interpretation = st.text_area(
    "Interpretation",
    value="""
This boxplot shows how fertilizer application rates vary across countries for nitrogen, phosphorus, and potassium. Nitrogen displays the widest spread and the highest outliers, reflecting large differences in how intensively countries apply it. Phosphorus and potassium have lower, tighter distributions, indicating more consistent use across countries.
""",
    height=220,
    key="distribution_interpretation"
)

# Top 10 Countries by Nitrogen Use (kg/ha)
st.subheader("Top 10 Countries by Nitrogen Use (kg/ha)")

st.code("""
top_n = (
    fert_npk_clean[fert_npk_clean["Nutrient"] == "Nitrogen (N)"]
    .groupby("Area")["Value"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

top_n_df = top_n.reset_index()

sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(12, 6))

fig.patch.set_facecolor("#DDEEE3")

ax.set_facecolor("white")

sns.barplot(
    data=top_n_df,
    x="Value",
    y="Area",
    color="#2E7D32",
    ax=ax
)

ax.grid(
    True,
    which="both",
    axis="x",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

ax.set_axisbelow(True)

for i, v in enumerate(top_n_df["Value"]):
    ax.text(v + 5, i, f"{v:.0f}", va="center")

plt.title(
    "Top 10 Countries by Nitrogen Use (kg/ha)",
    fontweight="bold"
)

plt.xlabel("Nitrogen Use (kg/ha)")
plt.ylabel("Country")

plt.xticks(
    range(
        0,
        int(top_n_df["Value"].max()) + 50,
        50
    )
)

plt.tight_layout()
""")
top_n = (
    fert_npk_clean[fert_npk_clean["Nutrient"] == "Nitrogen (N)"]
    .groupby("Area")["Value"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

top_n_df = top_n.reset_index()

sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(12, 6))

fig.patch.set_facecolor("#DDEEE3")

ax.set_facecolor("white")

sns.barplot(
    data=top_n_df,
    x="Value",
    y="Area",
    color="#2E7D32",
    ax=ax
)

ax.grid(
    True,
    which="both",
    axis="x",
    color="#D9D9D6",
    linewidth=0.6,
    alpha=0.6
)

ax.set_axisbelow(True)

for i, v in enumerate(top_n_df["Value"]):
    ax.text(v + 5, i, f"{v:.0f}", va="center")

plt.title(
    "Top 10 Countries by Nitrogen Use (kg/ha)",
    fontweight="bold"
)

plt.xlabel("Nitrogen Use (kg/ha)")
plt.ylabel("Country")

plt.xticks(
    range(
        0,
        int(top_n_df["Value"].max()) + 50,
        50
    )
)

plt.tight_layout()

st.pyplot(fig)

top_n_interpretation = st.text_area(
    "Interpretation",
    value="""
The Netherlands appears at the top of the ranking due to its extremely intensive agricultural system. Despite its small land area, it is one of the world's largest agricultural exporters, relying on high-yield greenhouse production, dense livestock operations, and nutrient-intensive farming practices. These factors drive nitrogen application far above global averages.
""",
    height=250,
    key="top_n_interpretation"
)

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
st.subheader("Global Fertilizer Use Over Time (kg/ha)")

st.code("""
df_fert = fert.groupby("Year")["Value"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))

fig.patch.set_facecolor("#DDEEE3")
ax.set_facecolor("white")

ax.plot(
    df_fert["Year"],
    df_fert["Value"],
    color="#4A7C59",
    linewidth=2.8,
    marker="o"
)

ax.set_title(
    "Global Fertilizer Use Over Time",
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("Fertilizer Use (kg/ha)")

ax.grid(True)
""")

df_fert = fert.groupby("Year")["Value"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))

fig.patch.set_facecolor("#DDEEE3")
ax.set_facecolor("white")

ax.plot(
    df_fert["Year"],
    df_fert["Value"],
    color="#4A7C59",
    linewidth=2.8,
    marker="o"
)

ax.set_title(
    "Global Fertilizer Use Over Time",
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("Fertilizer Use (kg/ha)")

ax.grid(True)

st.pyplot(fig)

fert_use_interpretation = st.text_area(
    "Interpretation",
    value="""
This line chart shows the long-term rise in global fertilizer application rates. Fertilizer use has increased steadily over the past several decades as global food production expanded, driven by population growth, intensification of agriculture, and the adoption of high-yield farming practices. This upward trend provides important context for understanding modern nutrient demand and environmental pressures.
""",
    height=250,
    key="fert_use_interpretation"
)

st.subheader("📜 Historical Context: Why Global Fertilizer Use Dropped (1990–2010)")
historical_context = st.text_area(
    "Historical Context",
    value="""
Global fertilizer use does not move randomly, it responds to major geopolitical, economic, and agricultural shifts. The decline from 1990 to 2010 aligns with several real-world events that disrupted fertilizer production, trade, and demand across multiple continents.

1. Collapse of the Soviet Union (1991)

The Soviet Union was one of the world's largest fertilizer producers and consumers. When it dissolved:

- fertilizer factories shut down
- subsidies disappeared
- millions of hectares of farmland were abandoned
- agricultural output collapsed across Russia, Ukraine, Kazakhstan, and Eastern Europe

This event alone caused a massive global drop in fertilizer use starting in the early 1990s.

2. Early 1990s Global Recession

Economic downturns reduced farm incomes worldwide. Farmers cut back on fertilizer inputs.

3. Agricultural Restructuring in the 1990s

- many countries shifted toward lower-input farming
- environmental regulations tightened
- fertilizer efficiency improved
- some regions intentionally reduced nitrogen use

These structural changes slowed fertilizer demand even as economies recovered.

4. Asian Financial Crisis (1997–1998)

Major agricultural economies experienced severe economic contraction. Fertilizer imports dropped sharply.

5. Global Financial Crisis (2008–2010)

- fertilizer prices spiked
- credit tightened
- farmers delayed or reduced fertilizer purchases
- global demand fell rapidly

This created the steepest short-term decline in the dataset.

6. Post-2010 Recovery

- fertilizer production increased
- agricultural investment grew
- demand rebounded

This explains the upward trend after 2010.
""",
    height=900,
    key="historical_context"
)


# -----------------------------
# CROP MODELING
# -----------------------------
st.header(f"🌾 Modeling: {crop} Yield vs Fertilizer")

st.subheader("Load Crop Yield Dataset")

st.code("""
crop_yield = kagglehub.dataset_download(
    "vijayveersingh/faostat-crops-and-livestock-data"
)
""")
crop_dataset_notes = st.text_area(
    "Dataset Explanation",
    value="""
This dataset provides global crop yield information from FAOSTAT. I load it here to prepare the data for cleaning, filtering, and merging with fertilizer application data in the next steps. This forms the foundation for the crop-specific modeling and analysis that follows.
""",
    height=180,
    key="crop_dataset_notes"
)

st.subheader("Load Raw Crop Yield Data")

st.code("""
df_yield = pd.read_csv(
    crop_yield +
    "/Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
)

df_yield
""")
raw_yield_notes = st.text_area(
    "Dataset Explanation",
    value="""
I load the full FAOSTAT crop and livestock dataset here so I can filter it down to crop yields only and prepare it for cleaning, merging, and analysis in the next steps.
""",
    height=150,
    key="raw_yield_notes"
)

st.subheader("Reshape and Filter Crop Yield Data")
reshape_notes = st.text_area(
    "Data Preparation Explanation",
    value="""
I reshape the FAOSTAT dataset from wide to long format so each row represents a single crop-year observation. After converting the year column, I filter the dataset to include only yield values for all usable crops. This prepares the data so users can select any crop they want to analyze and supports global aggregation for visualization and modeling in later steps.
""",
    height=250,
    key="reshape_notes"
)
st.code("""
df_long = df_yield.melt(
    id_vars=[
        "Area Code",
        "Area Code (M49)",
        "Area",
        "Item Code",
        "Item Code (CPC)",
        "Item",
        "Element Code",
        "Element",
        "Unit"
    ],
    var_name="Year",
    value_name="Value"
)

# Convert Y1961 -> 1961
df_long["Year"] = (
    df_long["Year"]
    .str.replace("Y", "")
    .astype(int)
)

crop_yield_only = (
    df_long[df_long["Element"] == "Yield"]
)

global_yield = (
    crop_yield_only
    .groupby(["Year", "Item"])["Value"]
    .sum()
    .reset_index()
)

df_long
""")

st.subheader("Choose Your Crop")
crop_selection_notes = st.text_area(
    "Crop Selection",
    value="""
The FAOSTAT dataset contains hundreds of available crop types. Users can select any crop included in the dataset for analysis. The selected crop will be used throughout the remainder of the project for visualization, modeling, and optimization.
""",
    height=220,
    key="crop_selection_notes"
)
st.code("""
sorted(crop_yield_only["Item"].unique())

# ↑ Uncomment this line to preview all available crops in the dataset

# ✏️ Type the crop you want to analyze
crop = "Wheat"

# Filter the dataset to the selected crop
selected_crop = crop_yield_only[
    crop_yield_only["Item"] == crop
]

selected_crop.head()
""")

# -----------------------------
# SECTION 3: CROP MODELING
# -----------------------------

st.header(f"🌾 Section 3: {crop} Modeling")
crop_modeling_intro = st.text_area(
    "Section Introduction",
    value=f"""
    In this section, I build three models to understand how fertilizer application influences {crop} yield. I begin with a simple linear regression to establish the baseline relationship between fertilizer use
    and yield. I then extend the model using polynomial features to capture curvature in the yield response. Finally, I identify the fertilizer level that maximizes predicted yield, providing a practical
    recommendation for optimal nutrient application.
    """,
    height=300,
    key="crop_modeling_intro"
)
st.subheader(f"Prepare {crop} Dataset for Regression")
crop_prep_notes = st.text_area(
    "Dataset Preparation",
    value=f"""
    To prepare the {crop} modeling dataset, I merge global {crop} yield data with global fertilizer application data. This creates a single dataframe where each row represents a year, the corresponding {crop}
    yield, and the fertilizer applied that year. This merged dataset serves as the foundation for all three {crop} models that follow.
    """,
    height=250,
    key="crop_prep_notes"
)
st.code("""
# Filter to the user-selected crop
selected_crop_df = crop_yield_only[
    crop_yield_only["Item"] == crop
]

# Global yield for the selected crop
global_crop = (
    selected_crop_df.groupby("Year")["Value"]
    .mean()
    .reset_index()
)

# Add crop name column
global_crop["Crop"] = crop

# Global fertilizer
global_fert = (
    fert.groupby("Year")["Value"]
    .mean()
    .reset_index()
)

# Merge yield + fertilizer
df_crop = pd.merge(
    global_crop,
    global_fert,
    on="Year",
    how="inner",
    suffixes=("_yield", "_fert")
)

df_crop.head()
""")

if df_crop_main is None:
    st.error(f"Could not find crop `{crop}` in FAOSTAT yield data.")
else:
    # -----------------------------
    # Yield Over Time
    # -----------------------------
    st.subheader(f"Visualize {crop} Yield Over Time")
    st.code(f'''
    plt.figure(figsize=(10,5))
    
    # Base plot (neutral default)
    plt.plot(df_crop["Year"], df_crop["Value_yield"], linewidth=2.5)
    
    # OPTIONAL: Customize colors for this plot
    # line_color = "#B58A4A"
    # background_color = "#EDE7D4"
    # grid_color = "#CCCCCC"

    plt.title("Global {crop} Yield Over Time", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.grid(True)
    
    plt.show()
    ''')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(
        df_crop_main["Year"],
        df_crop_main["Value_yield"],
        linewidth=2.5
    )
    
    ax.set_title(
        f"Global {crop} Yield Over Time",
        fontweight="bold"
    )
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (kg/ha)")

    ax.grid(True)

    st.pyplot(fig)    

    yield_over_time_interpretation = st.text_area(
        "Chart Explanation",
        value=f"""
        Describe the yield trend for {crop} over time. Discuss periods of growth, stability, decline, and any notable patterns that may help explain how fertilizer use and agricultural practices influenced
        crop performance.
        """,
        height=220,
        key="yield_over_time_interpretation"
    )
    
    st.subheader(f"Visualize Fertilizer vs {crop} Yield")
    st.code("""
    plt.figure(figsize=(10,6))
    
    plt.scatter(
        df_crop["Value_fert"],
        df_crop["Value_yield"]
    )
    
    plt.show()
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.patch.set_facecolor("#DDEEE3")
    
    ax.set_facecolor("white")
    
    plt.scatter(
        df_crop_main["Value_fert"],
        df_crop_main["Value_yield"],
        color="#B58A4A",
        alpha=0.85,
        s=85,
        edgecolor="black",
        linewidth=0.5
    )
    
    x = np.linspace(
        df_crop_main["Value_fert"].min(),
        df_crop_main["Value_fert"].max(),
        100
    )
    
    m, b = np.polyfit(
        df_crop_main["Value_fert"],
        df_crop_main["Value_yield"],
        1
    )
    
    plt.plot(
        x,
        m * x + b,
        color="#7C6A4A",
        linewidth=2.2,
        alpha=0.8
    )
    
    for spine in ax.spines.values():
        spine.set_color("#A8C3B0")
        spine.set_linewidth(1.2)
    
    plt.title(
        f"Global {crop} Yield vs Fertilizer Use",
        fontweight="bold"
    )
    
    plt.xlabel("Fertilizer Use (kg/ha)")
    plt.ylabel("Yield (kg/ha)")
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    yield_vs_fert_interpretation = st.text_area(
        "Chart Explanation",
        value=f"""
        Explain the relationship between fertilizer use and {crop} yield. Discuss whether higher fertilizer use generally corresponds with higher yields and identify any signs of diminishing returns or
        variability in the data.
        """,
        height=250,
        key="yield_vs_fert_interpretation"
    )
    
    # Correlation
    st.subheader(
        f"Correlation Between Fertilizer Use and {crop} Yield"
    )
    st.code("""
    df_crop[
        ["Value_fert", "Value_yield"]
    ].corr()
    """)
    correlation_matrix = (
        df_crop_main[
            ["Value_fert", "Value_yield"]
        ].corr()
    )
    
    st.dataframe(correlation_matrix)
    
    correlation_explanation = st.text_area(
        "Correlation Interpretation",
        value=f"""
        Interpret the correlation coefficient between fertilizer use and {crop} yield. Discuss whether the relationship is strong, weak, positive, or negative and what that suggests about yield response to
        fertilizer application.
        """,
        height=220,
        key="correlation_explanation"
    )
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

st.header("🤖 About the AI Assistant")

st.markdown("""
This application includes a Retrieval-Augmented Generation (RAG) assistant that can answer questions about the project documentation, notes, source code, and supporting materials included in the repository.
""")
