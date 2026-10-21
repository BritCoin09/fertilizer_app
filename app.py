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
# RAG corpus for FAISS
# -----------------------------
PROJECT_DOCS = [
    """
    Sustainable Fertilizer Optimization project by Britny Chambers.
    It uses FAOSTAT fertilizer and crop yield data to model how fertilizer use (kg/ha)
    affects yield (kg/ha) for crops like Wheat and Maize. It fits linear and polynomial
    regression models and finds an optimal fertilizer level where yield is high but
    environmental risk and waste are minimized.
    """,
    """
    The project includes global NPK fertilizer trends, distribution plots, and correlation
    analysis between fertilizer use and yield. It highlights nitrogen-heavy application
    patterns and the risk of over-fertilization for soil health and water quality.
    """,
    """
    Model 1: Linear regression between fertilizer use and yield.
    Model 2: Polynomial regression (degree 2) to capture diminishing returns.
    Model 3: Uses the quadratic coefficients to compute the optimal fertilizer level
    at -b/(2a), then predicts yield at that point.
    """,
    """
    Second crop analysis compares how another crop (e.g., Maize) responds to fertilizer
    versus the main crop. The summary section explains how farmers can use these insights
    to avoid over-application and focus on sustainable fertilizer strategies.
    """
]

@st.cache_resource
def build_faiss_index(docs, _client):
    if _client is None:
        return None, None

    texts = [d.strip() for d in docs]
    resp = _client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vectors = np.array([e.embedding for e in resp.data], dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, texts

# -----------------------------
# Sidebar: navigation + inputs
# -----------------------------
st.sidebar.title("🌱 Navigation")

step = st.sidebar.radio(
    "Go to section:",
    [
        "1. Project Overview",
        "2. Data & EDA",
        "3. Crop Modeling (Main Crop)",
        "4. Optimal Fertilizer Level",
        "5. Second Crop Analysis",
        "6. Summary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("✏️ Crop Selection")

crop = st.sidebar.text_input(
    "Main crop name (exact FAOSTAT label)",
    value="Wheat"
)

crop2 = st.sidebar.text_input(
    "Second crop name (optional)",
    value="Maize"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password").strip()

# -----------------------------
# Chatbot (simple RAG-style)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Fertilizer Assistant")

chat_query = st.sidebar.text_input("Ask a question:")
# Initialize OpenAI client using the user's key ONLY
_client = None
if api_key:
    try:
        _client = OpenAI(api_key=api_key.strip())
    except Exception:
        st.sidebar.error("Your API key could not be initialized. Double‑check it.")
        _client = None

# Build FAISS index once (cached)
faiss_index, faiss_texts = build_faiss_index(PROJECT_DOCS, _client)


# Handle chatbot query with RAG
if _client and chat_query:
    if faiss_index is None:
        st.sidebar.error("RAG index is not available. Check your API key.")
    else:
        try:
            # Embed the question
            q_emb = _client.embeddings.create(
                model="text-embedding-3-small",
                input=[chat_query]
            )
            q_vec = np.array([q_emb.data[0].embedding], dtype="float32")

            # Search FAISS
            D, I = faiss_index.search(q_vec, k=3)
            context_chunks = [faiss_texts[i] for i in I[0]]

            context = "\n\n".join(context_chunks)

            resp = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant for Britny Chambers' "
                            "Sustainable Fertilizer Optimization project. "
                            "Use the following project context when answering:\n\n"
                            + context
                        ),
                    },
                    {"role": "user", "content": chat_query},
                ],
                temperature=0
            )
            st.sidebar.write(resp.choices[0].message.content.strip())
        except Exception:
            st.sidebar.error("Authentication or RAG retrieval failed. Check your API key.")
elif chat_query:
    st.sidebar.info("Enter your OpenAI API key to enable the assistant.")



# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data
def load_fertilizer_data():
    path = kagglehub.dataset_download("raaad3000/faostat-fertilizers-indicators")
    fert = pd.read_csv(
        os.path.join(
            path,
            "Environment_Fertilizers_E_All_Data_(Normalized)",
            "Environment_Fertilizers_E_All_Data_(Normalized).csv"
        ),
        encoding="latin1"
    )
    return fert

@st.cache_data
def load_crop_yield_data():
    crop_yield = kagglehub.dataset_download(
        "vijayveersingh/faostat-crops-and-livestock-data"
    )
    df_yield = pd.read_csv(
        crop_yield + "/Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
    )
    return df_yield

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
# Helper: build main crop dataset
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
# Section 1: Project Overview
# -----------------------------
if step == "1. Project Overview":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🌍 Sustainable Fertilizer Optimization")

        st.markdown("""
This project analyzes how fertilizer affects **crop yields** to find the optimal application point:
the level that delivers strong crop performance **without wasting money or causing nutrient pollution**.

Using linear and polynomial models, the analysis:

- measures how yield responds to fertilizer
- identifies where diminishing returns begin
- pinpoints the “sweet spot” where productivity stays high and environmental risk stays low
        """)

        st.markdown(f"**Current main crop:** `{crop}`")
        if crop2:
            st.markdown(f"**Second crop:** `{crop2}`")

        with col2:
            st.markdown("### 🤖 Assistant")
            st.markdown("Ask questions about fertilizer, crops, or the analysis in the sidebar.")

            if chat_query and not _client:
                st.info("Add your OpenAI API key in the sidebar to enable the assistant.")

# Section 2: Data & EDA
# -----------------------------
elif step == "2. Data & EDA":
    st.header("📊 Data & Exploratory Analysis")

    col1, col2 = st.columns(2)

    with col1:
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

        ax.grid(True, which="both", axis="both",
                color="#D9D9D6", linewidth=0.6, alpha=0.6)
        fig.patch.set_facecolor("#F2F2F0")
        ax.set_facecolor("#FAFAF7")

        leg = plt.legend()
        leg.get_frame().set_facecolor("#FAFAF7")
        leg.get_frame().set_edgecolor("#D9D9D6")

        plt.ylabel("Nutrient Application Rate (kg/ha)")
        plt.title("Global NPK Fertilizer Use Over Time (kg/ha)", fontweight="bold")
        st.pyplot(fig)

        st.caption(
            "Global fertilizer use has become increasingly nitrogen‑heavy over the past 60 years, "
            "raising long‑term concerns for soil health and environmental impact."
        )

    with col2:
        st.subheader("Distribution of NPK Fertilizer Use")

        fig2 = plt.figure(figsize=(10, 6))
        fig2.patch.set_facecolor("#F2F2F0")
        ax2 = plt.gca()
        ax2.set_facecolor("#FAFAF7")

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
            ax=ax2
        )

        ax2.grid(True, which="both", axis="y",
                 color="#D9D9D6", linewidth=0.6, alpha=0.6)
        ax2.set_axisbelow(True)

        plt.xlabel("Nutrient Type")
        plt.ylabel("Fertilizer Use (kg/ha)")
        plt.title("Distribution of NPK Fertilizer Use (kg/ha)", fontweight="bold")

        st.pyplot(fig2)

        st.caption(
            "Nitrogen shows the widest spread and highest outliers, reflecting large differences "
            "in how intensively countries apply it."
        )

    st.markdown("---")

    st.subheader("Global Fertilizer Use Over Time")

    fig3 = plt.figure(figsize=(10, 5))
    plt.gcf().patch.set_facecolor("#DDEEE3")

    plt.plot(
        df_fert_global["Year"],
        df_fert_global["Value"],
        color="#4A7C59",
        linewidth=2.8,
        marker="o"
    )

    plt.title("Global Fertilizer Use Over Time", fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Fertilizer Use (kg/ha)")
    plt.grid(True)

    st.pyplot(fig3)

    st.caption(
        "Fertilizer use has increased steadily as global food production expanded, "
        "with notable dips during major economic crises."
    )

# -----------------------------
# Section 3: Crop Modeling (Main Crop)
# -----------------------------
elif step == "3. Crop Modeling (Main Crop)":
    st.header(f"🌾 Modeling: {crop} Yield vs Fertilizer")

    if df_crop_main is None:
        st.error(f"Could not find crop `{crop}` in FAOSTAT yield data. Try another name.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Global {crop} Yield Over Time")

            fig = plt.figure(figsize=(10, 5))
            plt.plot(df_crop_main["Year"], df_crop_main["Value_yield"], linewidth=2.5)
            plt.title(f"Global {crop} Yield Over Time", fontweight="bold")
            plt.xlabel("Year")
            plt.ylabel("Yield (kg/ha)")
            plt.grid(True)

            st.pyplot(fig)

        with col2:
            st.subheader(f"{crop} Yield vs Fertilizer Use")

            fig2 = plt.figure(figsize=(10, 6))
            plt.gcf().patch.set_facecolor("#DDEEE3")
            ax = plt.gca()
            ax.set_facecolor("white")

            plt.scatter(
                df_crop_main["Value_fert"],
                df_crop_main["Value_yield"],
                color="#B58A4A",
                alpha=0.85,
                s=85,
                edgecolor="black",
                linewidth=0.5,
                label=f"{crop} Data Points"
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
                alpha=0.8,
                label="Trendline"
            )

            plt.legend(
                frameon=True,
                facecolor="white",
                edgecolor="#A8C3B0"
            )

            for spine in ax.spines.values():
                spine.set_color("#A8C3B0")
                spine.set_linewidth(1.2)

            plt.title(f"Global {crop} Yield vs Fertilizer Use", fontweight="bold")
            plt.xlabel("Fertilizer Use (kg/ha)")
            plt.ylabel("Yield (kg/ha)")
            plt.grid(True, alpha=0.3)

            st.pyplot(fig2)

        st.markdown("---")

        st.subheader("Correlation Between Fertilizer Use and Yield")

        corr = df_crop_main[["Value_fert", "Value_yield"]].corr()
        st.dataframe(corr)

        st.caption(
            "This correlation summarizes how strongly fertilizer use and yield move together "
            "for the selected crop."
        )

        st.markdown("---")

        st.subheader("Model 1: Linear Regression")

        X = df_crop_main[["Value_fert"]]
        y = df_crop_main["Value_yield"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.markdown(f"- **Slope:** {model.coef_[0]:.2f}")
        st.markdown(f"- **Intercept:** {model.intercept_:.2f}")
        st.markdown(f"- **RMSE:** {rmse:.2f}")
        st.markdown(f"- **R²:** {r2:.3f}")

        fig3 = plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor("#DDEEE3")
        ax3 = plt.gca()
        ax3.set_facecolor("white")

        plt.scatter(
            y_test,
            y_pred,
            color="#B58A4A",
            edgecolor="black",
            s=85
        )

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

        plt.legend(
            ["Perfect Prediction Line", "Model Predictions"],
            frameon=True,
            facecolor="white",
            edgecolor="#A8C3B0"
        )

        st.pyplot(fig3)

# -----------------------------
# Section 4: Optimal Fertilizer Level (Model 2 + 3)
# -----------------------------
elif step == "4. Optimal Fertilizer Level":
    st.header(f"🎯 Optimal Fertilizer Level for {crop}")

    if df_crop_main is None:
        st.error(f"Could not find crop `{crop}` in FAOSTAT yield data. Try another name.")
    else:
        X = df_crop_main[["Value_fert"]]
        y = df_crop_main["Value_yield"]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )

        model2 = LinearRegression()
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)

        rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
        r2_2 = r2_score(y_test, y_pred2)

        st.subheader("Model 2: Polynomial Regression (Degree 2)")
        st.markdown(f"- **Coefficients:** {model2.coef_}")
        st.markdown(f"- **Intercept:** {model2.intercept_:.2f}")
        st.markdown(f"- **RMSE:** {rmse2:.2f}")
        st.markdown(f"- **R²:** {r2_2:.3f}")

        fig1 = plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor("#DDEEE3")
        ax1 = plt.gca()
        ax1.set_facecolor("white")

        plt.scatter(
            y_test,
            y_pred2,
            color="#B58A4A",
            edgecolor="black",
            s=85
        )

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="#7C6A4A",
            linewidth=2.2
        )

        plt.title(f"Model 2: Actual vs Predicted {crop} Yield", fontweight="bold")
        plt.xlabel("Actual Yield (kg/ha)")
        plt.ylabel("Predicted Yield (kg/ha)")
        plt.grid(True, alpha=0.3)

        plt.legend(
            ["Perfect Prediction Line", "Model Predictions"],
            frameon=True,
            facecolor="white",
            edgecolor="#A8C3B0"
        )

        st.pyplot(fig1)

        st.markdown("---")

        st.subheader("Model 3: Optimal Fertilizer Level")

        x_range = np.linspace(
            df_crop_main["Value_fert"].min(),
            df_crop_main["Value_fert"].max(),
            200
        )

        a = model2.coef_[1]   # quadratic term
        b = model2.coef_[0]   # linear term
        x_opt = -b / (2 * a)

        x_opt_df = pd.DataFrame({"Value_fert": [x_opt]})
        x_opt_poly = poly.transform(x_opt_df)
        y_opt = model2.predict(x_opt_poly)[0]

        st.markdown(f"**Optimal Fertilizer (kg/ha):** `{x_opt:.2f}`")
        st.markdown(f"**Predicted Yield at Optimum (kg/ha):** `{y_opt:.2f}`")

        x_range_df = pd.DataFrame({"Value_fert": x_range})
        X_curve = poly.transform(x_range_df)
        y_curve = model2.predict(X_curve)

        fig2 = plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor("#DDEEE3")
        ax2 = plt.gca()
        ax2.set_facecolor("white")

        plt.plot(
            x_range,
            y_curve,
            color="#7C6A4A",
            linewidth=2.5,
            label="Polynomial Curve"
        )

        plt.scatter(
            df_crop_main["Value_fert"],
            df_crop_main["Value_yield"],
            color="#B58A4A",
            edgecolor="black",
            s=85,
            alpha=0.8,
            label="Data Points"
        )

        plt.scatter(
            x_opt,
            y_opt,
            color="#4A7C59",
            s=120,
            edgecolor="black",
            zorder=5,
            label=f"Optimal Point ({x_opt:.1f} kg/ha)"
        )

        plt.title(f"Model 3: Optimal Fertilizer Level for Maximum {crop} Yield", fontweight="bold")
        plt.xlabel("Fertilizer Use (kg/ha)")
        plt.ylabel("Predicted Yield (kg/ha)")
        plt.grid(True, alpha=0.3)

        plt.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#A8C3B0"
        )

        st.pyplot(fig2)

# -----------------------------
# Section 5: Second Crop Analysis
# -----------------------------
elif step == "5. Second Crop Analysis":
    st.header(f"🌾 Second Crop Analysis: {crop2}")

    if not crop2:
        st.info("Enter a second crop name in the sidebar to enable this section.")
    elif df_crop_second is None:
        st.error(f"Could not find crop `{crop2}` in FAOSTAT yield data. Try another name.")
    else:
        st.subheader(f"Global {crop2} Yield Over Time")

        fig = plt.figure(figsize=(10, 5))
        plt.plot(
            df_crop_second["Year"],
            df_crop_second["Value_yield"],
            color="#d5b37e",
            linewidth=2.5
        )

        plt.gcf().patch.set_facecolor("#D5B37E")
        plt.title(f"Global {crop2} Yield Over Time", fontweight="bold")
        plt.xlabel("Year")
        plt.ylabel("Yield (kg/ha)")
        plt.grid(True)

        st.pyplot(fig)

        st.markdown("---")

        st.subheader(f"{crop2} Yield vs Fertilizer Use")

        fig2 = plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor("#DDEEE3")
        ax = plt.gca()
        ax.set_facecolor("white")

        plt.scatter(
            df_crop_second["Value_fert"],
            df_crop_second["Value_yield"],
            color="#B58A4A",
            alpha=0.85,
            s=85,
            edgecolor="black",
            linewidth=0.5,
            label=f"{crop2} Data Points"
        )

        x = np.linspace(
            df_crop_second["Value_fert"].min(),
            df_crop_second["Value_fert"].max(),
            100
        )
        m, b = np.polyfit(
            df_crop_second["Value_fert"],
            df_crop_second["Value_yield"],
            1
        )

        plt.plot(
            x,
            m * x + b,
            color="#7C6A4A",
            linewidth=2.2,
            alpha=0.8,
            label="Trendline"
        )

        plt.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#A8C3B0"
        )

        for spine in ax.spines.values():
            spine.set_color("#A8C3B0")
            spine.set_linewidth(1.2)

        plt.title(f"Global {crop2} Yield vs Fertilizer Use", fontweight="bold")
        plt.xlabel("Fertilizer Use (kg/ha)")
        plt.ylabel("Yield (kg/ha)")
        plt.grid(True, alpha=0.3)

        st.pyplot(fig2)

# -----------------------------
# Section 6: Summary
# -----------------------------
elif step == "6. Summary":
    st.header("📜 Summary & Export")

    st.markdown("""
This analysis:

- links global fertilizer use to crop yield
- shows how nitrogen-heavy application patterns have emerged
- quantifies the fertilizer–yield relationship for your selected crops
- identifies an optimal fertilizer level where yield is maximized before diminishing returns
    """)

    st.markdown("You can copy this summary into your report or portfolio.")

    # Optional: add a download button for a text summary
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



