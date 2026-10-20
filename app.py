import streamlit as st
import openai

openai.api_key = "YOUR_KEY_HERE"

st.set_page_config(
    page_title="Sustainable Fertilizer Optimization",
    page_icon="🌱",
    layout="wide"
)

# Layout: left = template, right = chatbot
col_template, col_chat = st.columns([2, 1])

with col_template:
    st.title("🌍 Sustainable Fertilizer Optimization Template")

    st.markdown("Fill this out while you chat with the assistant on the right.")

    crop = st.text_input("Primary crop (e.g., Wheat)", "")
    crop2 = st.text_input("Second crop (optional)", "")

    st.subheader("Problem & Impact")
    problem_text = st.text_area(
        "Describe the problem and environmental impact",
        "This project analyzes how fertilizer affects crop yields..."
    )

    st.subheader("Chart Explanations")
    fert_trend_expl = st.text_area(
        "Global NPK Fertilizer Trends explanation",
        ""
    )
    dist_expl = st.text_area(
        "Distribution of Fertilizer Use explanation",
        ""
    )
    top_n_expl = st.text_area(
        "Top 10 Countries by Nitrogen Use explanation",
        ""
    )
    global_fert_expl = st.text_area(
        "Global Fertilizer Use Over Time explanation",
        ""
    )

    st.subheader(f"{crop or '[Your Crop]'} Modeling Explanations")
    model1_expl = st.text_area("Model 1 (Linear Regression) explanation", "")
    model2_expl = st.text_area("Model 2 (Polynomial Regression) explanation", "")
    model3_expl = st.text_area("Model 3 (Optimal Fertilizer Level) explanation", "")
    crop_summary = st.text_area(f"{crop or '[Your Crop]'} model section summary", "")

    st.subheader(f"{crop2 or '[2nd Crop]'} Section")
    crop2_yield_expl = st.text_area(f"{crop2 or '[2nd Crop]'} yield over time explanation", "")

    st.subheader("Final Summary")
    final_summary = st.text_area("Project summary / conclusions", "")

    st.button("Save (placeholder)", help="You can later wire this to export or save.")

with col_chat:
    st.header("🌱 Fertilizer Intelligence Assistant")

    st.markdown(
        "Ask questions while you fill out the template—about crops, models, fertilizer, or wording."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask a question:")

    if user_q:
        st.session_state.chat_history.append(("user", user_q))

        prompt = (
            "You are a fertilizer optimization and data modeling assistant. "
            "Help the user fill out a sustainable fertilizer optimization template. "
            "Be clear, concise, and practical.\n\n"
            f"User question: {user_q}"
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = resp["choices"][0]["message"]["content"].strip()
        st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")


