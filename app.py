import streamlit as st
import openai
import textwrap

openai.api_key = "YOUR_KEY_HERE"

st.set_page_config(
    page_title="Sustainable Fertilizer Optimization",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# 🌿 Layout
# -----------------------------
col_template, col_chat = st.columns([2, 1])

# -----------------------------
# 🌱 LEFT: Multi-step template
# -----------------------------
with col_template:
    st.title("🌍 Sustainable Fertilizer Optimization")

    st.markdown(
        "Fill out the key fields below. Crop names and sections auto-update across the template."
    )

    step = st.radio(
        "Template step",
        ["1️⃣ Project setup", "2️⃣ Crop modeling", "3️⃣ Second crop", "4️⃣ Summary & export"],
        horizontal=True
    )

    # Shared core inputs
    crop = st.text_input("Primary crop (replaces [YOUR CROP(S) HERE] / [Your Crop])", "")
    crop2 = st.text_input("Second crop (replaces [2nd Crop])", "")

    # Defaults for display
    crop_display = crop or "[Your Crop]"
    crop_plural_display = crop or "[YOUR CROP(S) HERE]"
    crop2_display = crop2 or "[2nd Crop]"

    if step == "1️⃣ Project setup":
        st.subheader("Project Summary")
        st.markdown(
            textwrap.dedent(
                f"""
                This project analyzes how fertilizer affects **{crop_plural_display}** yields to find the optimal application point (the level that delivers strong crop performance without wasting money or causing nutrient pollution).

                Using linear and polynomial models, it identifies where fertilizer provides meaningful returns and where diminishing returns begin. The goal is to support farmers in using fertilizer efficiently while protecting soil, water, and the environment.
                """
            )
        )

        st.subheader("Problem & Impact")
        st.markdown(
            textwrap.dedent(
                f"""
                Fertilizer is one of the most powerful tools in agriculture. It boosts crop growth, increases food production, and supports global food security.

                But too much fertilizer becomes pollution, wastes money, and harms soil and water systems. The challenge is finding the point where fertilizer is helpful, not harmful.

                This project asks one core question:

                ♻️ **What fertilizer level delivers strong yields for {crop_display} without crossing into pollution‑causing excess?** ♻️
                """
            )
        )

    elif step == "2️⃣ Crop modeling":
        st.subheader(f"{crop_display} Modeling Overview")
        st.markdown(
            textwrap.dedent(
                f"""
                In this section, you model how fertilizer application influences **{crop_display}** yield using:

                - Model 1: Linear Regression  
                - Model 2: Polynomial Regression (Degree 2)  
                - Model 3: Optimal Fertilizer Level

                These models help identify the fertilizer “sweet spot” where yield is strong but environmental risk stays low.
                """
            )
        )

        model1_expl = st.text_area(
            "Model 1 (Linear Regression) explanation",
            f"Linear regression establishes the baseline relationship between fertilizer use and {crop_display} yield..."
        )
        model2_expl = st.text_area(
            "Model 2 (Polynomial Regression) explanation",
            f"Polynomial regression allows the model to capture curvature in the fertilizer–yield relationship for {crop_display}..."
        )
        model3_expl = st.text_area(
            "Model 3 (Optimal Fertilizer Level) explanation",
            f"Model 3 identifies the fertilizer level where predicted {crop_display} yield reaches its maximum before diminishing returns begin..."
        )
        crop_summary = st.text_area(
            f"{crop_display} model section summary",
            f"This section shows how fertilizer affects {crop_display} yield and identifies an environmentally responsible optimum..."
        )

    elif step == "3️⃣ Second crop":
        st.subheader(f"{crop2_display} Analysis")
        st.markdown(
            textwrap.dedent(
                f"""
                This section mirrors the {crop_display} workflow for **{crop2_display}**.

                It prepares a merged dataset of global {crop2_display} yield and fertilizer use, then applies the same modeling logic so both crops can be compared consistently.
                """
            )
        )

        crop2_yield_expl = st.text_area(
            f"{crop2_display} yield over time explanation",
            f"This plot shows how global {crop2_display} yield has changed across decades..."
        )

    elif step == "4️⃣ Summary & export":
        st.subheader("Final Summary")
        final_summary = st.text_area(
            "Write your final project summary / conclusions:",
            f"This project demonstrates how fertilizer affects {crop_plural_display} yields and identifies an optimal application level..."
        )

        st.subheader("Download Completed Template")

        # Build a markdown version of the filled template
        template_md = textwrap.dedent(
            f"""
            # Sustainable Fertilizer Optimization

            ## Project Summary
            This project analyzes how fertilizer affects **{crop_plural_display}** yields to find the optimal application point.

            ## Problem & Impact
            What fertilizer level delivers strong yields for **{crop_display}** without crossing into pollution‑causing excess?

            ## {crop_display} Modeling Summary
            {crop_summary or '[Add model section summary here]'}

            ## {crop2_display} Section
            {crop2_yield_expl if 'crop2_yield_expl' in locals() else '[Add second crop explanation here]'}

            ## Final Summary
            {final_summary}
            """
        )

        st.download_button(
            "📥 Download template as Markdown",
            data=template_md,
            file_name="sustainable_fertilizer_optimization_template.md",
            mime="text/markdown",
        )

# -----------------------------
# 🌿 RIGHT: Chatbot
# -----------------------------
with col_chat:
    st.header("🌱 Fertilizer Intelligence Assistant")
    st.markdown("Ask questions while you fill out the template.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask a question:")

    if user_q:
        st.session_state.chat_history.append(("user", user_q))

        prompt = (
            "You are a fertilizer optimization and data modeling assistant. "
            "Help the user fill out a Sustainable Fertilizer Optimization template. "
            "Use their crop names when helpful.\n\n"
            f"Primary crop: {crop_display}\n"
            f"Second crop: {crop2_display}\n\n"
            f"User question: {user_q}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response["choices"][0]["message"]["content"].strip()
        st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")



