
import streamlit as st
from rag_pipeline import generate_answer
import pandas as pd

st.set_page_config(page_title="Semantic Quote Retriever")
st.title("📜 Semantic Quote Retrieval with RAG")

query = st.text_input("Enter your query:", "Quotes about courage by women authors")
import openai




if st.button("Get Quotes"):
    answer, ctx, scores = generate_answer(query)
    st.subheader("📌 Generated Answer")
    st.markdown(answer)

    st.subheader("🔍 Retrieved Quotes")
    ctx['Similarity'] = [round(float(s), 3) for s in scores]
    st.dataframe(ctx[['quote', 'author', 'tags', 'Similarity']])

    st.download_button(
        label="📥 Download Results as JSON",
        data=ctx.to_json(orient="records", indent=2),
        file_name="quotes_result.json",
        mime="application/json"
    )