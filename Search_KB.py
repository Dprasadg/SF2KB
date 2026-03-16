import streamlit as st
import os

KB_FOLDER = "/Users/durgaprasad/Documents/SFtoKB/KB_Articles"

st.title("AI Support Knowledge Base")

query = st.text_input("Search KB articles")

# Load KB files
articles = []

for file in os.listdir(KB_FOLDER):
    if file.endswith(".md"):
        path = os.path.join(KB_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        articles.append({
            "name": file,
            "content": content
        })

# Search logic
if query:

    results = []

    for article in articles:
        if query.lower() in article["content"].lower():
            results.append(article)

    if results:

        st.subheader("Search Results")

        for article in results:

            with st.expander(article["name"]):
                st.write(article["content"])

    else:
        st.warning("No matching KB articles found.")