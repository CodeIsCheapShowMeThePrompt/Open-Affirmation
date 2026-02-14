import streamlit as st
from transformers import pipeline

# Load the text classification model pipeline
classifier = pipeline("text-classification",
                      model='isom5240ust/bert-base-uncased-emotion',
                      return_all_scores=True)

# Streamlit application title
st.title("Text Classification for you")
st.write("Classification for 6 emotions: sadness, joy, love, anger, fear, surprise")

# Text input for user to enter the text to classify
text = st.text_area("Enter the text to classify", "")

# Perform text classification when the user clicks the "Classify" button
if st.button("Classify"):
    if text.strip():
        # Perform text classification on the input text
        raw_results = classifier(text)

        # 处理结果：检查是否需要取第一个元素
        if isinstance(raw_results, list) and len(raw_results) > 0:
            # 检查第一个元素是列表还是字典
            if isinstance(raw_results[0], list):
                # 结构是 [[{}, {}, ...]]
                results = raw_results[0]
            elif isinstance(raw_results[0], dict):
                # 结构是 [{}, {}, ...]
                results = raw_results
            else:
                st.error("Unexpected result format")
                results = []
        else:
            results = []

        # Display the classification result
        if results:
            max_score = float('-inf')
            max_label = ''

            for result in results:
                if result['score'] > max_score:
                    max_score = result['score']
                    max_label = result['label']

            st.write("Text:", text)
            st.write("Label:", max_label)
            st.write("Score:", f"{max_score:.4f}")

            # 显示所有情感的分数
            st.write("\nAll scores:")
            for result in results:
                st.write(f"- {result['label']}: {result['score']:.4f}")
        else:
            st.error("No results returned")
    else:
        st.warning("Please enter some text to classify")
