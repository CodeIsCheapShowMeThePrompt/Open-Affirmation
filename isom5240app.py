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
    if text.strip():  # 检查文本不为空
        # Perform text classification on the input text
        raw_results = classifier(text)

        # 调试：查看实际返回的数据结构
        st.write("Debug - Raw results:", raw_results)
        st.write("Debug - Type:", type(raw_results))

        # 获取结果列表
        results = raw_results[0] if isinstance(raw_results, list) else raw_results

        # Display the classification result
        max_score = float('-inf')
        max_label = ''

        for result in results:
            if isinstance(result, dict) and 'score' in result:
                if result['score'] > max_score:
                    max_score = result['score']
                    max_label = result['label']

        st.write("Text:", text)
        st.write("Label:", max_label)
        st.write("Score:", max_score)
    else:
        st.warning("Please enter some text to classify")

