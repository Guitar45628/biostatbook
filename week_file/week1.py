import streamlit as st
import numpy as np
import io
import sys

def display_week1():
    # Title
    st.title("Week 1: Introduction to Statistical Thinking")

    # Table of Contents
    st.markdown("## Table of Contents")
    st.markdown("""
    1. [Definition and Scope of Biostatistics](#definition-and-scope-of-biostatistics)  
    2. [Differences Between Populations and Samples](#differences-between-populations-and-samples)  
    3. [Types of Data](#types-of-data)  
    4. [Summation Notation and Basic Mathematical Tools](#summation-notation-and-basic-mathematical-tools)  
    5. [Introduction to Descriptive Statistics](#introduction-to-descriptive-statistics)  
    6. [Activity: Basic Statistical Tools in Python](#activity-basic-statistical-tools-in-python)  
    7. [Quiz: Test Your Knowledge](#quiz-test-your-knowledge)  
    """, unsafe_allow_html=True)

    # Section: Introduction
    st.header("Definition and Scope of Biostatistics")
    st.write("Biostatistics is the application of statistical methods to biological and health sciences. It helps in making data-driven decisions in medicine, epidemiology, and public health.")

    # Section: Populations vs Samples
    st.header("Differences Between Populations and Samples")
    st.write("A population includes all members of a defined group, while a sample is a subset of that population. Statistical analysis often relies on samples to infer characteristics of the population.")

    # Section: Types of Data
    st.header("Types of Data")
    st.write("1. **Categorical Data**: Data that can be grouped into categories (e.g., blood type, gender).\n2. **Ordinal Data**: Categorical data with an inherent order (e.g., pain level: low, medium, high).\n3. **Interval Data**: Numeric data with meaningful differences but no true zero (e.g., temperature in Celsius).\n4. **Ratio Data**: Numeric data with a meaningful zero (e.g., weight, height, age).")

    # Section: Summation Notation
    st.header("Summation Notation and Basic Mathematical Tools")
    st.latex(r"""\sum_{i=1}^{n} x_i""")
    st.write("Summation notation is a compact way to represent the sum of a sequence of numbers.")

    # Section: Descriptive Statistics
    st.header("Introduction to Descriptive Statistics")
    st.write("Descriptive statistics summarize and describe data. Key measures include:")
    st.write("- **Mean**: Average value\n- **Median**: Middle value\n- **Mode**: Most frequent value\n- **Variance**: Measure of data spread\n- **Standard Deviation**: Square root of variance")

    # Activity Section
    st.header("Activity: Basic Statistical Tools in Python")
    st.write("Modify and run the Python code below to compute basic statistical measures.")

    # Default code
    default_code = '''\
    import numpy as np

    data = [10, 20, 30, 40, 50]

    mean_value = np.mean(data)
    median_value = np.median(data)
    variance_value = np.var(data)
    std_dev_value = np.std(data)

    print(f'Mean: {mean_value}')
    print(f'Median: {median_value}')
    print(f'Variance: {variance_value}')
    print(f'Standard Deviation: {std_dev_value}')
    '''

    st.code(default_code, language="python")

    # Code Editor
    user_code = st.text_area("Edit your Python code here:", default_code, height=100)

    # Run the code
    if st.button("Run Code"):
        try:
            # Capture the output
            output = io.StringIO()
            sys.stdout = output  # Redirect stdout

            exec(user_code)  # Execute the user code

            sys.stdout = sys.__stdout__  # Reset stdout

            st.success("Code executed successfully!")
            st.text_area("Output:", output.getvalue(), height=150)
        
        except Exception as e:
            st.error(f"Error: {e}")

    # Quiz Section
    st.header("Quiz: Test Your Knowledge")
    questions = [
        ("What is biostatistics mainly used for?", ["Physics", "Biology and health sciences", "Engineering", "Astronomy"], "Biology and health sciences"),
        ("Which type of data has a true zero?", ["Categorical", "Ordinal", "Interval", "Ratio"], "Ratio"),
        ("What is the median of the dataset [3, 5, 7, 9, 11]?", ["5", "7", "9", "6"], "7"),
        ("What does summation notation (∑) represent?", ["Product", "Sum", "Difference", "Ratio"], "Sum"),
        ("Which measure represents data spread?", ["Mean", "Median", "Variance", "Mode"], "Variance")
    ]

    score = 0
    for i, (q, options, correct) in enumerate(questions):
        user_answer = st.radio(q, options, key=f"q{i}")
        if user_answer == correct:
            score += 1

    if st.button("Submit Quiz"):
        st.success(f"You scored {score}/5!")
