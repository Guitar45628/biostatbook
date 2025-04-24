import numpy as np
import streamlit as st
from modules.nav import Navbar
import statistics
from modules.code_editor_all import code_editor_for_all

# Page Title
st.set_page_config(page_title="Week 01 | Introduction to Biostatistics")


def section_table_of_contents():
    st.header("📚 Table of Contents")
    st.markdown(""" 
    1. [Definition and Scope of Biostatistics](#definition-and-scope-of-biostatistics)  
    2. [Differences Between Populations and Samples](#differences-between-populations-and-samples)  
    3. [Types of Data](#types-of-data)  
    4. [Summation Notation and Basic Mathematical Tools](#summation-notation-and-basic-mathematical-tools)  
    5. [Introduction to Descriptive Statistics](#introduction-to-descriptive-statistics)  
    6. [Quiz: Test Your Knowledge](#quiz-test-your-knowledge) 
    7. [Activity: Basic Statistical Tools in Python](#activity-basic-statistical-tools-in-python)
    8. [Role-Play Activity: Biostatistics Investigator](#role-play-activity-biostatistics-investigator) 
    """, unsafe_allow_html=True)


def section_recap():
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
    st.write(
        "Summation notation is a compact way to represent the sum of a sequence of numbers.")

    # Section: Descriptive Statistics
    st.header("Introduction to Descriptive Statistics")
    st.write(
        "Descriptive statistics summarize and describe data. Key measures include:")
    st.write("- **Mean**: Average value\n- **Median**: Middle value\n- **Mode**: Most frequent value\n- **Variance**: Measure of data spread\n- **Standard Deviation**: Square root of variance")


def section_quiz():
    # Quiz Section
    st.header("Quiz: Test Your Knowledge")
    questions = [
        ("What is biostatistics mainly used for?", [
         "Physics", "Biology and health sciences", "Engineering", "Astronomy"], "Biology and health sciences"),
        ("Which type of data has a true zero?", [
         "Categorical", "Ordinal", "Interval", "Ratio"], "Ratio"),
        ("What is the median of the dataset [3, 5, 7, 9, 11]?", [
         "5", "7", "9", "6"], "7"),
        ("What does summation notation (∑) represent?", [
         "Product", "Sum", "Difference", "Ratio"], "Sum"),
        ("Which measure represents data spread?", [
         "Mean", "Median", "Variance", "Mode"], "Variance")
    ]

    score = 0
    for i, (q, options, correct) in enumerate(questions):
        user_answer = st.radio(q, options, key=f"q{i}", index=None,)
        if user_answer == correct:
            score += 1

    if st.button("Submit Quiz"):
        st.success(f"You scored {score}/5!")
        if score == 5:
            st.balloons()


def section_activity1():
    # Activity Section
    st.header("Activity: Basic Statistical Tools in Python")
    st.write(
        "Modify and run the Python code below to compute basic statistical measures.")

    # Default code
    default_code = '''\
import numpy as np
import statistics

data = [10, 20, 30, 40, 50, 20, 10, 20, 30, 60]

mean_value = np.mean(data)
median_value = np.median(data)
mode_value = statistics.mode(np.array(data))
variance_value = np.var(data)
std_dev_value = np.std(data)

print(f'Mean: {mean_value}')
print(f'Median: {median_value}')
print(f'Mode: {mode_value}')
print(f'Variance: {variance_value}')
print(f'Standard Deviation: {std_dev_value}')
    '''

    st.code(default_code, language="python")

    # อธิบายกิจกรรมนี้ โดยสั้นๆคือให้ผู้ใช้คำนวณค่าทางสถิติจากข้อมูลที่สุ่มขึ้นมา แล้วตอบในช่องให้ถูกต้อง โดยสามารถเขียนโค้ดได้
    st.write("In this activity, you will calculate basic statistical measures from a randomly generated dataset. You can write your own code to compute the mean, median, mode, variance, and standard deviation.")

    # Check if rand_numbers already exists in session state, if not, generate it
    if "rand_numbers" not in st.session_state:
        # Random numbers between 10 and 20 elements, values between 0 and 100
        # Random number of elements between 10 and 20
        num_elements = np.random.randint(10, 15)
        st.session_state.rand_numbers = np.random.randint(0, 101, num_elements)

    # Use the value from session state
    rand_numbers = st.session_state.rand_numbers

    # Display the generated numbers
    st.info(f"random_number = {rand_numbers.tolist()}")

    # Add button to generate new random numbers
    if st.button("Generate New Random Numbers"):
        # Generate new random numbers with random length between 10 and 20
        num_elements = np.random.randint(10, 15)
        st.session_state.rand_numbers = np.random.randint(0, 101, num_elements)

    # Code Editor
    code_editor_for_all(default_code=default_code, key="code_editor_activity1")

    st.subheader("Your Answers")
    st.info(f"random_number = {rand_numbers.tolist()}")

    # Create a form for the user to input answers
    with st.form("stats_form"):
        user_mean = st.number_input("Mean", format="%.2f", step=0.1)
        user_median = st.number_input("Median", format="%.2f", step=0.1)
        user_mode = st.number_input("Mode", format="%.2f", step=0.1)
        user_variance = st.number_input("Variance", format="%.2f", step=0.1)
        user_sd = st.number_input("Standard Deviation", format="%.2f", step=0.1)
        submit_button = st.form_submit_button("Submit Answers")

    if submit_button:
        # Calculate correct answers
        correct_mean = np.mean(rand_numbers)
        correct_median = np.median(rand_numbers)
        try:
            correct_mode = statistics.mode(rand_numbers)
        except statistics.StatisticsError:
            correct_mode = "No unique mode"
        correct_variance = np.var(rand_numbers)
        correct_sd = np.std(rand_numbers)

        # Tolerance for floating point comparisons
        tolerance = 0.01

        # Convert both user and correct mode values to the same type (float or string for "No unique mode")
        if isinstance(user_mode, (int, float)):
            user_mode = float(user_mode)
        if isinstance(correct_mode, (int, float)):
            correct_mode = float(correct_mode)

        
        # Compare Mean
        if abs(user_mean - correct_mean) < tolerance:
            st.success(f"Mean: Correct! (Your answer: {user_mean:.2f}, Correct: {correct_mean:.2f})")
        else:
            st.error(f"Mean: Incorrect! (Your answer: {user_mean:.2f}, Correct: {correct_mean:.2f})")

        # Compare Median
        if abs(user_median - correct_median) < tolerance:
            st.success(f"Median: Correct! (Your answer: {user_median:.2f}, Correct: {correct_median:.2f})")
        else:
            st.error(f"Median: Incorrect! (Your answer: {user_median:.2f}, Correct: {correct_median:.2f})")

        # Handle the mode being "No unique mode" case
        if user_mode == "No unique mode" and correct_mode == "No unique mode":
            st.success(f"Mode: Correct! (Your answer: {user_mode}, Correct: {correct_mode})")
        elif user_mode != "No unique mode" and correct_mode != "No unique mode":
            if abs(user_mode - correct_mode) < tolerance:
                st.success(f"Mode: Correct! (Your answer: {user_mode}, Correct: {correct_mode})")
            else:
                st.error(f"Mode: Incorrect! (Your answer: {user_mode}, Correct: {correct_mode})")
        else:
            st.error(f"Mode: Incorrect! (Your answer: {user_mode}, Correct: {correct_mode})")


        # Compare Variance
        if abs(user_variance - correct_variance) < tolerance:
            st.success(f"Variance: Correct! (Your answer: {user_variance:.2f}, Correct: {correct_variance:.2f})")
        else:
            st.error(f"Variance: Incorrect! (Your answer: {user_variance:.2f}, Correct: {correct_variance:.2f})")

        # Compare Standard Deviation
        if abs(user_sd - correct_sd) < tolerance:
            st.success(f"Standard Deviation: Correct! (Your answer: {user_sd:.2f}, Correct: {correct_sd:.2f})")
        else:
            st.error(f"Standard Deviation: Incorrect! (Your answer: {user_sd:.2f}, Correct: {correct_sd:.2f})")

def section_activity2():
    # Activity 2 Section
    st.header("Role-Play Activity: Biostatistics Investigator")

    # Introduction to the activity
    st.write("""
    Welcome to the **Introduction to Biostatistics** role-play! In this activity, you will take on the role of a biostatistician
    and work through some important concepts in biostatistics that are covered in Week 1.
    Along the way, you will encounter scenarios that ask you to make decisions based on statistical concepts like mean, median, mode,
    variance, and types of data. Let's get started!
    """)

    # Scenario 1: Understanding the Scope of Biostatistics
    st.write("""
    **Scenario 1:** You have been hired as a biostatistician to work on a research project for a public health institute.
    Your first task is to define the **scope** of biostatistics and explain what it is used for. In a few words, biostatistics is
    the application of statistical methods to biological and health sciences to make informed decisions in public health, medicine, and epidemiology.
    """)

    biostat_answer = st.radio("What is the primary field of application for biostatistics?", [
                              "Physics", "Biology and Health Sciences", "Engineering", "Astronomy"], index=None)

    # Submit button for scenario 1
    if st.button("Submit for Scenario 1"):
        if biostat_answer == "Biology and Health Sciences":
            st.success(
                "Correct! 😊 Biostatistics is mainly used in the biological and health sciences.")
            st.balloons()
        else:
            st.error(
                "Oops! 😞 That's incorrect. The correct answer is 'Biology and Health Sciences'. Try again!")

    # Scenario 2: Differences Between Populations and Samples
    st.write("""
    **Scenario 2:** Now, you need to explain the difference between **populations** and **samples** to a colleague.
    A population includes all members of a group, while a sample is a smaller subset taken from the population for analysis.
    """)

    population_answer = st.radio("Which of the following describes a population?", [
                                 "A group of randomly selected individuals from a larger population", "All members of a defined group", "A single person in a study", "None of the above"], index=None)

    # Submit button for scenario 2
    if st.button("Submit for Scenario 2"):
        if population_answer == "All members of a defined group":
            st.success(
                "Correct! 😊 A population includes all members of a defined group.")
            st.balloons()
        else:
            st.error("Oops! 😞 Try again!")

    # Scenario 3: Types of Data
    st.write("""
    **Scenario 3:** In your research, you come across different types of data. You need to categorize the following datasets into the
    appropriate type: **categorical, ordinal, interval, or ratio**.
    
    Which type of data does the following represent?
    
    1. **Blood type (A, B, AB, O)**
    2. **Pain level (low, medium, high)**
    3. **Temperature in Celsius**
    4. **Weight of a person**
    """)

    data_1 = st.selectbox("1. Blood type (A, B, AB, O)", [
                          "Categorical", "Ordinal", "Interval", "Ratio"])
    data_2 = st.selectbox("2. Pain level (low, medium, high)", [
                          "Categorical", "Ordinal", "Interval", "Ratio"])
    data_3 = st.selectbox("3. Temperature in Celsius", [
                          "Categorical", "Ordinal", "Interval", "Ratio"])
    data_4 = st.selectbox("4. Weight of a person", [
                          "Categorical", "Ordinal", "Interval", "Ratio"])

    # Submit button for scenario 3
    if st.button("Submit for Scenario 3"):
        score = 0
        correct_data_1 = "Categorical"
        correct_data_2 = "Ordinal"
        correct_data_3 = "Interval"
        correct_data_4 = "Ratio"

        if data_1 == correct_data_1:
            score += 1
        if data_2 == correct_data_2:
            score += 1
        if data_3 == correct_data_3:
            score += 1
        if data_4 == correct_data_4:
            score += 1

        if score == 4:
            st.success("Correct! 😊 You got all the data type questions right!")
            st.balloons()
        else:
            st.error(
                f"Oops! 😞 You got {4 - score} answer(s) wrong. Please review your answers.")

    # Scenario 4: Summation Notation and Basic Tools
    st.write("""
    **Scenario 4:** You are now reviewing some basic mathematical tools, specifically **summation notation**.
    The sum of the numbers from 1 to 5 can be represented as:

    $$\sum_{i=1}^{5} i$$
    
    What is the sum of numbers from 1 to 5?

    """)

    summation_answer = st.radio("Choose your answer:", [
                                "10", "15", "20", "5"], index=None)

    # Submit button for scenario 4
    if st.button("Submit for Scenario 4"):
        if summation_answer == "15":
            st.success("Correct! 😊 The sum of numbers from 1 to 5 is 15.")
            st.balloons()
        else:
            st.error("Oops! 😞 That's incorrect. Try again!")

    # Scenario 5: Descriptive Statistics - Mean, Median, Mode, Variance, and Standard Deviation
    st.write("""
    **Scenario 5:** You're now ready to perform some basic descriptive statistics.
    You have the following data for blood pressure reductions after treatment: [5, 10, 15, 20, 25].

    What is the **mean** of this data?

    Enter your answer:
    """)

    user_mean = st.number_input(
        "Enter the mean value:", format="%.2f", step=0.1)

    # Submit button for scenario 5
    if st.button("Submit for Scenario 5"):
        correct_mean = np.mean([5, 10, 15, 20, 25])
        if user_mean == correct_mean:
            st.success(f"Correct! 😊 The mean value is {correct_mean}.")
            st.balloons()
        else:
            st.error(f"Oops! 😞 Try again!")


def main():
    Navbar()

    # Title
    st.title("Week 01 | Introduction to Biostatistics")

    section_table_of_contents()

    st.divider()

    section_recap()

    st.divider()

    section_quiz()

    st.divider()

    section_activity1()

    st.divider()

    section_activity2()

    st.divider()

    st.markdown("""
<div style='
    text-align: center;
    padding: 0px;
    margin: 0px auto;
    max-width: 600px;
    border-radius: 10px;
    ;
'>
    <h2>End of Week 1 🎉</h2>
</div>
""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
