import os
import pandas as pd
import streamlit as st
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw
from scipy import stats
import numpy as np

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 05 | Confidence Intervals and Hypothesis Testing",
)


@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code,
                        key='codepad-week3-graph', warning_text=warning_text)


def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#introduction-to-confidence-intervals">Introduction to Confidence Intervals</a></li>
            <li><a href="#constructing-confidence-intervals">Constructing Confidence Intervals</a></li>
            <li><a href="#introduction-to-hypothesis-testing">Introduction to Hypothesis Testing</a></li>
            <li><a href="#one-sample-hypothesis-tests">One-Sample Hypothesis Tests</a></li>
            <li><a href="#steps-for-hypothesis-testing">Steps for Hypothesis Testing</a></li>
            <li><a href="#activity-one-quiz">Activity 1: Quiz</a></li>
            <li><a href="#activity-two-construct-ci">Activity 2: Confidence Intervals Practice</a></li>
            <li><a href="#activity-three-one-sample-tests">Activity 3: One-Sample Hypothesis Tests</a></li>
            <li><a href="#activity-four-hypothesis-decision">Activity 4: Hypothesis Decision Making</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_one_introduction_to_sampling():
    st.header("Introduction to Confidence Intervals")
    st.write("""
        Confidence intervals provide a range of values within which we can expect the true population parameter to lie, based on sample data.
        They are constructed using the sample mean and standard error, and they help us understand the precision of our estimates.
    """)

def section_two_constructing_confidence_intervals():
    st.header("Constructing Confidence Intervals")
    st.write("""
        To construct a confidence interval, we need to know the sample mean, standard deviation, and the desired level of confidence (e.g., 95%).
        The formula for a confidence interval is:
    """)
    st.latex(r"CI = \text{sample mean} \pm (\text{critical value} \times \text{standard error})")

def section_three_introduction_to_hypothesis_testing():
    st.header("Introduction to Hypothesis Testing")
    st.write("""
        Hypothesis testing is a statistical method used to make decisions about population parameters based on sample data.
        It involves formulating a null hypothesis (H0) and an alternative hypothesis (H1), and then using sample data to determine whether to reject or fail to reject the null hypothesis.
    """)

def section_four_one_sample_hypothesis_tests():
    st.header("One-Sample Hypothesis Tests")
    st.write("""
        One-sample hypothesis tests are used to compare a sample mean to a known population mean.
        Common tests include the t-test and z-test, depending on the sample size and whether the population standard deviation is known.
    """)

def section_five_steps_for_hypothesis_testing():
    st.header("Steps for Hypothesis Testing")
    st.write("""
        The steps for hypothesis testing include:
    """)
    st.markdown("""
        1. Formulate the null and alternative hypotheses.
        2. Choose the significance level (alpha).
        3. Collect sample data and calculate the test statistic.
        4. Determine the critical value or p-value.
        5. Make a decision to reject or fail to reject the null hypothesis.
    """)

def activity_one_quiz():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)
    with cent_co:
        st.header("Activity 1: Quiz")
        st.write("Test your knowledge on Confidence Intervals and Hypothesis Testing.")

    # Create a form for the quiz
    with st.form(key='confidence_hypothesis_quiz'):
        st.subheader("Please answer the following questions about Confidence Intervals and Hypothesis Testing:")

        q1 = st.radio("1. A confidence interval provides:",
                        ("A single best estimate of a population parameter",
                        "A range of values within which the sample mean is expected to lie",
                        "A range of values within which the true population parameter is expected to lie",
                        "A measure of the certainty of a point estimate"), index=None)

        q2 = st.radio("2. What does the confidence level (e.g., 95%) represent?",
                        ("The probability that the sample mean falls within the interval",
                        "The percentage of times we expect the population parameter to fall within the interval if we repeat the sampling process many times",
                        "The probability that our conclusion about the population parameter is correct",
                        "The degree of accuracy of our sample statistic"), index=None)

        q3 = st.radio("3. Which of the following will lead to a wider confidence interval (assuming other factors are constant)?",
                        ("A larger sample size",
                        "A lower confidence level",
                        "A smaller standard error",
                        "A higher confidence level"), index=None)

        q4 = st.radio("4. In hypothesis testing, the null hypothesis (H0) typically represents:",
                        ("The researcher's belief about the population",
                        "The status quo or no effect",
                        "The alternative hypothesis",
                        "The outcome that the researcher wants to prove"), index=None)

        q5 = st.radio("5. What is the purpose of the alternative hypothesis (H1)?",
                        ("To support the null hypothesis",
                        "To provide a specific value for the population parameter",
                        "To represent what the researcher is trying to find evidence for",
                        "To set the significance level"), index=None)

        q6 = st.radio("6. The p-value in hypothesis testing is:",
                        ("The probability of the null hypothesis being true",
                        "The probability of observing the sample data (or more extreme data) if the null hypothesis is true",
                        "The significance level chosen by the researcher",
                        "1 minus the confidence level"), index=None)

        q7 = st.radio("7. What is the significance level (alpha) in hypothesis testing?",
                        ("The probability of making a Type II error",
                        "The probability of making a Type I error",
                        "The power of the test",
                        "The confidence level"), index=None)

        q8 = st.radio("8. A one-sample t-test is typically used when:",
                        ("Comparing the means of two independent samples and the population standard deviations are known",
                        "Comparing the means of two related samples",
                        "Comparing a sample mean to a known population mean and the population standard deviation is unknown",
                        "Testing the variance of a single population"), index=None)

        q9 = st.radio("9. If the p-value is less than the significance level (alpha), we should:",
                        ("Accept the null hypothesis",
                        "Fail to reject the null hypothesis",
                        "Reject the alternative hypothesis",
                        "Reject the null hypothesis"), index=None)

        q10 = st.radio("10. Which of the following is a step in hypothesis testing?",
                        ("Calculating the confidence interval",
                        "Formulating the null and alternative hypotheses",
                        "Estimating the population size",
                        "Randomly selecting a significance level"), index=None)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        score = 0
        correct_answers = {
            "q1": "A range of values within which the true population parameter is expected to lie",
            "q2": "The percentage of times we expect the population parameter to fall within the interval if we repeat the sampling process many times",
            "q3": "A higher confidence level",
            "q4": "The status quo or no effect",
            "q5": "To represent what the researcher is trying to find evidence for",
            "q6": "The probability of observing the sample data (or more extreme data) if the null hypothesis is true",
            "q7": "The probability of making a Type I error",
            "q8": "Comparing a sample mean to a known population mean and the population standard deviation is unknown",
            "q9": "Reject the null hypothesis",
            "q10": "Formulating the null and alternative hypotheses"
        }

        user_answers = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5,
            "q6": q6,
            "q7": q7,
            "q8": q8,
            "q9": q9,
            "q10": q10
        }

        feedback = []

        for i, key in enumerate(correct_answers, 1):
            if user_answers[key] == correct_answers[key]:
                score += 1
                feedback.append(f"✔ {i}. Correct answer. The correct answer is: {correct_answers[key]}")
            else:
                feedback.append(f"❌ {i}. Wrong answer. The correct answer is: {correct_answers[key]}")

        st.write(f"Your score: {score}/10")

        with st.expander("See correct answers"):
            for answer in feedback:
                st.write(answer)

import random

def generate_random_problem():
    # Generate random values for the problem
    sample_size = random.randint(30, 100)  # Random sample size between 30 and 100
    mean = random.uniform(70, 100)  # Random mean between 70 and 100
    std_dev = random.uniform(5, 15)  # Random standard deviation between 5 and 15
    confidence_level = random.choice([80, 90, 95, 99])  # Random confidence level

    # Determine Z-value based on confidence level
    z_values = {
        80: 1.282,
        90: 1.645,
        95: 1.960,
        99: 2.576
    }
    z_value = z_values[confidence_level]

    return sample_size, mean, std_dev, confidence_level, z_value

def activity_two_construct_ci():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)
    with cent_co:
        st.header("Activity 2: Confidence Intervals Practice")
        st.write("Learn how to calculate, interpret, and apply confidence intervals in clinical contexts.")

    st.markdown("---")
    st.subheader("🧪 Practice Problem")

    # Function to generate a random problem and store it in session state
    def generate_problem():
        if "problem" not in st.session_state:
            st.session_state.problem = {
                "sample_size": random.randint(30, 100),
                "mean": random.uniform(70, 100),
                "std_dev": random.uniform(5, 15),
                "confidence_level": random.choice([80, 90, 95, 99])
            }

        # Determine Z-value based on confidence level
        z_values = {
            80: 1.282,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }
        st.session_state.problem["z_value"] = z_values[st.session_state.problem["confidence_level"]]

    # Function to reset the problem
    def reset_problem():
        if "problem" in st.session_state:
            del st.session_state.problem

    # Ensure a problem is generated only once per session
    if "problem" not in st.session_state:
        generate_problem()

    # Add a button to reset the problem
    if st.button("Generate New Problem"):
        reset_problem()
        generate_problem()

    # Use the problem from session state
    problem = st.session_state.problem

    # Display the problem
    st.markdown(f"""
    **Scenario:**  
    A researcher collects a sample of {problem['sample_size']} patients' diastolic blood pressure readings.  
    The sample mean is **{problem['mean']:.2f} mmHg**, with a standard deviation of **{problem['std_dev']:.2f} mmHg**.  
    Assume the population is normally distributed.

    **Task:**  
    Calculate the **{problem['confidence_level']}% Confidence Interval** for the true population mean.
    """)

    # Add hints dynamically
    with st.expander("💡 Click to see hints"):
        st.markdown(f"""
        - Standard Error (SE) = `std_dev / √n`
        - Z-value for {problem['confidence_level']}% confidence = **{problem['z_value']}**
        - CI = `mean ± z * SE`
        """)

    # Add a code editor for users to execute the solution
    default_code = f"""
import numpy as np

# Given values
mean = {problem['mean']:.2f}
std_dev = {problem['std_dev']:.2f}
n = {problem['sample_size']}
z = {problem['z_value']}  # for {problem['confidence_level']}% confidence

# Step 1: Standard Error
se = std_dev / np.sqrt(n)

# Step 2: Margin of Error
moe = z * se

# Step 3: Confidence Interval
lower = mean - moe
upper = mean + moe

print(f"{problem['confidence_level']}% CI: ({{lower:.2f}}, {{upper:.2f}})")
    """

    st.subheader("💻 Code Editor")
    st.write("Run the code below to calculate the confidence interval. Copy the results and input them in the fields below to check your answer.")

    code_editor_for_all(default_code=default_code, key="ci_code_editor", warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)")

    st.markdown("---")
    st.subheader("✏️ Your Turn")

    st.write("Fill in the values below to calculate the confidence interval:")

    lower_input = st.number_input("Lower Bound", value=0.0, step=0.01)
    upper_input = st.number_input("Upper Bound", value=0.0, step=0.01)

    if st.button("Check Answer"):
        # Correct answers based on the default code
        se = problem['std_dev'] / (problem['sample_size'] ** 0.5)
        moe = problem['z_value'] * se
        correct_lower = problem['mean'] - moe
        correct_upper = problem['mean'] + moe

        if abs(lower_input - correct_lower) < 0.01 and abs(upper_input - correct_upper) < 0.01:
            st.success(f"Correct! Your {problem['confidence_level']}% confidence interval is accurate.")
        else:
            st.error(f"Incorrect. Please try again or re-run the code to verify your calculations.")

def activity_three_one_sample_tests():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)
    with cent_co:
        st.header("Activity 3: One-Sample Hypothesis Tests")
        st.write("Learn how to perform and interpret one-sample hypothesis tests.")

    st.markdown("---")
    st.subheader("🧪 Practice Problem")

    st.markdown("""
    **Scenario:**  
    A hospital claims that the average diastolic blood pressure of its patients is **80 mmHg**.  
    A researcher collects a random sample of **30 patients** and finds a sample mean of **85 mmHg** with a standard deviation of **12 mmHg**.  

    **Task:**  
    Perform a one-sample t-test to determine if the hospital's claim is valid at a **5% significance level**.

    ### ✍️ Try it Yourself!
    """)

    with st.expander("💡 Click to see hints"):
        st.markdown("""
        - Null Hypothesis (H₀): The population mean is 80 mmHg.
        - Alternative Hypothesis (H₁): The population mean is not 80 mmHg.
        - Test Statistic (t) = `(sample_mean - population_mean) / (std_dev / √n)`
        - Degrees of Freedom (df) = `n - 1`
        - Compare the p-value with the significance level (α = 0.05).
        """)

    # Add a code editor for users to execute the solution
    default_code = """
import scipy.stats as stats
import numpy as np

# Given values
population_mean = 80
sample_mean = 85
std_dev = 12
n = 30
alpha = 0.05

# Step 1: Calculate the test statistic (t)
se = std_dev / np.sqrt(n)
t_statistic = (sample_mean - population_mean) / se

# Step 2: Degrees of Freedom
df = n - 1

# Step 3: Calculate the p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

# Step 4: Compare p-value with alpha
if p_value < alpha:
    result = "Reject the null hypothesis."
else:
    result = "Fail to reject the null hypothesis."

print(f"t-statistic: {t_statistic:.2f}")
print(f"p-value: {p_value:.4f}")
print(result)
"""

    st.subheader("💻 Code Editor")
    st.write("Run the code below to perform the one-sample t-test. Copy the results and input them in the fields below to check your answer.")

    code_editor_for_all(default_code=default_code, key="t_test_code_editor", warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)")

    st.markdown("---")
    st.subheader("✏️ Your Turn")

    st.write("Fill in the values below to check your understanding:")

    t_input = st.number_input("t-statistic", value=0.0, step=0.01)
    p_input = st.number_input("p-value", value=0.0, step=0.0001)

    if st.button("Check Answer", key="check_answer_activity_three"):
        # Correct answers based on the default code
        correct_se = 12 / (30 ** 0.5)
        correct_t = (85 - 80) / correct_se
        correct_df = 30 - 1
        correct_p = 2 * (1 - stats.t.cdf(abs(correct_t), correct_df))

        if abs(t_input - correct_t) < 0.01 and abs(p_input - correct_p) < 0.0001:
            st.success("Correct! Your calculations are accurate.")
        else:
            st.error("Incorrect. Please try again or re-run the code to verify your calculations.")

def activity_four_hypothesis_decision():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)
    with cent_co:
        st.header("Activity 4: Hypothesis Decision Making")
        st.write("Test your ability to decide whether to reject or fail to reject the null hypothesis based on given statistical values.")

    st.markdown("---")
    st.subheader("🧪 Practice Problem")

    st.markdown("""
    **Instructions:**  
    For each question, decide whether to **reject** or **fail to reject** the null hypothesis (H₀) based on the provided p-value and significance level (α).
    """)

    # Questions
    questions = [
        {"p_value": 0.03, "alpha": 0.05, "correct": "Reject"},
        {"p_value": 0.10, "alpha": 0.05, "correct": "Fail to Reject"},
        {"p_value": 0.001, "alpha": 0.01, "correct": "Reject"},
        {"p_value": 0.06, "alpha": 0.05, "correct": "Fail to Reject"},
        {"p_value": 0.049, "alpha": 0.05, "correct": "Reject"}
    ]

    user_answers = []
    score = 0

    with st.form(key="hypothesis_decision_form"):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Question {i}:**")
            st.write(f"p-value: {q['p_value']}, α: {q['alpha']}")
            user_answers.append(st.radio(f"What is your decision for Question {i}?", ["Reject", "Fail to Reject"], key=f"q{i}"))

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        feedback = []

        for i, q in enumerate(questions):
            if user_answers[i] == q["correct"]:
                score += 1
                feedback.append(f"✔ Question {i+1}: Correct! The correct answer is **{q['correct']}**.")
            else:
                feedback.append(f"❌ Question {i+1}: Incorrect. The correct answer is **{q['correct']}**.")

        st.write(f"Your score: {score}/{len(questions)}")

        with st.expander("See Feedback"):
            for f in feedback:
                st.write(f)

def main():
    # Initialize Tailwind CSS
    tw.initialize_tailwind()

    # Navbar
    Navbar()

    # Title
    st.title("Week 05 | Confidence Intervals and Hypothesis Testing")

    # Table of contents
    section_table_of_contents()

    # Section 1: Introduction to Confidence Intervals
    section_one_introduction_to_sampling()

    # Section 2: Constructing Confidence Intervals
    section_two_constructing_confidence_intervals()

    # Section 3: Introduction to Hypothesis Testing
    section_three_introduction_to_hypothesis_testing()

    # Section 4: One-Sample Hypothesis Tests
    section_four_one_sample_hypothesis_tests()

    # Section 5: Steps for Hypothesis Testing
    section_five_steps_for_hypothesis_testing()

    st.divider()

    # Activities
    st.markdown("<a id='activity-one-quiz'></a>", unsafe_allow_html=True)  # Anchor for Activity 1
    activity_one_quiz()
    st.divider()

    st.markdown("<a id='activity-two-construct-ci'></a>", unsafe_allow_html=True)  # Anchor for Activity 2
    activity_two_construct_ci()
    st.divider()

    st.markdown("<a id='activity-three-one-sample-tests'></a>", unsafe_allow_html=True)  # Anchor for Activity 3
    activity_three_one_sample_tests()
    st.divider()

    st.markdown("<a id='activity-four-hypothesis-decision'></a>", unsafe_allow_html=True)  # Anchor for Activity 4
    activity_four_hypothesis_decision()

    Footer(5)


if __name__ == "__main__":
    main()
