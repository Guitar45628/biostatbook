import pandas as pd
import streamlit as st
from modules.nav import Navbar
from modules.foot import Footer
from scipy.stats import chi2_contingency, chisquare

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 07 | Chi-Square Tests",
)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#chi-square-test-for-independence">Chi-Square Test for Independence</a></li>
            <li><a href="#chi-square-goodness-of-fit-test">Chi-Square Goodness-of-Fit Test</a></li>
            <li><a href="#applications-in-categorical-data-analysis">Applications in Categorical Data Analysis</a></li>
            <li><a href="#activities">Activities</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_chi_square_independence():
    st.header("Chi-Square Test for Independence")
    st.write("""
        The chi-square test for independence is used to determine whether two categorical variables are independent of each other.
    """)
    st.subheader("Example: Contingency Table Analysis")
    st.code("""
# Example contingency table
data = [[50, 30], [20, 100]]
stat, p, dof, expected = chi2_contingency(data)

print(f"Chi-Square Statistic: {stat:.2f}")
print(f"P-Value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)
""", language="python")

def section_chi_square_goodness_of_fit():
    st.header("Chi-Square Goodness-of-Fit Test")
    st.write("""
        The chi-square goodness-of-fit test is used to determine whether a sample matches the expected distribution.
    """)
    st.subheader("Example: Goodness-of-Fit Test")
    st.code("""
# Observed and expected frequencies
observed = [50, 30, 20]
expected = [40, 40, 20]

stat, p = chisquare(f_obs=observed, f_exp=expected)

print(f"Chi-Square Statistic: {stat:.2f}")
print(f"P-Value: {p:.4f}")
""", language="python")

def section_applications():
    st.header("Applications in Categorical Data Analysis")
    st.write("""
        Chi-square tests are widely used in categorical data analysis, such as:
        - Analyzing survey data.
        - Testing for independence in contingency tables.
        - Evaluating goodness-of-fit for distributions.
    """)

def activity_quiz():
    st.header("Activity 1: Quiz")
    st.write("Test your knowledge on Chi-Square Tests.")

    # Create a form for the quiz
    with st.form(key='chi_square_quiz'):
        st.subheader("Please answer the following questions about Chi-Square Tests:")

        q1 = st.radio("1. What is the purpose of a chi-square test for independence?",
                        ("To compare means of two groups",
                         "To determine if two categorical variables are independent",
                         "To test for normality in a dataset",
                         "To evaluate the goodness-of-fit of a regression model"), index=None)

        q2 = st.radio("2. Which of the following is NOT an assumption of the chi-square test?",
                        ("The data must be categorical",
                         "The expected frequency in each cell should be at least 5",
                         "The sample size must be greater than 30",
                         "The observations must be independent"), index=None)

        q3 = st.radio("3. What does a p-value less than 0.05 indicate in a chi-square test?",
                        ("The null hypothesis is rejected",
                         "The null hypothesis is accepted",
                         "The test is inconclusive",
                         "The data is normally distributed"), index=None)

        q4 = st.radio("4. In a chi-square goodness-of-fit test, what is being compared?",
                        ("Observed frequencies to expected frequencies",
                         "Means of two groups",
                         "Variances of two groups",
                         "Correlation between two variables"), index=None)

        q5 = st.radio("5. Which of the following is a limitation of the chi-square test?",
                        ("It cannot be used for categorical data",
                         "It requires a large sample size",
                         "It assumes the data is normally distributed",
                         "It cannot handle more than two variables"), index=None)

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        correct_answers = {
            "q1": "To determine if two categorical variables are independent",
            "q2": "The sample size must be greater than 30",
            "q3": "The null hypothesis is rejected",
            "q4": "Observed frequencies to expected frequencies",
            "q5": "It requires a large sample size"
        }

        user_answers = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5,
        }

        score = 0
        feedback = []
        for i in range(1, 6):
            key = f"q{i}"
            if user_answers[key] == correct_answers[key]:
                score += 1
                feedback.append(f"✔ Question {i}: Correct.")
            else:
                feedback.append(f"❌ Question {i}: Incorrect. The correct answer is: {correct_answers[key]}.")

        st.write(f"Your score: {score}/5")
        with st.expander("See detailed feedback"):
            for comment in feedback:
                st.write(comment)

def activity_performing_chi_square():
    st.header("Activity 2: Performing Chi-Square Tests on Contingency Tables")
    st.write("""
        In this activity, you will analyze a real-world scenario using a chi-square test for independence.
    """)

    # Scenario
    st.subheader("Scenario: Customer Preferences")
    st.write("""
        A company wants to determine if there is a relationship between customer age groups and their preference for a new product. 
        The survey data is summarized in the contingency table below:
    """)

    # Display Table
    st.table({
        "Age Group": ["18-25", "26-35", "36-45", "46+"],
        "Prefer": [50, 60, 40, 30],
        "Do Not Prefer": [30, 40, 50, 60]
    })

    # Formula Expander
    with st.expander("Chi-Square Test Formula"):
        st.write("The chi-square statistic is calculated as:")
        st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
        st.write("Where:")
        st.write("- \(O\): Observed frequency")
        st.write("- \(E\): Expected frequency")

    # Code Editor
    st.write("""
        Use the code editor below to perform a chi-square test for independence on the given data.
    """)
    default_code = """
from scipy.stats import chi2_contingency

# Survey data: Age Group vs Product Preference
data = [[50, 30], [60, 40], [40, 50], [30, 60]]  # Rows: Age Groups; Columns: Prefer, Do Not Prefer
stat, p, dof, expected = chi2_contingency(data)

print(f"Chi-Square Statistic: {stat:.2f}")
print(f"P-Value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

if p < 0.05:
    print("Reject the null hypothesis: There is a significant relationship between age group and product preference.")
else:
    print("Fail to reject the null hypothesis: No significant relationship between age group and product preference.")
"""
    code_editor_for_all(default_code=default_code, key="chi_square_activity_2")

    # Quiz
    st.subheader("Quick Quiz")
    st.write("Answer the following question based on the activity above:")

    q1 = st.radio("1. What does a p-value less than 0.05 indicate in this context?",
                  ["There is no relationship between age group and product preference",
                   "There is a significant relationship between age group and product preference",
                   "The data is normally distributed"], index=None)

    if st.button("Submit Quiz"):
        if q1 == "There is a significant relationship between age group and product preference":
            st.success("Correct! A p-value less than 0.05 indicates a significant relationship.")
        else:
            st.error("Incorrect. Review the chi-square test interpretation.")

def activity_interpreting_p_values_with_health_data():
    st.header("Activity 3: Analyzing Heart Attack Cases After Covid Vaccine")
    st.write("""
        In this activity, you will analyze the dataset "Heart Attack Cases After Covid Vaccine in India" to determine if there is a relationship between heart attack cases and the number of vaccine doses received.
    """)

    # Display sample data from CSV
    st.subheader("Sample Data")
    st.write("""
        Below is a sample of the dataset loaded from the CSV file:
    """)
    file_path = "data/heart_attack_vaccine_data.csv"
    vaccine_data = pd.read_csv(file_path)
    st.dataframe(vaccine_data)

    # Scenario
    st.subheader("Scenario: Relationship Between Heart Attack Cases and Vaccine Dose")
    st.write("""
        Using the dataset "Heart Attack Cases After Covid Vaccine in India", perform a chi-square test for independence to determine if there is a significant relationship 
        between heart attack cases (based on the 'Heart Attack Date' column) and the number of vaccine doses received.
    """)

    # Code Editor
    st.write("""
        Use the code editor below to perform the chi-square test for independence.
    """)
    default_code = """
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
file_path = "data/heart_attack_vaccine_data.csv"
vaccine_data = pd.read_csv(file_path)

# Add a new column to indicate whether a heart attack occurred
vaccine_data['Heart Attack Occurred'] = vaccine_data['Heart Attack Date'].notnull()

# Create a contingency table
contingency_table = pd.crosstab(vaccine_data['Heart Attack Occurred'], vaccine_data['Vaccine Dose'])

# Perform the chi-square test
stat, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Statistic:", stat)
print("P-Value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

if p < 0.05:
    print("Reject the null hypothesis: There is a significant relationship between heart attack cases and vaccine dose.")
else:
    print("Fail to reject the null hypothesis: No significant relationship between heart attack cases and vaccine dose.")
"""
    code_editor_for_all(default_code=default_code, key="chi_square_heart_attack_vaccine")

def main():
    Navbar()
    st.title("Week 07 | Chi-Square Tests")
    section_table_of_contents()
    section_chi_square_independence()
    section_chi_square_goodness_of_fit()
    section_applications()
    st.divider()
    st.header("Activities")
    activity_quiz()
    st.divider()
    activity_performing_chi_square()
    st.divider()
    activity_interpreting_p_values_with_health_data()
    Footer(7)

if __name__ == "__main__":
    main()