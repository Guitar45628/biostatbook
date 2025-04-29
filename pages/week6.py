import streamlit as st
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math


from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 06 | One-Sample and Two-Sample t-Tests",
)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#one-sample-t-test">One-Sample t-Test</a></li>
            <li><a href="#two-sample-t-test">Two-Sample t-Test</a></li>
            <li><a href="#assumptions-of-t-tests-checking-for-normality">Assumptions &amp; Checking for Normality</a></li>
            <li><a href="#activity-quiz-true-false-questions-on-t-tests">Activity Quiz: True/False Questions on t-Tests</a></li>
            <li><a href="#activity-2-conducting-t-tests-and-interpreting-results">Activity 2: Conducting t-Tests and Interpreting Results</a></li>
            <li><a href="#activity-3-interpreting-t-test-results-and-writing-conclusions">Activity 3: Interpreting t-Test Results and Writing Conclusions</a></li>
            <li><a href="#activity-4-coding-t-tests">Activity 4: Coding t-Tests</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_one_sample_t_test():
    st.header("One-Sample t-Test")
    st.write("""
        A one-sample t-test is used to determine if the sample mean is significantly different from a hypothesized population mean.
    """)

    st.subheader("When to use one-sample t-test?")
    st.write("""
        - When you have a single sample and want to compare its mean to a known or hypothesized population mean.
        - The data should be continuous and approximately normally distributed.
    """)
    st.subheader("Hypothesis Testing")
    st.markdown("""
    - **Null Hypothesis** ($H_0$): The sample mean is equal to the population mean.
    - **Alternative Hypothesis** ($H_a$): The sample mean is not equal to the population mean.
    """)

    st.subheader("t-Test Formula")
    st.latex(r"t = \frac{\bar{x} - \mu}{s/\sqrt{n}}")
    st.write("Where:")
    st.markdown(r"""
    - **$\bar{x}$**: Sample mean  
    - **$\mu$**: Hypothesized population mean  
    - **$s$**: Sample standard deviation  
    - **$n$**: Sample size  
    """)
    st.subheader("Degrees of Freedom")
    st.write("Degrees of freedom for a one-sample t-test is calculated as \(df = n - 1\).")
    st.subheader("t-Distribution")
    st.write("""
        The t-distribution is used instead of the normal distribution when the sample size is small (typically \(n < 30\)) or when the population standard deviation is unknown.
        It is similar to the normal distribution but has heavier tails, which allows for more variability in the sample mean.
        As the sample size increases, the t-distribution approaches the normal distribution.
    """)
    st.subheader("Determining Significance")
    st.markdown(r"""
    - Calculate the t-statistic using the formula.
    - Compare the calculated t-statistic to the critical t-value from the t-distribution table based on the chosen significance level ($\alpha$) and degrees of freedom.
    - If the absolute value of the t-statistic is greater than the critical t-value, reject the null hypothesis.
    - Alternatively, calculate the p-value and compare it to the significance level ($\alpha$). If $p < \alpha$, reject the null hypothesis.
    """)

def section_two_sample_t_test():
    st.header("Two-Sample t-Test")
    st.write("""
        A two-sample t-test is used to compare the means of two independent groups. 
        It helps determine whether there is a statistically significant difference between the means of two groups.
    """)
    
    st.subheader("1. Independent Samples t-Test")
    st.write("""
        The independent samples t-test compares the means of two independent groups. 
        The following assumptions must be met:
        - The two samples are independent.
        - Each group is approximately normally distributed.
        - The variances of the two groups are equal (for the equal variances version of the test).
    """)
    st.markdown(r"""
        **Equal Variances Assumed:**
        
        The t-statistic is calculated as:
        
        $$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$
        
        where the pooled standard deviation is:
        
        $$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$
    """, unsafe_allow_html=True)
    
    
    st.subheader("2. Paired Samples t-Test")
    st.write("""
        The paired samples t-test is used when the two groups are related, such as measurements taken before and after an intervention on the same subjects.
        In this test, the differences between paired observations are analyzed.
    """)
    st.markdown(r"""
        The t-statistic for paired tests is calculated as:
        
        $$t = \frac{\bar{d}}{s_d/\sqrt{n}}$$
        
        where:
        - $$\bar{d}$$ is the mean difference between paired observations,
        - $$s_d$$ is the standard deviation of the differences,
        - $$n$$ is the number of pairs.
    """, unsafe_allow_html=True)

def section_assumptions_normality():
    st.header("Assumptions of t-Tests & Checking for Normality")
    st.write("""
        The t-test assumes that:
        - The data are continuous (interval/ratio scale).
        - The data in each group are approximately normally distributed.
        - Groups have equal variances (for the independent t-test with equal variances).
        - Observations are independent.
    """)
    st.write("Example: Check normality using the Shapiro-Wilk test")
    st.code("""
from scipy.stats import shapiro
stat, p = shapiro(data)
if p > 0.05:
    print("Data is normally distributed")
else:
    print("Data is not normally distributed")
    """, language="python")

    st.subheader("Visualizing Normality")
    st.write("Below are example graphs using sample data generated from a normal distribution.")

    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=200)

    # Histogram with KDE
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    sns.histplot(data, kde=True, ax=ax_hist, color='skyblue')
    ax_hist.set_title("Histogram with KDE")
    st.pyplot(fig_hist)

    # Q-Q Plot
    fig_qq, ax_qq = plt.subplots(figsize=(6, 4))
    sm.qqplot(data, line='s', ax=ax_qq)
    ax_qq.set_title("Q-Q Plot")
    st.pyplot(fig_qq)

    st.write("The histogram with KDE and Q-Q plot help visualize the normality of the data. The Q-Q plot should show points along the diagonal line if the data is normally distributed.")
    st.subheader("How to addressing non-normality?")
    st.write("""
        - **Transformations**: Apply transformations (e.g., log, square root) to the data to achieve normality.
        - **Non-parametric tests**: Use non-parametric alternatives (e.g., Mann-Whitney U test, Wilcoxon signed-rank test) that do not assume normality.
        - **Bootstrap methods**: Use resampling techniques to estimate the sampling distribution of the statistic.
        - **Increase sample size**: Larger samples tend to be more normally distributed due to the Central Limit Theorem.
        - **Check for outliers**: Identify and address outliers that may affect normality.     
    """)

def activity_quiz():
    st.header("Activity Quiz: True/False Questions on t-Tests")
    st.write("Answer the following True/False questions based on what you learned about one-sample and two-sample t-tests.")

    with st.form(key="t_test_quiz_tf"):
        q1 = st.radio("1. A one-sample t-test compares the sample mean to a known population mean.", ("True", "False"), index=None)
        q2 = st.radio("2. For a one-sample t-test, the data must be normally distributed.", ("True", "False"), index=None)
        q3 = st.radio("3. The null hypothesis for a one-sample t-test always states that the sample mean is greater than the population mean.", ("True", "False"), index=None)
        q4 = st.radio("4. In a two-sample t-test, the two samples must be independent if using the independent method.", ("True", "False"), index=None)
        q5 = st.radio("5. Welch's t-test is used when the variances of the two groups are assumed to be equal.", ("True", "False"), index=None)
        q6 = st.radio("6. A paired samples t-test is appropriate when comparing measurements taken on the same subjects before and after an intervention.", ("True", "False"), index=None)
        q7 = st.radio("7. The t-distribution has heavier tails than the normal distribution, which allows for more variability with small samples.", ("True", "False"), index=None)
        q8 = st.radio("8. If the calculated p-value is less than the significance level (α), you fail to reject the null hypothesis.", ("True", "False"), index=None)
        q9 = st.radio("9. The degrees of freedom for a one-sample t-test is calculated as n - 1.", ("True", "False"), index=None)
        q10 = st.radio("10. A two-sample t-test can be used to compare the means of both independent and paired samples without any adjustments.", ("True", "False"), index=None)
        
        submit_quiz = st.form_submit_button("Submit Quiz")
    
    if submit_quiz:
        correct_answers = {
            "q1": "True",
            "q2": "True",
            "q3": "False",  # The null hypothesis states that the sample mean equals the population mean.
            "q4": "True",
            "q5": "False",  # Welch's t-test is used when variances are NOT assumed equal.
            "q6": "True",
            "q7": "True",
            "q8": "False",  # p < α means reject the null hypothesis.
            "q9": "True",
            "q10": "False"  # Paired t-test is used for related samples.
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
            "q10": q10,
        }
        
        score = 0
        feedback = []
        for i in range(1, 11):
            key = f"q{i}"
            if user_answers[key] == correct_answers[key]:
                score += 1
                feedback.append(f"✔ {i}. Correct.")
            else:
                feedback.append(f"❌ {i}. Incorrect. The correct answer is: {correct_answers[key]}.")
        
        st.write(f"Your score: {score}/10")
        with st.expander("See detailed feedback"):
            for comment in feedback:
                st.write(comment)

def activity_t_tests():
    st.header("Activity 2: Conducting t-Tests and Interpreting Results")
    st.write("In this activity, you'll work on a scenario to perform a one-sample t-test calculation. Adjust the provided code and run your analysis.")

    st.subheader("Scenario: Evaluating a New Treatment")
    st.write("""
A hospital has introduced a new treatment, and historical data shows that the average recovery time is **10 days**.  
A trial with **30 patients** using the new treatment resulted in:  
- Sample mean (𝑥̄): **9.2 days**  
- Sample standard deviation (s): **1.8 days**

Your task is to perform a one-sample t-test to determine if the new treatment significantly reduces the recovery time.
""")

    st.write("""
**Given Data:**
- **Sample size (n):** 30  
- **Sample mean (𝑥̄):** 9.2 days  
- **Sample standard deviation (s):** 1.8 days  
- **Historical mean (μ):** 10 days  
- **Significance level (α):** 0.05  
""")

    with st.expander("📘 See Formula and Example Code"):
        st.markdown(
    """
    ### One-Sample t-Test Formula
    """)
        st.latex(r"t = \frac{\bar{x} - \mu}{s/\sqrt{n}}")
        st.markdown(
    r"""
    **Where:**
    - **$\bar{x}$:** Sample mean  
    - **$\mu$:** Population mean (under H₀)  
    - **$s$:** Sample standard deviation  
    - **$n$:** Sample size

    **Degrees of Freedom:**
    \[
    df = n - 1
    \]

    ---
    **Example Code:**
    """, unsafe_allow_html=True)

        st.code('''
import math
from scipy import stats

# Given data
n = 30
sample_mean = 9.2
sample_std = 1.8
historical_mean = 10.0
alpha = 0.05

# Calculate t-statistic and p-value
t_stat = (sample_mean - historical_mean) / (sample_std / math.sqrt(n))
df = n - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"t-statistic: {t_stat:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The new treatment significantly reduces recovery time.")
else:
    print("Fail to reject the null hypothesis: No significant reduction in recovery time.")
        ''', language="python")
        code_snippet = '''
import math
from scipy import stats

# Given data
n = 30
sample_mean = 9.2
sample_std = 1.8
historical_mean = 10.0
alpha = 0.05

# Calculate t-statistic and p-value
t_stat = (sample_mean - historical_mean) / (sample_std / math.sqrt(n))
df = n - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"t-statistic: {t_stat:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The new treatment significantly reduces recovery time.")
else:
    print("Fail to reject the null hypothesis: No significant reduction in recovery time.")
'''

    
        @st.dialog("CodePad", width="large")
        def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
            code_editor_for_all(default_code=default_code, key='codepad-week6', warning_text=warning_text)

        # Replace the existing CodePad integration in activity_t_tests
        if st.button("Open in CodePad", key="week5-act-t-tests"):
            codeeditor_popup(default_code=code_snippet, warning_text="Don't forget to save your code before running it!")
        
    st.subheader("Answer the following questions:")
    with st.form(key="t_test_form_week5"):
        sample_mean_input = st.number_input("Enter Sample Mean (days):", step=0.1)
        sample_std_input = st.number_input("Enter Sample Standard Deviation (days):", step=0.1)
        n_input = st.number_input("Enter Sample Size:", step=1)
        historical_mean_input = st.number_input("Enter Historical Mean (days):", step=0.1)
        alpha_input = st.number_input("Enter Significance Level (α):", step=0.01)
        t_stat_input = st.number_input("Enter Calculated t-statistic:", step=0.001)
        p_value_input = st.number_input("Enter Calculated P-value:", step=0.001)
        reject_h0 = st.radio("Do you reject the null hypothesis (H0)?", ("Yes", "No"))
        submit_answers = st.form_submit_button("Submit")

    if submit_answers:
        feedback = []
        correct_count = 0

        # Validate user inputs against expected values
        expected_t_stat = (sample_mean_input - historical_mean_input) / (sample_std_input / math.sqrt(n_input))
        expected_df = n_input - 1
        expected_p_value = 2 * (1 - stats.t.cdf(abs(expected_t_stat), expected_df))

        if not math.isclose(t_stat_input, expected_t_stat, rel_tol=1e-3):
            feedback.append(f"❌ Incorrect t-statistic. Expected: {expected_t_stat:.3f}")
        else:
            feedback.append("✔ Correct t-statistic.")
            correct_count += 1

        if not math.isclose(p_value_input, expected_p_value, rel_tol=1e-3):
            feedback.append(f"❌ Incorrect P-value. Expected: {expected_p_value:.3f}")
        else:
            feedback.append("✔ Correct P-value.")
            correct_count += 1

        correct_decision = "Yes" if expected_p_value < alpha_input else "No"
        if reject_h0 != correct_decision:
            feedback.append(f"❌ Incorrect decision on rejecting H0. Expected: {correct_decision}")
        else:
            feedback.append("✔ Correct decision on rejecting H0.")
            correct_count += 1

        st.subheader("Feedback")
        for comment in feedback:
            st.write(comment)

        st.write(f"You got {correct_count}/3 correct answers.")
    
def activity_three_interpreting_results_mcq():
    st.header("Activity 3: Interpreting t-Test Results and Writing Conclusions")
    st.write("This activity will test your ability to interpret the results of t-tests and choose the appropriate conclusions.")

    st.subheader("Scenario 1: One-Sample t-Test")
    st.write("""
A researcher conducted a study to see if the average systolic blood pressure of patients in a specific demographic is different from the national average of **120 mmHg**. They collected data from a random sample of **40 patients** and performed a one-sample t-test. The results are:
    """)
    st.markdown("""
    - **Sample Mean (𝑥̄):** 125.5 mmHg
    - **Sample Standard Deviation (s):** 10.2 mmHg
    - **Sample Size (n):** 40
    - **Degrees of Freedom (df):** 39
    - **Calculated t-statistic:** 3.43
    - **P-value (two-tailed):** 0.0015
    - **Significance Level (α):** 0.05
    """)

    st.subheader("Questions:")
    with st.form(key="interpret_t_test_1_mcq"):
        st.write("1. Based on the p-value and the significance level, should you reject or fail to reject the null hypothesis?")
        q1_choice = st.radio("Question 1", 
                            options=["Reject the null hypothesis", "Fail to reject the null hypothesis"],
                            index=None, 
                            label_visibility='hidden')

        st.write("2. What does the null hypothesis state in this scenario?")
        q2_choice = st.radio("Question 2", 
                            options=[
                                "The average systolic blood pressure of patients in this demographic is greater than 120 mmHg.",
                                "The average systolic blood pressure of patients in this demographic is less than 120 mmHg.",
                                "The average systolic blood pressure of patients in this demographic is equal to 120 mmHg.",
                                "The sample mean is equal to the population standard deviation."
                            ],
                            index=None, 
                            label_visibility='hidden')

        st.write("3. Choose the most appropriate conclusion based on the results of this t-test:")
        q3_choice = st.selectbox("Question 3", 
                                options=[
                                    "The average systolic blood pressure in this demographic is not significantly different from the national average.",
                                    "The average systolic blood pressure in this demographic is significantly higher than the national average of 120 mmHg.",
                                    "The sample size was too small to make a conclusion.",
                                    "The standard deviation is significantly different from the mean."
                                ],
                                index=None,
                                label_visibility='hidden')

        submit_interpretation_1 = st.form_submit_button("Submit Answers for Scenario 1")

    if submit_interpretation_1:
        correct_q1 = "Reject the null hypothesis"
        correct_q2 = "The average systolic blood pressure of patients in this demographic is equal to 120 mmHg."
        correct_q3 = "The average systolic blood pressure in this demographic is significantly higher than the national average of 120 mmHg."

        feedback_1 = []
        if q1_choice == correct_q1:
            feedback_1.append("✔ Question 1: Correct!")
        else:
            feedback_1.append(f"❌ Question 1: Incorrect. The p-value (0.0015) is less than the significance level (0.05), so you should **{correct_q1}**.")

        if q2_choice == correct_q2:
            feedback_1.append("✔ Question 2: Correct!")
        else:
            feedback_1.append(f"❌ Question 2: Incorrect. The null hypothesis states: **{correct_q2}**.")

        if q3_choice == correct_q3:
            feedback_1.append("✔ Question 3: Correct!")
        else:
            feedback_1.append(f"❌ Question 3: Incorrect. The correct conclusion is: **{correct_q3}**.")

        st.subheader("Feedback for Scenario 1:")
        for fb in feedback_1:
            st.write(fb)
        st.divider()

    st.subheader("Scenario 2: Two-Sample Independent t-Test")
    st.write("""
A clinical trial compared a new drug to a standard drug for reducing cholesterol levels. Two independent groups of patients were given either the new drug or the standard drug. After 8 weeks, the change in LDL cholesterol levels was measured. The results of an independent samples t-test are:
    """)
    st.markdown("""
    - **Group 1 (New Drug):** Sample Mean Change = -25.3 mg/dL, Sample Size = 35, Sample Standard Deviation = 8.5 mg/dL
    - **Group 2 (Standard Drug):** Sample Mean Change = -21.8 mg/dL, Sample Size = 40, Sample Standard Deviation = 9.2 mg/dL
    - **Degrees of Freedom (df):** 73
    - **Calculated t-statistic:** -1.85
    - **P-value (two-tailed):** 0.068
    - **Significance Level (α):** 0.05
    """)

    st.subheader("Questions:")
    with st.form(key="interpret_t_test_2_mcq"):
        st.write("1. Based on the p-value and the significance level, should you reject or fail to reject the null hypothesis?")
        q4_choice = st.radio("Question 1", 
                            options=["Reject the null hypothesis", "Fail to reject the null hypothesis"],
                            index=None, 
                            label_visibility='hidden')

        st.write("2. What does the alternative hypothesis suggest in this scenario (for a two-tailed test)?")
        q5_choice = st.radio("Question 2", 
                            options=[
                                "The new drug reduces LDL cholesterol levels more than the standard drug.",
                                "The standard drug reduces LDL cholesterol levels more than the new drug.",
                                "The new drug has a different effect on reducing LDL cholesterol levels compared to the standard drug.",
                                "There is no difference in the effect of the two drugs."
                            ],
                            index=None, 
                            label_visibility='hidden')

        st.write("3. Choose the most appropriate conclusion based on the results of this t-test:")
        q6_choice = st.selectbox("", (
            "There is a statistically significant difference in the effect of the new drug and the standard drug on LDL cholesterol levels.",
            "There is no statistically significant difference in the effect of the new drug and the standard drug on LDL cholesterol levels.",
            "The new drug is clearly superior to the standard drug.",
            "More research is needed to compare the two drugs."
        ), index=None)

        submit_interpretation_2 = st.form_submit_button("Submit Answers for Scenario 2")

    if submit_interpretation_2:
        correct_q4 = "Fail to reject the null hypothesis"
        correct_q5 = "The new drug has a different effect on reducing LDL cholesterol levels compared to the standard drug."
        correct_q6 = "There is no statistically significant difference in the effect of the new drug and the standard drug on LDL cholesterol levels."

        feedback_2 = []
        if q4_choice == correct_q4:
            feedback_2.append("✔ Question 1: Correct!")
        else:
            feedback_2.append(f"❌ Question 1: Incorrect. The p-value (0.068) is greater than the significance level (0.05), so you should **{correct_q4}**.")

        if q5_choice == correct_q5:
            feedback_2.append("✔ Question 2: Correct!")
        else:
            feedback_2.append(f"❌ Question 2: Incorrect. The alternative hypothesis suggests: **{correct_q5}**.")

        if q6_choice == correct_q6:
            feedback_2.append("✔ Question 3: Correct!")
        else:
            feedback_2.append(f"❌ Question 3: Incorrect. The correct conclusion is: **{correct_q6}**.")

        st.subheader("Feedback for Scenario 2:")
        for fb in feedback_2:
            st.write(fb)
        st.divider()

def activity_four_coding_t_test_tabs_corrected():
    st.header("Activity 4: Coding t-Tests")
    st.write("In this activity, you will write Python code to perform different types of t-tests and interpret the results.")

    tab1, tab2, tab3 = st.tabs(["One-Sample t-Test", "Independent Samples t-Test", "Paired Samples t-Test (Optional)"])

    with tab1:
        st.subheader("Scenario: One-Sample t-Test - Evaluating a New Drug")
        st.write("""
A pharmaceutical company has developed a new drug to lower blood pressure. Historical data shows the average systolic blood pressure of patients with hypertension is **140 mmHg**. A clinical trial with **35 patients** using the new drug resulted in an average systolic blood pressure of **135 mmHg** with a standard deviation of **8 mmHg**. Use a significance level (α) of 0.05 to test if the new drug significantly lowers blood pressure.
        """)
        example_code_tab1 = """
from scipy import stats
import numpy as np

sample_mean = 135
population_mean = 140
sample_std = 8
n = 35
alpha = 0.05

# Calculate the t-statistic and p-value
t_statistic, p_value = stats.ttest_1samp(a=np.repeat(sample_mean, n), popmean=population_mean) # Using np.repeat to simulate sample data

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The new drug significantly lowers blood pressure.")
else:
    print("Fail to reject the null hypothesis: The new drug does not significantly lower blood pressure.")
        """
        @st.dialog("CodePad", width="large")
        def codeeditor_popup_tab1(default_code=None, key_suffix="tab1", warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
            code_editor_for_all(default_code=default_code, key=f'codepad-week6-act4-{key_suffix}', warning_text=warning_text)

        if st.button("Open Code Editor for One-Sample t-Test"):
            codeeditor_popup_tab1(default_code=example_code_tab1)

        # Calculate the actual solution for Tab 1
        sample_mean = 135
        population_mean = 140
        sample_std = 8
        n = 35
        alpha = 0.05
        
        # Use one-sample t-test formula to calculate t-statistic
        t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
        df = n - 1
        # Calculate p-value (two-tailed)
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        
        correct_conclusion_tab1 = "Yes" if p_val < alpha else "No"

        st.subheader("Check Your Understanding")
        with st.form(key="check_coding_t_test_tab1"):
            conclusion_choice_tab1 = st.radio("Based on the p-value, does the new drug significantly lower blood pressure?", ("Yes", "No"), index=None)
            p_value_answer_tab1 = st.number_input("Approximate p-value (round to three decimal places):", step=0.001, format="%.3f")
            submit_tab1 = st.form_submit_button("Submit Answers for One-Sample t-Test")

        if submit_tab1:
            feedback_tab1 = []
            if conclusion_choice_tab1 == correct_conclusion_tab1:
                feedback_tab1.append("✔ Correct!")
            else:
                feedback_tab1.append(f"❌ Incorrect. The p-value ({p_val:.3f}) is {'less' if p_val < alpha else 'greater'} than 0.05.")
            if abs(p_value_answer_tab1 - p_val) < 0.005:
                feedback_tab1.append("✔ Correct (approximately)!")
            else:
                feedback_tab1.append(f"❌ Incorrect. Expected p-value is approximately {p_val:.3f}.")
            st.write("Feedback:")
            for fb in feedback_tab1:
                st.write(fb)

    with tab2:
        st.subheader("Scenario: Independent Samples t-Test - Comparing Two Diets on Weight Loss")
        st.write("""
A researcher is investigating the effectiveness of two different diets (Diet A and Diet B) on weight loss. The data is as follows:
        """)
        st.markdown("""
        **Diet A Weight Loss (kg):** `[2.1, 3.5, 1.8, 2.9, 4.0, 2.5, 3.2]`
        **Diet B Weight Loss (kg):** `[1.5, 0.8, 1.2, 2.0, 1.0, 1.7]`
        """)
        example_code_tab2 = """
from scipy import stats
import numpy as np

diet_a = np.array([2.1, 3.5, 1.8, 2.9, 4.0, 2.5, 3.2])
diet_b = np.array([1.5, 0.8, 1.2, 2.0, 1.0, 1.7])
alpha = 0.05

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(diet_a, diet_b, equal_var=True)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in average weight loss between Diet A and Diet B.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average weight loss between Diet A and Diet B.")
        """
        @st.dialog("CodePad", width="large")
        def codeeditor_popup_tab2(default_code=None, key_suffix="tab2", warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
            code_editor_for_all(default_code=default_code, key=f'codepad-week6-act4-{key_suffix}', warning_text=warning_text)

        if st.button("Open Code Editor for Independent Samples t-Test"):
            codeeditor_popup_tab2(default_code=example_code_tab2)

        # Calculate the actual solution for Tab 2
        diet_a = np.array([2.1, 3.5, 1.8, 2.9, 4.0, 2.5, 3.2])
        diet_b = np.array([1.5, 0.8, 1.2, 2.0, 1.0, 1.7])
        alpha = 0.05

        # Perform independent samples t-test
        t_stat_tab2, p_val_tab2 = stats.ttest_ind(diet_a, diet_b, equal_var=True)
        correct_conclusion_tab2 = "Yes" if p_val_tab2 < alpha else "No"

        st.subheader("Check Your Understanding")
        with st.form(key="check_coding_t_test_tab2"):
            conclusion_choice_tab2 = st.radio("Based on the p-value, is there a significant difference in weight loss between Diet A and Diet B?", ("Yes", "No"), index=None)
            p_value_answer_tab2 = st.number_input("Approximate p-value (round to three decimal places):", step=0.001, format="%.3f")
            submit_tab2 = st.form_submit_button("Submit Answers for Independent Samples t-Test")

        if submit_tab2:
            feedback_tab2 = []
            if conclusion_choice_tab2 == correct_conclusion_tab2:
                feedback_tab2.append("✔ Correct!")
            else:
                feedback_tab2.append(f"❌ Incorrect. The p-value ({p_val_tab2:.3f}) is {'less' if p_val_tab2 < alpha else 'greater'} than 0.05.")
            if abs(p_value_answer_tab2 - p_val_tab2) < 0.005:
                feedback_tab2.append("✔ Correct (approximately)!")
            else:
                feedback_tab2.append(f"❌ Incorrect. Expected p-value is approximately {p_val_tab2:.3f}.")
            st.write("Feedback:")
            for fb in feedback_tab2:
                st.write(fb)

    with tab3:
        st.subheader("Scenario: Paired Samples t-Test (Optional) - Effect of Exercise on Heart Rate")
        st.write("""
A researcher wants to study the effect of a specific exercise program on resting heart rate. They measured the resting heart rate of 20 participants before and after participating in the program for one month. The data is as follows (heart rate in beats per minute):
        """)
        st.markdown("""
        **Heart Rate Before:** `[72, 75, 68, 80, 71, 78, 70, 74, 69, 76, 73, 77, 67, 79, 72, 74, 68, 75, 71, 78]`
        **Heart Rate After:** `[68, 70, 65, 75, 68, 72, 67, 70, 66, 73, 70, 74, 64, 76, 69, 71, 65, 72, 68, 75]`
        """)
        example_code_tab3 = """
from scipy import stats
import numpy as np

before = np.array([72, 75, 68, 80, 71, 78, 70, 74, 69, 76, 73, 77, 67, 79, 72, 74, 68, 75, 71, 78])
after = np.array([68, 70, 65, 75, 68, 72, 67, 70, 66, 73, 70, 74, 64, 76, 69, 71, 65, 72, 68, 75])
alpha = 0.05

# Perform paired samples t-test
t_statistic, p_value = stats.ttest_rel(before, after)

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The exercise program significantly reduced resting heart rate.")
else:
    print("Fail to reject the null hypothesis: The exercise program did not significantly reduce resting heart rate.")
        """
        @st.dialog("CodePad", width="large")
        def codeeditor_popup_tab3(default_code=None, key_suffix="tab3", warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
            code_editor_for_all(default_code=default_code, key=f'codepad-week6-act4-{key_suffix}', warning_text=warning_text)

        if st.button("Open Code Editor for Paired Samples t-Test"):
            codeeditor_popup_tab3(default_code=example_code_tab3)

        # Calculate the actual solution for Tab 3
        before = np.array([72, 75, 68, 80, 71, 78, 70, 74, 69, 76, 73, 77, 67, 79, 72, 74, 68, 75, 71, 78])
        after = np.array([68, 70, 65, 75, 68, 72, 67, 70, 66, 73, 70, 74, 64, 76, 69, 71, 65, 72, 68, 75])
        alpha = 0.05

        # Perform paired samples t-test
        t_stat_tab3, p_val_tab3 = stats.ttest_rel(before, after)
        correct_conclusion_tab3 = "Yes" if p_val_tab3 < alpha else "No"

        st.subheader("Check Your Understanding")
        with st.form(key="check_coding_t_test_tab3"):
            conclusion_choice_tab3 = st.radio("Based on the p-value, did the exercise program significantly reduce resting heart rate?", ("Yes", "No"), index=None)
            p_value_answer_tab3 = st.number_input("Approximate p-value (round to three decimal places):", step=0.001, format="%.3f")
            submit_tab3 = st.form_submit_button("Submit Answers for Paired Samples t-Test")

        if submit_tab3:
            feedback_tab3 = []
            if conclusion_choice_tab3 == correct_conclusion_tab3:
                feedback_tab3.append("✔ Correct!")
            else:
                feedback_tab3.append(f"❌ Incorrect. The p-value ({p_val_tab3:.3f}) is {'less' if p_val_tab3 < alpha else 'greater'} than 0.05.")
            if abs(p_value_answer_tab3 - p_val_tab3) < 0.005:
                feedback_tab3.append("✔ Correct (approximately)!")
            else:
                feedback_tab3.append(f"❌ Incorrect. Expected p-value is approximately {p_val_tab3:.3f}.")
            st.write("Feedback:")
            for fb in feedback_tab3:
                st.write(fb)

def main():
    tw.initialize_tailwind()
    Navbar()
    st.title("Week 06 | One-Sample and Two-Sample t-Tests")
    section_table_of_contents()
    section_one_sample_t_test()
    section_two_sample_t_test()
    section_assumptions_normality()
    st.divider()
    activity_quiz()
    activity_t_tests()
    activity_three_interpreting_results_mcq()
    activity_four_coding_t_test_tabs_corrected()
    Footer(6)

if __name__== "__main__":
    main()