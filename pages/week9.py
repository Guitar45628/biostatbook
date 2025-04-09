import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from modules.nav import Navbar
from modules.foot import Footer
from scipy import stats
import st_tailwind as tw
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 09 | Analysis of Variance (ANOVA)",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code, key='codepad-week9', warning_text=warning_text)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#introduction-to-anova">Introduction to ANOVA</a></li>
            <li><a href="#one-way-anova">One-Way ANOVA</a></li>
            <li><a href="#assumptions-of-anova">Assumptions of ANOVA</a></li>
            <li><a href="#post-hoc-tests">Post-hoc Tests</a></li>
            <li><a href="#activities">Activities</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_introduction_to_anova():
    st.header("Introduction to ANOVA")
    st.markdown("""
        Analysis of Variance (ANOVA) is a statistical method used to compare means of three or more groups. It extends the t-test
        which is used to compare means between two groups.
    """)
    
    st.subheader("Why ANOVA?")
    st.write("""
        When we need to compare means across multiple groups, conducting multiple t-tests would increase the probability of 
        Type I error (false positive). ANOVA solves this problem by conducting a single test to determine if there are any 
        significant differences between the groups.
    """)
    
    st.subheader("How ANOVA Works")
    st.write("""
        ANOVA compares the variance between groups (between-group variance) to the variance within groups (within-group variance).
        If the between-group variance is significantly larger than the within-group variance, we can conclude that there are 
        significant differences between at least some of the groups.
    """)
    
    st.info("""
        The F-statistic in ANOVA is the ratio of between-group variance to within-group variance. A large F-value suggests 
        significant differences between groups.
    """)
    
    # Visual representation of ANOVA
    st.subheader("Visual Representation")
    
    # Creating sample data for visualization
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 30)
    group2 = np.random.normal(13, 2, 30)
    group3 = np.random.normal(8, 2, 30)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplots to visualize the data
    data = [group1, group2, group3]
    ax.boxplot(data)
    ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3'])
    ax.set_ylabel('Value')
    ax.set_title('Example of Data for ANOVA Analysis')
    
    st.pyplot(fig)
    
    st.write("""
        In the figure above, we can visually see differences between the groups. ANOVA helps us determine if these 
        differences are statistically significant.
    """)

def section_one_way_anova():
    st.header("One-Way ANOVA")
    
    st.write("""
        One-way ANOVA is used when we have a single categorical independent variable (factor) with three or more levels 
        and a continuous dependent variable.
    """)
    
    st.subheader("Hypotheses")
    st.markdown("""
        - **Null Hypothesis (H₀):** All group means are equal (μ₁ = μ₂ = μ₃ = ... = μₖ)
        - **Alternative Hypothesis (H₁):** At least one group mean is different from the others
    """)
    
    st.subheader("ANOVA Table Interpretation")
    st.write("""
        The ANOVA table provides the following information:
        
        1. **Sum of Squares (SS):** 
           - **Between-groups SS:** Variation due to differences between group means
           - **Within-groups SS:** Variation due to differences within each group
           - **Total SS:** Total variation in the data
        
        2. **Degrees of Freedom (df):** 
           - **Between-groups df:** Number of groups - 1
           - **Within-groups df:** Total number of observations - number of groups
           
        3. **Mean Square (MS):** SS divided by df
        
        4. **F-statistic:** Ratio of between-groups MS to within-groups MS
        
        5. **p-value:** Probability of observing the F-statistic or a more extreme value if the null hypothesis is true
    """)
    
    st.info("""
        If the p-value is less than our significance level (typically 0.05), we reject the null hypothesis 
        and conclude that at least one group mean is different.
    """)
    
    # Example code for one-way ANOVA
    st.subheader("Example Code for One-Way ANOVA")
    
    example_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Create sample data
np.random.seed(42)
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(13, 2, 30)
group3 = np.random.normal(8, 2, 30)

# Organize data for analysis
data = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['Group 1'] * 30 + ['Group 2'] * 30 + ['Group 3'] * 30
})

# Visualize the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='value', data=data)
plt.title('Distribution of Values by Group')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()

# One-way ANOVA using scipy
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# One-way ANOVA using statsmodels
model = ols('value ~ C(group)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\\nANOVA Table:")
print(anova_table)
"""
    
    st.code(example_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (One-Way ANOVA)", key='week9-codepad-one-way', use_container_width=True):
            codeeditor_popup(example_code)

def section_assumptions_of_anova():
    st.header("Assumptions of ANOVA")
    
    st.write("""
        For ANOVA results to be valid, several assumptions must be met:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Independence")
        st.write("""
            Observations in each group should be independent of observations in other groups 
            and within the same group.
        """)
        
        st.subheader("2. Normality")
        st.write("""
            The data within each group should be approximately normally distributed. 
            This can be checked using:
            - Histograms
            - Q-Q plots
            - Formal tests like Shapiro-Wilk test
        """)
    
    with col2:
        st.subheader("3. Homogeneity of Variance")
        st.write("""
            The variance within each group should be approximately equal. 
            This can be checked using:
            - Levene's test
            - Bartlett's test
            - Box plots
        """)
    
    st.subheader("Checking Assumptions - Example Code")
    
    assumption_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Sample data (already created in previous example)
np.random.seed(42)
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(13, 2, 30)
group3 = np.random.normal(8, 2, 30)

data = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['Group 1'] * 30 + ['Group 2'] * 30 + ['Group 3'] * 30
})

# 1. Check Normality
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
group_data = [group1, group2, group3]
group_names = ['Group 1', 'Group 2', 'Group 3']

for i, (group, name) in enumerate(zip(group_data, group_names)):
    # Q-Q plot
    stats.probplot(group, plot=axes[i])
    axes[i].set_title(f'Q-Q Plot: {name}')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
print("Shapiro-Wilk Test for Normality:")
for i, (group, name) in enumerate(zip(group_data, group_names)):
    stat, p_value = stats.shapiro(group)
    print(f"{name}: W = {stat:.4f}, p = {p_value:.4f}")
    if p_value > 0.05:
        print(f"  ✓ {name} is approximately normal (fail to reject H0)")
    else:
        print(f"  ✗ {name} deviates from normality (reject H0)")

# 2. Check Homogeneity of Variance
# Levene's test
stat, p_value = stats.levene(group1, group2, group3)
print("\\nLevene's Test for Homogeneity of Variance:")
print(f"W = {stat:.4f}, p = {p_value:.4f}")
if p_value > 0.05:
    print("  ✓ Variances are homogeneous (fail to reject H0)")
else:
    print("  ✗ Variances are not homogeneous (reject H0)")

# Visualize variance with boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='value', data=data)
plt.title('Boxplots to Check Homogeneity of Variance')
plt.show()
"""
    
    st.code(assumption_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (ANOVA Assumptions)", key='week9-codepad-assumptions', use_container_width=True):
            codeeditor_popup(assumption_code)
    
    st.subheader("What If Assumptions Are Violated?")
    
    st.write("""
        If ANOVA assumptions are violated, you can consider:
        
        1. **Data Transformation:** Apply transformations like log, square root, or Box-Cox to make the data closer to normal.
        
        2. **Non-parametric Alternatives:** Use Kruskal-Wallis test as a non-parametric alternative when normality is violated.
        
        3. **Welch's ANOVA:** Use when homogeneity of variance is violated. It's more robust to unequal variances.
    """)
    
    # Example code for alternatives
    st.code("""
# Kruskal-Wallis test (non-parametric alternative)
stat, p_value = stats.kruskal(group1, group2, group3)
print(f"Kruskal-Wallis H = {stat:.4f}, p = {p_value:.4f}")

# Welch's ANOVA (for unequal variances)
from scipy.stats import f_oneway
# Implementation can be done using statsmodels or pingouin packages
""", language="python")

def section_post_hoc_tests():
    st.header("Post-hoc Tests")
    
    st.write("""
        When ANOVA indicates significant differences between groups, post-hoc tests are used to determine 
        which specific groups differ from each other. These tests adjust for multiple comparisons to control 
        the familywise error rate.
    """)
    
    st.subheader("Common Post-hoc Tests")
    
    st.markdown("""
        - **Tukey's Honestly Significant Difference (HSD)**: Most common, compares all possible pairs of groups
        - **Bonferroni Correction**: Controls Type I error by adjusting the significance level
        - **Scheffé's Method**: Conservative method that allows complex comparisons
        - **Duncan's New Multiple Range Test**: Less conservative than Tukey's
        - **Newman-Keuls Method**: Significance levels vary depending on the step
    """)
    
    st.subheader("Tukey's HSD - Example Code")
    
    tukey_code = """
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sample data (continuing from previous examples)
np.random.seed(42)
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(13, 2, 30)
group3 = np.random.normal(8, 2, 30)

data = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['Group 1'] * 30 + ['Group 2'] * 30 + ['Group 3'] * 30
})

# First, run one-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_value:.4f}")

# If ANOVA is significant, proceed with Tukey's HSD
if p_value < 0.05:
    print("\\nANOVA is significant, proceeding with Tukey's HSD test")
    
    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(data['value'], data['group'], alpha=0.05)
    print("\\nTukey's HSD Results:")
    print(tukey_results)
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=data)
    plt.title('Group Comparisons with Tukey HSD Results')
    
    # Add significance annotations (simplified)
    # In a real application, you would extract and add the significant differences
    y_max = data['value'].max() + 1
    plt.plot([0, 1], [y_max, y_max], 'k-')
    plt.text(0.5, y_max + 0.2, '***', ha='center')
    
    plt.plot([1, 2], [y_max - 0.5, y_max - 0.5], 'k-')
    plt.text(1.5, y_max - 0.3, '***', ha='center')
    
    plt.plot([0, 2], [y_max - 1, y_max - 1], 'k-')
    plt.text(1, y_max - 0.8, '***', ha='center')
    
    plt.tight_layout()
    plt.show()
else:
    print("\\nANOVA is not significant, no need for post-hoc tests")
"""
    
    st.code(tukey_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Tukey's HSD)", key='week9-codepad-tukey', use_container_width=True):
            codeeditor_popup(tukey_code)
    
    st.subheader("Interpreting Post-hoc Results")
    
    st.write("""
        When interpreting the results of a post-hoc test like Tukey's HSD:
        
        1. Look at the p-value for each pairwise comparison.
        
        2. If the p-value is less than alpha (typically 0.05), the group means are significantly different.
        
        3. The confidence interval should not contain zero for significantly different group means.
        
        4. The 'reject' column in the Tukey results indicates whether the null hypothesis 
           (that the group means are equal) is rejected.
    """)

def activity_quiz():
    st.header("Activity 1: ANOVA Quiz")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # Reusing image from week 5
    with cent_co:
        st.write("Test your knowledge about Analysis of Variance (ANOVA) with this quiz.")
    
    with st.form(key='anova_quiz'):
        st.subheader("Please answer the following questions about ANOVA:")

        q1 = st.radio("1. What is the main purpose of ANOVA?",
                       ("To compare means between two groups",
                        "To compare means across three or more groups",
                        "To analyze correlation between variables",
                        "To test for normality in data distributions"), index=None)

        q2 = st.radio("2. Which test statistic is used in ANOVA?",
                       ("t-statistic",
                        "F-statistic",
                        "Chi-square statistic",
                        "z-statistic"), index=None)

        q3 = st.radio("3. Which of the following is NOT an assumption of ANOVA?",
                       ("Normality of data within each group",
                        "Homogeneity of variance across groups",
                        "Independence of observations",
                        "Equal sample sizes across all groups"), index=None)

        q4 = st.radio("4. In a one-way ANOVA, what does the null hypothesis state?",
                       ("At least one group mean is different from the others",
                        "All group means are different from each other",
                        "All group means are equal",
                        "The groups have equal variances"), index=None)

        q5 = st.radio("5. What is a post-hoc test used for in ANOVA?",
                       ("To check for assumption violations",
                        "To determine which specific groups differ from each other",
                        "To calculate the overall F-statistic",
                        "To transform non-normal data"), index=None)

        q6 = st.radio("6. What does a large F-value in ANOVA indicate?",
                       ("Smaller variance within groups relative to variance between groups",
                        "Larger variance within groups relative to variance between groups",
                        "Equal variances between all groups",
                        "Equal means across all groups"), index=None)

        q7 = st.radio("7. If ANOVA's assumption of homogeneity of variance is violated, which alternative test could be used?",
                       ("Paired t-test",
                        "Chi-square test",
                        "Welch's ANOVA",
                        "McNemar's test"), index=None)

        q8 = st.radio("8. What is the degrees of freedom (df) for the between-groups variance in a one-way ANOVA with 4 groups?",
                       ("3",
                        "4",
                        "n-4",
                        "n-1"), index=None)

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        score = 0
        correct_answers = {
            "q1": "To compare means across three or more groups",
            "q2": "F-statistic",
            "q3": "Equal sample sizes across all groups",
            "q4": "All group means are equal",
            "q5": "To determine which specific groups differ from each other",
            "q6": "Smaller variance within groups relative to variance between groups",
            "q7": "Welch's ANOVA",
            "q8": "3"
        }

        user_answers = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5,
            "q6": q6,
            "q7": q7,
            "q8": q8
        }

        feedback = []
        
        for i, key in enumerate(correct_answers, 1):
            if user_answers[key] == correct_answers[key]:
                score += 1
                feedback.append(f"✔ {i}. Correct answer. The correct answer is: {correct_answers[key]}")
            else:
                feedback.append(f"❌ {i}. Wrong answer. The correct answer is: {correct_answers[key]}")

        st.write(f"Your score: {score}/8")
        
        with st.expander("See correct answers"):
            for answer in feedback:
                st.write(answer)

def activity_performing_anova():
    st.header("Activity 2: Performing ANOVA with Lung Cancer Data")
    
    st.write("""
        In this activity, you will practice performing one-way ANOVA analysis using a lung cancer dataset. 
        The dataset contains information about different treatment groups and their effects on tumor size.
    """)
    
    # Load the lung cancer dataset
    lung_data_path = os.path.join(os.getcwd(), "data/lung_cancer.csv")
    
    # Check if file exists
    try:
        lung_data = pd.read_csv(lung_data_path)
        st.write("Successfully loaded the lung cancer dataset:")
        st.dataframe(lung_data)
    except FileNotFoundError:
        # If file doesn't exist, create sample data
        st.warning("The lung_cancer.csv file was not found. Creating sample data for this activity.")
        np.random.seed(42)
        treatment_a = np.random.normal(25, 5, 30)
        treatment_b = np.random.normal(22, 4, 30)
        treatment_c = np.random.normal(30, 6, 30)
        placebo = np.random.normal(35, 7, 30)
        
        lung_data = pd.DataFrame({
            'tumor_size': np.concatenate([treatment_a, treatment_b, treatment_c, placebo]),
            'treatment_group': ['Treatment A'] * 30 + ['Treatment B'] * 30 + ['Treatment C'] * 30 + ['Placebo'] * 30,
            'patient_age': np.random.randint(40, 75, 120),
            'smoking_status': np.random.choice(['Current', 'Former', 'Never'], 120)
        })
        st.dataframe(lung_data)
        
    st.subheader("Your Task")
    
    st.write("""
        Perform a one-way ANOVA to determine if there are significant differences in tumor size across the different treatment groups.
        Follow these steps:
        
        1. Explore the data visually
        2. Check ANOVA assumptions
        3. Perform one-way ANOVA
        4. If significant, conduct post-hoc tests
        5. Interpret the results
    """)
    
    # Provide sample code
    sample_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the lung cancer data
# In actual code, change this path to the correct location
lung_data = pd.read_csv("data/lung_cancer.csv")

# Step 1: Explore the data visually
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='treatment_group', y='tumor_size', data=lung_data)
ax.set_title('Tumor Size by Treatment Group')
ax.set_xlabel('Treatment Group')
ax.set_ylabel('Tumor Size (mm)')
plt.show()

# Step 2: Check ANOVA assumptions
# Create separate groups for testing
groups = lung_data.groupby('treatment_group')['tumor_size'].apply(list)

# Check normality for each group
print("Shapiro-Wilk Test for Normality:")
for name, group in groups.items():
    stat, p_value = stats.shapiro(group)
    print(f"{name}: W = {stat:.4f}, p = {p_value:.4f}")

# Check homogeneity of variance
group_lists = [groups[group_name] for group_name in groups.index]
stat, p_value = stats.levene(*group_lists)
print(f"\\nLevene's Test: W = {stat:.4f}, p = {p_value:.4f}")

# Step 3: Perform one-way ANOVA
model = ols('tumor_size ~ C(treatment_group)', data=lung_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\\nANOVA Table:")
print(anova_table)

# Step 4: Post-hoc tests (if ANOVA is significant)
if anova_table['PR(>F)'][0] < 0.05:
    print("\\nPerforming Tukey's HSD post-hoc test:")
    tukey = pairwise_tukeyhsd(endog=lung_data['tumor_size'],
                             groups=lung_data['treatment_group'],
                             alpha=0.05)
    print(tukey)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='treatment_group', y='tumor_size', data=lung_data)
    
    # Add title and labels
    ax.set_title('Tumor Size by Treatment Group with Significant Differences')
    ax.set_xlabel('Treatment Group')
    ax.set_ylabel('Tumor Size (mm)')
    
    # Show the plot
    plt.show()
else:
    print("\\nNo significant differences found between treatment groups.")

# Step 5: Interpret the results - would be written in text or comments
"""
    
    st.code(sample_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (ANOVA Practice)", key='week9-codepad-practice', use_container_width=True):
            codeeditor_popup(sample_code)
    
    st.subheader("Interpretation Guide")
    
    st.write("""
        After running the analysis, here's how to interpret the results:
        
        1. **ANOVA p-value < 0.05**: This indicates that there are significant differences between at least some of the treatment groups.
        
        2. **Post-hoc results**: Look at the Tukey's HSD output to determine which specific groups are different from each other.
            - If the p-adj value for a pair is < 0.05, those two groups are significantly different.
            - If the confidence interval doesn't contain zero, the difference is significant.
        
        3. **Clinical interpretation**: Consider what the statistical differences mean in the context of cancer treatment:
            - Which treatment showed the smallest tumor size?
            - How much better is the best treatment compared to placebo?
            - Are the differences large enough to be clinically meaningful?
    """)
    
    # Challenge exercise
    with st.expander("Challenge Exercise"):
        st.write("""
            Extend your analysis by:
            
            1. Creating a bar chart showing mean tumor size with error bars
            
            2. Adding patient age as a covariate in your analysis (using ANCOVA)
            
            3. Creating a grouped analysis to see if treatment effects differ by smoking status
            
            Sample code to get you started:
        """)
        
        st.code("""
# Mean tumor size with error bars
plt.figure(figsize=(10, 6))
sns.barplot(x='treatment_group', y='tumor_size', data=lung_data, ci=95)
plt.title('Mean Tumor Size by Treatment Group (with 95% CI)')
plt.xlabel('Treatment Group')
plt.ylabel('Mean Tumor Size (mm)')
plt.show()

# ANCOVA with age as a covariate
model_ancova = ols('tumor_size ~ C(treatment_group) + patient_age', data=lung_data).fit()
ancova_table = sm.stats.anova_lm(model_ancova, typ=2)
print("ANCOVA Table (with age as covariate):")
print(ancova_table)

# Grouped analysis by smoking status
plt.figure(figsize=(12, 8))
sns.boxplot(x='treatment_group', y='tumor_size', hue='smoking_status', data=lung_data)
plt.title('Tumor Size by Treatment Group and Smoking Status')
plt.xlabel('Treatment Group')
plt.ylabel('Tumor Size (mm)')
plt.legend(title='Smoking Status')
plt.show()
""", language="python")

def main():
    # Initialize Tailwind CSS if needed
    try:
        tw.initialize_tailwind()
    except:
        pass
    
    # Navbar
    Navbar()
    
    # Title
    st.title("Week 09 | Analysis of Variance (ANOVA)")
    
    # Table of contents
    section_table_of_contents()
    
    # Main content sections
    section_introduction_to_anova()
    st.divider()
    section_one_way_anova()
    st.divider()
    section_assumptions_of_anova()
    st.divider()
    section_post_hoc_tests()
    st.divider()
    
    # Activities
    st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
    st.header("Activities")
    activity_quiz()
    st.divider()
    activity_performing_anova()
    
    # Footer
    Footer(9)

if __name__ == "__main__":
    main()