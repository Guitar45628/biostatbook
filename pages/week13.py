import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 13 | Advanced Techniques and Scientific Communication",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code, key='codepad-week13', warning_text=warning_text)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#advanced-techniques-in-biostatistics">Advanced Techniques in Biostatistics</a>
                <ul>
                    <li><a href="#multiple-regression">Multiple Regression</a></li>
                    <li><a href="#survival-analysis">Survival Analysis</a></li>
                    <li><a href="#non-parametric-tests">Non-parametric Tests</a></li>
                </ul>
            </li>
            <li><a href="#reading-and-interpreting-statistical-results">Reading and Interpreting Statistical Results in Scientific Papers</a></li>
            <li><a href="#communicating-statistical-findings">Communicating Statistical Findings Effectively</a></li>
            <li><a href="#activities">Activities</a>
                <ul>
                    <li><a href="#activity-1-critiquing-a-scientific-paper">Activity 1: Critiquing a Scientific Paper for Statistical Methods</a></li>
                    <li><a href="#activity-2-presenting-statistical-results">Activity 2: Presenting Statistical Results Clearly</a></li>
                </ul>
            </li>
        </ol>
    """, unsafe_allow_html=True)

def section_advanced_techniques():
    st.markdown("<a id='advanced-techniques-in-biostatistics'></a>", unsafe_allow_html=True)
    st.header("Advanced Techniques in Biostatistics")
    
    st.write("""
        In this section, we'll explore three important advanced statistical techniques commonly used in biomedical research:
        multiple regression, survival analysis, and non-parametric tests. These methods extend our analytical capabilities
        beyond basic hypothesis testing and allow us to address more complex research questions.
    """)
    
    # Multiple Regression
    st.markdown("<a id='multiple-regression'></a>", unsafe_allow_html=True)
    st.subheader("Multiple Regression")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # Using existing image as placeholder
    with cent_co:
        st.write("""
        Multiple regression extends simple linear regression by allowing us to examine how multiple 
        independent variables simultaneously affect a dependent variable. This is particularly useful in 
        biostatistics because health outcomes are rarely influenced by a single factor.
        
        For example, a researcher might want to understand how a patient's blood pressure is affected by 
        age, weight, exercise habits, and medication use—all at the same time.
        """)
    
    st.markdown("##### Key Features of Multiple Regression:")
    st.write("""
        - Allows for control of confounding variables
        - Can determine the relative importance of different predictors
        - Helps detect interactions between variables
        - Provides a more complete model of complex biological systems
    """)
    
    st.markdown("##### The Multiple Regression Equation:")
    st.latex(r'Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \varepsilon')
    st.write("""
        Where:
        - Y is the dependent variable
        - X₁, X₂, ..., Xₚ are the independent variables
        - β₀ is the y-intercept
        - β₁, β₂, ..., βₚ are the regression coefficients
        - ε is the error term
    """)
    
    st.markdown("##### Example Code: Multiple Regression")
    multiple_regression_code = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the health dataset
health_data = pd.read_csv('data/health_data.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(health_data.head())

# Create a correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = health_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Health Variables')
plt.tight_layout()
plt.show()

# Fit multiple regression model to predict BMI
X = health_data[['age', 'height', 'weight', 'cholesterol', 'blood_pressure']]
X = sm.add_constant(X)  # Add a constant for the intercept
y = health_data['bmi']

# Fit the model
model = sm.OLS(y, X).fit()

# Display the summary
print(model.summary())

# Create a scatter plot of predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y, model.predict(), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.title('Actual vs Predicted BMI Values')
plt.tight_layout()
plt.show()
"""
    st.code(multiple_regression_code, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Multiple Regression)", key='week13-codepad-mr', use_container_width=True):
            codeeditor_popup(multiple_regression_code)
    
    st.info("""
        **Interpreting Coefficients in Multiple Regression:**
        
        Each coefficient (β) represents the expected change in the dependent variable for a one-unit 
        increase in the corresponding independent variable, *while holding all other independent variables constant*.
        This "holding other variables constant" is what makes multiple regression so powerful for understanding
        complex relationships.
    """)
    
    # Survival Analysis
    st.markdown("<a id='survival-analysis'></a>", unsafe_allow_html=True) 
    st.subheader("Survival Analysis")
    
    st.write("""
        Survival analysis is a set of statistical methods for analyzing data where the outcome variable is 
        the time until an event of interest occurs. In biostatistics, this "event" is often death, disease 
        progression, or treatment failure—but it can be any defined endpoint.
    """)
    
    st.markdown("##### Key Concepts in Survival Analysis:")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Survival Function**")
            st.write("""
                The probability that a subject survives longer than time t.
                
                S(t) = P(T > t)
            """)
            
            st.markdown("**Hazard Function**")
            st.write("""
                The instantaneous rate of occurrence of the event at time t.
                
                h(t) = lim[P(t ≤ T < t+Δt | T ≥ t)/Δt] as Δt→0
            """)
        
        with col2:
            st.markdown("**Censoring**")
            st.write("""
                When follow-up ends before the event occurs, the observation is "censored."
                
                Types:
                - Right censoring: Subject leaves study or study ends before event
                - Left censoring: Event occurs before observation begins
                - Interval censoring: Event occurs between two observation times
            """)
    
    st.markdown("##### Common Survival Analysis Methods:")
    st.write("""
        1. **Kaplan-Meier Estimator:** Non-parametric method for estimating the survival function
        2. **Cox Proportional Hazards Model:** Semi-parametric regression model that examines effects of variables on survival
        3. **Parametric Models:** Assumes survival times follow a specific distribution (e.g., Weibull, exponential)
    """)
    
    st.markdown("##### Example Code: Kaplan-Meier Survival Curve")
    survival_code = """
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

# Create a simple survival dataset
np.random.seed(42)
N = 100

# Generate survival data
data = pd.DataFrame({
    'time': np.random.exponential(scale=10, size=N),
    'event': np.random.binomial(n=1, p=0.7, size=N),  # 1=event occurred, 0=censored
    'treatment': np.random.binomial(n=1, p=0.5, size=N),  # 1=treatment, 0=control
    'age': np.random.normal(loc=65, scale=10, size=N),
    'sex': np.random.binomial(n=1, p=0.5, size=N)  # 1=female, 0=male
})

# Fit Kaplan-Meier curves
kmf = KaplanMeierFitter()
kmf.fit(data['time'], event_observed=data['event'], label='Overall Population')

# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True, alpha=0.3)

# Compare survival curves by treatment group
treatment_groups = data['treatment'].unique()
plt.figure(figsize=(10, 6))

for treatment in treatment_groups:
    mask = data['treatment'] == treatment
    label = f'Treatment Group: {"Treatment" if treatment == 1 else "Control"}'
    kmf = KaplanMeierFitter()
    kmf.fit(data.loc[mask, 'time'], event_observed=data.loc[mask, 'event'], label=label)
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by Treatment Group')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Fit a Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(data, duration_col='time', event_col='event')
print(cph.summary)

# Plot the hazard ratio and confidence intervals
plt.figure(figsize=(10, 6))
cph.plot()
plt.title('Hazard Ratios with 95% Confidence Intervals')
plt.tight_layout()
plt.show()
"""
    st.code(survival_code, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Survival Analysis)", key='week13-codepad-survival', use_container_width=True):
            codeeditor_popup(survival_code)
    
    # Non-parametric Tests
    st.markdown("<a id='non-parametric-tests'></a>", unsafe_allow_html=True)
    st.subheader("Non-parametric Tests")
    
    st.write("""
        Non-parametric tests don't assume that data follow a specific probability distribution (like the normal distribution).
        They're particularly useful in biostatistics when:
        
        - Sample sizes are small
        - Data don't meet the assumptions of parametric tests
        - Working with ordinal data or ranks
        - Dealing with outliers
    """)
    
    st.markdown("##### Common Non-parametric Tests and Their Parametric Equivalents:")
    
    non_parametric_tests = pd.DataFrame({
        "Non-parametric Test": ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis Test", "Friedman Test", "Spearman's Rank Correlation"],
        "Parametric Equivalent": ["Independent t-test", "Paired t-test", "One-way ANOVA", "Repeated measures ANOVA", "Pearson Correlation"],
        "When to Use": [
            "Compare two independent groups when normality is violated",
            "Compare two related samples when normality is violated",
            "Compare three or more independent groups when normality is violated",
            "Compare three or more related groups when normality is violated",
            "Measure association between ranked variables"
        ]
    })
    
    st.dataframe(non_parametric_tests, use_container_width=True)
    
    st.markdown("##### Example Code: Mann-Whitney U Test vs. t-test")
    non_parametric_code = """
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate normally distributed data
normal_group1 = np.random.normal(loc=10, scale=2, size=30)
normal_group2 = np.random.normal(loc=12, scale=2, size=30)

# Generate skewed data (log-normal distribution)
skewed_group1 = np.random.lognormal(mean=2, sigma=0.5, size=30)
skewed_group2 = np.random.lognormal(mean=2.3, sigma=0.5, size=30)

# Visualize the data distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot normal data
sns.histplot(normal_group1, kde=True, ax=axes[0, 0], color='blue', label='Group 1')
sns.histplot(normal_group2, kde=True, ax=axes[0, 0], color='red', alpha=0.7, label='Group 2')
axes[0, 0].set_title('Normal Distributions')
axes[0, 0].legend()

# Plot skewed data
sns.histplot(skewed_group1, kde=True, ax=axes[0, 1], color='blue', label='Group 1')
sns.histplot(skewed_group2, kde=True, ax=axes[0, 1], color='red', alpha=0.7, label='Group 2')
axes[0, 1].set_title('Skewed Distributions')
axes[0, 1].legend()

# Q-Q plots to test for normality
stats.probplot(normal_group1, plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Normal Data')

stats.probplot(skewed_group1, plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Skewed Data')

plt.tight_layout()
plt.show()

# Perform t-tests and Mann-Whitney U tests
print("--- Normal Data ---")
t_stat, t_p = stats.ttest_ind(normal_group1, normal_group2)
print(f"t-test: t={t_stat:.4f}, p={t_p:.4f}")

u_stat, u_p = stats.mannwhitneyu(normal_group1, normal_group2)
print(f"Mann-Whitney U test: U={u_stat:.4f}, p={u_p:.4f}")

print("\\n--- Skewed Data ---")
t_stat, t_p = stats.ttest_ind(skewed_group1, skewed_group2)
print(f"t-test: t={t_stat:.4f}, p={t_p:.4f}")

u_stat, u_p = stats.mannwhitneyu(skewed_group1, skewed_group2)
print(f"Mann-Whitney U test: U={u_stat:.4f}, p={u_p:.4f}")

# Create boxplots to compare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Normal data boxplot
axes[0].boxplot([normal_group1, normal_group2])
axes[0].set_xticklabels(['Group 1', 'Group 2'])
axes[0].set_title('Normal Data')

# Skewed data boxplot
axes[1].boxplot([skewed_group1, skewed_group2])
axes[1].set_xticklabels(['Group 1', 'Group 2'])
axes[1].set_title('Skewed Data')

plt.tight_layout()
plt.show()
"""
    st.code(non_parametric_code, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Non-parametric Tests)", key='week13-codepad-nonparam', use_container_width=True):
            codeeditor_popup(non_parametric_code)
    
    st.info("""
        **When to use non-parametric tests:**
        
        1. When your data violate assumptions of parametric tests (especially normality)
        2. With small sample sizes where normality is difficult to assess
        3. With ordinal or ranked data
        4. When dealing with data with many outliers
        
        Remember that non-parametric tests generally have less statistical power than parametric tests
        when parametric test assumptions are met.
    """)

def section_reading_statistical_results():
    st.markdown("<a id='reading-and-interpreting-statistical-results'></a>", unsafe_allow_html=True)
    st.header("Reading and Interpreting Statistical Results in Scientific Papers")
    
    st.write("""
        Biomedical literature is filled with statistical analyses, but interpreting these results can be challenging.
        This section will guide you through how to critically read and interpret statistical findings in scientific papers.
    """)
    
    # Key Elements to Look for in Statistical Results
    st.subheader("Key Elements to Look for in Statistical Results")
    
    with st.container(border=True):
        elements = [
            {
                "title": "1️⃣ Study Design",
                "description": """
                    - What type of study was conducted? (RCT, cohort, case-control, cross-sectional, etc.)
                    - What was the sample size and selection process?
                    - Were there appropriate controls?
                    - Was there randomization, blinding, or matching?
                """
            },
            {
                "title": "2️⃣ Statistical Tests",
                "description": """
                    - Which tests were used and are they appropriate for the data?
                    - Were assumptions of the tests checked and met?
                    - Were multiple comparisons accounted for?
                """
            },
            {
                "title": "3️⃣ Effect Size",
                "description": """
                    - What is the magnitude of the effect (not just statistical significance)?
                    - Are confidence intervals reported?
                    - Is the effect clinically meaningful even if statistically significant?
                """
            },
            {
                "title": "4️⃣ P-values and Significance",
                "description": """
                    - Are exact p-values reported or just thresholds (p<0.05)?
                    - Is the significance threshold appropriate for the context?
                    - Was the study adequately powered to detect meaningful differences?
                """
            },
            {
                "title": "5️⃣ Presentation of Results",
                "description": """
                    - Are data visualized clearly and appropriately?
                    - Are raw data or summary statistics provided?
                    - Are uncertainties (standard deviations, standard errors, confidence intervals) clearly reported?
                """
            }
        ]
        
        for element in elements:
            st.markdown(f"#### {element['title']}")
            st.markdown(element['description'])
    
    # Common Statistical Sections in Papers
    st.subheader("Common Statistical Sections in Scientific Papers")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week4-i3.png", width=200)  # Using existing image
    with cent_co:
        st.write("""
        Scientific papers typically organize statistical information in similar ways across journals.
        Understanding this structure can help you efficiently extract and evaluate the statistical content.
        
        Here's how statistical information is typically distributed throughout a paper:
        """)
    
    sections = {
        "Methods Section": [
            "Statistical tests used",
            "Software packages",
            "Sample size calculations",
            "Randomization procedures",
            "Handling of missing data",
            "Significance thresholds"
        ],
        "Results Section": [
            "Descriptive statistics",
            "Test statistics and p-values",
            "Effect sizes and confidence intervals",
            "Tables and figures summarizing data",
            "Subgroup analyses"
        ],
        "Discussion Section": [
            "Interpretation of findings",
            "Comparison with previous research",
            "Discussion of limitations",
            "Clinical or practical significance",
            "Generalizability of results"
        ]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Methods Section")
        for item in sections["Methods Section"]:
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown("##### Results Section")
        for item in sections["Results Section"]:
            st.markdown(f"- {item}")
    
    with col3:
        st.markdown("##### Discussion Section")
        for item in sections["Discussion Section"]:
            st.markdown(f"- {item}")
    
    # Red Flags in Statistical Reporting
    st.subheader("Red Flags in Statistical Reporting")
    
    red_flags = [
        "**P-hacking:** Analyzing data in multiple ways until statistically significant results are found",
        "**Cherry-picking:** Reporting only significant results and ignoring non-significant ones",
        "**Missing data handling:** Not explaining how missing data were addressed",
        "**Inappropriate tests:** Using statistical tests whose assumptions are violated by the data",
        "**Multiple testing:** Not correcting for multiple comparisons when appropriate",
        "**Unclear methodology:** Vague descriptions of statistical methods that can't be replicated",
        "**Overgeneralizing:** Extending conclusions beyond what the data support",
        "**Confusion of causation and correlation:** Interpreting correlational findings as causal"
    ]
    
    for flag in red_flags:
        st.markdown(f"- {flag}")
    
    # Understanding Statistical Terminology
    st.subheader("Understanding Statistical Terminology")
    
    st.write("""
        Scientific papers often use technical statistical terms. Here's a quick guide to commonly 
        used terms you might encounter:
    """)
    
    terms = pd.DataFrame({
        "Term": [
            "Adjusted odds ratio",
            "Hazard ratio",
            "Confidence interval",
            "Statistical power",
            "Effect size",
            "Multivariate vs. Multivariable",
            "Relative risk",
            "Number needed to treat (NNT)"
        ],
        "Definition": [
            "An odds ratio that accounts for confounding variables",
            "Ratio of hazard rates between two groups in survival analysis",
            "A range of values that is likely to contain the true population parameter",
            "Probability of detecting an effect if it truly exists",
            "Quantified magnitude of a phenomenon (independent of sample size)",
            "Multivariate refers to multiple outcomes; multivariable refers to multiple predictors",
            "Ratio of the probability of an event occurring in an exposed group to the probability in a non-exposed group",
            "Number of patients who need to be treated to prevent one additional bad outcome"
        ]
    })
    
    st.dataframe(terms, use_container_width=True)
    
    st.info("""
        **Pro Tip:** When reading statistical results in papers, it's often helpful to start by examining 
        the figures and tables first to get a visual understanding of the data, then read the detailed text 
        for full context.
    """)

def section_communicating_statistical_findings():
    st.markdown("<a id='communicating-statistical-findings'></a>", unsafe_allow_html=True)
    st.header("Communicating Statistical Findings Effectively")
    
    st.write("""
        Being able to clearly communicate statistical findings is crucial for biomedical professionals. 
        Whether writing papers, presenting at conferences, or explaining results to patients, how you 
        communicate statistical information impacts understanding and decision-making.
    """)
    
    # Principles of Effective Statistical Communication
    st.subheader("Principles of Effective Statistical Communication")
    
    principles = [
        {
            "principle": "Know your audience",
            "description": "Adapt your level of technical detail based on whether you're speaking to statisticians, clinicians, or patients"
        },
        {
            "principle": "Focus on the research question",
            "description": "Always relate statistics back to the original research question and clinical significance"
        },
        {
            "principle": "Use clear visuals",
            "description": "Well-designed graphs and tables communicate complex relationships more effectively than text alone"
        },
        {
            "principle": "Present uncertainty",
            "description": "Always include measures of uncertainty (confidence intervals, standard errors) alongside point estimates"
        },
        {
            "principle": "Avoid statistical jargon",
            "description": "Explain technical terms if you must use them, especially for non-technical audiences"
        },
        {
            "principle": "Contextualize significance",
            "description": "Explain both statistical significance (p-values) and practical/clinical significance"
        }
    ]
    
    cols = st.columns(3)
    for i, item in enumerate(principles):
        with cols[i % 3]:
            st.container(border=True).markdown(f"""
            #### {item['principle']}
            
            {item['description']}
            """)
    
    # Effective Data Visualization
    st.subheader("Effective Data Visualization")
    
    st.write("""
        Data visualization is one of the most powerful tools for communicating statistical findings.
        However, poorly designed visualizations can mislead or confuse your audience.
    """)
    
    st.markdown("##### Guidelines for Creating Effective Data Visualizations:")
    
    visualization_tips = [
        "**Choose the right chart type** for your data and message",
        "**Minimize chart junk** (unnecessary decorative elements)",
        "**Label axes clearly** with units of measurement",
        "**Use color strategically** and consider accessibility (colorblindness)",
        "**Show data in context** (reference lines, comparisons)",
        "**Include sample sizes** on all charts",
        "**Show error bars** or confidence intervals",
        "**Use consistent scales** when comparing multiple charts"
    ]
    
    for tip in visualization_tips:
        st.markdown(f"- {tip}")
    
    # Example code for data visualization
    st.markdown("##### Example Code: Creating Effective Figures for Scientific Communication")
    
    visualization_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set the aesthetic style of the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")  # Use a colorblind-friendly palette

# Generate example data (simulating a clinical trial outcome)
np.random.seed(42)
n_patients = 200

data = pd.DataFrame({
    'Treatment': np.repeat(['Drug A', 'Drug B', 'Placebo'], n_patients//3),
    'Gender': np.random.choice(['Female', 'Male'], size=n_patients, p=[0.55, 0.45]),
    'Age': np.random.normal(65, 12, n_patients).astype(int),
    'Response': np.random.binomial(1, 0.5, n_patients)  # 1 = responded to treatment, 0 = no response
})

# Add response rates that differ by treatment
data.loc[data['Treatment'] == 'Drug A', 'Response'] = np.random.binomial(1, 0.75, sum(data['Treatment'] == 'Drug A'))
data.loc[data['Treatment'] == 'Drug B', 'Response'] = np.random.binomial(1, 0.60, sum(data['Treatment'] == 'Drug B'))
data.loc[data['Treatment'] == 'Placebo', 'Response'] = np.random.binomial(1, 0.30, sum(data['Treatment'] == 'Placebo'))

# Calculate response rates by treatment and confidence intervals
response_stats = data.groupby('Treatment')['Response'].agg(['mean', 'count'])
response_stats['se'] = np.sqrt(response_stats['mean'] * (1 - response_stats['mean']) / response_stats['count'])
response_stats['ci_lower'] = response_stats['mean'] - 1.96 * response_stats['se']
response_stats['ci_upper'] = response_stats['mean'] + 1.96 * response_stats['se']

# Example 1: Bar chart with confidence intervals
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Treatment', y='Response', data=data, estimator=np.mean, 
                      ci=95, capsize=0.2, errwidth=2)

# Add sample size to labels
for i, treatment in enumerate(response_stats.index):
    n = response_stats.loc[treatment, 'count']
    bar_plot.annotate(f'n={n}', xy=(i, 0.02), ha='center', fontweight='bold')

plt.title('Treatment Response Rate with 95% Confidence Intervals', fontsize=14)
plt.xlabel('Treatment Group', fontsize=12)
plt.ylabel('Response Rate', fontsize=12)
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

# Add exact percentages on top of bars
for i, p in enumerate(response_stats['mean']):
    plt.annotate(f'{p:.1%}', xy=(i, p + 0.02), ha='center')

plt.tight_layout()
plt.show()

# Example 2: Forest plot for odds ratios (common in medical literature)
from scipy.stats import fisher_exact
import matplotlib.patches as mpatches

# Calculate odds ratios compared to placebo
treatments = ['Drug A', 'Drug B']
reference = 'Placebo'

# Create a function to calculate odds ratio and CI
def calculate_or(data, treatment, reference='Placebo'):
    # Create contingency table
    treatment_data = data[data['Treatment'] == treatment]
    reference_data = data[data['Treatment'] == reference]
    
    treatment_success = sum(treatment_data['Response'])
    treatment_failure = len(treatment_data) - treatment_success
    
    reference_success = sum(reference_data['Response'])
    reference_failure = len(reference_data) - reference_success
    
    # Calculate odds ratio and p-value
    table = [[treatment_success, treatment_failure], 
             [reference_success, reference_failure]]
    odds_ratio, p_value = fisher_exact(table)
    
    # Calculate 95% CI for odds ratio
    import math
    se = math.sqrt(1/treatment_success + 1/treatment_failure + 1/reference_success + 1/reference_failure)
    ci_lower = math.exp(math.log(odds_ratio) - 1.96 * se)
    ci_upper = math.exp(math.log(odds_ratio) + 1.96 * se)
    
    return {'treatment': treatment, 'odds_ratio': odds_ratio, 
            'ci_lower': ci_lower, 'ci_upper': ci_upper, 'p_value': p_value}

# Calculate odds ratios
odds_ratios = [calculate_or(data, treatment) for treatment in treatments]
or_df = pd.DataFrame(odds_ratios)

# Create forest plot
plt.figure(figsize=(10, 6))
y_pos = range(len(or_df))

# Reference line
plt.axvline(x=1, color='gray', linestyle='--')

# Plot odds ratios and confidence intervals
plt.errorbar(or_df['odds_ratio'], y_pos, 
             xerr=[or_df['odds_ratio'] - or_df['ci_lower'], or_df['ci_upper'] - or_df['odds_ratio']],
             fmt='o', markersize=10, capsize=10, elinewidth=2, capthick=2)

# Add labels and annotations
for i, row in enumerate(or_df.itertuples()):
    plt.annotate(f'OR: {row.odds_ratio:.2f} (95% CI: {row.ci_lower:.2f}-{row.ci_upper:.2f}), p={row.p_value:.4f}', 
                xy=(row.odds_ratio, i), xytext=(row.odds_ratio + 0.2, i),
                va='center')

plt.yticks(y_pos, or_df['treatment'])
plt.xlabel('Odds Ratio (vs. Placebo)', fontsize=12)
plt.title('Forest Plot: Treatment Effect vs. Placebo', fontsize=14)

# Add interpretation legend
legend_handles = [
    mpatches.Patch(color='white', label='OR > 1: Treatment better than placebo'),
    mpatches.Patch(color='white', label='OR < 1: Treatment worse than placebo'),
    mpatches.Patch(color='white', label='CI crosses 1: Not statistically significant')
]
plt.legend(handles=legend_handles, loc='upper right')

plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale is common for odds ratios
plt.xlim(0.5, 15)
plt.tight_layout()
plt.show()
"""
    
    st.code(visualization_code, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Data Visualization)", key='week13-codepad-visualization', use_container_width=True):
            codeeditor_popup(visualization_code)
    
    # Writing About Statistical Results
    st.subheader("Writing About Statistical Results")
    
    st.write("""
        When writing about your statistical results in papers, reports, or presentations, 
        following some best practices will help ensure your audience understands your findings.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### What to Include:")
        st.markdown("""
        - Description of statistical methods used
        - Sample sizes
        - Effect sizes with confidence intervals
        - Exact p-values (when possible)
        - Units of measurement
        - Both raw data and summary statistics
        - Interpretation in clinical/practical context
        """)
    
    with col2:
        st.markdown("##### Common Mistakes to Avoid:")
        st.markdown("""
        - Reporting only p-values without effect sizes
        - Stating "not significant" without providing actual p-values
        - Incorrectly stating p=0.000 (use p<0.001 instead)
        - Overinterpreting marginally significant results (p=0.049)
        - Using causal language for correlational findings
        - Presenting too many decimal places
        - Making claims beyond what the data support
        """)
    
    # Examples of Good vs. Poor Statistical Communication
    st.subheader("Examples: Good vs. Poor Statistical Communication")
    
    examples = [
        {
            "poor": "The treatment was significant (p<0.05).",
            "good": "Patients receiving the treatment had a 25% higher response rate (95% CI: 18%-32%, p=0.003) than those in the control group."
        },
        {
            "poor": "The correlation between BMI and blood pressure was highly significant.",
            "good": "BMI was moderately correlated with systolic blood pressure (r=0.42, 95% CI: 0.36-0.48, p<0.001), suggesting that higher BMI is associated with higher blood pressure."
        },
        {
            "poor": "Our drug caused a reduction in tumor size compared to the placebo.",
            "good": "Patients receiving our drug showed a mean tumor size reduction of 3.2 cm (95% CI: 2.1-4.3 cm) compared to 0.8 cm (95% CI: 0.3-1.3 cm) in the placebo group (p<0.001)."
        }
    ]
    
    for i, example in enumerate(examples, 1):
        with st.container(border=True):
            st.markdown(f"##### Example {i}")
            cols = st.columns(2)
            with cols[0]:
                st.error(f"**Poor:** {example['poor']}")
            with cols[1]:
                st.success(f"**Good:** {example['good']}")
    
    st.info("""
        **Remember:** Effective communication of statistical results combines technical accuracy with clarity. 
        Your goal should be to help your audience understand not just what you found, but what it means in the 
        larger context of the research question.
    """)

def section_activities():
    st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
    st.header("Activities")
    
    # Activity 1: Scientific Paper Statistical Critique - Multiple Choice
    st.markdown("<a id='activity-1-critiquing-a-scientific-paper'></a>", unsafe_allow_html=True)
    st.subheader("Activity 1: Statistical Methods Evaluation Quiz")
    
    st.write("""
        Instead of writing a critique, you'll evaluate specific aspects of statistical methods 
        in research through a series of multiple choice questions. This will help develop your 
        critical evaluation skills in a more structured way.
    """)
    
    paper_abstract = """
        **Title**: Effect of Plant-Based Diet on Blood Pressure Control in Hypertensive Patients
        
        **Abstract**: This study investigates the effects of a plant-based diet on blood pressure in hypertensive adults. 
        120 participants (aged 40-65 years) with stage 1 or 2 hypertension were randomized to either a plant-based 
        diet intervention (n=60) or standard diet control group (n=60) for 12 weeks. Blood pressure was measured at 
        baseline, 6 weeks, and 12 weeks. Results showed a significant reduction in systolic blood pressure in the 
        intervention group (-8.4 mmHg, p<0.01) compared to the control group (-1.2 mmHg, p=0.45). Similar trends were 
        observed for diastolic blood pressure. Multiple regression analysis controlling for age, BMI, and baseline 
        medication use confirmed the independent effect of the dietary intervention (β=-7.2, p<0.01). These findings 
        suggest that a plant-based diet may be an effective non-pharmacological approach for blood pressure management 
        in hypertensive patients.
    """
    
    with st.expander("Sample Scientific Paper Abstract"):
        st.markdown(paper_abstract)
    
    st.write("Based on the abstract provided, answer the following questions:")
    
    # Creating a form for multiple-choice questions
    with st.form("statistics_critique_quiz"):
        # Question 1
        q1 = st.radio(
            "1. What type of study design is described in this abstract?",
            ["Observational study", "Case-control study", "Randomized controlled trial", "Cross-sectional study"]
        )
        
        # Question 2
        q2 = st.radio(
            "2. What statistical method was used to control for confounding variables?",
            ["ANOVA", "Chi-square test", "Multiple regression", "Logistic regression"]
        )
        
        # Question 3
        q3 = st.multiselect(
            "3. Which of the following were reported in the results? (Select all that apply)",
            ["Effect size", "Exact p-values", "Confidence intervals", "Mean differences", "Sample size"]
        )
        
        # Question 4
        q4 = st.radio(
            "4. Which important statistical element is missing from the results reporting?",
            ["Sample size", "Statistical test used", "Confidence intervals", "P-values"]
        )
        
        # Question 5
        q5 = st.radio(
            "5. Is the conclusion drawn by the authors supported by the statistical evidence presented?",
            ["Yes, fully supported", "Yes, but with important limitations", "No, overstated conclusions", "Cannot determine from abstract"]
        )
        
        # Submit button
        submitted = st.form_submit_button("Check Answers")
        
        if submitted:
            score = 0
            feedback = []
            
            # Check Question 1
            if q1 == "Randomized controlled trial":
                score += 1
                feedback.append("✅ Question 1: Correct! This is a randomized controlled trial as participants were randomized to either intervention or control.")
            else:
                feedback.append("❌ Question 1: Incorrect. The study explicitly mentions randomization of participants to either plant-based diet or control group.")
            
            # Check Question 2
            if q2 == "Multiple regression":
                score += 1
                feedback.append("✅ Question 2: Correct! Multiple regression was used to control for confounders like age, BMI, and medication use.")
            else:
                feedback.append("❌ Question 2: Incorrect. The abstract states 'Multiple regression analysis controlling for age, BMI, and baseline medication use...'")
            
            # Check Question 3 (multiple correct answers)
            correct_q3 = ["Effect size", "Mean differences", "Sample size"]
            if set(q3) == set(correct_q3):
                score += 1
                feedback.append("✅ Question 3: Correct! The abstract reported effect sizes (-8.4 mmHg), mean differences between groups, and sample size (n=60 per group).")
            else:
                feedback.append("❌ Question 3: Incorrect. The correct answers are: Effect size, Mean differences, and Sample size. The abstract does not report exact p-values (only p<0.01) or confidence intervals.")
            
            # Check Question 4
            if q4 == "Confidence intervals":
                score += 1
                feedback.append("✅ Question 4: Correct! Confidence intervals are missing, which are important for understanding the precision of the estimate.")
            else:
                feedback.append("❌ Question 4: Incorrect. Confidence intervals are missing from the results, though they are important for interpreting the precision of effect estimates.")
            
            # Check Question 5
            if q5 == "Yes, but with important limitations":
                score += 1
                feedback.append("✅ Question 5: Correct! The findings do suggest the diet may be effective, but there are statistical limitations (e.g., no confidence intervals, incomplete p-value reporting).")
            else:
                feedback.append("❌ Question 5: Incorrect. The data supports the conclusion but with statistical reporting limitations that affect our ability to fully evaluate the claim.")
            
            # Display results
            st.success(f"Your score: {score}/5")
            for fb in feedback:
                st.markdown(fb)
    
    # Activity 2: Interactive Data Analysis with Automated Feedback
    st.markdown("<a id='activity-2-presenting-statistical-results'></a>", unsafe_allow_html=True)
    st.subheader("Activity 2: Interactive Data Analysis")
    
    st.write("""
        In this activity, you'll analyze real health data through a series of guided steps.
        Instead of open-ended responses, you'll make specific analytical choices and get 
        immediate feedback on your approach.
    """)
    
    # Option for using different datasets
    dataset_choice = st.radio(
        "Choose a dataset to work with:",
        ["General Health Data", "Heart Attack & Vaccination Data"]
    )
    
    if dataset_choice == "General Health Data":
        st.markdown("### General Health Dataset Analysis")
        
        # Load the health data
        try:
            health_data = pd.read_csv('/workspaces/biostatbook/data/health_data.csv')
            
            # Display a sample of the data
            with st.expander("View Dataset Sample"):
                st.dataframe(health_data.head(10))
            
            # Interactive analysis exercise
            st.markdown("#### Interactive Analysis Exercise")
            st.write("For this dataset, you'll perform a guided analysis with choices at each step:")
            
            # Step 1: Choose variables to analyze
            st.markdown("##### Step 1: Variable Selection")
            st.write("Which variables would you like to analyze for their relationship with weight?")
            
            selected_vars = st.multiselect(
                "Select independent variables to include in analysis:",
                ["age", "height", "chronic_disease", "exercise_habits", "smoking_habit"],
                default=["height", "age"]
            )
            
            if not selected_vars:
                st.warning("Please select at least one variable to continue.")
            else:
                # Step 2: Choose analysis method
                st.markdown("##### Step 2: Analysis Method")
                analysis_method = st.selectbox(
                    "Select an appropriate statistical method for this analysis:",
                    ["Simple linear regression", "Multiple linear regression", "Chi-square test", "ANOVA"]
                )
                
                # Provide feedback on method choice
                if analysis_method == "Multiple linear regression" and len(selected_vars) > 1:
                    st.success("✅ Good choice! Multiple regression is appropriate when analyzing multiple potential predictors of a continuous outcome like weight.")
                elif analysis_method == "Simple linear regression" and len(selected_vars) == 1:
                    st.success("✅ Good choice for a single predictor! Simple linear regression works when examining one continuous predictor with a continuous outcome.")
                elif analysis_method == "Simple linear regression" and len(selected_vars) > 1:
                    st.warning("⚠️ You've selected multiple predictors but chosen simple linear regression. This will only allow you to examine one relationship at a time.")
                elif analysis_method == "Chi-square test":
                    st.error("❌ Chi-square test is for categorical variables and isn't suitable for predicting a continuous outcome like weight.")
                elif analysis_method == "ANOVA":
                    st.info("ℹ️ ANOVA could be appropriate for categorical predictors like chronic_disease, but would need to be used separately for each categorical variable.")
            
                # Step 3: Choose visualization
                st.markdown("##### Step 3: Visualization Choice")
                viz_type = st.selectbox(
                    "Which visualization would best present the relationship between your selected variables and weight?",
                    ["Scatter plot", "Bar chart", "Box plot", "Line graph", "Heatmap"]
                )
                
                # Prepare data based on selections
                if len(selected_vars) > 0:
                    # Categorical variables need special handling
                    categorical_vars = ["chronic_disease", "exercise_habits", "smoking_habit"]
                    continuous_vars = [var for var in selected_vars if var not in categorical_vars]
                    categ_selected = [var for var in selected_vars if var in categorical_vars]
                    
                    # Feedback on visualization choice
                    if viz_type == "Scatter plot" and len(continuous_vars) > 0 and len(categ_selected) == 0:
                        st.success("✅ Good choice! Scatter plots work well for showing relationships between continuous variables.")
                    elif viz_type == "Box plot" and len(categ_selected) > 0:
                        st.success("✅ Good choice! Box plots are excellent for comparing distributions across categories.")
                    elif viz_type == "Bar chart" and len(categ_selected) > 0:
                        st.success("✅ Bar charts can work well for comparing mean values across categories.")
                    elif viz_type == "Heatmap" and len(selected_vars) >= 2:
                        st.success("✅ Heatmaps can be useful for visualizing correlations between multiple variables.")
                    elif viz_type == "Line graph" and "age" in selected_vars:
                        st.info("ℹ️ Line graphs typically work best for time series data, but could show trends across age groups.")
                    else:
                        st.warning("⚠️ Consider whether this visualization type is optimal for your selected variables.")
                    
                    # Generate a sample visualization based on selections
                    st.markdown("##### Your Analysis Results:")
                    
                    try:
                        if viz_type == "Scatter plot" and len(continuous_vars) > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            main_var = continuous_vars[0]
                            sns.scatterplot(x=main_var, y="weight", data=health_data, alpha=0.7)
                            plt.xlabel(main_var)
                            plt.ylabel("Weight (kg)")
                            plt.title(f"Relationship between {main_var} and Weight")
                            
                            # Add regression line
                            sns.regplot(x=main_var, y="weight", data=health_data, scatter=False, ax=ax)
                            
                            # Calculate correlation
                            corr = health_data[[main_var, "weight"]].corr().iloc[0, 1]
                            plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
                            
                            st.pyplot(fig)
                            
                            if main_var == "height":
                                st.info("As expected, there appears to be a positive correlation between height and weight.")
                                
                        elif viz_type == "Box plot" and len(categ_selected) > 0:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            main_var = categ_selected[0]
                            
                            # For chronic_disease, filter out None values and get top categories
                            if main_var == "chronic_disease":
                                filtered_data = health_data[health_data[main_var].notna() & (health_data[main_var] != "None")]
                                top_categories = filtered_data[main_var].value_counts().nlargest(5).index.tolist()
                                plot_data = filtered_data[filtered_data[main_var].isin(top_categories)]
                                title_addition = " (Top 5 Categories)"
                            else:
                                plot_data = health_data
                                title_addition = ""
                                
                            sns.boxplot(x=main_var, y="weight", data=plot_data)
                            plt.title(f"Weight Distribution by {main_var}{title_addition}")
                            plt.xlabel(main_var)
                            plt.ylabel("Weight (kg)")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        elif viz_type == "Bar chart" and len(categ_selected) > 0:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            main_var = categ_selected[0]
                            
                            if main_var == "chronic_disease":
                                filtered_data = health_data[health_data[main_var].notna() & (health_data[main_var] != "None")]
                                top_categories = filtered_data[main_var].value_counts().nlargest(5).index.tolist()
                                grouped_data = filtered_data[filtered_data[main_var].isin(top_categories)].groupby(main_var)["weight"].mean().sort_values(ascending=False)
                                title_addition = " (Top 5 Categories)"
                            else:
                                grouped_data = health_data.groupby(main_var)["weight"].mean().sort_values(ascending=False)
                                title_addition = ""
                                
                            grouped_data.plot(kind='bar', ax=ax)
                            plt.title(f"Average Weight by {main_var}{title_addition}")
                            plt.xlabel(main_var)
                            plt.ylabel("Average Weight (kg)")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        elif viz_type == "Heatmap" and len(selected_vars) >= 2:
                            # Create correlation heatmap for selected variables plus weight
                            analysis_vars = selected_vars.copy()
                            if "weight" not in analysis_vars:
                                analysis_vars.append("weight")
                                
                            # Filter for only numeric variables
                            numeric_vars = health_data[analysis_vars].select_dtypes(include=[np.number]).columns.tolist()
                            
                            if len(numeric_vars) < 2:
                                st.warning("Need at least 2 numeric variables for a correlation heatmap. Selected categorical variables will be excluded.")
                            else:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                corr_matrix = health_data[numeric_vars].corr()
                                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                                plt.title('Correlation Heatmap of Selected Variables')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                        elif viz_type == "Line graph":
                            if "age" in selected_vars:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                # Group by age and calculate mean weight
                                age_groups = health_data.groupby("age")["weight"].mean()
                                age_groups.plot(marker='o', ax=ax)
                                plt.xlabel("Age")
                                plt.ylabel("Average Weight (kg)")
                                plt.title("Average Weight by Age")
                                plt.grid(alpha=0.3)
                                st.pyplot(fig)
                            else:
                                st.warning("Line graphs are typically used for time series or ordinal data like age. Consider selecting age as a variable.")
                    
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
                        
                    # Step 4: Statistical interpretation quiz
                    st.markdown("##### Step 4: Statistical Interpretation")
                    st.write("Based on what you've observed, answer these interpretation questions:")
                    
                    # Create some interpretation questions based on the data and selections
                    q1_answered = False
                    
                    with st.form("interpretation_questions"):
                        if "height" in selected_vars:
                            q1 = st.radio(
                                "What is the relationship between height and weight in this dataset?",
                                ["Strong negative correlation", "Weak or no correlation", "Strong positive correlation", "Cannot determine from the visualization"]
                            )
                            q1_answered = True
                            
                        if "chronic_disease" in selected_vars:
                            q2 = st.radio(
                                "Based on the data, which statement is most accurate?",
                                [
                                    "There is no relationship between chronic disease status and weight",
                                    "Different chronic diseases appear to have different associations with weight",
                                    "All chronic diseases are associated with lower weight",
                                    "All chronic diseases are associated with higher weight"
                                ]
                            )
                        else:
                            q2 = st.radio(
                                "When analyzing categorical variables like chronic disease, which visualization would be most appropriate?",
                                ["Scatter plot", "Line graph", "Box plot", "Pie chart"]
                            )
                            
                        q3 = st.radio(
                            "What statistical approach would be most appropriate to control for multiple factors affecting weight?",
                            ["Chi-square test", "t-test", "Multiple linear regression", "Kaplan-Meier analysis"]
                        )
                        
                        interpret_submitted = st.form_submit_button("Check Interpretation")
                        
                        if interpret_submitted:
                            interpret_score = 0
                            interpret_feedback = []
                            
                            # Check height-weight question if answered
                            if q1_answered:
                                if q1 == "Strong positive correlation":
                                    interpret_score += 1
                                    interpret_feedback.append("✅ Correct! Height and weight typically have a strong positive correlation.")
                                else:
                                    interpret_feedback.append("❌ Incorrect. The data shows a strong positive correlation between height and weight.")
                            
                            # Check chronic disease question
                            if "chronic_disease" in selected_vars:
                                if q2 == "Different chronic diseases appear to have different associations with weight":
                                    interpret_score += 1
                                    interpret_feedback.append("✅ Correct! Different chronic conditions show different patterns of association with weight.")
                                else:
                                    interpret_feedback.append("❌ Incorrect. The data shows that different chronic diseases have varying associations with weight.")
                            else:
                                if q2 == "Box plot":
                                    interpret_score += 1
                                    interpret_feedback.append("✅ Correct! Box plots are excellent for comparing distributions across categories.")
                                else:
                                    interpret_feedback.append("❌ Incorrect. Box plots are most appropriate for visualizing distributions across categorical groups.")
                            
                            # Check approach question
                            if q3 == "Multiple linear regression":
                                interpret_score += 1
                                interpret_feedback.append("✅ Correct! Multiple regression is the appropriate approach for controlling multiple factors affecting weight.")
                            else:
                                interpret_feedback.append("❌ Incorrect. Multiple linear regression is the appropriate method for analyzing multiple factors affecting a continuous outcome like weight.")
                            
                            # Display results
                            max_score = 3 if q1_answered else 2
                            st.success(f"Your interpretation score: {interpret_score}/{max_score}")
                            for fb in interpret_feedback:
                                st.markdown(fb)
                                
            # Additional resources
            with st.expander("Additional Resources on Health Data Analysis"):
                st.markdown("""
                    - **Handling missing data:** [Strategies for missing data in health research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6536756/)
                    - **Selecting appropriate visualizations:** [Data visualization in healthcare](https://www.healthcatalyst.com/insights/healthcare-data-visualization-concepts-best-practices)
                    - **Controlling for confounding:** [Understanding confounding in health research](https://www.bmj.com/content/347/bmj.f5577)
                """)
                
        except Exception as e:
            st.error(f"Error loading health data: {e}")
            st.write("Please make sure the file '/workspaces/biostatbook/data/health_data.csv' exists and is accessible.")
    
    else:  # Heart Attack & Vaccination Data
        st.markdown("### Heart Attack & Vaccination Dataset Analysis")
        
        try:
            heart_data = pd.read_csv('/workspaces/biostatbook/data/heart_attack_vaccine_data.csv')
            
            # Display a sample of the data
            with st.expander("View Dataset Sample"):
                st.dataframe(heart_data.head(10))
            
            # Count how many had heart attacks
            heart_attack_count = heart_data['Heart Attack Date'].notna().sum()
            total_patients = len(heart_data)
            
            st.metric("Patients with Heart Attack Events", f"{heart_attack_count} / {total_patients}", 
                     f"{heart_attack_count/total_patients:.1%}")
            
            # Interactive analysis exercise
            st.markdown("#### Interactive Survival Analysis Exercise")
            
            # Step 1: Research question formulation
            st.markdown("##### Step 1: Research Question")
            st.write("In survival analysis, it's important to clearly define your research question.")
            
            research_question = st.selectbox(
                "Which research question would be most appropriate for survival analysis of this dataset?",
                [
                    "Select a research question...",
                    "What proportion of patients develop heart attacks after vaccination?",
                    "Is there an association between vaccine dose and heart attack occurrence?",
                    "What factors affect the time from vaccination to heart attack event?",
                    "What is the average age of patients who received different vaccine doses?"
                ]
            )
            
            if research_question == "What factors affect the time from vaccination to heart attack event?":
                st.success("✅ Excellent choice! This question directly addresses the time-to-event nature of survival analysis.")
            elif research_question == "Is there an association between vaccine dose and heart attack occurrence?":
                st.info("ℹ️ This is a good question, but doesn't fully utilize the time-to-event data. It could be answered with simpler methods like chi-square tests.")
            elif research_question == "What proportion of patients develop heart attacks after vaccination?":
                st.info("ℹ️ This is descriptive and doesn't utilize the time component of the data.")
            elif research_question == "What is the average age of patients who received different vaccine doses?":
                st.warning("⚠️ This question doesn't relate to the survival outcome (time to heart attack).")
            
            # Step 2: Variable selection for survival analysis
            st.markdown("##### Step 2: Variable Selection")
            st.write("Select variables you think might influence time to heart attack event:")
            
            selected_predictors = st.multiselect(
                "Select potential predictors:",
                ["Age", "Gender", "Vaccine Dose", "Pre-existing Conditions", "Blood Pressure", "Cholesterol Level", "BMI", "Smoking History"],
                default=["Age", "Gender"]
            )
            
            if len(selected_predictors) > 0:
                # Step 3: Method selection
                st.markdown("##### Step 3: Survival Analysis Method")
                method = st.selectbox(
                    "Which survival analysis method would be most appropriate for your research question?",
                    ["Kaplan-Meier estimation", "Cox Proportional Hazards model", "Log-rank test", "All of the above"]
                )
                
                if method == "All of the above":
                    st.success("✅ Correct! A comprehensive analysis would use all these methods: Kaplan-Meier for visualization, log-rank test for comparing groups, and Cox model for multivariable analysis.")
                elif method == "Cox Proportional Hazards model":
                    st.success("✅ Good choice! Cox models are excellent for assessing multiple predictors simultaneously.")
                elif method == "Kaplan-Meier estimation":
                    st.info("ℹ️ Kaplan-Meier is useful for visualization and univariate analysis, but won't account for multiple factors simultaneously.")
                elif method == "Log-rank test":
                    st.info("ℹ️ Log-rank tests compare survival curves between groups but don't adjust for multiple predictors.")
                
                # Create the event indicator
                heart_data['event'] = heart_data['Heart Attack Date'].notna().astype(int)
                
                # Step 4: Results interpretation
                st.markdown("##### Step 4: Interpreting Sample Results")
                st.write("Below is a visualization of heart attack rates by vaccine dose. Use this to answer the interpretation questions.")
                
                # Create a simple visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                doses = heart_data['Vaccine Dose'].unique()
                heart_attack_by_dose = {}
                
                for dose in doses:
                    subset = heart_data[heart_data['Vaccine Dose'] == dose]
                    heart_attack_by_dose[dose] = subset['event'].mean() * 100
                
                # Sort by heart attack rate
                heart_attack_by_dose = {k: v for k, v in sorted(heart_attack_by_dose.items(), key=lambda item: item[1])}
                
                plt.barh(list(heart_attack_by_dose.keys()), list(heart_attack_by_dose.values()), color='coral')
                plt.xlabel('Heart Attack Rate (%)')
                plt.title('Heart Attack Rate by Vaccine Dose Type')
                plt.grid(axis='x', alpha=0.3)
                
                # Add percentage labels
                for i, (dose, rate) in enumerate(heart_attack_by_dose.items()):
                    plt.text(rate + 0.1, i, f"{rate:.1f}%", va='center')
                
                st.pyplot(fig)
                
                # Interpretation questions
                with st.form("survival_interpretation"):
                    st.write("Answer these questions based on the chart above and your understanding of survival analysis:")
                    
                    q1 = st.radio(
                        "1. What statement best describes the relationship between vaccine dose and heart attack rate?",
                        [
                            "Booster doses have substantially higher risk than other doses",
                            "All doses have approximately similar heart attack rates",
                            "1st doses have substantially lower risk than other doses",
                            "The data doesn't show any clear pattern of association"
                        ]
                    )
                    
                    q2 = st.radio(
                        "2. If a log-rank test comparing these groups yields p=0.763, what can we conclude?",
                        [
                            "The vaccine definitely causes heart attacks",
                            "The differences in heart attack timing between dose groups are not statistically significant",
                            "All doses are equally safe",
                            "Vaccine type has no effect on heart attack risk"
                        ]
                    )
                    
                    q3 = st.radio(
                        "3. What is the main advantage of Cox Proportional Hazards models over simple comparisons of event rates?",
                        [
                            "They're easier to interpret",
                            "They require less data",
                            "They can control for multiple confounding factors simultaneously",
                            "They always produce statistically significant results"
                        ]
                    )
                    
                    q4 = st.radio(
                        "4. In survival analysis, what is the correct interpretation of hazard ratio = 1.04 for Age?",
                        [
                            "For each additional year of age, there's a 4% increased risk at any given time point",
                            "Age increases risk by 104%",
                            "Age has no effect on survival",
                            "The effect of age cannot be determined from a hazard ratio"
                        ]
                    )
                    
                    survival_submitted = st.form_submit_button("Check Answers")
                    
                    if survival_submitted:
                        survival_score = 0
                        survival_feedback = []
                        
                        # Check q1
                        if q1 == "All doses have approximately similar heart attack rates":
                            survival_score += 1
                            survival_feedback.append("✅ Correct! The chart shows relatively similar rates across doses.")
                        else:
                            survival_feedback.append("❌ Incorrect. The visualization shows similar rates across different doses, with minor variations.")
                        
                        # Check q2
                        if q2 == "The differences in heart attack timing between dose groups are not statistically significant":
                            survival_score += 1
                            survival_feedback.append("✅ Correct! The p-value >0.05 indicates no statistically significant difference in survival curves.")
                        else:
                            survival_feedback.append("❌ Incorrect. A p-value of 0.763 indicates that we cannot reject the null hypothesis of no difference.")
                        
                        # Check q3
                        if q3 == "They can control for multiple confounding factors simultaneously":
                            survival_score += 1
                            survival_feedback.append("✅ Correct! Cox models allow for multiple predictors and control for confounding variables.")
                        else:
                            survival_feedback.append("❌ Incorrect. The primary advantage is the ability to adjust for multiple factors simultaneously.")
                        
                        # Check q4
                        if q4 == "For each additional year of age, there's a 4% increased risk at any given time point":
                            survival_score += 1
                            survival_feedback.append("✅ Correct! HR=1.04 means approximately 4% increased hazard for each additional year of age.")
                        else:
                            survival_feedback.append("❌ Incorrect. A hazard ratio of 1.04 for age indicates a 4% increase in hazard for each additional year.")
                        
                        # Display results
                        st.success(f"Your score: {survival_score}/4")
                        for fb in survival_feedback:
                            st.markdown(fb)
                            
                # Provide additional context
                with st.expander("Key Concepts in Survival Analysis"):
                    st.markdown("""
                        ### Important Concepts
                        
                        **Censoring**: When subjects don't experience the event during study follow-up, their data is "censored"
                        
                        **Kaplan-Meier curves**: Non-parametric visualization of survival probability over time
                        
                        **Hazard ratio**: Relative effect of a variable on the hazard or risk of an event
                        
                        **Proportional hazards assumption**: The assumption that hazard ratios remain constant over time
                        
                        **Competing risks**: When subjects can experience different types of events that compete with each other
                    """)
                
        except Exception as e:
            st.error(f"Error loading heart attack data: {e}")
            st.write("Please make sure the file '/workspaces/biostatbook/data/heart_attack_vaccine_data.csv' exists and is accessible.")

def main():
    Navbar()
    st.title("Week 13: Advanced Techniques and Scientific Communication")
    section_table_of_contents()
    section_advanced_techniques()
    section_reading_statistical_results()
    section_communicating_statistical_findings()
    section_activities()  # Now using our updated section with CSV data
    Footer(num_weeks=13)

if __name__ == "__main__":
    main()