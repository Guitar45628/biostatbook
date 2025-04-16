import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower, TTestPower
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 12 | Power Analysis and Study Design",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code, key='codepad-week12', warning_text=warning_text)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#statistical-power">Statistical Power</a></li>
            <li><a href="#factors-affecting-power">Factors Affecting Power</a></li>
            <li><a href="#calculating-sample-size">Calculating Sample Size</a></li>
            <li><a href="#activities">Activities</a>
                <ol>
                    <li><a href="#activity-1-quiz-on-power-analysis-and-study-design">Activity 1: Quiz on Power Analysis</a></li>
                    <li><a href="#activity-2-performing-power-analysis-using-software">Activity 2: Performing Power Analysis</a></li>
                    <li><a href="#activity-3-designing-a-study-with-sample-size-calculation">Activity 3: Designing a Study</a></li>
                </ol>
            </li>
        </ol>
    """, unsafe_allow_html=True)

def section_statistical_power():
    st.header("Statistical Power and Its Importance")

    # เพิ่มกรณีศึกษาที่น่าสนใจ
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # ใช้ภาพจากไฟล์อื่นที่มีอยู่ก่อน
    with cent_co:
        st.subheader("The Case of a Failed Drug Trial")
        st.write("""
        A pharmaceutical company spent $50 million on a new drug to reduce cholesterol levels. 
        Their clinical trial showed no significant effect, but was it because:
        
        1. The drug truly doesn't work? 🤔
        2. They didn't test enough patients to detect the effect? 🧪
        
        This is where **statistical power** comes in!
        """)

    st.write("""
        Statistical power is the probability that a statistical test will correctly reject a false null hypothesis (i.e., detect a true effect).
        
        - **High power** reduces the risk of Type II errors (false negatives).
        - Power is typically set at 0.8 (80%) or higher in study design.
        
        Power analysis helps researchers determine the sample size needed to detect an effect of a given size with a certain degree of confidence.
    """)
    
    # เพิ่มตัวอย่างการใช้ power analysis ในชีวิตจริง
    with st.expander("✨ Real-world examples of power analysis in medicine"):
        st.write("""
        1. **COVID-19 Vaccine Trials**: Researchers needed to determine how many participants to include in vaccine trials to reliably detect efficacy.
        
        2. **Cancer Treatment Studies**: When testing new cancer treatments, scientists must ensure they have enough patients to detect even small improvements in survival rates.
        
        3. **Rare Disease Research**: Studies on rare conditions are especially challenging because of limited patient populations - power analysis helps determine if meaningful research is possible.
        
        4. **Genetic Association Studies**: When studying how genetic variants relate to disease risk, scientists need large sample sizes to detect small effects.
        """)
    
    st.info("Power analysis is a critical step in planning a study to ensure results are reliable and resources are used efficiently.")

    # เพิ่มการ visualize ความสัมพันธ์ระหว่าง power, alpha, sample size
    st.subheader("Visualizing Statistical Errors")
    
    # Create a dataframe for the plot
    viz_code = """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st
    from scipy import stats
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values
    x = np.linspace(-4, 4, 1000)
    
    # Null and alternative distributions
    null_mean = 0
    alt_mean = 1.5
    std_dev = 1
    
    # Generate PDFs
    null_pdf = stats.norm.pdf(x, null_mean, std_dev)
    alt_pdf = stats.norm.pdf(x, alt_mean, std_dev)
    
    # Plot PDFs
    ax.plot(x, null_pdf, 'b-', label='Null Hypothesis Distribution')
    ax.plot(x, alt_pdf, 'r-', label='Alternative Hypothesis Distribution')
    
    # Critical value
    alpha = 0.05
    critical_value = stats.norm.ppf(1-alpha, null_mean, std_dev)
    
    # Fill alpha area (Type I error)
    x_alpha = x[x >= critical_value]
    y_alpha = stats.norm.pdf(x_alpha, null_mean, std_dev)
    ax.fill_between(x_alpha, y_alpha, color='blue', alpha=0.3, label='Type I Error (α)')
    
    # Fill beta area (Type II error)
    x_beta = x[x <= critical_value]
    y_beta = stats.norm.pdf(x_beta, alt_mean, std_dev)
    ax.fill_between(x_beta, y_beta, color='red', alpha=0.3, label='Type II Error (β)')
    
    # Fill power area
    x_power = x[x >= critical_value]
    y_power = stats.norm.pdf(x_power, alt_mean, std_dev)
    ax.fill_between(x_power, y_power, color='green', alpha=0.3, label='Power (1-β)')
    
    # Add vertical line for critical value
    ax.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical Value')
    
    # Labels and legend
    ax.set_title('Statistical Power, Type I and Type II Errors', fontsize=14)
    ax.set_xlabel('Test Statistic Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend()
    
    # Show the plot
    st.pyplot(fig)
    """
    
    st.code(viz_code, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("▶️ Run Code to See Visualization", key='week12-codepad-visual', use_container_width=True):
            codeeditor_popup(viz_code)

def section_factors_affecting_power():
    st.header("Factors Affecting Power")
    st.write("""
        The main factors that influence statistical power are:
        
        1. **Sample Size (n):** Larger samples increase power.
        2. **Effect Size:** Larger effects are easier to detect.
        3. **Significance Level (α):** Higher alpha increases power but also increases Type I error risk.
        4. **Variance:** Lower variance increases power.
        
        Power analysis can be used to calculate the required sample size for a desired power, or to estimate power given a sample size.
    """)
    
    st.subheader("Visualizing Power vs. Sample Size")
    st.write("""
        The following graph demonstrates how power increases as sample size increases, for different effect sizes.
        Notice that for larger effect sizes, we need fewer samples to achieve the same power.
    """)
    
    # Code for plotting power curves
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.stats.power import TTestIndPower
    
    # Initialize power analysis
    power_analysis = TTestIndPower()
    
    # Parameters for power analysis
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effect sizes
    sample_sizes = np.arange(10, 300, 5)
    alpha = 0.05
    
    # Calculate power for each effect size and sample size
    plt.figure(figsize=(10, 6))
    for effect_size in effect_sizes:
        power = [power_analysis.solve_power(effect_size=effect_size, 
                                           nobs1=n, 
                                           alpha=alpha, 
                                           ratio=1.0, 
                                           alternative='two-sided') 
                 for n in sample_sizes]
        plt.plot(sample_sizes, power, label=f'Effect size = {effect_size}')
    
    # Add reference line for common 0.8 power threshold
    plt.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
    
    plt.xlabel('Sample Size (per group)')
    plt.ylabel('Statistical Power')
    plt.title('Statistical Power vs. Sample Size for Two-Sample t-Test')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    
    st.code(code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Power Curves)", key='week12-codepad-curves', use_container_width=True):
            codeeditor_popup(code)

def section_calculating_sample_size():
    st.header("Calculating Sample Size for Common Tests")
    st.write("""
        Sample size calculations depend on the type of test, effect size, significance level, and desired power.
        
        **Example: Two-sample t-test**
        
        The formula for sample size estimation can be complex, but Python libraries like `statsmodels` make it easy.
    """)
    st.code(
        '''from statsmodels.stats.power import TTestIndPower\n\npower_analysis = TTestIndPower()\n# Calculate sample size for effect size=0.5, alpha=0.05, power=0.8\nsample_size = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided')\nprint(f"Required sample size per group: {sample_size:.2f}")''',
        language="python"
    )
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Sample Size Calculation)", key='week12-codepad-sample-size', use_container_width=True):
            codeeditor_popup('''from statsmodels.stats.power import TTestIndPower\n\npower_analysis = TTestIndPower()\nsample_size = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided')\nprint(f"Required sample size per group: {sample_size:.2f}")''')
    
    st.subheader("Sample Size for Different Tests")
    st.write("""
        Different statistical tests require different approaches to sample size calculation:
        
        1. **One-sample t-test**
        ```python
        from statsmodels.stats.power import TTestPower
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
        ```
        
        2. **Two-sample t-test**
        ```python
        from statsmodels.stats.power import TTestIndPower
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
        ```
        
        3. **Paired t-test**
        ```python
        from statsmodels.stats.power import TTestPower
        power_analysis = TTestPower()
        # For paired tests, effect size is often smaller
        sample_size = power_analysis.solve_power(effect_size=0.3, alpha=0.05, power=0.8)
        ```
        
        4. **ANOVA**
        ```python
        from statsmodels.stats.power import FTestAnovaPower
        power_analysis = FTestAnovaPower()
        # k = number of groups
        sample_size = power_analysis.solve_power(effect_size=0.25, k_groups=3, alpha=0.05, power=0.8)
        ```
    """)
    
    st.subheader("Interactive Power Analysis Example")
    
    code_interactive = """
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st
    from statsmodels.stats.power import TTestIndPower
    
    # User inputs
    st.write("### Power Analysis Parameters")
    effect_size = st.slider("Effect size (Cohen's d)", 0.2, 1.0, 0.5, 0.1)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    desired_power = st.slider("Desired power", 0.7, 0.95, 0.8, 0.05)
    
    # Initialize power analysis
    power_analysis = TTestIndPower()
    
    # Calculate required sample size
    sample_size = power_analysis.solve_power(
        effect_size=effect_size, 
        alpha=alpha, 
        power=desired_power, 
        alternative='two-sided'
    )
    
    st.write(f"Required sample size per group: {sample_size:.0f}")
    
    # Power curve
    sample_sizes = np.arange(10, 200, 5)
    power = [power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        alternative='two-sided') for n in sample_sizes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, power)
    ax.axhline(y=desired_power, color='r', linestyle='--', label=f'Power = {desired_power}')
    ax.axvline(x=sample_size, color='g', linestyle='--', label=f'Sample size = {sample_size:.0f}')
    ax.set_xlabel('Sample Size (per group)')
    ax.set_ylabel('Statistical Power')
    ax.set_title(f'Power Curve (Effect size = {effect_size}, α = {alpha})')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    """
    
    st.code(code_interactive, language="python")
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Interactive Power Analysis)", key='week12-codepad-interactive', use_container_width=True):
            codeeditor_popup(code_interactive)

def activity_quiz_power_analysis():
    st.header("Activity 1: Quiz on Power Analysis and Study Design")
    st.write("Test your understanding of statistical power, sample size, and study design.")

    with st.form(key='power_quiz'):
        st.subheader("Please answer the following questions:")

        q1 = st.radio("1. What is statistical power?",
            ("The probability of making a Type I error",
             "The probability of correctly rejecting a false null hypothesis",
             "The probability of accepting the null hypothesis",
             "The probability of making a Type II error"), index=None)

        q2 = st.radio("2. Which factor does NOT affect statistical power?",
            ("Sample size",
             "Effect size",
             "Significance level (alpha)",
             "Color of the data points"), index=None)

        q3 = st.radio("3. Increasing the sample size will:",
            ("Decrease power",
             "Increase power",
             "Have no effect on power",
             "Increase Type II error"), index=None)

        q4 = st.radio("4. What is the typical minimum acceptable value for power in study design?",
            ("0.2",
             "0.5",
             "0.8",
             "1.0"), index=None)

        q5 = st.radio("5. If the effect size is small, what should you do to maintain power?",
            ("Decrease sample size",
             "Increase sample size",
             "Increase alpha to 0.5",
             "Decrease power"), index=None)

        q6 = st.radio("6. What does a Type II error mean?",
            ("Rejecting a true null hypothesis",
             "Failing to reject a false null hypothesis",
             "Accepting a true alternative hypothesis",
             "None of the above"), index=None)

        q7 = st.radio("7. Which of the following increases power?",
            ("Lowering the significance level (alpha)",
             "Increasing sample size",
             "Increasing variance",
             "Decreasing effect size"), index=None)

        q8 = st.radio("8. What is the main purpose of power analysis before a study?",
            ("To determine the required sample size",
             "To calculate the mean",
             "To test for normality",
             "To increase Type I error"), index=None)

        q9 = st.radio("9. If alpha is increased from 0.05 to 0.10, what happens to power (all else equal)?",
            ("Power increases",
             "Power decreases",
             "Power stays the same",
             "Type II error increases"), index=None)

        q10 = st.radio("10. Which software can be used for power analysis in Python?",
            ("statsmodels",
             "matplotlib",
             "seaborn",
             "pandas"), index=None)

        submit_button = st.form_submit_button("Submit Quiz")

    if submit_button:
        score = 0
        correct_answers = {
            "q1": "The probability of correctly rejecting a false null hypothesis",
            "q2": "Color of the data points",
            "q3": "Increase power",
            "q4": "0.8",
            "q5": "Increase sample size",
            "q6": "Failing to reject a false null hypothesis",
            "q7": "Increasing sample size",
            "q8": "To determine the required sample size",
            "q9": "Power increases",
            "q10": "statsmodels"
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

def activity_power_analysis():
    st.header("Activity 2: Performing Power Analysis Using Software")
    st.write("""
        In this activity, you will use Python to perform a power analysis for a two-sample t-test.\n\n        **Dataset:** We'll use the `lung_cancer.csv` dataset.\n\n        **Task:**\n        1. Load the dataset and compare tumor sizes between two groups.\n        2. Estimate the effect size (Cohen's d) between the groups.\n        3. Use `statsmodels` to calculate the required sample size for 80% power.\n    """)
    with st.expander("Show Example Code"):
        st.code(
            '''import pandas as pd\nfrom statsmodels.stats.power import TTestIndPower\nimport numpy as np\n\ndata = pd.read_csv('data/lung_cancer.csv')\ngroup1 = data[data['treatment_group'] == 'Treatment A']['tumor_size']\ngroup2 = data[data['treatment_group'] == 'Placebo']['tumor_size']\n\n# Calculate effect size (Cohen's d)\nmean1, mean2 = group1.mean(), group2.mean()\nsd1, sd2 = group1.std(), group2.std()\npooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)\neffect_size = (mean1 - mean2) / pooled_sd\n\npower_analysis = TTestIndPower()\nsample_size = power_analysis.solve_power(effect_size=abs(effect_size), alpha=0.05, power=0.8)\nprint(f"Estimated effect size (Cohen's d): {effect_size:.2f}")\nprint(f"Required sample size per group: {sample_size:.2f}")''',
            language="python"
        )
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in CodePad (Power Analysis)", key='week12-codepad-power', use_container_width=True):
                codeeditor_popup('''import pandas as pd\nfrom statsmodels.stats.power import TTestIndPower\nimport numpy as np\n\ndata = pd.read_csv('data/lung_cancer.csv')\ngroup1 = data[data['treatment_group'] == 'Treatment A']['tumor_size']\ngroup2 = data[data['treatment_group'] == 'Placebo']['tumor_size']\n\n# Calculate effect size (Cohen's d)\nmean1, mean2 = group1.mean(), group2.mean()\nsd1, sd2 = group1.std(), group2.std()\npooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)\neffect_size = (mean1 - mean2) / pooled_sd\n\npower_analysis = TTestIndPower()\nsample_size = power_analysis.solve_power(effect_size=abs(effect_size), alpha=0.05, power=0.8)\nprint(f"Estimated effect size (Cohen's d): {effect_size:.2f}")\nprint(f"Required sample size per group: {sample_size:.2f}")''')

def activity_study_design():
    st.header("Activity 3: Designing a Study with Sample Size Calculation")
    
    # Role-play introduction
    left_co, cent_co = st.columns([1, 3])
    with left_co:
        st.image("assets/week4-i3.png", width=200)
    with cent_co:
        st.subheader("🎭 Role Play: You are the Lead Biostatistician!")
        st.write("""
        Your team has developed a revolutionary vaccine that might prevent heart attacks. 
        The CEO wants to launch a clinical trial but needs your expertise to determine *exactly* 
        how many participants to recruit!
        
        The entire project and millions of dollars depend on your power analysis...
        """)
    
    st.write("""
        **The Challenge:** Design a clinical trial comparing your vaccine to a placebo.
        
        **Your Tools:**
        - The `heart_attack_vaccine_data.csv` dataset contains preliminary data
        - Your knowledge of statistical power and sample size calculation
        - Python's statsmodels library
        
        **Your Mission:**
        1. Choose an outcome variable (e.g., cholesterol reduction)
        2. Calculate the effect size from preliminary data
        3. Determine the required sample size for 90% power and α=0.05
        4. Present your findings in a professional way
    """)
    
    with st.expander("📊 Sample Data Exploration"):
        st.write("Let's first explore the heart attack vaccine dataset:")
        
        example_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower

# Load the data
try:
    data = pd.read_csv('data/heart_attack_vaccine_data.csv')
    st.write("### Data Overview")
    st.write(f"Dataset shape: {data.shape}")
    st.write(data.head())
    
    # Check available groups and variables
    st.write("### Available Groups")
    st.write(data['group'].unique() if 'group' in data.columns else "No 'group' column found")
    
    # Count patients in each group
    if 'group' in data.columns:
        group_counts = data['group'].value_counts()
        st.write(pd.DataFrame(group_counts).rename(columns={'group': 'count'}))
    
    # Display a sample comparison between groups
    st.write("### Example: Cholesterol Levels by Group")
    if 'cholesterol' in data.columns and 'group' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='group', y='cholesterol', data=data, ax=ax)
        ax.set_title('Cholesterol Levels by Group')
        st.pyplot(fig)
        
        # Calculate basic stats for cholesterol by group
        group_stats = data.groupby('group')['cholesterol'].agg(['mean', 'std', 'count'])
        st.write(group_stats)
    
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
"""
        st.code(example_code, language="python")
        if "codeeditor_popup" not in st.session_state:
            if st.button("🚀 Explore Dataset", key='week12-codepad-explore', use_container_width=True):
                codeeditor_popup(example_code)
    
    # ตัวอย่างการคำนวณ power analysis แบบง่ายๆ
    with st.expander("📋 Power Analysis Template"):
        st.write("Use this template for your power analysis:")
        
        power_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower

# Load data
data = pd.read_csv('data/heart_attack_vaccine_data.csv')

# Select variables for analysis
outcome_var = 'cholesterol'  # You can change this to blood_pressure, inflammation_marker, etc.
group1_name = 'Vaccine'
group2_name = 'Placebo'
power_target = 0.90  # 90% power
alpha_value = 0.05  # 5% significance level

# Extract data based on selection
group1 = data[data['group'] == group1_name][outcome_var]
group2 = data[data['group'] == group2_name][outcome_var]

# Calculate effect size (Cohen's d)
mean1, mean2 = group1.mean(), group2.mean()
sd1, sd2 = group1.std(), group2.std()
pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
effect_size = abs((mean1 - mean2) / pooled_sd)

# Power analysis
power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(
    effect_size=effect_size, 
    alpha=alpha_value, 
    power=power_target, 
    alternative='two-sided'
)

# Print results
print(f"Outcome variable: {outcome_var}")
print(f"Group 1: {group1_name}, mean = {mean1:.2f}, SD = {sd1:.2f}")
print(f"Group 2: {group2_name}, mean = {mean2:.2f}, SD = {sd2:.2f}")
print(f"Estimated effect size (Cohen's d): {effect_size:.3f}")
print(f"Required sample size per group for {power_target*100}% power: {int(np.ceil(sample_size))}")

# Plot power curve
sample_sizes = np.arange(5, int(sample_size*2.5), max(1, int(sample_size/20)))
powers = [power_analysis.solve_power(
    effect_size=effect_size, 
    nobs1=n, 
    alpha=alpha_value, 
    alternative='two-sided') 
    for n in sample_sizes]

# Create plot    
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sample_sizes, powers, 'b-', linewidth=2)
ax.axhline(y=power_target, color='r', linestyle='--', label=f'Target Power = {power_target}')
ax.axvline(x=sample_size, color='g', linestyle='--', label=f'Required n = {int(np.ceil(sample_size))}')
ax.set_xlabel('Sample Size (per group)')
ax.set_ylabel('Statistical Power')
ax.set_title(f'Power Curve for {group1_name} vs {group2_name} ({outcome_var})')
ax.grid(True, alpha=0.3)
ax.legend()

# Display using matplotlib
plt.tight_layout()
plt.show()

# Interpret results
impact_text = "substantial" if effect_size > 0.8 else "moderate" if effect_size > 0.5 else "small"
print(f"The effect size is considered {impact_text}.")
print(f"To achieve {power_target*100}% power with α={alpha_value}, we need {int(np.ceil(sample_size))} participants per group.")
print(f"Total required sample size: {int(np.ceil(sample_size))*2} participants.")
"""
        st.code(power_code, language="python")
        if "codeeditor_popup" not in st.session_state:
            if st.button("▶️ Run Power Analysis", key='week12-codepad-analysis', use_container_width=True):
                codeeditor_popup(power_code)
    
    # คำแนะนำสำหรับเเนวทางตอบคำถามสำหรับการอภิปราย
    with st.expander("📝 Discussion Guide"):
        st.subheader("Points to consider in your power analysis discussion:")
        
        st.write("""
        1. **Effect of Sample Size on Power:**
           - How does doubling the sample size affect power?
           - Is there a point of diminishing returns?
        
        2. **Effect of Effect Size on Power:**
           - How would detecting a smaller effect change your sample size requirements?
           - If your intervention was more effective, how would that change your study design?
        
        3. **Practical Considerations:**
           - What are the ethical implications of using a smaller sample size?
           - What are the resource implications of using a larger sample size?
           - How would you balance statistical rigor against practical limitations?
        """)
        
        st.info("💡 **Pro Tip:** In your final report to the CEO, include both the statistical findings and the practical implications. For example, if your sample size is very large, discuss recruitment strategies and budget considerations.")
        
    # Simple example code
    with st.expander("🔍 Basic Example Solution"):
        st.code(
            '''import pandas as pd
from statsmodels.stats.power import TTestIndPower
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/heart_attack_vaccine_data.csv')

# Compare cholesterol between vaccine and placebo groups
group1 = data[data['group'] == 'Vaccine']['cholesterol']
group2 = data[data['group'] == 'Placebo']['cholesterol']

# Calculate effect size (Cohen's d)
mean1, mean2 = group1.mean(), group2.mean()
sd1, sd2 = group1.std(), group2.std()
pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
effect_size = abs((mean1 - mean2) / pooled_sd)

# Calculate required sample size for 90% power
power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.9)

print(f"Vaccine group: Mean cholesterol = {mean1:.2f}, SD = {sd1:.2f}")
print(f"Placebo group: Mean cholesterol = {mean2:.2f}, SD = {sd2:.2f}")
print(f"Estimated effect size (Cohen's d): {effect_size:.2f}")
print(f"Required sample size per group: {int(np.ceil(sample_size))}")
print(f"Total sample size needed: {int(np.ceil(sample_size))*2}")''',
            language="python"
        )
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in CodePad (Basic Example)", key='week12-codepad-study', use_container_width=True):
                codeeditor_popup('''import pandas as pd
from statsmodels.stats.power import TTestIndPower
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/heart_attack_vaccine_data.csv')

# Compare cholesterol between vaccine and placebo groups
group1 = data[data['group'] == 'Vaccine']['cholesterol']
group2 = data[data['group'] == 'Placebo']['cholesterol']

# Calculate effect size (Cohen's d)
mean1, mean2 = group1.mean(), group2.mean()
sd1, sd2 = group1.std(), group2.std()
pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
effect_size = abs((mean1 - mean2) / pooled_sd)

# Calculate required sample size for 90% power
power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.9)

print(f"Vaccine group: Mean cholesterol = {mean1:.2f}, SD = {sd1:.2f}")
print(f"Placebo group: Mean cholesterol = {mean2:.2f}, SD = {sd2:.2f}")
print(f"Estimated effect size (Cohen's d): {effect_size:.2f}")
print(f"Required sample size per group: {int(np.ceil(sample_size))}")
print(f"Total sample size needed: {int(np.ceil(sample_size))*2}")''')

def main():
    try:
        tw.initialize_tailwind()
    except Exception as e:
        st.warning(f"Tailwind initialization issue: {e}")
        pass
        
    try:
        Navbar()
        st.title("Week 12 | Power Analysis and Study Design")
        section_table_of_contents()
        st.divider()
        
        # Main content sections with proper anchors
        st.markdown("<a id='statistical-power'></a>", unsafe_allow_html=True)
        section_statistical_power()
        st.divider()
        
        st.markdown("<a id='factors-affecting-power'></a>", unsafe_allow_html=True)
        section_factors_affecting_power()
        st.divider()
        
        st.markdown("<a id='calculating-sample-size'></a>", unsafe_allow_html=True)
        section_calculating_sample_size()
        st.divider()
        
        # Activities section with proper anchors
        st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
        st.header("Activities")
        
        st.markdown("<a id='activity-1-quiz-on-power-analysis-and-study-design'></a>", unsafe_allow_html=True)
        activity_quiz_power_analysis()
        st.divider()
        
        st.markdown("<a id='activity-2-performing-power-analysis-using-software'></a>", unsafe_allow_html=True)
        activity_power_analysis()
        st.divider()
        
        st.markdown("<a id='activity-3-designing-a-study-with-sample-size-calculation'></a>", unsafe_allow_html=True)
        activity_study_design()
        
        Footer(12)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {str(e)}")
        st.error("โปรดแจ้งผู้พัฒนาระบบเพื่อแก้ไขปัญหานี้")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
