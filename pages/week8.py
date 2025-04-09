import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from modules.nav import Navbar
from modules.foot import Footer

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 08 | Risk Ratios, Odds Ratios, and Experimental Design",
)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#risk-ratio-and-odds-ratio">Risk Ratio and Odds Ratio</a></li>
            <li><a href="#interpreting-risk-and-odds-ratios">Interpreting Risk and Odds Ratios</a></li>
            <li><a href="#study-designs-observational-vs-experimental">Study Designs: Observational vs. Experimental</a></li>
            <li><a href="#biases-in-study-design">Biases in Study Design</a></li>
            <li><a href="#activities">Activities</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_risk_odds_ratios():
    st.header("Risk Ratio and Odds Ratio")
    st.write("""
        Risk ratios and odds ratios are measures of association between exposure and outcome in epidemiological studies.
        They are commonly used to quantify the strength of an association between a risk factor and a disease or outcome.
    """)
    
    st.subheader("2×2 Contingency Table")
    st.write("""
        Risk and odds ratios are typically calculated from a 2×2 contingency table like the one below:
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Example 2×2 Table:**")
        table_data = {
            "": ["Exposed", "Not Exposed"],
            "Disease": ["a", "c"],
            "No Disease": ["b", "d"]
        }
        st.table(table_data)
    
    with col2:
        st.markdown("""
        Where:
        - **a** = Number of exposed individuals with the disease
        - **b** = Number of exposed individuals without the disease
        - **c** = Number of unexposed individuals with the disease
        - **d** = Number of unexposed individuals without the disease
        """)
    
    st.subheader("Risk Ratio (Relative Risk)")
    st.write("""
        Risk ratio (RR) is the ratio of the probability of an event occurring in an exposed group versus a non-exposed group.
    """)
    
    st.latex(r"RR = \frac{\text{Risk in exposed}}{\text{Risk in unexposed}} = \frac{a/(a+b)}{c/(c+d)}")
    
    st.subheader("Odds Ratio")
    st.write("""
        Odds ratio (OR) is the ratio of the odds of an event occurring in one group to the odds of it occurring in another group.
    """)
    
    st.latex(r"OR = \frac{\text{Odds in exposed}}{\text{Odds in unexposed}} = \frac{a/b}{c/d} = \frac{ad}{bc}")
    
    st.info("""
        **When to use which ratio?**
        - **Risk Ratio**: Typically used in cohort studies and randomized controlled trials
        - **Odds Ratio**: Typically used in case-control studies and can be calculated in any study design
    """)

def section_interpreting_ratios():
        
    st.subheader("Risk Ratio Interpretation")
    
    # Add your content for risk ratio interpretation here
    st.write("""
        Risk ratios (RR) measure how many times more likely an outcome is in an exposed group compared to an unexposed group:
        
        - **RR = 1**: No association between exposure and outcome
        - **RR > 1**: Positive association (increased risk with exposure)
        - **RR < 1**: Negative association (decreased risk with exposure)
    """)
    
    # Create a table for easier visualization
    interpretation_data = {
        "RR Value": ["RR = 1", "RR > 1", "RR < 1"],
        "Association": ["None", "Positive", "Negative"],
        "Interpretation": [
            "Equal risk in exposed and unexposed groups", 
            "Higher risk in exposed group", 
            "Lower risk in exposed group (protective)"
        ]
    }
    
    st.table(interpretation_data)
    
    st.subheader("Odds Ratio Interpretation")
    
    st.write("""
        Odds ratios (OR) compare the odds of an outcome in the exposed group to the odds in the unexposed group:
        
        - **OR = 1**: No association between exposure and outcome
        - **OR > 1**: Positive association (increased odds with exposure)
        - **OR < 1**: Negative association (decreased odds with exposure)
    """)
    
    # Create a table for easier visualization
    interpretation_data_or = {
        "OR Value": ["OR = 1", "OR > 1", "OR < 1"],
        "Association": ["None", "Positive", "Negative"],
        "Interpretation": [
            "Equal odds in exposed and unexposed groups", 
            "Higher odds in exposed group", 
            "Lower odds in exposed group (protective)"
        ]
    }
    
    st.table(interpretation_data_or)
    
    st.subheader("Comparing Risk Ratio and Odds Ratio")
    
    st.write("""
        - When the outcome is rare (< 10%), RR and OR are approximately equal
        - As the outcome becomes more common, OR tends to overestimate RR
        - OR is often used in case-control studies where RR cannot be calculated directly
    """)

def section_study_designs():
    st.header("Study Designs: Observational vs. Experimental")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Observational Studies")
        st.write("""
            In observational studies, researchers observe subjects without manipulation or intervention.
            
            **Types:**
            - **Cross-sectional**: Measures exposure and outcome at the same time
            - **Case-control**: Compares cases (with disease) to controls (without disease)
            - **Cohort**: Follows subjects over time from exposure to outcome
            
            **Advantages:**
            - More feasible for rare diseases
            - Ethical when exposure is harmful
            - Can study multiple exposures and outcomes
            
            **Limitations:**
            - Cannot establish causation definitively
            - Subject to various biases
            - Cannot control for unknown confounders
        """)
    
    with col2:
        st.subheader("Experimental Studies")
        st.write("""
            In experimental studies, researchers actively intervene and assign exposures to subjects.
            
            **Types:**
            - **Randomized Controlled Trial (RCT)**: Gold standard with random assignment
            - **Non-randomized trial**: Assignment not randomized
            - **Crossover design**: Subjects serve as their own controls
            
            **Advantages:**
            - Can establish causation
            - Randomization controls for confounders
            - More controlled environment
            
            **Limitations:**
            - Ethical concerns for harmful exposures
            - Often expensive and time-consuming
            - May have limited external validity (generalizability)
        """)
    
    st.subheader("Hierarchy of Evidence")
    
    # Create a pyramid diagram to show hierarchy of evidence
    labels = ['Systematic Reviews\n& Meta-Analyses', 'Randomized\nControlled Trials', 
              'Cohort Studies', 'Case-Control Studies', 'Case Series/Reports', 
              'Expert Opinion']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create triangle/pyramid shape
    for i, (label, color) in enumerate(zip(labels, colors)):
        height = len(labels) - i
        width = (i + 1) * 2
        left = (len(labels) * 2 - width) / 2
        ax.bar(left + width/2, height, width=width, color=color, edgecolor='white', linewidth=1)
        ax.text(len(labels), height - 0.5, label, ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, len(labels) * 2)
    ax.set_ylim(0, len(labels) + 0.5)
    ax.set_title('Hierarchy of Evidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strength of Evidence', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    st.pyplot(fig)
    
    st.info("""
        The hierarchy of evidence represents the strength and quality of different study designs,
        with systematic reviews and meta-analyses at the top providing the strongest evidence,
        and expert opinion at the bottom providing the weakest evidence.
    """)

def section_biases():
    st.header("Biases in Study Design")
    
    st.write("""
        Bias is a systematic error in the design, conduct, or analysis of a study that results in a mistaken estimate
        of an exposure's effect on the risk of disease. Recognizing and addressing biases is crucial for valid research.
    """)
    
    st.subheader("Selection Bias")
    st.write("""
        Selection bias occurs when the selection of subjects into a study or their retention in the study leads to a result that is different from what would be obtained in the full population.
        
        **Examples:**
        - **Berkson's bias**: When hospitalized patients are used as cases, as hospitalization depends on both exposure and disease
        - **Healthy worker effect**: When employed individuals are healthier than the general population
        - **Loss to follow-up**: When participants who drop out differ systematically from those who remain
        
        **Prevention strategies:**
        - Random sampling from the target population
        - High participation rates
        - Minimal loss to follow-up
    """)
    
    st.subheader("Information Bias")
    st.write("""
        Information bias results from inaccurate measurement or classification of key variables (exposure, outcome, or covariates).
        
        **Examples:**
        - **Recall bias**: Differential recall of information by cases and controls
        - **Observer bias**: When knowledge of subject's disease status influences measurement
        - **Misclassification**: Incorrect categorization of exposure or disease status
        
        **Prevention strategies:**
        - Standardized data collection procedures
        - Blinding of observers to study hypotheses
        - Validation of measurement instruments
    """)
    
    st.subheader("Confounding Bias")
    st.write("""
        Confounding occurs when the relationship between exposure and outcome is influenced by a third variable that is associated with both.
        
        **Criteria for a confounder:**
        - Associated with the exposure
        - Independent risk factor for the outcome
        - Not on the causal pathway between exposure and outcome
        
        **Prevention and control strategies:**
        - Randomization (in experimental studies)
        - Matching cases and controls on potential confounders
        - Stratification during analysis
        - Statistical adjustment (e.g., regression models)
    """)
    
    # Visual representation of confounding
    st.subheader("Visual Representation of Confounding")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create nodes
    exposure_pos = (0.2, 0.5)
    outcome_pos = (0.8, 0.5)
    confounder_pos = (0.5, 0.8)
    
    # Draw arrows
    ax.annotate("", xy=outcome_pos, xytext=exposure_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color='blue'))
    ax.annotate("", xy=outcome_pos, xytext=confounder_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color='red'))
    ax.annotate("", xy=exposure_pos, xytext=confounder_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color='red'))
    
    # Add labels
    ax.text(exposure_pos[0], exposure_pos[1]-0.05, "Exposure", ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    ax.text(outcome_pos[0], outcome_pos[1]-0.05, "Outcome", ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    ax.text(confounder_pos[0], confounder_pos[1]+0.05, "Confounder", ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    # Set plot limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    st.pyplot(fig)

def activity_quiz():
    st.header("Activity 1: Quiz on Risk Ratios, Odds Ratios, and Study Design")
    st.write("Test your understanding of this week's concepts.")

    # Create a form for the quiz
    with st.form(key='rr_or_quiz'):
        st.subheader("Please answer the following questions:")

        q1 = st.radio("1. Which measure is most appropriate for a case-control study?",
                        ("Risk Ratio", 
                         "Odds Ratio",
                         "Both are equally appropriate",
                         "Neither is appropriate"), index=None)

        q2 = st.radio("2. If a risk ratio (RR) equals 0.75, this indicates:",
                        ("The exposed group has 75% higher risk than the unexposed group",
                         "The exposed group has 25% lower risk than the unexposed group", 
                         "The exposed group has 75% lower risk than the unexposed group",
                         "The exposed group has 25% higher risk than the unexposed group"), index=None)

        q3 = st.radio("3. When would risk ratio and odds ratio values be approximately equal?",
                        ("When the disease is common",
                         "When the disease is rare",
                         "When the sample size is large",
                         "When the exposure is rare"), index=None)

        q4 = st.radio("4. Which of the following is an experimental study design?",
                        ("Case-control study",
                         "Cross-sectional study",
                         "Randomized controlled trial",
                         "Cohort study"), index=None)

        q5 = st.radio("5. Which bias occurs when participants who drop out of a study differ systematically from those who remain?",
                        ("Selection bias",
                         "Information bias",
                         "Recall bias",
                         "Observer bias"), index=None)

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        correct_answers = {
            "q1": "Odds Ratio",
            "q2": "The exposed group has 25% lower risk than the unexposed group",
            "q3": "When the disease is rare",
            "q4": "Randomized controlled trial",
            "q5": "Selection bias"
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

def activity_calculate_rr_or():
    st.header("Activity 2: Calculate and Interpret Risk Ratios and Odds Ratios")

    # Load and display the Lung Cancer dataset
    try:
        lung_cancer_data = pd.read_csv("data/lung_cancer.csv")
        st.subheader("Lung Cancer Dataset")
        st.write("This dataset contains information about patients and their lung cancer status along with various risk factors.")
        with st.expander("View Dataset"):
            st.dataframe(lung_cancer_data)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    # Compute contingency table counts from the dataset
    smokers_with_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] == 2) & (lung_cancer_data['LUNG_CANCER'] == 'YES')].shape[0]
    smokers_without_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] == 2) & (lung_cancer_data['LUNG_CANCER'] == 'NO')].shape[0]
    nonsmokers_with_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] != 2) & (lung_cancer_data['LUNG_CANCER'] == 'YES')].shape[0]
    nonsmokers_without_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] != 2) & (lung_cancer_data['LUNG_CANCER'] == 'NO')].shape[0]

    st.subheader("Scenario: Analyzing Smoking and Lung Cancer")
    st.write("""
    Using the dataset, we can analyze the relationship between smoking and lung cancer. In this activity, you will:
    1. Create a 2×2 contingency table
    2. Calculate risk and odds ratios
    3. Interpret the results
    
    Let's start by constructing our 2×2 contingency table:
    """)

    # Show the structure of a 2×2 contingency table
    st.write("#### 2×2 Contingency Table Structure")
    col1, col2 = st.columns([2,2])
    with col1:
        table_structure = pd.DataFrame({
            "": ["Exposed (Smokers)", "Unexposed (Non-smokers)"],
            "Disease (Lung Cancer)": ["a", "c"],
            "No Disease": ["b", "d"]
        })
        st.table(table_structure)
    
    with col2:
        st.markdown("""
        Where:
        - **a** = Smokers with lung cancer
        - **b** = Smokers without lung cancer
        - **c** = Non-smokers with lung cancer
        - **d** = Non-smokers without lung cancer
        """)

    # Dataset values table
    st.write("#### Values from Dataset")
    table_ref = pd.DataFrame({
        "": ["Smokers", "Non-smokers", "Total"],
        "Lung Cancer": [smokers_with_cancer, nonsmokers_with_cancer, smokers_with_cancer + nonsmokers_with_cancer],
        "No Lung Cancer": [smokers_without_cancer, nonsmokers_without_cancer, smokers_without_cancer + nonsmokers_without_cancer],
        "Total": [smokers_with_cancer + smokers_without_cancer, 
                  nonsmokers_with_cancer + nonsmokers_without_cancer,
                  smokers_with_cancer + smokers_without_cancer + nonsmokers_with_cancer + nonsmokers_without_cancer]
    })
    st.table(table_ref)

    # Set values from the dataset
    user_a = smokers_with_cancer
    user_b = smokers_without_cancer
    user_c = nonsmokers_with_cancer
    user_d = nonsmokers_without_cancer

    # Calculate button
    calculate_btn = st.button("📊 Calculate Risk and Odds Ratios", type="primary", use_container_width=True)

    if calculate_btn:
        st.write("### Calculation Results")
        
        # Calculate measures
        risk_exposed = user_a / (user_a + user_b) if (user_a + user_b) > 0 else 0
        risk_unexposed = user_c / (user_c + user_d) if (user_c + user_d) > 0 else 0
        risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
        
        odds_exposed = user_a / user_b if user_b > 0 else float('inf')
        odds_unexposed = user_c / user_d if user_d > 0 else float('inf')
        odds_ratio = (user_a * user_d) / (user_b * user_c) if user_b > 0 and user_c > 0 else float('inf')

        # Display calculations with formulas
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Risk Ratio Calculation")
            st.latex(r"RR = \frac{\text{Risk in exposed}}{\text{Risk in unexposed}} = \frac{a/(a+b)}{c/(c+d)}")
            
            st.write(f"Risk in exposed (smokers): {user_a}/{user_a + user_b} = {risk_exposed:.4f}")
            st.write(f"Risk in unexposed (non-smokers): {user_c}/{user_c + user_d} = {risk_unexposed:.4f}")
            st.write(f"Risk Ratio (RR): {risk_exposed:.4f}/{risk_unexposed:.4f} = **{risk_ratio:.2f}**")
        
        with col2:
            st.write("#### Odds Ratio Calculation")
            st.latex(r"OR = \frac{\text{Odds in exposed}}{\text{Odds in unexposed}} = \frac{a/b}{c/d} = \frac{ad}{bc}")
            
            st.write(f"Odds in exposed (smokers): {user_a}/{user_b} = {odds_exposed:.4f}")
            st.write(f"Odds in unexposed (non-smokers): {user_c}/{user_d} = {odds_unexposed:.4f}")
            st.write(f"Odds Ratio (OR): {odds_exposed:.4f}/{odds_unexposed:.4f} = **{odds_ratio:.2f}**")

        # Interpretation
        st.write("### Interpretation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Risk Ratio Interpretation")
            if risk_ratio > 1.1:
                st.success(f"Risk Ratio = {risk_ratio:.2f} > 1: Smokers have {risk_ratio:.2f} times higher risk of lung cancer compared to non-smokers.")
            elif risk_ratio < 0.9:
                st.success(f"Risk Ratio = {risk_ratio:.2f} < 1: Smoking appears protective with {1/risk_ratio:.2f} times lower risk of lung cancer.")
            else:
                st.success(f"Risk Ratio ≈ 1: No substantial association between smoking and lung cancer risk.")
        
        with col2:
            st.write("#### Odds Ratio Interpretation")
            if odds_ratio > 1.1:
                st.success(f"Odds Ratio = {odds_ratio:.2f} > 1: The odds of lung cancer are {odds_ratio:.2f} times higher in smokers.")
            elif odds_ratio < 0.9:
                st.success(f"Odds Ratio = {odds_ratio:.2f} < 1: The odds of lung cancer are {1/odds_ratio:.2f} times lower in smokers.")
            else:
                st.success(f"Odds Ratio ≈ 1: No substantial association between smoking and lung cancer odds.")

        # Visualization
        st.write("### Visualization")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Risk comparison
        risks = [risk_unexposed, risk_exposed]
        ax1.bar(['Non-smokers', 'Smokers'], risks, color=['skyblue', 'salmon'])
        ax1.set_title('Risk Comparison')
        ax1.set_ylabel('Risk of Lung Cancer')
        for i, v in enumerate(risks):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Odds comparison
        odds = [odds_unexposed, odds_exposed]
        ax2.bar(['Non-smokers', 'Smokers'], odds, color=['lightgreen', 'plum'])
        ax2.set_title('Odds Comparison')
        ax2.set_ylabel('Odds of Lung Cancer')
        for i, v in enumerate(odds):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        st.pyplot(fig)

    # Provide sample code for reference
    st.write("### Reference Code for Calculations")
    
    sample_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data into a Pandas DataFrame
lung_cancer_data = pd.read_csv("data/lung_cancer.csv")

# Compute contingency table counts from the dataset
smokers_with_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] == 2) & (lung_cancer_data['LUNG_CANCER'] == 'YES')].shape[0]
smokers_without_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] == 2) & (lung_cancer_data['LUNG_CANCER'] == 'NO')].shape[0]
nonsmokers_with_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] != 2) & (lung_cancer_data['LUNG_CANCER'] == 'YES')].shape[0]
nonsmokers_without_cancer = lung_cancer_data[(lung_cancer_data['SMOKING'] != 2) & (lung_cancer_data['LUNG_CANCER'] == 'NO')].shape[0]

# Print the contingency table
print("Contingency Table:")
print(f"                  Lung Cancer   No Lung Cancer   Total")
print(f"Smokers           {smokers_with_cancer:<13} {smokers_without_cancer:<15} {smokers_with_cancer + smokers_without_cancer}")
print(f"Non-smokers       {nonsmokers_with_cancer:<13} {nonsmokers_without_cancer:<15} {nonsmokers_with_cancer + nonsmokers_without_cancer}")
print(f"Total             {smokers_with_cancer + nonsmokers_with_cancer:<13} {smokers_without_cancer + nonsmokers_without_cancer:<15} {smokers_with_cancer + smokers_without_cancer + nonsmokers_with_cancer + nonsmokers_without_cancer}")

def calculate_measures(a, b, c, d):
    # Calculate risks
    risk_exposed = a / (a + b) if (a + b) > 0 else 0
    risk_unexposed = c / (c + d) if (c + d) > 0 else 0
    risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
    
    # Calculate odds
    odds_exposed = a / b if b > 0 else float('inf')
    odds_unexposed = c / d if d > 0 else float('inf')
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
    
    return {
        'risk_exposed': risk_exposed,
        'risk_unexposed': risk_unexposed,
        'risk_ratio': risk_ratio,
        'odds_exposed': odds_exposed, 
        'odds_unexposed': odds_unexposed,
        'odds_ratio': odds_ratio
    }

# Values from the Lung Cancer dataset
a = smokers_with_cancer
b = smokers_without_cancer
c = nonsmokers_with_cancer
d = nonsmokers_without_cancer

# Calculate all measures
results = calculate_measures(a, b, c, d)

# Print results with interpretation
print("\\nCalculated Results:")
print(f"Risk in Smokers: {results['risk_exposed']:.4f}")
print(f"Risk in Non-smokers: {results['risk_unexposed']:.4f}")
print(f"Risk Ratio (RR): {results['risk_ratio']:.2f}")
print(f"Odds in Smokers: {results['odds_exposed']:.4f}")
print(f"Odds in Non-smokers: {results['odds_unexposed']:.4f}")
print(f"Odds Ratio (OR): {results['odds_ratio']:.2f}")

# Interpretation
print("\\nInterpretation:")
if results['risk_ratio'] > 1:
    print(f"Risk Ratio > 1: Smokers have {results['risk_ratio']:.2f} times higher risk of lung cancer")
elif results['risk_ratio'] < 1:
    print(f"Risk Ratio < 1: Smoking appears protective with {1/results['risk_ratio']:.2f} times lower risk")
else:
    print("Risk Ratio = 1: No association between smoking and lung cancer risk")

if results['odds_ratio'] > 1:
    print(f"Odds Ratio > 1: The odds of lung cancer are {results['odds_ratio']:.2f} times higher in smokers")
elif results['odds_ratio'] < 1:
    print(f"Odds Ratio < 1: The odds of lung cancer are {1/results['odds_ratio']:.2f} times lower in smokers")
else:
    print("Odds Ratio = 1: No association between smoking and lung cancer odds")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Risk comparison
risks = [results['risk_unexposed'], results['risk_exposed']]
ax1.bar(['Non-smokers', 'Smokers'], risks, color=['skyblue', 'salmon'])
ax1.set_title('Risk Comparison')
ax1.set_ylabel('Risk of Lung Cancer')
for i, v in enumerate(risks):
    ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')

# Odds comparison
odds = [results['odds_unexposed'], results['odds_exposed']]
ax2.bar(['Non-smokers', 'Smokers'], odds, color=['lightgreen', 'plum'])
ax2.set_title('Odds Comparison')
ax2.set_ylabel('Odds of Lung Cancer')
for i, v in enumerate(odds):
    ax2.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()
"""

    with st.expander("View Reference Code"):
        st.code(sample_code, language="python")

    # CodePad integration
    @st.dialog("CodePad", width="large")
    def codeeditor_popup(default_code=None):
        code_editor_for_all(default_code=default_code, key='week8-rr-or-code', warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)")

    if st.button("Open CodePad", key="open_codepad_rr_or", use_container_width=True):
        codeeditor_popup(default_code=sample_code)


def main():
    Navbar()
    st.title("Week 08 | Risk Ratios, Odds Ratios, and Experimental Design")
    section_table_of_contents()
    section_risk_odds_ratios()
    section_interpreting_ratios()
    section_study_designs()
    section_biases()
    st.divider()
    st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
    st.header("Activities")
    activity_quiz()
    st.divider()
    activity_calculate_rr_or()
    Footer(8)

if __name__ == "__main__":
    main()