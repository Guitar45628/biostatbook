import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import statsmodels.api as sm
import statsmodels.formula.api as smf

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 11 | Logistic Regression and Binary Outcomes",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code, key='codepad-week11', warning_text=warning_text)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#introduction-to-logistic-regression">Introduction to Logistic Regression</a></li>
            <li><a href="#odds-ratios-interpretation">Odds Ratios and Their Interpretation</a></li>
            <li><a href="#model-diagnostics">Model Diagnostics and Goodness-of-Fit</a></li>
            <li><a href="#activities">Activities</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_introduction_to_logistic_regression():
    st.header("Introduction to Logistic Regression")
    
    st.write("""
        Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent 
        variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there 
        are only two possible outcomes).
    """)
    
    st.subheader("Why Use Logistic Regression?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
            Logistic regression is particularly useful in medical and health sciences where:
            
            - We want to predict binary outcomes (e.g., disease present/absent)
            - We need to understand risk factors and their impact on outcomes
            - We want to calculate odds and odds ratios for different variables
            - We need to adjust for confounding variables
        """)
    
    with col2:
        st.image("assets/week5-1.png", width=300)  # Reusing image from week 5
    
    st.subheader("The Logistic Model")
    
    st.write("""
        Unlike linear regression which directly predicts a continuous outcome, logistic regression models the probability 
        of an outcome using the logistic function:
    """)
    
    st.latex(r"P(Y=1) = \frac{e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n}}{1 + e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n}}")
    
    st.write("""
        This can also be written in terms of the logit transformation (log odds):
    """)
    
    st.latex(r"\text{logit}(P) = \ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n")
    
    st.info("""
        The logistic function ensures that our predicted probabilities are always between 0 and 1, unlike 
        linear regression which can produce predictions outside this range.
    """)
    
    st.subheader("Example Code: Fitting a Simple Logistic Regression")
    
    logistic_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")

# Convert the target variable to binary (0/1)
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)

# Select a predictor variable (e.g., SMOKING)
X = data[['SMOKING']]
y = data['LUNG_CANCER_BINARY']

# Fit the logistic regression model using sklearn
model = LogisticRegression()
model.fit(X, y)

# Print the coefficients
print("Coefficient:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# Calculate predicted probabilities
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_proba = model.predict_proba(X_range)[:, 1]

# Plot the logistic curve
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_range, y_proba, color='red', linewidth=2)
plt.xlabel('Smoking Status')
plt.ylabel('Probability of Lung Cancer')
plt.title('Simple Logistic Regression Model')
plt.grid(True, alpha=0.3)
plt.show()

# Using statsmodels for more statistical details
X_with_const = sm.add_constant(X)
sm_model = sm.Logit(y, X_with_const).fit()
print(sm_model.summary())
"""

    st.code(logistic_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-intro', use_container_width=True):
            codeeditor_popup(logistic_code)

def section_odds_ratios_interpretation():
    st.header("Odds Ratios and Their Interpretation")
    
    st.write("""
        A key advantage of logistic regression is the ability to interpret results in terms of odds ratios,
        which are much more intuitive for binary outcomes than coefficients in linear regression.
    """)
    
    st.subheader("Understanding Odds vs. Probability")
    
    st.write("""
        - **Probability**: The chance of an event occurring, ranging from 0 to 1
        - **Odds**: The ratio of the probability of success to the probability of failure
    """)
    
    st.latex(r"\text{Odds} = \frac{P}{1-P}")
    
    st.subheader("The Odds Ratio")
    
    st.write("""
        The odds ratio (OR) represents the odds that an outcome will occur given a particular exposure, 
        compared to the odds of the outcome occurring in the absence of that exposure.
    """)
    
    st.latex(r"\text{OR} = \frac{\text{Odds for exposed}}{\text{Odds for unexposed}} = e^{\beta}")
    
    st.write("""
        Where β is the coefficient from the logistic regression model. This is a key advantage of 
        logistic regression - the exponentiated coefficients directly give you odds ratios.
    """)
    
    st.info("""
        - OR = 1: Exposure does not affect odds of outcome
        - OR > 1: Exposure associated with higher odds of outcome
        - OR < 1: Exposure associated with lower odds of outcome
    """)
    
    st.subheader("Example: Interpreting Odds Ratios")
    
    odds_ratio_code = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")

# Convert the target variable to binary
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)

# Fit a logistic regression model with multiple predictors
formula = 'LUNG_CANCER_BINARY ~ SMOKING + AGE + GENDER + ANXIETY'
model = smf.logit(formula=formula, data=data).fit()

# Display the summary
print(model.summary())

# Calculate odds ratios and 95% confidence intervals
params = model.params
conf = model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
conf['OR'] = np.exp(conf['OR'])
conf['2.5%'] = np.exp(conf['2.5%'])
conf['97.5%'] = np.exp(conf['97.5%'])

# Display the odds ratios
print("\\nOdds Ratios:")
print(conf)

# Interpretation example
print("\\nInterpretation:")
for var in ['SMOKING', 'AGE', 'GENDER', 'ANXIETY']:
    if var in conf.index:
        or_value = conf.loc[var, 'OR']
        ci_low = conf.loc[var, '2.5%']
        ci_high = conf.loc[var, '97.5%']
        
        if or_value > 1:
            direction = "increased"
        elif or_value < 1:
            direction = "decreased"
        else:
            direction = "did not change"
            
        print(f"{var}: OR = {or_value:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})")
        print(f"  Interpretation: A one-unit increase in {var} is associated with {direction} odds of lung cancer.")
"""

    st.code(odds_ratio_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-odds', use_container_width=True):
            codeeditor_popup(odds_ratio_code)
    
    st.subheader("Confounding and Adjusted Odds Ratios")
    
    st.write("""
        One of the main advantages of logistic regression is the ability to adjust for confounding variables.
        
        - **Crude OR**: Calculated from a univariate logistic regression model with a single predictor
        - **Adjusted OR**: Calculated from a multivariate model that includes potential confounders
        
        If there's a substantial difference between crude and adjusted ORs, it suggests the presence of confounding.
    """)

def section_model_diagnostics():
    st.header("Model Diagnostics and Goodness-of-Fit")
    
    st.write("""
        After fitting a logistic regression model, it's important to assess how well the model fits the data
        and its predictive performance.
    """)
    
    st.subheader("1. Assessing Model Fit")
    
    st.write("""
        **Likelihood Ratio Test**
        
        Compares the likelihood of the data under the full model against the likelihood of the data under a model with only an intercept.
        
        **Hosmer-Lemeshow Test**
        
        A statistical test for goodness of fit for logistic regression models. A p-value greater than 0.05
        suggests that the model fits the data well.
    """)
    
    st.subheader("2. Classification Metrics")
    
    st.write("""
        **Confusion Matrix**
        
        A table that is often used to describe the performance of a classification model:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
            - True Positives (TP): Correctly predicted positive cases
            - True Negatives (TN): Correctly predicted negative cases
            - False Positives (FP): Incorrectly predicted positive cases
            - False Negatives (FN): Incorrectly predicted negative cases
        """)
    
    with col2:
        confusion_data = [
            ["", "Predicted Positive", "Predicted Negative"],
            ["Actual Positive", "True Positive (TP)", "False Negative (FN)"],
            ["Actual Negative", "False Positive (FP)", "True Negative (TN)"]
        ]
        
        confusion_df = pd.DataFrame(confusion_data[1:], columns=confusion_data[0])
        st.table(confusion_df)
    
    st.write("""
        **Common Metrics**
        
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Sensitivity: TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - Precision: TP / (TP + FP)
    """)
    
    st.subheader("3. ROC Curve and AUC")
    
    st.write("""
        The **Receiver Operating Characteristic (ROC) curve** plots the true positive rate (sensitivity)
        against the false positive rate (1-specificity) at various threshold settings.
        
        The **Area Under the Curve (AUC)** measures the ability of the model to distinguish between
        positive and negative classes.
        
        - AUC = 0.5: No discrimination (equivalent to random guessing)
        - 0.7 ≤ AUC < 0.8: Acceptable discrimination
        - 0.8 ≤ AUC < 0.9: Excellent discrimination
        - AUC ≥ 0.9: Outstanding discrimination
    """)
    
    st.subheader("Example: Model Diagnostics Code")
    
    diagnostics_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import seaborn as sns
from scipy.stats import chi2

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)

# Select features
features = ['SMOKING', 'AGE', 'GENDER', 'ANXIETY', 'FATIGUE', 'SHORTNESS OF BREATH']
X = data[features]
y = data['LUNG_CANCER_BINARY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 2. Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\\n", report)

# 3. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC-ROC: {roc_auc:.3f}")

# 4. Examine coefficients and their significance
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
sm_model = sm.Logit(y_train, X_train_sm).fit()
print(sm_model.summary())

print("\\nOdds Ratios:")
print(np.exp(sm_model.params))
"""

    st.code(diagnostics_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-diagnostics', use_container_width=True):
            codeeditor_popup(diagnostics_code)

def activity_fitting_logistic_models():
    st.header("Activity 1: Fitting Logistic Regression Models")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # Reusing image from week 5
    with cent_co:
        st.subheader("Predicting Lung Cancer with Logistic Regression")
        st.write("In this activity, you'll practice fitting logistic regression models to the lung cancer dataset.")
    
    st.markdown("---")
    
    st.write("""
        You'll be working with the lung cancer dataset, which contains various features about patients and whether they have
        been diagnosed with lung cancer.
        
        Your goal is to build a logistic regression model to predict the presence of lung cancer based on various risk factors.
    """)
    
    st.subheader("Step 1: Explore the Data")
    
    # Load the dataset for demonstration
    file_path = os.path.join(os.getcwd(), "data/lung_cancer.csv")
    try:
        data = pd.read_csv(file_path)
        st.write("First few rows of the dataset:")
        st.dataframe(data.head())
        
        st.write("Summary statistics:")
        st.dataframe(data.describe())
        
        # Count of lung cancer cases
        cancer_counts = data['LUNG_CANCER'].value_counts()
        st.write("Lung cancer distribution:")
        st.write(f"YES: {cancer_counts.get('YES', 0)} cases")
        st.write(f"NO: {cancer_counts.get('NO', 0)} cases")
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.write("Please make sure the file exists at the specified path.")
    
    st.subheader("Step 2: Fit a Basic Logistic Regression Model")
    
    basic_model_code = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")

# Convert categorical variables to numeric
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)
data['GENDER_BINARY'] = (data['GENDER'] == 'M').astype(int)  # Male = 1, Female = 0

# Select a single predictor for a simple model
X = sm.add_constant(data[['SMOKING']])
y = data['LUNG_CANCER_BINARY']

# Fit the logistic regression model
model = sm.Logit(y, X).fit()

# Print the summary
print(model.summary())

# Calculate odds ratio
odds_ratio = np.exp(model.params['SMOKING'])
print(f"\\nOdds Ratio for SMOKING: {odds_ratio:.3f}")
print(f"Interpretation: For each unit increase in smoking status, the odds of lung cancer increase by {(odds_ratio-1)*100:.1f}%")

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.regplot(x='SMOKING', y='LUNG_CANCER_BINARY', data=data, logistic=True, ci=None)
plt.xlabel('Smoking Status')
plt.ylabel('Probability of Lung Cancer')
plt.title('Logistic Regression: Smoking vs. Lung Cancer')
plt.grid(True, alpha=0.3)
plt.show()
"""
    
    st.code(basic_model_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-basic-model', use_container_width=True):
            codeeditor_popup(basic_model_code)
    
    st.subheader("Step 3: Multiple Logistic Regression")
    
    st.write("""
        Now let's fit a more complex model with multiple predictors to better understand 
        the factors associated with lung cancer.
    """)
    
    multiple_model_code = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")

# Convert categorical variables to numeric
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)
data['GENDER_BINARY'] = (data['GENDER'] == 'M').astype(int)  # Male = 1, Female = 0

# Select multiple predictors
predictors = ['GENDER_BINARY', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 
              'ANXIETY', 'CHRONIC DISEASE', 'FATIGUE', 'SHORTNESS OF BREATH']

X = sm.add_constant(data[predictors])
y = data['LUNG_CANCER_BINARY']

# Fit the logistic regression model
model = sm.Logit(y, X).fit()

# Print the summary
print(model.summary())

# Calculate and display odds ratios with confidence intervals
params = model.params
conf = model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
conf['OR'] = np.exp(conf['OR'])
conf['2.5%'] = np.exp(conf['2.5%'])
conf['97.5%'] = np.exp(conf['97.5%'])

print("\\nOdds Ratios and 95% Confidence Intervals:")
print(conf)

# Create a visualization of odds ratios
plt.figure(figsize=(10, 8))
conf_subset = conf.iloc[1:].sort_values(by='OR')  # Exclude the constant term
plt.errorbar(
    conf_subset['OR'], 
    range(len(conf_subset)), 
    xerr=[conf_subset['OR'] - conf_subset['2.5%'], conf_subset['97.5%'] - conf_subset['OR']],
    fmt='o'
)

# Add a vertical line at OR = 1
plt.axvline(x=1, color='red', linestyle='--')

# Set the y-tick labels to the variable names
plt.yticks(range(len(conf_subset)), conf_subset.index)
plt.xlabel('Odds Ratio (log scale)')
plt.xscale('log')
plt.title('Odds Ratios with 95% Confidence Intervals')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
    
    st.code(multiple_model_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-multiple-model', use_container_width=True):
            codeeditor_popup(multiple_model_code)
    
    st.subheader("Step 4: Model Evaluation and Interpretation")
    
    st.write("""
        Finally, let's evaluate our model's performance and interpret the results.
    """)
    
    evaluation_code = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("data/lung_cancer.csv")

# Convert categorical variables to numeric
data['LUNG_CANCER_BINARY'] = (data['LUNG_CANCER'] == 'YES').astype(int)
data['GENDER_BINARY'] = (data['GENDER'] == 'M').astype(int)  # Male = 1, Female = 0

# Select features
predictors = ['GENDER_BINARY', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 
              'ANXIETY', 'CHRONIC DISEASE', 'FATIGUE', 'SHORTNESS OF BREATH']

X = data[predictors]
y = data['LUNG_CANCER_BINARY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit the model using sklearn (easier for prediction)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No Cancer', 'Cancer'],
           yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:\\n")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC-ROC: {roc_auc:.3f}")

# Feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})
print("\\nFeature Importance:")
print(coefficients.sort_values('Coefficient', ascending=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
coefficients_sorted = coefficients.sort_values('Coefficient')
plt.barh(coefficients_sorted['Feature'], coefficients_sorted['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients')
plt.grid(True, alpha=0.3)
plt.show()
"""
    
    st.code(evaluation_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-evaluation', use_container_width=True):
            codeeditor_popup(evaluation_code)
    
    st.subheader("Questions to Answer:")
    
    with st.form(key="logistic_questions_form"):
        st.write("**1. Based on the multiple logistic regression model, which factor has the strongest association with lung cancer?**")
        q1 = st.radio(
            "Choose the best answer:",
            [
                "Gender",
                "Age",
                "Smoking",
                "Anxiety",
                "Cannot determine without running the code"
            ],
            index=None
        )
        
        st.write("**2. How would you interpret an odds ratio of 2.5 for the 'SMOKING' variable?**")
        q2 = st.radio(
            "Choose the best interpretation:",
            [
                "Smokers are 2.5 times more likely to get lung cancer",
                "Smokers are 2.5% more likely to get lung cancer",
                "The odds of lung cancer are 2.5 times higher for smokers compared to non-smokers",
                "The probability of lung cancer increases by 2.5 units for smokers"
            ],
            index=None
        )
        
        st.write("**3. What does an AUC-ROC value of 0.85 indicate about your model?**")
        q3 = st.radio(
            "Choose the best answer:",
            [
                "The model correctly classifies 85% of all cases",
                "The model has excellent discriminative ability",
                "The model explains 85% of the variance in the outcome",
                "85% of predictions are true positives"
            ],
            index=None
        )
        
        submit = st.form_submit_button("Submit Answers")
    
    if submit:
        score = 0
        feedback = []
        
        # Question 1
        correct_q1 = "Cannot determine without running the code"
        if q1 == correct_q1:
            score += 1
            feedback.append("✅ Question 1: Correct! You need to run the code to determine which factor has the strongest association.")
        else:
            feedback.append("❌ Question 1: Incorrect. The correct answer is: Cannot determine without running the code. You need to analyze the actual coefficients.")
        
        # Question 2
        correct_q2 = "The odds of lung cancer are 2.5 times higher for smokers compared to non-smokers"
        if q2 == correct_q2:
            score += 1
            feedback.append("✅ Question 2: Correct! The odds ratio represents how the odds change with a unit change in the predictor.")
        else:
            feedback.append(f"❌ Question 2: Incorrect. The correct interpretation is: {correct_q2}")
        
        # Question 3
        correct_q3 = "The model has excellent discriminative ability"
        if q3 == correct_q3:
            score += 1
            feedback.append("✅ Question 3: Correct! An AUC-ROC value of 0.85 indicates excellent discriminative ability.")
        else:
            feedback.append(f"❌ Question 3: Incorrect. The correct answer is: {correct_q3}")
        
        st.write(f"Your score: {score}/3")
        for fb in feedback:
            st.write(fb)

def activity_interpreting_odds_ratios():
    st.header("Activity 2: Interpreting Odds Ratios and Confidence Intervals")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week4-i2.png", width=300)  # Reusing image
    with cent_co:
        st.subheader("Clinical Interpretation of Logistic Regression Results")
        st.write("Learn how to interpret odds ratios and their confidence intervals in a clinical context.")
    
    st.markdown("---")
    
    st.write("""
        In this activity, you'll practice interpreting the results of a logistic regression analysis,
        focusing on odds ratios and their confidence intervals in a clinical context.
    """)
    
    st.subheader("Scenario: Lung Cancer Risk Factors Study")
    
    st.write("""
        A research team conducted a study to identify risk factors for lung cancer. They collected data from 
        300 participants and performed a logistic regression analysis. The results are shown below.
    """)
    
    # Create a mock results table for the activity
    results_data = {
        'Variable': ['SMOKING', 'AGE (per 10 years)', 'GENDER (Male vs. Female)', 'SHORTNESS OF BREATH', 'CHEST PAIN'],
        'Odds Ratio': [3.12, 1.45, 1.22, 2.87, 2.31],
        '95% CI Lower': [2.18, 1.21, 0.78, 1.92, 1.54],
        '95% CI Upper': [4.46, 1.73, 1.91, 4.29, 3.47],
        'p-value': [0.0001, 0.0004, 0.38, 0.0001, 0.0001]
    }
    
    results_df = pd.DataFrame(results_data)
    st.table(results_df)
    
    st.subheader("Tasks:")
    
    interpretation_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the results dataframe
results_data = {
    'Variable': ['SMOKING', 'AGE (per 10 years)', 'GENDER (Male vs. Female)', 'SHORTNESS OF BREATH', 'CHEST PAIN'],
    'Odds Ratio': [3.12, 1.45, 1.22, 2.87, 2.31],
    '95% CI Lower': [2.18, 1.21, 0.78, 1.92, 1.54],
    '95% CI Upper': [4.46, 1.73, 1.91, 4.29, 3.47],
    'p-value': [0.0001, 0.0004, 0.38, 0.0001, 0.0001]
}

results_df = pd.DataFrame(results_data)

# Create a forest plot of odds ratios
plt.figure(figsize=(12, 8))

# Plot the odds ratios and their confidence intervals
y_pos = range(len(results_df))
plt.errorbar(
    x=results_df['Odds Ratio'],
    y=y_pos,
    xerr=[
        results_df['Odds Ratio'] - results_df['95% CI Lower'], 
        results_df['95% CI Upper'] - results_df['Odds Ratio']
    ],
    fmt='o',
    capsize=5
)

# Add a vertical line at OR = 1
plt.axvline(x=1, color='red', linestyle='--', label='OR = 1 (No effect)')

# Set the y-tick labels to the variable names
plt.yticks(y_pos, results_df['Variable'])

# Annotate with p-values
for i, (or_val, p_val) in enumerate(zip(results_df['Odds Ratio'], results_df['p-value'])):
    significant = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    plt.text(or_val + 0.1, i, f"p={p_val:.4f} {significant}", va='center')

# Add labels and title
plt.xlabel('Odds Ratio (log scale)')
plt.xscale('log')
plt.title('Forest Plot of Odds Ratios for Lung Cancer Risk Factors')
plt.grid(True, alpha=0.3)

# Add a legend
plt.legend()
plt.tight_layout()
plt.show()

# Interpret each variable
for i, row in results_df.iterrows():
    var = row['Variable']
    or_val = row['Odds Ratio']
    ci_lower = row['95% CI Lower']
    ci_upper = row['95% CI Upper']
    p_val = row['p-value']
    
    # Interpret the significance
    if p_val < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
    
    # Interpret the direction of association
    if or_val > 1:
        direction = "increased"
        percent_increase = (or_val - 1) * 100
        direction_text = f"increases by approximately {percent_increase:.1f}%"
    elif or_val < 1:
        direction = "decreased"
        percent_decrease = (1 - or_val) * 100
        direction_text = f"decreases by approximately {percent_decrease:.1f}%"
    else:
        direction = "unchanged"
        direction_text = "remains unchanged"
    
    # Print interpretation
    print(f"\\n{var}:")
    print(f"Odds Ratio: {or_val:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}), p={p_val:.4f}")
    
    if var == "AGE (per 10 years)":
        print(f"Interpretation: For each 10-year increase in age, the odds of lung cancer {direction_text}.")
    elif var == "GENDER (Male vs. Female)":
        print(f"Interpretation: Being male {direction_text} the odds of lung cancer compared to females.")
    else:
        print(f"Interpretation: The presence of {var.lower()} {direction_text} the odds of lung cancer.")
    
    print(f"This finding is {significance}.")
    
    # Clinical significance interpretation
    if p_val < 0.05:
        if var == "SMOKING":
            print("Clinical Significance: Smoking is a major modifiable risk factor for lung cancer.")
        elif var == "SHORTNESS OF BREATH" or var == "CHEST PAIN":
            print("Clinical Significance: This symptom may be an important clinical indicator for lung cancer screening.")
        elif var == "AGE (per 10 years)":
            print("Clinical Significance: Age is an important factor to consider in lung cancer screening guidelines.")
        else:
            print("Clinical Significance: This factor may be important in clinical assessment.")
    else:
        print("Clinical Significance: This factor may not be a primary consideration in clinical assessment for lung cancer risk.")
"""
    
    st.code(interpretation_code, language="python")
    
    # Button to open code in CodePad
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad", key='week11-codepad-interpretation', use_container_width=True):
            codeeditor_popup(interpretation_code)
    
    with st.form(key="odds_ratio_interpretation_form"):
        st.write("**1. Which variable has the strongest association with lung cancer?**")
        q1 = st.selectbox(
            "Select one:",
            ["SMOKING", "AGE", "GENDER", "SHORTNESS OF BREATH", "CHEST PAIN"],
            index=None
        )
        
        st.write("**2. Is the association between gender and lung cancer statistically significant?**")
        q2 = st.radio(
            "Choose one:",
            ["Yes", "No"],
            index=None
        )
        
        st.write("**3. What does the 95% confidence interval for SMOKING (2.18-4.46) tell us?**")
        q3 = st.radio(
            "Choose the best interpretation:",
            [
                "95% of smokers will develop lung cancer",
                "We are 95% confident that the true odds ratio for smoking is between 2.18 and 4.46",
                "The p-value is 0.05",
                "The risk of lung cancer is between 2.18% and 4.46% for smokers"
            ],
            index=None
        )
        
        st.write("**4. For a 70-year-old person compared to a 60-year-old person, how much higher are the odds of lung cancer?**")
        q4 = st.radio(
            "Choose the best interpretation:",
            [
                "1.45 times higher",
                "14.5% higher",
                "45% higher",
                "Cannot be determined from the data"
            ],
            index=None
        )
        
        st.write("**5. Which of these risk factors is non-modifiable?**")
        q5 = st.radio(
            "Choose one:",
            [
                "SMOKING",
                "AGE",
                "SHORTNESS OF BREATH",
                "CHEST PAIN"
            ],
            index=None
        )
        
        submit = st.form_submit_button("Submit Answers")
    
    if submit:
        score = 0
        feedback = []
        
        # Question 1
        correct_q1 = "SMOKING"
        if q1 == correct_q1:
            score += 1
            feedback.append("✅ Question 1: Correct! Smoking has the highest odds ratio (3.12).")
        else:
            feedback.append(f"❌ Question 1: Incorrect. The correct answer is {correct_q1}, which has the highest odds ratio (3.12).")
        
        # Question 2
        correct_q2 = "No"
        if q2 == correct_q2:
            score += 1
            feedback.append("✅ Question 2: Correct! The p-value for gender is 0.38, which is greater than 0.05.")
        else:
            feedback.append(f"❌ Question 2: Incorrect. The p-value for gender is 0.38, which is greater than 0.05, so it's not statistically significant.")
        
        # Question 3
        correct_q3 = "We are 95% confident that the true odds ratio for smoking is between 2.18 and 4.46"
        if q3 == correct_q3:
            score += 1
            feedback.append("✅ Question 3: Correct! The 95% CI represents our confidence in the range that contains the true odds ratio.")
        else:
            feedback.append(f"❌ Question 3: Incorrect. The correct interpretation is: {correct_q3}")
        
        # Question 4
        correct_q4 = "45% higher"
        if q4 == correct_q4:
            score += 1
            feedback.append("✅ Question 4: Correct! For each 10-year increase in age, the odds increase by 45% (OR = 1.45).")
        else:
            feedback.append(f"❌ Question 4: Incorrect. The odds ratio for a 10-year increase in age is 1.45, which means 45% higher odds.")
        
        # Question 5
        correct_q5 = "AGE"
        if q5 == correct_q5:
            score += 1
            feedback.append("✅ Question 5: Correct! Age is a non-modifiable risk factor.")
        else:
            feedback.append(f"❌ Question 5: Incorrect. Age is the only non-modifiable risk factor among these options.")
        
        st.write(f"Your score: {score}/5")
        for fb in feedback:
            st.write(fb)

def main():
    # Initialize Tailwind CSS if needed
    try:
        tw.initialize_tailwind()
    except:
        pass
    
    # Navbar
    Navbar()
    
    # Title
    st.title("Week 11 | Logistic Regression and Binary Outcomes")
    
    # Table of contents
    section_table_of_contents()
    
    # Main content sections
    section_introduction_to_logistic_regression()
    st.divider()
    section_odds_ratios_interpretation()
    st.divider()
    section_model_diagnostics()
    st.divider()
    
    # Activities
    st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
    st.header("Activities")
    activity_fitting_logistic_models()
    st.divider()
    activity_interpreting_odds_ratios()
    
    # Footer
    Footer(11)

if __name__ == "__main__":
    main()