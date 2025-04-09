import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw
import statsmodels.api as sm
from statsmodels.formula.api import ols

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 10 | Regression and Correlation",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None, warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code, key='codepad-week10', warning_text=warning_text)

def section_table_of_contents():
    st.markdown("""
        <h2>📚 Table of Contents</h2>
        <ol>
            <li><a href="#introduction-to-regression-and-correlation">Introduction to Regression and Correlation</a></li>
            <li><a href="#simple-linear-regression">Simple Linear Regression</a></li>
            <li><a href="#pearsons-correlation-coefficient">Pearson's Correlation Coefficient</a></li>
            <li><a href="#assumptions-of-linear-regression">Assumptions of Linear Regression</a></li>
            <li><a href="#activities">Activities</a></li>
        </ol>
    """, unsafe_allow_html=True)

def section_introduction_to_regression():
    st.header("Introduction to Regression and Correlation")
    
    st.write("""
        Regression and correlation analyses are powerful statistical methods used to understand relationships 
        between variables. While they are related concepts, they serve different purposes in statistical analysis.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regression Analysis")
        st.write("""
            Regression analysis is used to model the relationship between a dependent variable and one or more 
            independent variables. It helps us understand how the dependent variable changes when any of the 
            independent variables change.
            
            **Key Features:**
            - Predicts values of the dependent variable
            - Shows the strength and nature of relationships
            - Can be used for forecasting
            - Identifies which variables have the most impact
        """)
    
    with col2:
        st.subheader("Correlation Analysis")
        st.write("""
            Correlation analysis measures the strength and direction of the relationship between two variables. 
            It tells us how closely the variables are related without implying causation.
            
            **Key Features:**
            - Ranges from -1 to +1
            - Shows direction (positive or negative)
            - Indicates strength of association
            - Does not imply causation
            - Symmetric relationship between variables
        """)
    
    # Example visualization
    st.subheader("Example Visualization")
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 2, 100)
    
    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6)
    
    # Add regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='red', linestyle='-', linewidth=2)
    
    ax.set_title('Example: Scatter Plot with Regression Line')
    ax.set_xlabel('Independent Variable (X)')
    ax.set_ylabel('Dependent Variable (Y)')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Correlation Coefficient (r): {corr:.2f}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)

def section_simple_linear_regression():
    st.header("Simple Linear Regression")
    
    st.write("""
        Simple linear regression is a statistical method that allows us to model the relationship between 
        two continuous variables by fitting a linear equation to the observed data. One variable is considered 
        the explanatory variable (independent variable), and the other is the response variable (dependent variable).
    """)
    
    st.subheader("The Linear Regression Equation")
    st.latex(r'Y = \beta_0 + \beta_1 X + \varepsilon')
    
    st.write("""
        Where:
        - Y is the dependent variable (what we're trying to predict)
        - X is the independent variable (what we're using to predict Y)
        - β₀ is the y-intercept (the value of Y when X = 0)
        - β₁ is the slope (the change in Y for each unit increase in X)
        - ε (epsilon) is the error term (representing unexplained variation)
    """)
    
    st.subheader("Model Fitting")
    st.write("""
        The most common method for fitting a regression line is the method of **least squares**. This method 
        calculates the best-fitting line by minimizing the sum of the squared differences between the observed 
        values and the values predicted by the linear approximation.
    """)
    
    # Visualize least squares principle
    st.subheader("Least Squares Principle")
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 3, 20)
    
    # Fit regression line
    m, b = np.polyfit(x, y, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='blue', s=60, label='Data Points')
    
    # Plot regression line
    line_x = np.array([min(x), max(x)])
    line_y = m * line_x + b
    ax.plot(line_x, line_y, 'r-', linewidth=2, label='Regression Line')
    
    # Plot residuals
    for i in range(len(x)):
        y_pred = m * x[i] + b
        ax.plot([x[i], x[i]], [y[i], y_pred], 'g--', linewidth=1)
    
    ax.set_title('Least Squares Method: Visualizing Residuals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.subheader("Interpreting Regression Output")
    st.write("""
        When interpreting a simple linear regression model, we typically look at:
        
        1. **Slope (β₁)**: Indicates the change in Y for each unit change in X
        2. **Intercept (β₀)**: The value of Y when X = 0
        3. **R-squared**: The proportion of variance in Y explained by X (ranges from 0 to 1)
        4. **p-value**: Tests the null hypothesis that the coefficient = 0
        5. **Standard Error**: The average distance that the observed values fall from the regression line
    """)
    
    # Example code
    st.subheader("Example Code: Fitting a Linear Regression Model in Python")
    
    regression_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

# Fit the model using scikit-learn
model = LinearRegression()
model.fit(X, y)

# Print results
print(f"Slope (β₁): {model.coef_[0]:.4f}")
print(f"Intercept (β₀): {model.intercept_:.4f}")

# For more detailed statistics, we can use statsmodels
X_sm = sm.add_constant(X)  # Add a constant for the intercept
model_sm = sm.OLS(y, X_sm).fit()
print("\\nDetailed Statistics:")
print(model_sm.summary().tables[1])

# Create a scatter plot with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data Points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
    
    st.code(regression_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Simple Linear Regression)", key='week10-codepad-regression', use_container_width=True):
            codeeditor_popup(regression_code)

def section_correlation():
    st.header("Pearson's Correlation Coefficient")
    
    st.write("""
        Pearson's correlation coefficient (r) measures the linear relationship between two continuous variables. 
        It ranges from -1 to +1, where:
        
        - **+1** indicates a perfect positive linear relationship
        - **0** indicates no linear relationship
        - **-1** indicates a perfect negative linear relationship
    """)
    
    st.subheader("Formula for Pearson's Correlation Coefficient")
    st.latex(r'r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}')
    
    st.write("""
        Where:
        - x₁, x₂, ..., xₙ and y₁, y₂, ..., yₙ are the values of the two variables
        - x̄ and ȳ are the means of the two variables
    """)
    
    # Visualizing correlation
    st.subheader("Visualizing Different Correlation Values")
    
    # Create data with different correlation values
    np.random.seed(42)
    
    # Create a figure with 3 subplots showing different correlations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate correlated data
    def generate_correlated_data(r, n=100):
        # Generate x
        x = np.random.normal(0, 1, n)
        
        # Generate y with correlation r to x
        y = r * x + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
        
        return x, y
    
    # Strong positive correlation (r ≈ 0.9)
    x1, y1 = generate_correlated_data(0.9)
    axes[0].scatter(x1, y1, alpha=0.6)
    axes[0].set_title(f'Strong Positive Correlation\nr = {np.corrcoef(x1, y1)[0, 1]:.2f}')
    axes[0].grid(True, alpha=0.3)
    
    # No correlation (r ≈ 0)
    x2, y2 = generate_correlated_data(0)
    axes[1].scatter(x2, y2, alpha=0.6)
    axes[1].set_title(f'No Correlation\nr = {np.corrcoef(x2, y2)[0, 1]:.2f}')
    axes[1].grid(True, alpha=0.3)
    
    # Strong negative correlation (r ≈ -0.9)
    x3, y3 = generate_correlated_data(-0.9)
    axes[2].scatter(x3, y3, alpha=0.6)
    axes[2].set_title(f'Strong Negative Correlation\nr = {np.corrcoef(x3, y3)[0, 1]:.2f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Interpreting Correlation Coefficient")
    
    st.write("""
        The strength of the correlation can be interpreted as follows:
        
        | Correlation Value | Strength |
        | --- | --- |
        | 0.00 - 0.19 | Very weak |
        | 0.20 - 0.39 | Weak |
        | 0.40 - 0.59 | Moderate |
        | 0.60 - 0.79 | Strong |
        | 0.80 - 1.00 | Very strong |
    """)
    
    st.warning("""
        **Important Reminder:** Correlation does not imply causation!
        
        A high correlation between two variables does not necessarily mean that one variable causes the other. 
        There might be other factors influencing both variables or the relationship might be coincidental.
    """)
    
    st.subheader("Example Code: Calculating Correlation in Python")
    
    correlation_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create sample data for demonstration
np.random.seed(42)
n = 100
x = np.random.normal(0, 1, n)
y1 = 0.8 * x + np.random.normal(0, 0.5, n)  # Strong positive correlation
y2 = -0.8 * x + np.random.normal(0, 0.5, n)  # Strong negative correlation
y3 = np.random.normal(0, 1, n)  # No correlation

# Create dataframe
df = pd.DataFrame({
    'x': x,
    'y1': y1,
    'y2': y2,
    'y3': y3
})

# Calculate correlations using different methods
print("Using NumPy:")
print(f"Correlation between x and y1: {np.corrcoef(x, y1)[0, 1]:.4f}")
print(f"Correlation between x and y2: {np.corrcoef(x, y2)[0, 1]:.4f}")
print(f"Correlation between x and y3: {np.corrcoef(x, y3)[0, 1]:.4f}")

print("\\nUsing Pandas:")
print(df.corr()['x'])

print("\\nUsing SciPy:")
print(f"Correlation between x and y1: {stats.pearsonr(x, y1)[0]:.4f}, p-value: {stats.pearsonr(x, y1)[1]:.4f}")
print(f"Correlation between x and y2: {stats.pearsonr(x, y2)[0]:.4f}, p-value: {stats.pearsonr(x, y2)[1]:.4f}")
print(f"Correlation between x and y3: {stats.pearsonr(x, y3)[0]:.4f}, p-value: {stats.pearsonr(x, y3)[1]:.4f}")

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
"""
    
    st.code(correlation_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Correlation Analysis)", key='week10-codepad-correlation', use_container_width=True):
            codeeditor_popup(correlation_code)

def section_assumptions_regression():
    st.header("Assumptions of Linear Regression")
    
    st.write("""
        To ensure the validity of a linear regression model, several assumptions should be met. Violations of these 
        assumptions can lead to biased estimates, inefficient predictions, or misleading inferences.
    """)
    
    st.subheader("Key Assumptions")
    
    assumptions = [
        {
            "name": "Linearity",
            "description": "The relationship between the independent and dependent variables should be linear.",
            "check": "Scatter plots of the dependent variable against each independent variable.",
            "violation": "Non-linear transformations of variables, polynomial terms, or non-linear models."
        },
        {
            "name": "Independence",
            "description": "Observations should be independent of each other.",
            "check": "Time series plots for time-related data; Durbin-Watson test.",
            "violation": "Use time series models or account for clustering in the data."
        },
        {
            "name": "Homoscedasticity",
            "description": "The variance of the residuals should be constant across all levels of the independent variables.",
            "check": "Plot residuals vs. predicted values; Breusch-Pagan test.",
            "violation": "Weighted least squares; transform the dependent variable."
        },
        {
            "name": "Normality of Residuals",
            "description": "The residuals should be normally distributed.",
            "check": "Q-Q plot; Shapiro-Wilk test; histogram of residuals.",
            "violation": "With large samples, this is less concerning due to the Central Limit Theorem."
        },
        {
            "name": "No Multicollinearity",
            "description": "Independent variables should not be highly correlated with each other.",
            "check": "Correlation matrix; Variance Inflation Factor (VIF).",
            "violation": "Remove one of the correlated variables; use ridge regression or other regularization methods."
        }
    ]
    
    for i, assumption in enumerate(assumptions):
        with st.container(border=True):
            st.subheader(f"{i+1}. {assumption['name']}")
            st.write(f"**Description**: {assumption['description']}")
            st.write(f"**How to check**: {assumption['check']}")
            st.write(f"**What to do if violated**: {assumption['violation']}")
    
    # Diagnostic plots visualization
    st.subheader("Diagnostic Plots for Linear Regression")
    
    # Generate sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 2, 100)
    
    # Fit model
    model = sm.OLS(y, sm.add_constant(x)).fit()
    
    # Get predictions and residuals
    predictions = model.predict()
    residuals = model.resid
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(predictions, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title('Residuals vs. Fitted Values')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Scale-Location Plot
    axes[1, 0].scatter(predictions, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 0].set_title('Scale-Location Plot')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Residuals|')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=15, alpha=0.6, edgecolor='black')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Example code for diagnostics
    st.subheader("Example Code: Diagnostic Plots in Python")
    
    diagnostic_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import seaborn as sns

# Generate sample data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 2, 100)
df = pd.DataFrame({'x': x, 'y': y})

# Fit the model
X = sm.add_constant(df['x'])
model = sm.OLS(df['y'], X).fit()
print(model.summary())

# Get predictions and residuals
df['predicted'] = model.predict()
df['residuals'] = df['y'] - df['predicted']
df['abs_residuals'] = np.abs(df['residuals'])
df['sqrt_abs_residuals'] = np.sqrt(df['abs_residuals'])

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs. Fitted Values
axes[0, 0].scatter(df['predicted'], df['residuals'], alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='-')
axes[0, 0].set_title('Residuals vs. Fitted Values')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')

# 2. Q-Q Plot
QQ = ProbPlot(df['residuals'])
QQ.qqplot(line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# 3. Scale-Location Plot
axes[1, 0].scatter(df['predicted'], df['sqrt_abs_residuals'], alpha=0.6)
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Residuals|')

# 4. Histogram of Residuals
axes[1, 1].hist(df['residuals'], bins=15, alpha=0.6, edgecolor='black')
axes[1, 1].set_title('Histogram of Residuals')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
"""
    
    st.code(diagnostic_code, language="python")
    
    if "codeeditor_popup" not in st.session_state:
        if st.button("Open in CodePad (Regression Diagnostics)", key='week10-codepad-diagnostics', use_container_width=True):
            codeeditor_popup(diagnostic_code)

def activity_lung_cancer_regression():
    st.header("Activity 1: Fitting Regression Models and Interpreting Coefficients")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # Reusing image from week 5
    with cent_co:
        st.write("""
        In this activity, you will explore the relationship between different variables in the lung cancer dataset 
        by fitting regression models and interpreting the coefficients.
        """)
    
    # Load the lung cancer data
    file_path = os.path.join(os.getcwd(), "data/lung_cancer.csv")
    df_lung = pd.read_csv(file_path)
    
    # Display the first few rows and info about the dataset
    st.subheader("Lung Cancer Dataset")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df_lung.head())
    
    st.write(f"Dataset shape: {df_lung.shape[0]} rows, {df_lung.shape[1]} columns")
    
    # Prepare data for regression
    # Convert categorical variables for the example
    df_lung['LUNG_CANCER_BINARY'] = (df_lung['LUNG_CANCER'] == 'YES').astype(int)
    
    # Fix: Strip whitespace from column names to handle the spaces in the CSV headers
    df_lung.columns = df_lung.columns.str.strip()
    
    # Select only numeric columns for analysis
    numeric_cols = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                    'CHRONIC DISEASE', 'WHEEZING', 
                    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                    'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER_BINARY']
    
    df_numeric = df_lung[numeric_cols]
    
    # Show correlation matrix
    st.subheader("Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    plt.title('Correlation Matrix of Lung Cancer Dataset')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Predict Lung Cancer Risk")
    st.write("""
        Let's create a regression model to predict lung cancer risk based on selected variables.
        Choose one or more variables to include in your regression model:
    """)
    
    # Feature selection for regression
    feature_options = [col for col in numeric_cols if col != 'LUNG_CANCER_BINARY']
    selected_features = st.multiselect(
        "Select features for the regression model:", 
        options=feature_options,
        default=['AGE', 'SMOKING']
    )
    
    if selected_features:
        # Create regression code example with the selected features
        regression_code_example = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the lung cancer data
df_lung = pd.read_csv('data/lung_cancer.csv')

# Convert the target variable to binary
df_lung['LUNG_CANCER_BINARY'] = (df_lung['LUNG_CANCER'] == 'YES').astype(int)

# Select features and target
X = df_lung[[{', '.join(f"'{{feature}}'" for feature in selected_features)}]]
y = df_lung['LUNG_CANCER_BINARY']

# Add constant for statsmodels
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Display model summary
print(model.summary())

# Visualize coefficients
plt.figure(figsize=(10, 6))
coefs = model.params[1:].sort_values(ascending=False)
plt.bar(coefs.index, coefs.values)
plt.title('Regression Coefficients')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coefficient Value')
plt.axhline(y=0, color='r', linestyle='-')
plt.tight_layout()
plt.show()

# Split data for prediction evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on the training data
model_train = sm.OLS(y_train, X_train).fit()

# Predict on the test data
y_pred = model_train.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {{mse:.4f}}')
print(f"R-squared: {{r2:.4f}}")
"""

        
        st.code(regression_code_example, language="python")
        
        # Add CodePad button
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in CodePad (Lung Cancer Regression)", key='week10-activity1-regression', use_container_width=True):
                codeeditor_popup(regression_code_example)
        
        # Actual model fitting and visualization
        try:
            # Fit the model
            X = df_numeric[selected_features]
            y = df_numeric['LUNG_CANCER_BINARY']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Display model summary
            st.subheader("Regression Model Summary")
            st.text(str(model.summary()))
            
            # Visualize coefficients
            st.subheader("Visualization of Regression Coefficients")
            fig, ax = plt.subplots(figsize=(10, 6))
            coefs = model.params[1:].sort_values(ascending=False)
            ax.bar(coefs.index, coefs.values)
            ax.set_title('Regression Coefficients')
            ax.set_ylabel('Coefficient Value')
            ax.axhline(y=0, color='r', linestyle='-')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Interpretation of Results")
            st.write("""
                Use the regression model summary and coefficient visualization above to answer the following questions:
            """)
            
            with st.form(key='regression_interpretation'):
                q1 = st.radio(
                    "1. Which variable has the strongest positive association with lung cancer risk?",
                    options=selected_features + ["None of the above"],
                    index=None
                )
                
                q2 = st.radio(
                    "2. Is the regression model statistically significant overall?",
                    options=["Yes", "No", "Can't determine"],
                    index=None
                )
                
                q3 = st.text_area(
                    "3. Write a brief interpretation of one of the coefficients in the context of lung cancer risk:",
                    height=100
                )
                
                submit_button = st.form_submit_button("Submit")
                
                if submit_button:
                    # Calculate the strongest positive variable
                    strongest_pos = coefs[coefs > 0].idxmax() if any(coefs > 0) else "None of the above"
                    
                    # Check model significance
                    model_significant = "Yes" if model.f_pvalue < 0.05 else "No"
                    
                    feedback = []
                    if q1 == strongest_pos:
                        feedback.append("✔ Question 1: Correct!")
                    else:
                        feedback.append(f"❌ Question 1: Incorrect. The variable with the strongest positive association is {strongest_pos}.")
                    
                    if q2 == model_significant:
                        feedback.append("✔ Question 2: Correct!")
                    else:
                        feedback.append(f"❌ Question 2: Incorrect. The model is {'statistically significant' if model_significant == 'Yes' else 'not statistically significant'} (p-value: {model.f_pvalue:.4f}).")
                    
                    if q3 and len(q3.strip()) > 10:
                        feedback.append("✔ Question 3: Thanks for your interpretation!")
                    else:
                        feedback.append("❌ Question 3: Please provide a more detailed interpretation of one of the coefficients.")
                    
                    st.subheader("Feedback")
                    for fb in feedback:
                        st.write(fb)
            
        except Exception as e:
            st.error(f"Error fitting the model: {e}. Please select different features or try again.")
    else:
        st.warning("Please select at least one feature for the regression model.")

def activity_correlation_analysis():
    st.header("Activity 2: Calculating and Interpreting Correlation Coefficients")
    
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week5-1.png", width=300)  # Reusing image from week 5
    with cent_co:
        st.write("""
        In this activity, you will explore the correlations between different variables in the lung cancer dataset 
        and interpret the correlation coefficients.
        """)
    
    # Load the lung cancer data
    file_path = os.path.join(os.getcwd(), "data/lung_cancer.csv")
    df_lung = pd.read_csv(file_path)
    
    # Convert binary target variable
    df_lung['LUNG_CANCER_BINARY'] = (df_lung['LUNG_CANCER'] == 'YES').astype(int)
    
    # Select only numeric columns for analysis
    numeric_cols = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                    'CHRONIC DISEASE', 'WHEEZING', 
                    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                    'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER_BINARY']
    
    df_numeric = df_lung[numeric_cols]
    
    st.subheader("Explore Correlations Between Variables")
    st.write("""
        Select any two variables from the dataset to explore their correlation:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select first variable:", options=numeric_cols, index=0)
    with col2:
        var2 = st.selectbox("Select second variable:", options=numeric_cols, index=1)
    
    if var1 and var2:
        # Calculate correlation
        correlation = df_numeric[var1].corr(df_numeric[var2])
        p_value = stats.pearsonr(df_numeric[var1], df_numeric[var2])[1]
        
        # Display correlation
        st.subheader(f"Correlation between {var1} and {var2}")
        st.metric("Pearson Correlation Coefficient", f"{correlation:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        
        # Interpret the correlation
        if abs(correlation) < 0.2:
            strength = "very weak"
        elif abs(correlation) < 0.4:
            strength = "weak"
        elif abs(correlation) < 0.6:
            strength = "moderate"
        elif abs(correlation) < 0.8:
            strength = "strong"
        else:
            strength = "very strong"
            
        direction = "positive" if correlation > 0 else "negative"
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
        
        st.write(f"""
            **Interpretation:** There is a {strength} {direction} correlation between {var1} and {var2} (r = {correlation:.4f}).
            This correlation is {significance} at α = 0.05 (p-value = {p_value:.4f}).
        """)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_numeric[var1], df_numeric[var2], alpha=0.6)
        
        # Add regression line
        m, b = np.polyfit(df_numeric[var1], df_numeric[var2], 1)
        ax.plot(df_numeric[var1], m * df_numeric[var1] + b, color='red', linestyle='-')
        
        ax.set_title(f'Scatter Plot of {var1} vs {var2}')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.grid(True, alpha=0.3)
        
        # Add correlation info
        ax.text(0.05, 0.95, f'r = {correlation:.4f}, p-value = {p_value:.4f}', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        st.pyplot(fig)
        
        # Code example for calculating correlation
        correlation_code_example = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the lung cancer data
df_lung = pd.read_csv('data/lung_cancer.csv')

# Convert the target variable to binary
df_lung['LUNG_CANCER_BINARY'] = (df_lung['LUNG_CANCER'] == 'YES').astype(int)

# Calculate correlation between {var1} and {var2}
correlation, p_value = stats.pearsonr(df_lung['{var1}'], df_lung['{var2}'])

print(f"Correlation between {var1} and {var2}: {{correlation:.4f}}")
print(f"P-value: {{p_value:.4f}}")

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df_lung['{var1}'], df_lung['{var2}'], alpha=0.6)

# Add regression line
m, b = np.polyfit(df_lung['{var1}'], df_lung['{var2}'], 1)
plt.plot(df_lung['{var1}'], m * df_lung['{var1}'] + b, color='red')

plt.title(f'Scatter Plot of {var1} vs {var2}')
plt.xlabel('{var1}')
plt.ylabel('{var2}')
plt.grid(True, alpha=0.3)

# Add correlation info
plt.text(0.05, 0.95, f'r = {{correlation:.4f}}, p-value = {{p_value:.4f}}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.show()
"""
        
        st.subheader("Code Example")
        st.code(correlation_code_example, language="python")
        
        # Add CodePad button
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in CodePad (Correlation Analysis)", key='week10-activity2-correlation', use_container_width=True):
                codeeditor_popup(correlation_code_example)
        
        # Quiz on correlation interpretation
        st.subheader("Test Your Understanding")
        with st.form(key='correlation_quiz'):
            st.write("Answer the following questions about correlation:")
            
            q1 = st.radio(
                "1. If two variables have a correlation of -0.8, this means:",
                options=[
                    "As one variable increases, the other tends to increase strongly",
                    "As one variable increases, the other tends to decrease strongly",
                    "The two variables are not related",
                    "The two variables have a causal relationship"
                ],
                index=None
            )
            
            q2 = st.radio(
                "2. A correlation coefficient of 0.3 indicates:",
                options=[
                    "A strong positive relationship",
                    "A moderate positive relationship",
                    "A weak positive relationship",
                    "No relationship"
                ],
                index=None
            )
            
            q3 = st.radio(
                f"3. Based on the correlation between {var1} and {var2}, we can conclude that:",
                options=[
                    f"{var1} causes changes in {var2}",
                    f"{var2} causes changes in {var1}",
                    f"There is a {direction} association between {var1} and {var2}, but we cannot determine causation",
                    "There is no meaningful relationship between these variables"
                ],
                index=None
            )
            
            q4 = st.radio(
                "4. What does a p-value less than 0.05 mean for a correlation coefficient?",
                options=[
                    "The correlation is practically significant",
                    "The correlation is statistically significant",
                    "The correlation is at least 0.05",
                    "The correlation is due to random chance"
                ],
                index=None
            )
            
            q5 = st.radio(
                "5. Which of these is NOT a valid use of correlation analysis?",
                options=[
                    "Identifying potentially related variables",
                    "Measuring the strength of a linear relationship",
                    "Determining the direction of association",
                    "Proving that one variable causes changes in another"
                ],
                index=None
            )
            
            submit_button = st.form_submit_button("Submit")
            
            if submit_button:
                correct_answers = {
                    "q1": "As one variable increases, the other tends to decrease strongly",
                    "q2": "A weak positive relationship",
                    "q3": f"There is a {direction} association between {var1} and {var2}, but we cannot determine causation",
                    "q4": "The correlation is statistically significant",
                    "q5": "Proving that one variable causes changes in another"
                }
                
                user_answers = {
                    "q1": q1,
                    "q2": q2,
                    "q3": q3,
                    "q4": q4,
                    "q5": q5
                }
                
                score = 0
                feedback = []
                
                for i in range(1, 6):
                    key = f"q{i}"
                    if user_answers[key] == correct_answers[key]:
                        score += 1
                        feedback.append(f"✔ Question {i}: Correct!")
                    else:
                        feedback.append(f"❌ Question {i}: Incorrect. The correct answer is: {correct_answers[key]}.")
                
                st.subheader("Quiz Results")
                st.write(f"Your score: {score}/5")
                
                with st.expander("See detailed feedback"):
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
    st.title("Week 10 | Regression and Correlation")
    
    # Table of contents
    section_table_of_contents()
    
    # Main content sections
    section_introduction_to_regression()
    st.divider()
    section_simple_linear_regression()
    st.divider()
    section_correlation()
    st.divider()
    section_assumptions_regression()
    st.divider()
    
    # Activities
    st.markdown("<a id='activities'></a>", unsafe_allow_html=True)
    st.header("Activities")
    activity_lung_cancer_regression()
    st.divider()
    activity_correlation_analysis()
    
    # Footer
    Footer(10)

if __name__ == "__main__":
    main()