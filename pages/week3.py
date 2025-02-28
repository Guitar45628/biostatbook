import streamlit as st
from modules.nav import Navbar
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import statistics

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 03 | Data Visualization and Descriptive Statistics",
)
@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None,warning_text=None):
    code_editor_for_all(default_code=default_code,key='codepad-week3-graph',warning_text=warning_text)


def section_table_of_contents():
    st.header("📚 Table of Contents")
    st.markdown("""
        <ol>
            <li><a href="#introduction-to-data-visualization">Introduction to Data Visualization</a></li>
            <li><a href="#types-of-graphs">Types of Graphs</a></li>
            <li><a href="#measures-of-central-tendency">Measures of Central Tendency</a></li>
            <li><a href="#measures-of-variability">Measures of Variability</a></li>
            <li><a href="#measures-of-distribution-shape">Measures of Distribution Shape</a></li>
        </ol>
    """, unsafe_allow_html=True)


def section_one_introduction_to_data_visualization():
    st.header("Introduction to Data Visualization")
    st.write("""
        Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
    """)
    st.subheader("Why is Data Visualization Important?")
    st.write("""
        Data visualization is important because it allows trends and patterns to be more easily seen. With the rise of big data upon us, we need to be able to interpret and understand data. Data visualization is a quick, easy way to convey concepts in a universal manner – and you can experiment with different scenarios by making slight adjustments.
    """)
    st.subheader("Benefits of Data Visualization")
    st.markdown("""
        - **Clarify & Understanding:** Helps break down complex data into clear and accessible visuals.
        - **Insight & Decision Making:** Empowers decision-makers with actionable insights drawn from data trends.
        - **Exploration:** Facilitates the identification of hidden patterns and potential anomalies in the data.
        - **Effective Communication:** Transforms data into compelling visuals that communicate key findings efficiently.
        - **Engagement:** Enhances audience interaction and interest through visually appealing data presentations.
    """)
    st.subheader("Example of Data Visualization")
    st.write("Below is an example of using a line chart to display random data trends. This example mimics how you might visualize trends in sales, marketing, and research data over time.")

    # Example code to generate a line chart using random data
    import pandas as pd
    import numpy as np

    # Create a DataFrame with random data
    data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Sales', 'Marketing', 'Research']
    )
    st.line_chart(data)


def section_two_types_of_graphs():
    warning_text = "Please comment `st.pyplot(fig)` and use `plt.show()` for running with CodePad.  **Don't forget to save before running.**"
    
    # Types of Graphs
    st.header("Types of Graphs")
    st.write("""
        Data visualization can be achieved through various types of graphs, each serving a different purpose. Below are some of the most common types of graphs used in data visualization.
    """)

    # Histogram
    st.subheader("Histogram")
    st.write("""
        A histogram is a graphical representation of the distribution of numerical data. It is an estimate of the probability distribution of a continuous variable and was first introduced by Karl Pearson.
    """)

    histogram_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
heights = np.random.normal(165, 8, 500)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(heights, bins=25, color='lightblue', edgecolor='black')
ax.set_title('Distribution of Student Heights', pad=20)
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Number of Students')
ax.grid(True, alpha=0.3)

# Add mean line and annotation
mean_height = heights.mean()
ax.axvline(mean_height, color='red', linestyle='dashed', linewidth=1,
            label=f'Mean: {mean_height:.1f} cm')
ax.legend()

st.pyplot(fig)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(histogram_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Histogram",key='week3-codepad-histogram',use_container_width=True):
            codeeditor_popup(histogram_code,warning_text=warning_text)

    # Description of Histogram
    st.info("""
        The histogram above shows the distribution of student heights in a school. The x-axis
        represents the height in centimeters, while the y-axis represents the number of students
        falling within each height range. The red dashed line represents the mean height of the
        students.
    """)

    # Box Plot
    st.subheader("Box Plot")
    st.write("""
        A box plot is a method for graphically depicting groups of numerical data through their quartiles.
        Box plots may also have lines extending from the boxes (whiskers) indicating variability outside
        the upper and lower quartiles.
    """)

    boxplot_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
male_heights = np.random.normal(175, 7, 300)    # Male students
female_heights = np.random.normal(162, 6, 200)   # Female students

# Create the boxplot
fig_box, ax_box = plt.subplots(figsize=(10, 6))
data = [male_heights, female_heights]
labels = ['Male', 'Female']

bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)

# Customize boxplot colors
colors = ['lightblue', 'pink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize plot (ignoring fontsize changes)
ax_box.set_title('Distribution of Student Heights by Gender', pad=20)
ax_box.set_ylabel('Height (cm)')
ax_box.grid(True, alpha=0.3, axis='y')

# Add jittered points for better visualization
for i, d in enumerate(data, 1):
    x = np.random.normal(i, 0.04, size=len(d))
    ax_box.plot(x, d, 'o', alpha=0.5, color='gray', markersize=4)

st.pyplot(fig_box)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(boxplot_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Box Plot", key='week3-codepad-boxplot', use_container_width=True):
            codeeditor_popup(boxplot_code, warning_text=warning_text)

    # Description of Box Plot
    st.info("""
        The box plot above shows the distribution of student heights by gender, comparing male and female height ranges and medians.
    """)

    # Scatter Plot
    st.subheader("Scatter Plot")
    st.write("""
        A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data.
    """)

    scatter_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
heights = np.random.normal(165, 8, 500)    # Heights in cm
weights = np.random.normal(70, 10, 500)     # Weights in kg

# Create the scatter plot
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
scatter = ax_scatter.scatter(
    heights, weights, alpha=0.6, edgecolors='w', s=100)
ax_scatter.set_title('Height vs. Weight of Students', pad=20)
ax_scatter.set_xlabel('Height (cm)')
ax_scatter.set_ylabel('Weight (kg)')
ax_scatter.grid(True, alpha=0.3)

# Add a trend line
z = np.polyfit(heights, weights, 1)
p = np.poly1d(z)
ax_scatter.plot(heights, p(heights), "r--")

st.pyplot(fig_scatter)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(scatter_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Scatter Plot", key='week3-codepad-scatter', use_container_width=True):
            codeeditor_popup(scatter_code, warning_text=warning_text)

    # Description of Scatter Plot
    st.info("""
        The scatter plot above shows the relationship between student heights and weights. Each point represents a student, with their height on the x-axis and weight on the y-axis. The red dashed line represents the trend line showing the general direction of the data.
    """)

    # Bar Chart
    st.subheader("Bar Chart")
    st.write("""
        A bar chart is a chart that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
    """)

    bar_chart_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
subjects = ['Math', 'Science', 'English', 'History', 'Art']
grades = np.random.randint(50, 100, size=len(subjects))

# Create the bar chart
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar.bar(subjects, grades, color='skyblue', edgecolor='black')
ax_bar.set_title('Student Grades by Subject', pad=20)
ax_bar.set_xlabel('Subjects')
ax_bar.set_ylabel('Grades')
ax_bar.set_ylim(0, 100)
ax_bar.grid(axis='y', alpha=0.3)

st.pyplot(fig_bar)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(bar_chart_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Bar Chart", key='week3-codepad-bar', use_container_width=True):
            codeeditor_popup(bar_chart_code, warning_text=warning_text)

    # Description of Bar Chart
    st.info("""
        The bar chart above shows the grades of students in different subjects. Each bar represents a subject, with the height of the bar indicating the average grade for that subject.
    """)

    # Pie Chart
    st.subheader("Pie Chart")
    st.write("""
        A pie chart is a circular statistical graphic, which is divided into slices to illustrate numerical proportion.
    """)

    pie_chart_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
categories = ['Sports', 'Music', 'Art', 'Science', 'Technology']
preferences = np.random.randint(10, 50, size=len(categories))

# Create the pie chart
fig_pie, ax_pie = plt.subplots(figsize=(10, 6))
ax_pie.pie(preferences, labels=categories, autopct='%1.1f%%',
           startangle=140, colors=plt.cm.Paired.colors)
ax_pie.set_title('Student Preferences by Category', pad=20)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax_pie.axis('equal')

st.pyplot(fig_pie)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(pie_chart_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Pie Chart", key='week3-codepad-pie', use_container_width=True):
            codeeditor_popup(pie_chart_code, warning_text=warning_text)

    # Description of Pie Chart
    st.info("""
        The pie chart above shows the preferences of students in different categories. Each slice represents a category, with the size of the slice indicating the proportion of students who prefer that category.
    """)

    # Line Chart
    st.subheader("Line Chart")
    st.write("""
        A line chart is a type of chart that displays information as a series of data points called 'markers' connected by straight line segments.
    """)

    line_chart_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
attendance = np.random.randint(50, 100, size=len(days))

# Create the line chart
fig_line, ax_line = plt.subplots(figsize=(10, 6))
ax_line.plot(days, attendance, marker='o', linestyle='-', color='orange')
ax_line.set_title('Student Attendance by Day', pad=20)
ax_line.set_xlabel('Days')
ax_line.set_ylabel('Number of Students')
ax_line.set_ylim(0, 100)
ax_line.grid(True, alpha=0.3)

st.pyplot(fig_line)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(line_chart_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Line Chart", key='week3-codepad-line', use_container_width=True):
            codeeditor_popup(line_chart_code, warning_text=warning_text)

    # Description of Line Chart
    st.info("""
        The line chart above shows the attendance of students by day. Each point represents the number of students present on a particular day, with lines connecting the points to show the trend over the week.
    """)

    # Population Pyramid
    st.subheader("Population Pyramid")
    st.write("""
        A population pyramid is a graphical illustration that shows the distribution of various age groups in a population, which forms the shape of a pyramid when the population is growing.
    """)

    population_pyramid_code = """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # For reproducibility
age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
              '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

# Generate random population data for males and females with more older age groups
total_population = 71702435
male_population = np.random.randint(100000, 2000000, size=len(age_groups))
female_population = np.random.randint(100000, 2000000, size=len(age_groups))

# Increase the population for older age groups
age_weight = np.linspace(1, 2, len(age_groups))
male_population = male_population * age_weight
female_population = female_population * age_weight

# Normalize to match the total population
male_population = (male_population / male_population.sum()) * (total_population / 2)
female_population = (female_population / female_population.sum()) * (total_population / 2)

# Create the population pyramid plot
fig_pyramid, ax_pyramid = plt.subplots(figsize=(10, 8))

# Plot male population
ax_pyramid.barh(age_groups, -male_population,
                color='skyblue', edgecolor='black', label='Male')

# Plot female population
ax_pyramid.barh(age_groups, female_population, color='pink',
                edgecolor='black', label='Female')

# Customize the plot
ax_pyramid.set_title(
    'Population Pyramid of Thailand (2023) *Random dataset', pad=20)
ax_pyramid.set_xlabel('Population')
ax_pyramid.set_ylabel('Age Groups')
ax_pyramid.legend()
ax_pyramid.grid(True, alpha=0.3)

st.pyplot(fig_pyramid)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
"""


    exec(population_pyramid_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Population Pyramid", key='week3-codepad-pyramid', use_container_width=True):
            codeeditor_popup(population_pyramid_code, warning_text=warning_text)

    # Description of Population Pyramid
    st.info("""
        The population pyramid above shows the distribution of the population of Thailand in 2023 by age group.
    """)

    # Heatmap
    st.subheader("Correlation Heatmap of Student Health Metrics")
    st.write("""
        A correlation heatmap is a graphical representation of the correlation matrix, where individual values contained in the matrix are represented as colors.
    """)

    heatmap_code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    'Height': np.random.normal(165, 10, 100),
    'Weight': np.random.normal(65, 15, 100),
    'BMI': np.random.normal(22, 3, 100),
    'Blood Pressure': np.random.normal(120, 10, 100),
    'Cholesterol': np.random.normal(200, 30, 100)
})

# Compute the correlation matrix
correlation_matrix = data.corr()

# Create the correlation heatmap
fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, ax=ax_heatmap, vmin=-1, vmax=1)

ax_heatmap.set_title('Correlation Heatmap of Student Health Metrics', pad=20)

st.pyplot(fig_heatmap)  # Turn off this line if running in CodePad then please use plt.show()
#plt.show()  # -> Used in regular Python programs
    """

    exec(heatmap_code)

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad for Heatmap", key='week3-codepad-heatmap', use_container_width=True):
            codeeditor_popup(heatmap_code, warning_text=warning_text)

    # Description of Heatmap
    st.info("""
        The heatmap above visualizes the correlations between various health metrics of students, including height, weight, BMI, blood pressure, and cholesterol.
    """)


def section_three_measures_of_central_tendency():
    st.header("Measures of Central Tendency")
    st.write("""
        Measures of central tendency are statistical measures that describe the center or typical value of a dataset. The three main measures of central tendency are the mean, median, and mode.
    """)

    # Mean
    st.subheader("Mean")
    st.write("""
        The mean is the average of a set of numbers. It is calculated by adding all the numbers in a dataset and dividing by the total number of values.
    """)

    # Median
    st.subheader("Median")
    st.write("""
        The median is the middle value in a dataset when the values are arranged in ascending order. If there is an even number of values, the median is the average of the two middle values.
    """)

    # Mode
    st.subheader("Mode")
    st.write("""
        The mode is the value that appears most frequently in a dataset. A dataset may have one mode, more than one mode, or no mode at all.
    """)


def section_four_measures_of_variability():
    st.header("Measures of Variability")
    st.write("""
        Measures of variability describe the spread or dispersion of a dataset. The two main measures of variability are the range and standard deviation.
    """)

    # Range
    st.subheader("Range")
    st.write("""
        The range is the difference between the highest and lowest values in a dataset. It provides a simple measure of the spread of values.
    """)
    # Equation for Range
    st.latex(r'Range = X_{max} - X_{min}')
    # description of range in equation
    st.info("""
        where Xmax is the maximum value and Xmin is the minimum value in the dataset.
    """)

    # Variance
    st.subheader("Variance")
    st.write("""
        Variance is a measure of how far each number in a dataset is from the mean and thus from every other number in the dataset. It is calculated by taking the average of the squared differences from the mean.
    """)
    # Equation for Variance
    st.latex(r'Var(X) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2')
    # description of var in equation
    st.info("""
        where Var(X) is the variance, N is the number of observations, Xi is each individual observation, and Mu(μ) is the mean of the dataset.
    """)

    # Standard Deviation
    st.subheader("Standard Deviation")
    st.write("""
        The standard deviation is a measure of the amount of variation or dispersion in a set of values. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.
    """)
    # Equation for Standard Deviation
    st.latex(r'\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}')
    # description of std in equation
    st.info("""
        where sigma(σ) is the standard deviation, N is the number of observations, Xi is each individual observation, and Mu(μ) is the mean of the dataset.
    """)

    # Interquartile Range
    st.subheader("Interquartile Range (IQR)")
    st.write("""
        The interquartile range (IQR) is a measure of statistical dispersion, or variability, and is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of a dataset. It represents the range within which the central 50% of the data points lie.
    """)
    # Equation for IQR
    st.latex(r'IQR = Q3 - Q1')
    # description of IQR in equation
    st.info("""
        where IQR is the interquartile range, Q3 is the 75th percentile, and Q1 is the 25th percentile of the dataset.
    """)


def section_five_measures_of_distribution_shape():
    st.header("Measures of Distribution Shape")
    st.write("""
        Measures of distribution shape describe the symmetry and peakedness of a dataset. The two main measures of distribution shape are skewness and kurtosis.
    """)

    # Skewness
    st.subheader("Skewness")
    st.write("""
        Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable. It can be positive, negative, or undefined.
    """)
    # Equation for Skewness
    st.latex(r'Skewness = \frac{E[(X - \mu)^3]}{\sigma^3}')
    # description of skewness in equation
    st.info("""
        where E is the expected value, X is the random variable, mu(μ) is the mean, and sigma(σ) is the standard deviation.
    """)

    # Tree colume of chart left show Positive Skewness, middle show Normal Distribution, right show Negative Skewness
    st.write("Below is an example of skewness in a dataset.")

    # Create sample data for skewness
    np.random.seed(42)  # For reproducibility
    data_positive_skew = np.random.exponential(scale=2, size=1000)
    data_negative_skew = -np.random.exponential(scale=2, size=1000)
    data_normal = np.random.normal(loc=0, scale=1, size=1000)
    fig_skewness, ax_skewness = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data_positive_skew, kde=True,
                 ax=ax_skewness[0], color='skyblue')
    ax_skewness[0].set_title('Positive Skewness')
    sns.histplot(data_normal, kde=True, ax=ax_skewness[1], color='lightgreen')
    ax_skewness[1].set_title('Normal Distribution')
    sns.histplot(data_negative_skew, kde=True,
                 ax=ax_skewness[2], color='salmon')
    ax_skewness[2].set_title('Negative Skewness')
    st.pyplot(fig_skewness)
    st.info("""
        The left chart shows a positively skewed distribution, where the tail on the right side is longer or fatter than the left side. The middle chart shows a normal distribution, which is symmetric around the mean. The right chart shows a negatively skewed distribution, where the tail on the left side is longer or fatter than the right side.
    """)

    st.header("Kurtosis")
    st.write("""
        Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable. It can be positive, negative, or undefined.
    """)
    # Equation for Kurtosis
    st.latex(r'Kurtosis = \frac{E[(X - \mu)^4]}{\sigma^4} - 3')
    # Description of kurtosis in equation
    st.info("""
        where E is the expected value, X is the random variable, mu(μ) is the mean, and sigma(σ) is the standard deviation.
    """)

    # Create sample data for kurtosis
    np.random.seed(42)  # For reproducibility

    # Leptokurtic (thick tails, sharp peak)
    leptokurtic_data = np.random.laplace(loc=0, scale=1, size=1000)
    leptokurtic_kurt = kurtosis(leptokurtic_data, fisher=False)

    # Mesokurtic (normal distribution)
    mesokurtic_data = np.random.normal(loc=0, scale=1, size=1000)
    mesokurtic_kurt = kurtosis(mesokurtic_data, fisher=False)

    # Platykurtic (thin tails, flat peak)
    platykurtic_data = np.random.uniform(low=-3, high=3, size=1000)
    platykurtic_kurt = kurtosis(platykurtic_data, fisher=False)

    # Create plots for each type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Leptokurtic
    axes[0].hist(leptokurtic_data, bins=30, alpha=0.7, color='skyblue')
    axes[0].set_title(f"Leptokurtic (Kurtosis: {leptokurtic_kurt:.2f})")

    # Mesokurtic
    axes[1].hist(mesokurtic_data, bins=30, alpha=0.7, color='lightgreen')
    axes[1].set_title(f"Mesokurtic (Kurtosis: {mesokurtic_kurt:.2f})")

    # Platykurtic
    axes[2].hist(platykurtic_data, bins=30, alpha=0.7, color='salmon')
    axes[2].set_title(f"Platykurtic (Kurtosis: {platykurtic_kurt:.2f})")

    st.pyplot(fig)

    # Description of Kurtosis
    st.info("""
        The charts above show different types of kurtosis. The left chart shows a leptokurtic distribution with thick tails and a sharp peak. The middle chart shows a mesokurtic distribution, which is a normal distribution. The right chart shows a platykurtic distribution with thin tails and a flat peak.
    """)


def section_activity_one():
    st.header("⚒ Activity")
    
    left_co, cent_co = st.columns([1,3])
    with left_co:
        st.image("assets/week3-01.png", width=150)
    with cent_co:
        st.subheader("Role Play activity")
        st.write("You are a junior biostatistician who has been assigned by your supervisor to survey the population at a hospital. You are traveling there with a heart full of joy.")
    
    st.write("Since you are a junior biostatistician, your supervisor wants to assess your basic skills to determine how prepared and capable you are for the task.")
    
    st.subheader("⚡ Let's test your statistical skills!")

    hint_code_01 = """
# Edit your code here!
# Save before running!

import statistics

# Mean
mean_value = statistics.mean([1, 3, 5, 7, 9, 11, 13])
print(f"Mean: {mean_value}")  

# Max
max_value = max([1, 2, 3])
print(f"Max: {max_value}")  

# Min
min_value = min([1, 2, 3])
print(f"Min: {min_value}")  

# Range
range_value = max_value - min_value
print(f"Range: {range_value}")  

# Median
median_value = statistics.median([1, 3, 5, 7, 9, 11, 13])
print(f"Median: {median_value}")  

# Mode
mode_value = statistics.mode([1, 3, 3, 7, 9, 9, 13])
print(f"Mode: {mode_value}")  
"""


    with st.expander("CodePad"):
        code_editor_for_all(default_code=hint_code_01, key='codepad-week3-1')

    with st.form(key='quiz_form'):
        st.write("**1. Mean Calculation**")
        st.write("The following are the ages of 6 patients at the hospital: 21, 28, 34, 45, 50, 57.")
        user_mean = st.number_input("What is the mean age of the patients?", min_value=0.0, format="%.2f", key="mean")

        st.write("**2. Max and Min**")
        st.write("A list of hospital bed temperatures in Celsius is: 36.5, 37.1, 37.8, 36.9, 37.3, 38.0.")
        max_temp = st.number_input("What is the maximum temperature recorded?", min_value=0.0, format="%.2f", key="max_temp")
        min_temp = st.number_input("What is the minimum temperature recorded?", min_value=0.0, format="%.2f", key="min_temp")
        
        st.write("**3. Range Calculation**")
        st.write("The following are the test scores of 7 students: 85, 90, 78, 92, 88, 84, 91.")
        user_range = st.number_input("What is the range of these scores?", min_value=0.0, format="%.2f", key="range")

        st.write("**4. Median**")
        st.write("The ages of 5 healthcare workers are: 30, 35, 28, 32, 29.")
        user_median = st.number_input("What is the median age of the healthcare workers?", min_value=0.0, format="%.2f", key="median")

        st.write("**5. Mode**")
        st.write("The following numbers represent the number of hospital visits per month for a group of patients: 3, 5, 3, 2, 4, 3, 6, 5, 3.")
        user_mode = st.number_input("What is the mode of the number of visits?", min_value=0, key="mode")

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if user_mean and max_temp and min_temp and user_range and user_median and user_mode:
            correct_answers = 0
            wrong_answers = []

            actual_mean = sum([21, 28, 34, 45, 50, 57]) / 6
            if user_mean == actual_mean:
                correct_answers += 1
                wrong_answers.append(f"✔ 1. Correct answer. The correct mean is {actual_mean:.2f}")
            else:
                wrong_answers.append(f"❌ 1. Wrong answer. The correct mean is {actual_mean:.2f}")

            actual_max, actual_min = max([36.5, 37.1, 37.8, 36.9, 37.3, 38.0]), min([36.5, 37.1, 37.8, 36.9, 37.3, 38.0])
            if max_temp == actual_max and min_temp == actual_min:
                correct_answers += 1
                wrong_answers.append(f"✔ 2. Correct answer. The correct Max temperature is {actual_max}, Min temperature is {actual_min}")
            else:
                wrong_answers.append(f"❌ 2. Wrong answer. The correct Max temperature is {actual_max}, Min temperature is {actual_min}")

            actual_range = max([85, 90, 78, 92, 88, 84, 91]) - min([85, 90, 78, 92, 88, 84, 91])
            if user_range == actual_range:
                correct_answers += 1
                wrong_answers.append(f"✔ 3. Correct answer. The correct range is {actual_range}")
            else:
                wrong_answers.append(f"❌ 3. Wrong answer. The correct range is {actual_range}")

            actual_median = statistics.median([30, 35, 28, 32, 29])
            if user_median == actual_median:
                correct_answers += 1
                wrong_answers.append(f"✔ 4. Correct answer. The correct median is {actual_median}")
            else:
                wrong_answers.append(f"❌ 4. Wrong answer. The correct median is {actual_median}")

            actual_mode = statistics.mode([3, 5, 3, 2, 4, 3, 6, 5, 3])
            if user_mode == actual_mode:
                correct_answers += 1
                wrong_answers.append(f"✔ 5. Correct answer. The correct mode is {actual_mode}")
            else:
                wrong_answers.append(f"❌ 5. Wrong answer. The correct mode is {actual_mode}")

            with st.expander(f"Your score: {correct_answers}/5  Click to see the correct answers"):
                for answer in wrong_answers:
                    st.write(answer)

        else:
            st.error("❌ Please answer all questions before submitting.")

def section_activity_two():
    left_co, cent_co = st.columns([1,3])
    with left_co:
        st.image("assets/week3-02.png", width=150)
    with cent_co:
        st.subheader("Mission 1 | Survey the Population")
        st.write("You have successfully passed the test and are now ready to survey the population at the hospital.")
    
    # คุณได้สำรวจประชากรทั้งหมด 2412 คน เป็นผู้ชาย 1157 เป็นผู้หญิง 1255 
    st.write("You have surveyed a total of 2412 people, with 1157 being male and 1255 being female. Your supervisor wants you to create a Bar Chart to display this data.")

    st.subheader("⚡ Let's create a Bar Chart!")

    hint_code_02 = """
import matplotlib.pyplot as plt

# Data
categories = ['Male', 'Female']
population = [1157, 1255]

# Create the bar chart
plt.bar(categories, population, color=['skyblue', 'pink'])

# Title and labels
plt.title('Population Breakdown')
plt.xlabel('Gender')
plt.ylabel('Number of People')

# Show the chart
plt.show()

"""
    st.code(hint_code_02, language='python')

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad",key='week3-codepad-1',use_container_width=True):
            codeeditor_popup(hint_code_02)

    st.write("After creating the Bar Chart, you present it to your supervisor. Your supervisor is impressed with your work and wants you to create a Pie Chart to display the same data.")
    st.subheader("⚡ Let's create a Pie Chart!")
    hint_code_03 = """
import matplotlib.pyplot as plt

# Data
categories = ['Male', 'Female']
population = [1157, 1255]

# Create the bar chart
plt.pie(population, labels=categories, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'pink'])

# Title and labels
plt.title('Population Breakdown')

# Show the chart
plt.show()
"""
    st.code(hint_code_03, language='python')

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad",key='week3-codepad-2',use_container_width=True):
            codeeditor_popup(hint_code_03)


    st.write("Your supervisor is very pleased with your work and wants you to show deeper insights into the data.")
    st.info("You surveyed the population and collected data on their age, height, weight, and blood pressure. Your supervisor wants you to create a Histogram to display the distribution of ages in the population.")
    st.subheader("⚡ Let's create a Histogram!")
    hint_code_04 = """
import matplotlib.pyplot as plt
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Data: Age distribution of the surveyed population
# Random age data between 0 and 120 years for 2412 people
ages = np.random.randint(0, 120, size=2412)  

# Define the age bins (groups)
age_bins = list(range(0, 121, 10))  # Bins from 0-9, 10-19, 20-29, ..., 110-119

# Create the histogram
plt.hist(ages, bins=age_bins, color='skyblue', edgecolor='black')

# Title and labels
plt.title('Age Distribution of Surveyed Population')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show the chart
plt.show()
    """

    st.code(hint_code_04, language='python')

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad",key='week3-codepad-4',use_container_width=True):
            codeeditor_popup(hint_code_04)

    # Box plot of wieght and hieght
    st.write("Your supervisor is very impressed with your work and wants you to create a Box Plot to display the distribution of weights and heights in the population.")
    st.subheader("⚡ Let's create a Box Plot!")
    hint_code_05 = """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Data: Weight and Height distribution of the surveyed population
# Random weight data between 3 and 200 kg for 2412 people
weights = np.random.randint(3, 200, size=2412)
# Random height data between 50 and 250 cm for 2412 people
heights = np.random.randint(50, 250, size=2412)

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=[weights, heights], palette=["skyblue", "lightgreen"])
plt.xticks([0, 1], ['Weight (kg)', 'Height (cm)'])
plt.title('Distribution of Weight and Height in Surveyed Population')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

# Add jittered points for better visualization
for i, d in enumerate([weights, heights]):
    x = np.random.normal(i, 0.04, size=len(d))
    plt.plot(x, d, 'x', alpha=0.2, color='gray', markersize=2)

plt.show()
    """

    st.code(hint_code_05, language='python')

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad",key='week3-codepad-5',use_container_width=True):
            codeeditor_popup(hint_code_05)

    st.write("Your supervisor is very impressed with your work and wants you to create a Scatter Plot to display the relationship between weight and height in the population.")
    st.subheader("⚡ Let's create a Scatter Plot!")
    hint_code_06 = """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Data: Weight and Height distribution of the surveyed population
# Random weight data between 3 and 200 kg for 2412 people
weights = np.random.randint(3, 200, size=2412)
# Random height data between 50 and 250 cm for 2412 people
heights = np.random.randint(50, 250, size=2412)
# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(heights, weights, alpha=0.6, edgecolors='w', s=100)
plt.title('Weight vs. Height of Surveyed Population')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True, alpha=0.3)
plt.show()
    """

    st.code(hint_code_06, language='python')

    if "codeeditor_popup" not in st.session_state:
        if st.button("Open CodePad",key='week3-codepad-6',use_container_width=True):
            codeeditor_popup(hint_code_06)

def footer():
    left_co, cent_co = st.columns([1,3])
    with left_co:
        st.image("assets/week3-03.png")
    with cent_co:
        st.subheader("End of Week03")
        st.write("**You are so tired that you fainted at your desk.**")
        st.write("You missed over 200 calls from your supervisor because you’ve been exhausted for a long time. As a result, your supervisor has given you a one-week break from work.")
    

def main():
    Navbar()

    # Title
    st.title("Week 03 | Data Visualization and Descriptive Statistics")

    # Table of Contents
    section_table_of_contents()

    # Content
    section_one_introduction_to_data_visualization()
    st.divider()
    section_two_types_of_graphs()
    st.divider()
    section_three_measures_of_central_tendency()
    st.divider()
    section_four_measures_of_variability()
    st.divider()
    section_five_measures_of_distribution_shape()

    st.divider()
    section_activity_one()
    section_activity_two()

    st.divider()
    footer()


if __name__ == "__main__":
    main()
