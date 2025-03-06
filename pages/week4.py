import os
import pandas as pd
import streamlit as st
from modules.nav import Navbar
from modules.foot import Footer
import st_tailwind as tw

from modules.code_editor_all import code_editor_for_all

# Page Titlebar
st.set_page_config(
    page_title="Week 04 | Sampling and the Central Limit Theorem",
)

@st.dialog("CodePad", width="large")
def codeeditor_popup(default_code=None,warning_text="*Don't forget to save your code before running it!* (ctrl+enter or save button)"):
    code_editor_for_all(default_code=default_code,key='codepad-week3-graph',warning_text=warning_text)



def section_table_of_contents():
    st.markdown("""
            <h2>📚 Table of Contents</h2>
            <ol>
                <li><a href="#introduction-to-sampling">Introduction to Sampling</a></li>
                <li><a href="#probability-sampling">Probability Sampling</a></li>
                <li><a href="#non-probability-sampling">Non-Probability Sampling</a></li>
                <li><a href="#sampling-distributions">Sampling Distributions</a></li>
                <li><a href="#the-central-limit-theorem-clt">The Central Limit Theorem (CLT)</a></li>
                <li><a href="#standard-error-of-the-mean">Standard Error of the Mean</a></li>
            </ol>
        </div>
    """, unsafe_allow_html=True)


def section_introduction_to_sampling():
    st.header("Introduction to Sampling")
    st.write("""
        Sampling is the process of selecting a subset of individuals from a population to estimate the characteristics of the population as a whole. 
        The individuals selected for the sample are called sample units, and the process of selecting the sample units is called sampling.
        There are two main types of sampling: probability sampling and non-probability sampling.
    """)
    st.info("Sampling is an essential part of statistical analysis, as it allows us to make inferences about a population based on a sample of the population.")

    st.header("Types of Sampling")
    st.subheader("1. Probability Sampling")
    st.write("""
        Probability sampling is a sampling technique in which every individual in the population has an equal chance of being selected for the sample. 
        """)
    st.info("""
Example of probability sampling methods include:
- Simple random sampling
- Stratified sampling
- Cluster sampling
- Systematic sampling
            """)

    st.subheader("2. Non-Probability Sampling")
    st.write("""
        Non-probability sampling is a sampling technique in which the probability of selecting an individual for the sample is not known. 
        
    """)
    st.info("""
Example of non-probability sampling methods include:
- Convenience sampling
- Judgmental sampling
- Snowball sampling
- Quota sampling
            """)

    with st.container(border=True):
        col_left, col_right = st.columns([4, 1])
        with col_left:
            tw.write("You can learn more about Sampling Methods on Scribbr.",classes="flex mt-2 text-bold") 
            
        with col_right:
            st.link_button("Read More",icon=":material/open_in_new:",url="https://www.scribbr.com/methodology/sampling-methods")

    st.header("Sampling Distributions")
    st.write("""
        A sampling distribution is a probability distribution that describes the likelihood of obtaining a particular sample statistic from a sample of a given size. 
        The shape of the sampling distribution depends on the sample size and the population distribution.
    """)
    st.header("The Central Limit Theorem (CLT)")
    st.write("""
        The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, 
        regardless of the shape of the population distribution. This property of the sample mean makes it a powerful tool for estimating population parameters.
    """)
    st.subheader("Standard Error of the Mean")
    st.write("""
        The standard error of the mean is a measure of the variability of the sample mean. 
        It is calculated as the standard deviation of the population divided by the square root of the sample size. 
        The standard error of the mean decreases as the sample size increases, which means that larger samples provide more precise estimates of the population mean.
    """)
    st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
    st.info("""
- SE = Standard Error of the Mean
- \(\sigma\) = Standard Deviation of the Population
- n = Sample Size
""")

def section_activity_one():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week4-i1.png", width=300)
    with cent_co:
        st.header("Activity 1: Sampling Methods")
        st.write("After you have rested enough and are ready to work, today you return to the hospital where you work regularly, still showing some signs of fatigue.")
        st.write("Your Supervisor has informed you to review Sampling because there will be a health data collection of the nearby population.")
        
    # Create a form for the quiz
    with st.form(key='sampling_quiz'):
        st.subheader("Please answer the following questions about Sampling Methods:")
        
        q1 = st.radio("1. Which of the following is a probability sampling method?", 
                        ("Convenience sampling", "Judgmental sampling", "Simple random sampling", "Quota sampling"), index=None)
        
        q2 = st.radio("2. What is the main characteristic of non-probability sampling?", 
                        ("Every individual has an equal chance of being selected", 
                        "The probability of selecting an individual is not known", 
                        "It is always more accurate than probability sampling", 
                        "It requires a larger sample size"), index=None)
        
        q3 = st.radio("3. Which sampling method involves dividing the population into subgroups and selecting samples from each subgroup?", 
                        ("Cluster sampling", "Systematic sampling", "Stratified sampling", "Snowball sampling"), index=None)
        
        q4 = st.radio("4. What does the Central Limit Theorem state?", 
                        ("The sample mean will always equal the population mean", 
                        "The sampling distribution of the sample mean approaches a normal distribution as the sample size increases", 
                        "The population distribution will become normal as the sample size increases", 
                        "The standard error of the mean increases with sample size"), index=None)
        
        q5 = st.radio("5. How is the standard error of the mean calculated?", 
                        ("Standard deviation of the sample divided by the sample size", 
                        "Standard deviation of the population divided by the square root of the sample size", 
                        "Mean of the sample divided by the population size", 
                        "Mean of the population divided by the sample size"), index=None)
        
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        score = 0
        correct_answers = {
            "q1": "Simple random sampling",
            "q2": "The probability of selecting an individual is not known",
            "q3": "Stratified sampling",
            "q4": "The sampling distribution of the sample mean approaches a normal distribution as the sample size increases",
            "q5": "Standard deviation of the population divided by the square root of the sample size"
        }
        
        user_answers = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5
        }
        
        feedback = []
        
        for i, key in enumerate(correct_answers, 1):
            if user_answers[key] == correct_answers[key]:
                score += 1
                feedback.append(f"✔ {i}. Correct answer. The correct answer is {correct_answers[key]}")
            else:
                feedback.append(f"❌ {i}. Wrong answer. The correct answer is {correct_answers[key]}")
        
        st.write(f"Your score: {score}/5")
        
        with st.expander("See correct answers"):
            for answer in feedback:
                st.write(answer)

def section_activity_two():
    left_co, cent_co = st.columns([2, 4])
    with left_co:
        st.image("assets/week4-i2.png", width=300)
    with cent_co:
        st.header("Activity 2: Sampling Coding")
        
        # คุณได้ไปเก็บข้อมูลสุขภาพจากกลุ่มคนในพื้นที่ใกล้เคียง จำนวน 10000 คน และได้รับข้อมูลดังนี้
        st.write("You have collected health data from a group of people in the nearby area, with a total of 10,000 people, and received the following data:")
        st.warning("This is mock data generated for educational purposes.")

    file_path_health_data = os.path.join(os.getcwd(), "data/health_data.csv")
    health_data = pd.read_csv(file_path_health_data)
    st.dataframe(data=health_data)
    st.info(file_path_health_data)

    

    # เนื่องจากคุณอยากการ Sampling แต่ละแบบให้ผลลัพธ์แตกต่างกันมากแค่ไหน คุณจึงต้องการทดสอบดู
    st.write("Since you want to see how different each sampling method is, you want to test it.")

    with st.container(border=True):
        st.subheader("Probability Sampling")
        st.write("1. Simple Random Sampling")
        simple_random_sampling_code = """
import os
import pandas as pd
import numpy as np
                
# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)
                
# Simple random sampling
sample_size = 100
                
simple_random_sample = health_data.sample(n=sample_size, random_state=42)
print("Simple Random Sampling:")
print(simple_random_sample)     

# Show beatiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=simple_random_sample)
"""
        st.code(simple_random_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-simple-random',use_container_width=True):
                codeeditor_popup(simple_random_sampling_code)

        st.write("2. Stratified Sampling")
        stratified_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Ensure that there is a 'Blood Type' column in the dataset
if 'Blood Type' not in health_data.columns:
    raise ValueError("The dataset must contain a 'Blood Type' column")

# Stratified sampling
sample_size = 100  # Define sample size here

# Perform stratified sampling by grouping the data by 'Blood Type'
def stratified_sample_group(group, sample_size):
    # Ensure we don't try to sample more than the group size
    return group.sample(min(len(group), sample_size), random_state=42)

# Apply stratified sampling
stratified_sample = health_data.groupby('Blood Type', group_keys=False).apply(stratified_sample_group, sample_size=sample_size)

# Show stratified sample
print("Stratified Sampling:")
print(stratified_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=stratified_sample)
"""
        st.code(stratified_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-stratified',use_container_width=True):
                codeeditor_popup(stratified_sampling_code)

        st.write("3. Cluster Sampling")
        cluster_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Ensure that there is a 'Blood Type' column in the dataset
if 'Blood Type' not in health_data.columns:
    raise ValueError("The dataset must contain a 'Blood Type' column")

# Cluster sampling based on 'Blood Type'
sample_size = 100  # Define sample size for each cluster

# Perform cluster sampling by grouping the data by 'Blood Type'
def cluster_sample_group(group, sample_size):
    # Ensure we don't try to sample more than the group size
    return group.sample(min(len(group), sample_size), random_state=42)

# Apply cluster sampling
cluster_sample = health_data.groupby('Blood Type', group_keys=False).apply(cluster_sample_group, sample_size=sample_size)

# Show cluster sample
print("Cluster Sampling:")
print(cluster_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=cluster_sample)
"""
        st.code(cluster_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-cluster',use_container_width=True):
                codeeditor_popup(cluster_sampling_code)

        st.write("4. Systematic Sampling")
        systematic_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Systematic sampling
sample_size = 100  # Define sample size here

# Calculate the sampling interval
sampling_interval = len(health_data) // sample_size

# Perform systematic sampling
systematic_sample = health_data.iloc[::sampling_interval]

# Show systematic sample
print("Systematic Sampling:")
print(systematic_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=systematic_sample)
"""
        st.code(systematic_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-systematic',use_container_width=True):
                codeeditor_popup(systematic_sampling_code)
        

    with st.container(border=True):
        st.subheader("Non-Probability Sampling")
        st.write("1. Convenience Sampling")
        convenience_sampling_code = """
import os
import pandas as pd
import numpy as np  

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Convenience sampling
sample_size = 100  # Define sample size here

# Perform convenience sampling
convenience_sample = health_data.sample(n=sample_size, random_state=42)

# Show convenience sample
print("Convenience Sampling:")
print(convenience_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=convenience_sample)
"""
        st.code(convenience_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-convenience',use_container_width=True):
                codeeditor_popup(convenience_sampling_code)

        st.write("2. Judgmental Sampling")
        judgmental_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Judgmental sampling
sample_size = 100  # Define sample size here

# Define the criteria for judgmental sampling
criteria = ['Age', 'Blood Pressure']

# Perform judgmental sampling
judgmental_sample = health_data.sample(n=sample_size, random_state=42)

# Show judgmental sample
print("Judgmental Sampling:")
print(judgmental_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=judgmental_sample)
"""
        st.code(judgmental_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-judgmental',use_container_width=True):
                codeeditor_popup(judgmental_sampling_code)

        st.write("3. Snowball Sampling")
        snowball_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Snowball sampling
sample_size = 100  # Define sample size

# Perform snowball sampling
snowball_sample = health_data.sample(n=sample_size, random_state=42)

# Show snowball sample
print("Snowball Sampling:")
print(snowball_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=snowball_sample)
"""
        st.code(snowball_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-snowball',use_container_width=True):
                codeeditor_popup(snowball_sampling_code)

        st.write("4. Quota Sampling")
        quota_sampling_code = """
import os
import pandas as pd
import numpy as np

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Quota sampling
sample_size = 100  # Define sample size

# Perform quota sampling
quota_sample = health_data.sample(n=sample_size, random_state=42)

# Show quota sample
print("Quota Sampling:")
print(quota_sample)

# Show beautiful dataframe in Streamlit
import streamlit as st
st.dataframe(data=quota_sample)
"""
        st.code(quota_sampling_code, language="python")
        
        # try it on code editor
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act1-quota',use_container_width=True):
                codeeditor_popup(quota_sampling_code)

import os
import pandas as pd
import numpy as np
import streamlit as st

def section_activity_three():
    # ในกิจกรรมนี้จะทดสอบว่าค่าทางสถิติของกลุ่มที่ Sampling มา และทั้งหมดมีความแตกต่างกันมากน้อยเพียงใด
    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.write("")
        st.write("")
        st.image("assets/week4-i3.png", width=300)
    with col_right:
        st.header("Activity 3: Sampling Statistics")
        st.write("In this activity, you will test how different the statistics of the sampled group are from the entire population.")
        st.write("You will calculate the mean, standard deviation, and standard error of the mean for the entire population and the sample.")
        st.write("You will then compare the statistics to see how they differ.")

    # Load the health data
    file_path = os.path.join(os.getcwd(), "data/health_data.csv")
    health_data = pd.read_csv(file_path)

    st.info("""
    **Note:** This is mock data generated for educational purposes.""")
    st.dataframe(data=health_data)

    # Calculate statistics for the population and sample
    with st.container():
        st.subheader("Calculate Statistics for Population and Sample")
        st.write("You will calculate the mean, standard deviation, and standard error of the mean for the entire population and the sample.")

        # Example code to calc them
        calculate_statistics_code = """
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load the health data
file_path = os.path.join(os.getcwd(), "data/health_data.csv")
health_data = pd.read_csv(file_path)

# Select column to analyze
column_to_analyze = 'Weight'  # You can change this to any column you'd like to analyze
if column_to_analyze not in health_data.columns:
    st.error(f"Column '{column_to_analyze}' not found in the dataset.")
else:
    # Entire population statistics
    population_mean = health_data[column_to_analyze].mean()
    population_std = health_data[column_to_analyze].std()
    population_size = len(health_data)
    population_se = population_std / np.sqrt(population_size)  # Standard error

    # Stratified sampling or random sampling of the data
    sample_size = 100
    stratified_sample = health_data.sample(n=sample_size, random_state=42)

    # Sample statistics
    sample_mean = stratified_sample[column_to_analyze].mean()
    sample_std = stratified_sample[column_to_analyze].std()
    sample_se = sample_std / np.sqrt(sample_size)

    # Display the statistics
    st.subheader("Population vs Sample Statistics Comparison")

    # Prepare data for the bar chart
    stats = ['Mean', 'Standard Deviation', 'Standard Error']
    population_values = [population_mean, population_std, population_se]
    sample_values = [sample_mean, sample_std, sample_se]

    # Create a DataFrame for easy plotting
    stats_df = pd.DataFrame({
        'Statistic': stats,
        'Population': population_values,
        'Sample': sample_values
    })

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use indices for bar positions
    indices = np.arange(len(stats_df))  # Create index positions for bars

    # Bar plot for both population and sample
    ax.bar(indices - 0.2, stats_df['Population'], width=0.4, label='Population', align='center')
    ax.bar(indices + 0.2, stats_df['Sample'], width=0.4, label='Sample', align='center')

    ax.set_xlabel('Statistic')
    ax.set_ylabel('Value')
    ax.set_title(f'Comparison of {column_to_analyze} Statistics')
    ax.set_xticks(indices)
    ax.set_xticklabels(stats_df['Statistic'])
    ax.legend()

    # Display the chart
    plt.show()

"""
        st.code(calculate_statistics_code, language="python")
        # Open in CodePad
        if "codeeditor_popup" not in st.session_state:
            if st.button("Open in Codepad",key='week4-act3-calc-stat',use_container_width=True):
                codeeditor_popup(calculate_statistics_code)

    st.write("You can compare the statistics of the population and the sample to see how they differ.")
    
    
    # Function to calculate sample statistics, with caching
    @st.cache_data
    def calculate_sample_statistics(sample_size, sampling_type, column_to_analyze):
        if sampling_type == "Random Sampling":
            # Perform random sampling
            stratified_sample = health_data.sample(n=sample_size, random_state=42)
        elif sampling_type == "Stratified Sampling":
            # For simplicity, let's assume we are stratifying by 'Blood Type' column
            stratified_sample = health_data.groupby('Blood Type', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size), random_state=42))
        
        # Calculate statistics
        sample_mean = stratified_sample[column_to_analyze].mean()
        sample_std = stratified_sample[column_to_analyze].std()
        sample_se = sample_std / np.sqrt(sample_size)
        
        return sample_mean, sample_std, sample_se

    with st.container(border=True):

        # ให้ User เลือก Column ที่ต้องการทดสอบ โดยเอาเฉพาะที่เป็นตัวเลข
        st.write("Select the column you'd like to analyze:")
        column_to_analyze = st.selectbox("Select column", health_data.select_dtypes(include=[np.number]).columns)
        
        # Slider สำหรับกำหนดขนาด Sample ดึงค่า min max จาก health data
        slider_sample_size = st.slider("Select the sample size:", min_value=1, max_value=len(health_data), value=100, step=1)

        # ให้ผู้ใช้เลือกประเภทของ Sampling
        sampling_type = st.radio("Select the type of sampling:", options=["Random Sampling", "Stratified Sampling"])

        with st.container():
            st.write("**Population statistics**")
            population_a, population_b, population_c = st.columns(3)

            # คำนวณค่าทางสถิติของ Population
            population_mean = health_data[column_to_analyze].mean()
            population_std = health_data[column_to_analyze].std()
            population_size = len(health_data)
            population_se = population_std / np.sqrt(population_size)

            population_a.metric("Mean", f"{population_mean:.2f}", border=True)
            population_b.metric("SD", f"{population_std:.2f}", border=True)
            population_c.metric("SE", f"{population_se:.2f}", border=True)

        with st.container():
            # คำนวณค่าสถิติเมื่อมีการเปลี่ยนแปลงขนาดตัวอย่างและประเภทการสุ่ม
            sample_mean, sample_std, sample_se = calculate_sample_statistics(slider_sample_size, sampling_type, column_to_analyze)

            st.write(f"**Sample statistics** (Sample size: {slider_sample_size})")
            sample_a, sample_b, sample_c = st.columns(3)

            sample_a.metric("Mean", f"{sample_mean:.2f}", delta=None, border=True)
            sample_b.metric("SD", f"{sample_std:.2f}", delta=None, border=True)
            sample_c.metric("SE", f"{sample_se:.2f}", delta=None, border=True)

        # อธิบายว่า Sampling เยอะแล้วยิ่งเข้าใกล้ Population
        st.info("""
        The larger the sample size, the closer the sample statistics will be to the population statistics.
        This is because larger samples reduce the standard error, providing a more accurate estimate of the population's true characteristics.
        """)
    
   
    

def main():
    # Initialize Tailwind CSS
    tw.initialize_tailwind()

    # Navbar
    Navbar()

    # Title
    st.title("Week 04 | Sampling and the Central Limit Theorem")

    # Table of contents
    section_table_of_contents()

    # Content of the page
    section_introduction_to_sampling()

    ("---")
    # Activity
    section_activity_one()
    section_activity_two()
    section_activity_three()

    Footer(4)


if __name__ == "__main__":
    main()
