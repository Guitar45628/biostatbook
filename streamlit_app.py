import streamlit as st
from modules.nav import Navbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="Homepage | Fundamentals of Statistics and Biostatistics",
    layout="wide"
)

# โหลดไฟล์ CSS
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

def main():
    # เรียกใช้ Navbar
    Navbar()

    # แสดงชื่อคอร์ส
    st.title('Fundamentals of Statistics and Biostatistics')
    
    # แสดงวันที่อัปเดตเนื้อหาเป็นกล่อง
    st.markdown("""<div class="update-box">
    <p><strong>Last updated:</strong> April 24, 2025</p>
    <p><strong>Course status:</strong> Active</p>
    </div>""", unsafe_allow_html=True)

    # Course description with enhanced styling
    st.markdown("""
    ### 📊 Interactive Course Overview
    
    This course provides a comprehensive introduction to the fundamental concepts of statistics and biostatistics. 
    It covers a wide range of topics, including descriptive statistics, probability theory, hypothesis testing, 
    regression analysis, and more. The course is designed for students with little or no prior knowledge of 
    statistics and is suitable for those interested in pursuing a career in data analysis, research, or related fields.
    """)

    # Course highlights section
    st.header("✨ Course Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧪 Applied Learning
        Interactive exercises using sample datasets to build practical skills
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Statistical Analysis
        Learn to perform and interpret various statistical tests with practical examples
        """)
    
    with col3:
        st.markdown("""
        ### 💻 Interactive Code
        Practice coding statistical analyses in a supportive, guided environment
        """)

    # Weekly topics preview with tabs
    st.header("📚 Weekly Topics")
    
    tab1, tab2, tab3 = st.tabs(["Weeks 1-5", "Weeks 6-10", "Weeks 11-13"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            for i in range(1, 4):
                with st.container():
                    st.markdown(f"""
                    ##### Week {i} 
                    {get_week_description(i)}
                    [Go to Week {i}](pages/week{i})
                    """)
        with col2:
            for i in range(4, 6):
                with st.container():
                    st.markdown(f"""
                    ##### Week {i} 
                    {get_week_description(i)}
                    [Go to Week {i}](pages/week{i})
                    """)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            for i in range(6, 9):
                with st.container():
                    st.markdown(f"""
                    ##### Week {i} 
                    {get_week_description(i)}
                    [Go to Week {i}](pages/week{i})
                    """)
        with col2:
            for i in range(9, 11):
                with st.container():
                    st.markdown(f"""
                    ##### Week {i} 
                    {get_week_description(i)}
                    [Go to Week {i}](pages/week{i})
                    """)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            for i in range(11, 14):
                with st.container():
                    st.markdown(f"""
                    ##### Week {i} 
                    {get_week_description(i)}
                    [Go to Week {i}](pages/week{i})
                    """)

    # Data visualization teaser
    st.header("📈 Data Visualization Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample data visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.random.normal(size=1000)
        ax.hist(x, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_title('Normal Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        st.markdown("*Example of normal distribution visualization from Week 2*")

    with col2:
        # Display a sample scatter plot with regression line
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.random.rand(100) * 10
        y = x * 0.5 + np.random.normal(size=100)
        ax.scatter(x, y, color='green', alpha=0.7)
        
        # Add regression line
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='red', linestyle='--')
        
        ax.set_title('Linear Regression Example')
        ax.set_xlabel('Independent Variable')
        ax.set_ylabel('Dependent Variable')
        st.pyplot(fig)
        st.markdown("*Example of regression analysis from Week 10*")

    # Getting started section
    st.header("🚀 Getting Started")
    
    st.markdown("""
    1. **Navigate the course**: Use the sidebar to access weekly modules
    2. **Complete exercises**: Each week includes interactive coding exercises
    3. **Take quizzes**: Test your understanding with practice quizzes
    4. **Engage with data**: Work with sample datasets for statistical practice
    5. **Track progress**: Monitor your advancement through the course
    """)

    # Call to action
    st.markdown("""
    ---
    ### Ready to begin your biostatistics journey?
    Start with [Week 1: Introduction to Biostatistics](pages/week1)
    """)

    # Footer
    st.markdown("""
    ---
    Made with ❤️ and Streamlit | © 2025 GUITAR45628
    """)

def get_week_description(week_number):
    descriptions = {
        1: "Introduction to Biostatistics",
        2: "Probability and Probability Distributions",
        3: "Data Visualization and Descriptive Statistics",
        4: "Sampling and the Central Limit Theorem",
        5: "Confidence Intervals and Hypothesis Testing",
        6: "One-Sample and Two-Sample t-Tests",
        7: "Chi-Square Tests",
        8: "Risk Ratios, Odds Ratios, and Experimental Design",
        9: "Analysis of Variance (ANOVA)",
        10: "Regression and Correlation",
        11: "Logistic Regression and Binary Outcomes",
        12: "Power Analysis and Study Design",
        13: "Advanced Techniques and Scientific Communication"
    }
    return descriptions.get(week_number, "Topic")

if __name__ == '__main__':
    main()
