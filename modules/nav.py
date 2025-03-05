import streamlit as st


def Navbar():
    with st.sidebar:
        # ลิงค์ไปยังหน้า HomePage
        st.page_link('streamlit_app.py', label='HomePage', icon='🏠')

        # ลิงค์ไปยังหน้า Week 1 (Introduction to Biostatistics)
        st.page_link('pages/week1.py', label='Week 01 | Introduction to Biostatistics', disabled=False)
        st.page_link('pages/week2.py', label='Week 02 | Probability and Probability Distributions', disabled=False)
        st.page_link('pages/week3.py', label='Week 03 | Data Visualization and Descriptive Statistics', disabled=False)
        st.page_link('pages/week4.py', label='Week 04 | Sampling and the Central Limit Theorem', disabled=False)
        st.page_link('pages/week5.py', label='Week 05 | Confidence Intervals and Hypothesis Testing', disabled=True)
        st.page_link('pages/week6.py', label='Week 06 | One-Sample and Two-Sample t-Tests', disabled=True)
        st.page_link('pages/week7.py', label='Week 07 | Chi-Square Tests', disabled=True)
        st.page_link('pages/week8.py', label='Week 08 | Risk Ratios, Odds Ratios, and Experimental Design', disabled=True)
        st.page_link('pages/week9.py', label='Week 09 | Analysis of Variance (ANOVA)', disabled=True)
        st.page_link('pages/week10.py', label='Week 10 | Regression and Correlation', disabled=True)
        st.page_link('pages/week11.py', label='Week 11 | Logistic Regression and Binary Outcomes', disabled=True)
        st.page_link('pages/week12.py', label='Week 12 | Power Analysis and Study Design', disabled=True)
        st.page_link('pages/week13.py', label='Week 13 | Advanced Techniques and Scientific Communication', disabled=True)
        st.page_link('pages/week14.py', label='Week 14 | Group Projects and Presentations', disabled=True)
        st.page_link('pages/week15.py', label='Week 15 | Review and Final Exam', disabled=True)

        st.divider()
        footer_html = """<div style='text-align: left;position: fixed;
            bottom: 0;
            width: 100%;
            font-size: 12px;
            margin-left: 2.5rem;'>
        <p>Developed with ❤️ by GUITAR45628</p>
        </div>"""


        st.markdown(footer_html, unsafe_allow_html=True)
