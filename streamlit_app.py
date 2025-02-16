import streamlit as st
from modules.nav import Navbar

st.set_page_config(page_title="Homepage | Fundamentals of Statistics and Biostatistics")

# โหลดไฟล์ CSS
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

def main():
    # เรียกใช้ Navbar
    Navbar()

    # แสดงชื่อคอร์ส
    st.title('Fundamentals of Statistics and Biostatistics')

    # แสดงคำอธิบายคอร์ส
    st.markdown("""This course provides a comprehensive introduction to the fundamental concepts of statistics and biostatistics. It covers a wide range of topics, including descriptive statistics, probability theory, hypothesis testing, regression analysis, and more. The course is designed for students with little or no prior knowledge of statistics and is suitable for those interested in pursuing a career in data analysis, research, or related fields.
    """)

    # แสดงวันที่อัปเดตเนื้อหาเป็นกล่อง
    st.markdown("""<div class="update-box">""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
