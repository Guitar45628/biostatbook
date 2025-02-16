import streamlit as st
from week_file.week1 import display_week1
# from week_file.week2 import display_week2
# คุณสามารถเพิ่มการนำเข้า (import) สำหรับ week3.py ไปจนถึง week15.py ได้

# Title
st.title("Statistical Thinking Course")

# ตัวเลือกให้ผู้ใช้เลือกสัปดาห์
week = st.selectbox("Choose a Week to View", ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7", "Week 8", "Week 9", "Week 10", "Week 11", "Week 12", "Week 13", "Week 14", "Week 15"])

# แสดงเนื้อหาตามการเลือกของผู้ใช้
if week == "Week 1":
    display_week1()
# elif week == "Week 2":
#     display_week2()
# elif week == "Week 3":
#     from week_file.week3 import display_week3
#     display_week3()
# elif week == "Week 4":
#     from week_file.week4 import display_week4
#     display_week4()
# elif week == "Week 5":
#     from week_file.week5 import display_week5
#     display_week5()
# elif week == "Week 6":
#     from week_file.week6 import display_week6
#     display_week6()
# elif week == "Week 7":
#     from week_file.week7 import display_week7
#     display_week7()
# elif week == "Week 8":
#     from week_file.week8 import display_week8
#     display_week8()
# elif week == "Week 9":
#     from week_file.week9 import display_week9
#     display_week9()
# elif week == "Week 10":
#     from week_file.week10 import display_week10
#     display_week10()
# elif week == "Week 11":
#     from week_file.week11 import display_week11
#     display_week11()
# elif week == "Week 12":
#     from week_file.week12 import display_week12
#     display_week12()
# elif week == "Week 13":
#     from week_file.week13 import display_week13
#     display_week13()
# elif week == "Week 14":
#     from week_file.week14 import display_week14
#     display_week14()
# elif week == "Week 15":
#     from week_file.week15 import display_week15
#     display_week15()
