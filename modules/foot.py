import streamlit as st

def Footer(num_weeks):
    st.divider()

    st.markdown(f"""
<div style='
    text-align: center;
    padding: 0px;
    margin: 0px auto;
    max-width: 600px;
    border-radius: 10px;
    ;
'>
    <h2>End of Week {num_weeks} 🎉</h2>
</div>
""", unsafe_allow_html=True)