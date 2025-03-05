import streamlit as st
from modules.nav import Navbar

# Page Titlebar
st.set_page_config(
    page_title="Week 04 | Sampling and the Central Limit Theorem",
)

def main():
    # Navbar
    Navbar()

    # Title
    st.title("Week 04 | Sampling and the Central Limit Theorem")

if __name__ == "__main__":
    main()