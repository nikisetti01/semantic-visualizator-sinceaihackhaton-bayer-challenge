import os
import streamlit as st
from dotenv import load_dotenv
from components import query_sequence, sidebar

def main():

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.session_state.filters = {}
    st.session_state.hard_limit = 300

    st.title("HSE Bot")
    st.markdown(":robot_face: _I'm AI powered - Ask me anything about HSE data!_ :robot_face:")
    st.markdown(
        """
        Voit kysyä myös suomeksi!
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Flag_of_Finland.svg" alt="Finnish Flag" width="20"/>
        """,
    unsafe_allow_html=True)
    
    user_input = st.text_area(" ",key="user_input")
    show_sidebar = st.checkbox("Show Filters", value=True)

    if show_sidebar:
        sidebar(st.session_state)

    if st.button("Run Query"):
        if user_input.strip():
            try:
                query_sequence(user_input, st.session_state)
            except Exception as e:
                st.error(f"Error running query: {e}")
        else:
            st.warning("Please enter a valid request.")

if __name__ == "__main__":

    load_dotenv()
    ATHENA_ACCESS_KEY = os.getenv('ATHENA_ACCESS_KEY')
    ATHENA_SECRET_ACCESS_KEY = os.getenv('ATHENA_SECRET_ACCESS_KEY')
    
    main()