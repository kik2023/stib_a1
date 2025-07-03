import streamlit as st


st.image('753728_poster.jpg',width=300)
st.title("Some random streamlit thing")

st.date_input("Transactional date")
st.radio("Department:",['AI','CFI PE', 'CFI PM', 'ISG NPI', 'CSG NPI'])

st.camera_input("Case Input:")