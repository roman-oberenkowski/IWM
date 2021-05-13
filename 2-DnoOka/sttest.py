import streamlit as st

def do():
    st.set_page_config(page_title="Classifier", page_icon="random")
    st.title("wtf")
    st.write("# Classifier RO KL")
    classic_button = st.sidebar.button("Classic!")
    unet_button = st.sidebar.button("Unet!")
    classifier_button = st.sidebar.button("Classifier!")
    show_image_button = st.sidebar.button("SHow image!")
    if classic_button:
        st.write("xd classic")

if __name__=='__main__':
    do()
    print("Ended")