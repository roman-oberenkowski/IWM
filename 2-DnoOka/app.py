
# import streamlit as st



# @st.cache
def classic(img, exp, fov):
    import UnetAndClassic as uc
    # classic processing
    classic_img = uc.classicProcessing((img, exp, fov))
    print("CLASSIC:\n" + uc.statsToString(uc.getAccuracySensitivitySpecificity(classic_img, exp)))


# @st.cache
def unet(img, exp, fov):
    import UnetAndClassic as uc
    # unet
    unet_img = uc.unetPreditct((img, exp, fov))
    print("UNET:\n" + uc.statsToString(uc.getAccuracySensitivitySpecificity(unet_img, exp)))




def main_function():
    # st.set_page_config(page_title="Classifier", page_icon="random")
    # st.write("# Classifier RO KL")
    # classic_button = st.sidebar.button("Classic!")
    # unet_button = st.sidebar.button("Unet!")
    # classifier_button = st.sidebar.button("Classifier!")
    # show_image_button = st.sidebar.button("SHow image!")

    # loading image
    import classifier
    img, exp, fov = classifier.loadImageNr(44, show=False)
    # classic()
    # unet()


    classifier_img = classifier.load_model_and_predict(img, exp, fov)
    print("Classifier:\n"+classifier.statsToString(classifier.getAccuracySensitivitySpecificity(classifier_img, exp)))

    classifier_img=classifier.postprocess_and_display_image(classifier_img, exp)
    print("Classifier:\n"+classifier.statsToString(classifier.getAccuracySensitivitySpecificity(classifier_img, exp)))

if __name__ =='__main__':
    main_function()