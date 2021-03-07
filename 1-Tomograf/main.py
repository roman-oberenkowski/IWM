import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.exposure as exp
import skimage.draw
import skimage.transform
import skimage
import math
import cv2

global tom
global state_showing_results


class Tomograf:
    def calcKernel(self):
        kernel_size = 21
        kernel = np.zeros(kernel_size)
        kernel[kernel_size // 2] = 1
        for i in range(1, kernel_size // 2 + 1):
            if (i % 2 == 0):
                kernel[kernel_size // 2 + i] = 0
            else:
                kernel[kernel_size // 2 + i] = (-4 / (np.pi) ** 2) / i / i
            kernel[kernel_size // 2 - i] = kernel[kernel_size // 2 + i]
        return kernel

    def __init__(self, img_param=None):
        self.isFiltered = True
        self.rescalleIntensity = False
        self.simulate_outer_circle= True
        if img_param is None:
            self.inputImg = io.imread("Kolo.jpg", as_gray=True)
        else:
            self.inputImg = img_param
        #make the image square
        (x, y) = self.inputImg.shape
        if (x > y):
            self.inputImg = cv2.copyMakeBorder(self.inputImg, 0, 0, (x - y) // 2, (x - y) // 2, cv2.BORDER_CONSTANT, value=0)
        elif (x < y):
            self.inputImg = cv2.copyMakeBorder(self.inputImg, (y - x) // 2, (y - x) // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if self.simulate_outer_circle:
            (x, y) = self.inputImg.shape
            print(self.inputImg.shape)
            offset = int((math.sqrt(2) * x) // 4)
            print(offset)
            self.inputImg = cv2.copyMakeBorder(self.inputImg, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)

        self.iterCount = 90
        self.n = 180
        self.l = np.pi
        self.sinogram = np.zeros((self.iterCount, self.n))
        self.outputImg = np.zeros(self.inputImg.shape)
        self.rescalledImg = np.zeros(self.inputImg.shape)
        self.partialImgs = np.zeros((10, self.inputImg.shape[0], self.inputImg.shape[1]))
        self.lines = []
        self.kernel = self.calcKernel()
        print(self.inputImg.shape)

    def calcEmiterPosition(self, alpha, r, imgShape):
        # alpha -= np.pi / 2
        x = r * math.cos(alpha) + imgShape[1] // 2
        y = r * math.sin(alpha) * -1 + imgShape[0] // 2
        return (y, x)

    def calcDetectorPosition(self, alpha, r, i):
        x = r * math.cos(alpha + math.pi - self.l / 2 + i * self.l / (self.n - 1)) + self.inputImg.shape[1] // 2
        y = r * math.sin(alpha + math.pi - self.l / 2 + i * self.l / (self.n - 1)) * -1 + self.inputImg.shape[0] // 2
        return (y, x)

    def doOneSinogramCell(self, i, iterationNumber, emiterPosition, alpha):
        detectorPos = self.calcDetectorPosition(alpha, np.min(self.inputImg.shape) // 2, i)
        self.lines.append(skimage.draw.line_nd(emiterPosition, detectorPos))
        res = np.mean(self.inputImg[self.lines[-1]])
        # self.outputImg[line] +=res
        self.sinogram[iterationNumber, i] = res

    def doOneSingoramRow(self, iterationNumber, alpha):
        self.lines = []
        emiterPos = self.calcEmiterPosition(alpha, np.min(self.inputImg.shape) // 2 - 1, self.inputImg.shape)
        for i in range(0, self.n):
            self.doOneSinogramCell(i, iterationNumber, emiterPos, alpha)
        if (self.isFiltered):
            self.filter_image(iterationNumber)
        for i in range(0, self.n):
            self.outputImg[self.lines[i]] += self.sinogram[iterationNumber, i]

    def filter_image(self, j):
        self.sinogram[j, :] = np.convolve(self.sinogram[j, :], self.kernel, mode='same')

    def makeSinogramWithParams(self,filter,rescalle_intensity):
        self.isFiltered=filter
        self.rescalleIntensity=rescalle_intensity
        self.makeSinogram()

    def makeSinogram(self):
        pom = self.iterCount // 10
        pomCounter = 0
        for i in range(0, self.iterCount):
            alpha = 2 * np.pi / self.iterCount * i
            self.doOneSingoramRow(i, alpha)
            # if (i % pom == 0 and pomCounter < 10):
            #   self.partialImgs[pomCounter] = self.outputImg
            #  plt.imshow(self.partialImgs[pomCounter], cmap="gray")
            # plt.show()
            # pomCounter+=1

        if(self.rescalleIntensity):
            print("rescalling intensity")
            float_img = self.outputImg
            p2, p98 = np.percentile(float_img, (1, 98))
            img_rescalled = exp.rescale_intensity(
                float_img,
                in_range=(p2, p98),
                out_range=(0, 1)
            )
            self.outputImg=img_rescalled

            float_img = self.sinogram
            p2, p98 = np.percentile(float_img, (1, 98))
            img_rescalled = exp.rescale_intensity(
                float_img,
                in_range=(p2, p98),
                out_range=(0, 1)
            )
            self.sinogram = img_rescalled
        #plt.imshow(self.outputImg,cmap="gray")
        #plt.show()

    def test1(self):
        emiter = self.calcEmiterPosition(0, np.min(self.inputImg.shape) // 2 - 1, self.inputImg.shape)
        for i in range(0, 180, 10):
            det = self.calcDetectorPosition(0, np.min(self.inputImg.shape) // 2 - 1, i)
            line = skimage.draw.line_nd(emiter, det)
            self.inputImg[line] = 1

#tom2=Tomograf()
#tom2.makeSinogramWithParams(True,True)

import streamlit as st

@st.cache
def start():
    global state_showing_results
    state_showing_results = False


@st.cache
def calculate_tomograph(img=None,filter=False,rescalle=False):
    global tom
    if (img is not None):
        tom = Tomograf(img)
    else:
        tom = Tomograf(None)
    tom.makeSinogramWithParams(filter,rescalle)
    return tom


st.set_page_config(page_title="Symulator tomografu RO KL", page_icon="random")
st.write("# Symulator tomografu RO KL")
start()

input_image = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
# input_image = io.imread("Kolo.jpg", as_gray=True)

if input_image is not None:

    # file_details = {"FileName": input_image.name, "FileType": input_image.type, "FileSize": input_image.size}
    # st.write(file_details)
    input_image_checkbox = st.checkbox("Show input image")
    if input_image_checkbox:
        st.image(input_image)
    is_filtered_checkbox = st.checkbox("Use Filter")
    rescalle_intensity_checkbox = st.checkbox("Rescalle intensity")
    run_button = st.button("Run")
    if (run_button):

        selected_iteration = st.slider("Iteration", 1, 100, 20)
        st.write(selected_iteration)
        print(is_filtered_checkbox,rescalle_intensity_checkbox)
        tomix = calculate_tomograph(io.imread(input_image, as_gray=True),is_filtered_checkbox,rescalle_intensity_checkbox)

        img = skimage.img_as_float(tomix.outputImg)
        sinogram = skimage.img_as_float(tomix.sinogram)
        st.image(skimage.transform.resize(sinogram, (600, 600)), clamp=True)
        st.image(skimage.transform.resize(img, (600, 600)), clamp=True)
    else:
        age_input = st.number_input('Age')
        gender_input = st.radio("Gender: ", ("Male", "Female"))
        first_name = st.text_input("First name")
        last_name = st.text_input("Last name")
        date_of_birth = st.date_input("date of birth")


else:
    st.text("Please upload image first")