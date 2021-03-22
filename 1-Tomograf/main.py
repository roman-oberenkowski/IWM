import base64
import datetime
import SessionState
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import skimage.io as io
import skimage.exposure as exp
import skimage.draw
import skimage.transform
import skimage
import math
import cv2


def calcRMSE(img1, img2):
    img1[img1 > 1] = 1
    img2[img2 > 1] = 1
    temp = img1 - img2
    temp = temp * temp
    return math.sqrt(np.mean(temp))


def calcStatistics():
    testImg = io.imread("Shepp_logan.jpg", as_gray=True)
    file = open("ZmianyN.txt", 'w+')
    file.write("N RMSE\n")
    for i in range(90, 721, 90):
        tomograf = Tomograf(testImg, i)
        tomograf.makeSinogramWithParams(False, False)
        rmse = calcRMSE(tomograf.inputImg, tomograf.outputImg)
        file.write(str(i) + " " + str(rmse) + "\n")
    file.close()
    file = open("ZmianyIlosciSkanow.txt", 'w+')
    file.write("N RMSE\n")
    for i in range(90, 721, 90):
        tomograf = Tomograf(testImg, 180, i)
        tomograf.makeSinogramWithParams(False, False)
        rmse = calcRMSE(tomograf.inputImg, tomograf.outputImg)
        file.write(str(i) + " " + str(rmse) + "\n")
    file.close()
    file = open("ZmianyL.txt", 'w+')
    file.write("N RMSE\n")
    for i in range(45, 271, 45):
        tomograf = Tomograf(testImg, 180, 90, np.radians(i))
        tomograf.makeSinogramWithParams(False, False)
        rmse = calcRMSE(tomograf.inputImg, tomograf.outputImg)
        file.write(str(i) + " " + str(rmse) + "\n")
    file.close()


def makeDicom(img, filename, patiantName, patientID, patientWeigth, patientSex, comment):
    # File meta info data elements
    file_meta = FileMetaDataset()

    file_meta.FileMetaInformationGroupLength = 192
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.5962.2'
    file_meta.ImplementationVersionName = 'DCTOOL100'
    file_meta.SourceApplicationEntityTitle = 'CLUNIE1'

    # Main data elements
    ds = Dataset()
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y-%m-%d')
    timeStr = dt.strftime('%H:%M')  # long format with micro seconds
    ds.ContentTime = timeStr

    ds.PatientName = patiantName
    ds.PatientID = patientID
    ds.PatientWeight = patientWeigth
    ds.PatientSex = patientSex

    ds.StudyInstanceUID = '1.3.6.1.4.1.5962.1.2.1.20040119072730.12322'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.5962.1.3.1.1.20040119072730.12322'
    ds.ImageComments = comment

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = img.shape[0]
    ds.Columns = img.shape[1]

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelPaddingValue = -2000
    ds.RescaleIntercept = "-1024.0"
    ds.RescaleSlope = "1.0"
    savedImg = img
    savedImg = savedImg / np.max(savedImg)
    savedImg = savedImg * (2 ^ 16 - 1)
    savedImg[savedImg < 0] = 0
    savedImg = savedImg.astype(np.uint16)
    ds.PixelData = savedImg

    ds.file_meta = file_meta
    ds.is_implicit_VR = True
    ds.is_little_endian = True
    ds.save_as(filename, write_like_original=False)


def readDicom(filename):
    st.write("# Dicom photo:")
    ds = pydicom.dcmread(filename)
    float_img= skimage.img_as_uint(ds.pixel_array)
    p2, p98 = np.percentile(float_img, (1, 98))
    img_rescalled = exp.rescale_intensity(
        float_img,
        in_range=(p2, p98),
        out_range=(0, 1)
    )
    resized = skimage.transform.resize(img_rescalled, (400, 400))

    st.image(resized,clamp=True)
    st.write("# Patient info:")
    st.write("Name: "+ str(ds.PatientName))
    st.write("ID: " + ds.PatientID)
    st.write(f'Weight: {ds.PatientWeight}kg')
    st.write(f'Sex: {ds.PatientSex}')
    st.write(f'Comment: {ds.ImageComments}')
    st.write(f'Date: {ds.ContentDate}')
    st.write(f'Time: {ds.ContentTime}')



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

    def __init__(self, img_param=None, n=180, iterNumber=90, l=np.pi, isFiltered=True,rescalleIntensity=True,simulateOuterCircle=True):
        self.isFiltered = isFiltered
        self.rescalleIntensity = rescalleIntensity
        self.simulate_outer_circle = simulateOuterCircle
        if img_param is None:
            self.inputImg = io.imread("Kolo.jpg", as_gray=True)
        else:
            self.inputImg = img_param
        # make the image square
        (x, y) = self.inputImg.shape
        if (x > y):
            self.inputImg = cv2.copyMakeBorder(self.inputImg, 0, 0, (x - y) // 2, (x - y) // 2, cv2.BORDER_CONSTANT,
                                               value=0)
        elif (x < y):
            self.inputImg = cv2.copyMakeBorder(self.inputImg, (y - x) // 2, (y - x) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                               value=0)

        if self.simulate_outer_circle:
            (x, y) = self.inputImg.shape
            # print(self.inputImg.shape)
            offset = int((math.sqrt(2) * x) // 4)
            # print(offset)
            self.inputImg = cv2.copyMakeBorder(self.inputImg, offset, offset, offset, offset, cv2.BORDER_CONSTANT,
                                               value=0)

        self.iterCount = iterNumber
        self.n = n
        self.l = l
        self.sinogram = np.zeros((self.iterCount, self.n))
        self.outputImg = np.zeros(self.inputImg.shape)
        self.rescalledImg = np.zeros(self.inputImg.shape)
        self.partialOutputImgs = []  # np.zeros((10, self.inputImg.shape[0], self.inputImg.shape[1]))
        self.partialSinogramImgs = []
        self.lines = []
        self.kernel = self.calcKernel()
        # print(self.inputImg.shape)

    def calcEmiterPosition(self, alpha, r, imgShape):
        x = r * math.cos(alpha) + imgShape[1] // 2
        y = r * math.sin(alpha) * -1 + imgShape[0] // 2
        return (y, x)

    def calcDetectorPosition(self, alpha, r, i):
        x = r * math.cos(alpha + math.pi - self.l / 2 + i * self.l / (self.n - 1)) + self.inputImg.shape[1] // 2
        y = r * math.sin(alpha + math.pi - self.l / 2 + i * self.l / (self.n - 1)) * -1 + self.inputImg.shape[0] // 2
        return (y, x)

    def doOneSinogramCell(self, i, iterationNumber, emiterPosition, alpha):
        detectorPos = self.calcDetectorPosition(alpha, np.min(self.inputImg.shape) // 2 - 1, i)
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


    def rescalle_image(self, in_img):
        # print("rescalling intensity")
        float_img = in_img
        p2, p98 = np.percentile(float_img, (1, 98))
        img_rescalled = exp.rescale_intensity(
            float_img,
            in_range=(p2, p98),
            out_range=(0, 1)
        )
        return img_rescalled

    def makeSinogram(self):
        for i in range(0, self.iterCount):
            prog_bar.progress(i / self.iterCount)
            alpha = 2 * np.pi / self.iterCount * i
            self.doOneSingoramRow(i, alpha)
            if i % (self.iterCount // 10) == 1:
                if self.rescalleIntensity:
                    self.partialOutputImgs.append(self.rescalle_image(self.outputImg))
                    self.partialSinogramImgs.append(self.rescalle_image(self.sinogram))
                else:
                    self.partialOutputImgs.append(self.outputImg.copy())
                    self.partialSinogramImgs.append(self.sinogram.copy())

        while len(self.partialSinogramImgs)<11:
            if self.rescalleIntensity:
                self.partialOutputImgs.append(self.rescalle_image(self.outputImg))
                self.partialSinogramImgs.append(self.rescalle_image(self.sinogram))
            else:
                self.partialOutputImgs.append(self.outputImg.copy())
                self.partialSinogramImgs.append(self.sinogram.copy())
        prog_bar.progress(1.0)
        self.outputImg=self.rescalle_image(self.outputImg)


    def test1(self):
        emiter = self.calcEmiterPosition(0, np.min(self.inputImg.shape) // 2 - 1, self.inputImg.shape)
        for i in range(0, 180, 10):
            det = self.calcDetectorPosition(0, np.min(self.inputImg.shape) // 2 - 1, i)
            line = skimage.draw.line_nd(emiter, det)
            self.inputImg[line] = 1


import streamlit as st

global tom


@st.cache #(suppress_st_warning=True)
def calculate_tomograph(img,a,b,c,d,e,f):
    global tom
    tom = Tomograf(img,detector_count,iterations,radial_length,is_filtered_checkbox,rescalle_intensity_checkbox,simulate_outer_circle_checkbox)
    tom.makeSinogram()
    return tom


def get_binary_file_downloader_html(bin_file, file_label='File'):
    import os
    import base64
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


# calcStatistics()

# https://towardsdatascience.com/pagination-in-streamlit-82b62de9f62b
ss = SessionState.get(page_number=0, showing_output=False, var_2="Streamlit demo!")
st.set_page_config(page_title="Symulator tomografu RO KL", page_icon="random")
st.write("# Symulator tomografu RO KL")

input_image = st.sidebar.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
input_image_checkbox = st.sidebar.checkbox("Show input image")
is_filtered_checkbox = st.sidebar.checkbox("Use Filter")
rescalle_intensity_checkbox = st.sidebar.checkbox("Rescalle intensity")
simulate_outer_circle_checkbox = st.sidebar.checkbox("Simulate outer circle")
detector_count = st.sidebar.number_input('Detector Count', value=180)
iterations = st.sidebar.number_input('Iterations', value=180)
radial_length = st.sidebar.number_input('Radial Length', value=3.14)
run_button = st.sidebar.button("Run CT simulation!")
if run_button and input_image is not None:
    ss.showing_output = True

# dicom information
uploaded_dicom=st.sidebar.file_uploader("Upload DICOM", type=['dicom'])
patient_id_input = st.sidebar.number_input('Patient ID', min_value=1, max_value=100000)
patient_weight = st.sidebar.number_input('Weight', min_value=10.0, max_value=10000.0)
gender_input = st.sidebar.radio("Gender: ", ("Male", "Female"))
first_name = st.sidebar.text_input("First name", value='Name')
last_name = st.sidebar.text_input("Last name", value='Last name')
date_of_birth = st.sidebar.date_input("date of birth")

def show_image(item):
    resized = skimage.transform.resize(item, (400, 400))
    float_img = skimage.img_as_float(resized)
    st.image(float_img, clamp=True)


if input_image is not None and ss.showing_output:

    global prog_bar
    prog_bar = st.progress(0)
    tomix = calculate_tomograph(io.imread(input_image, as_gray=True),is_filtered_checkbox,rescalle_intensity_checkbox,simulate_outer_circle_checkbox,detector_count,iterations,radial_length)
    if input_image_checkbox:
        show_image(tomix.inputImg)
    selected_iteration = st.slider("Percent level of computing", 0, 100, 100,step=10)
    index=selected_iteration // 10
    # print("partials: "+str(len(tomix.partialSinogramImgs)))
    show_image(tomix.partialOutputImgs[index])
    show_image(tomix.partialSinogramImgs[index])
    results=[calcRMSE(tomix.inputImg,curr_img) for curr_img in tomix.partialOutputImgs]

    fig, ax = plt.subplots()
    plt.ylim([0.0, 1.0])
    plt.plot([i for i in range(0, 11)], results)

    st.pyplot(fig,dpi=200)
    generate_dicom_button = st.sidebar.button("Generate Dicom file")
    if generate_dicom_button:
        filename = 'CT_' + str(patient_id_input) + '_' + first_name + '_' + last_name + '.dicom'
        makeDicom(tomix.outputImg, filename, first_name + ' ' + last_name, str(patient_id_input),
                  patient_weight, 'F', 'comment')
        st.sidebar.markdown(get_binary_file_downloader_html(filename, 'Dicom File'),
                            unsafe_allow_html=True)

else:
    st.text("Please input a file to start")

if uploaded_dicom is not None:
    readDicom(uploaded_dicom)