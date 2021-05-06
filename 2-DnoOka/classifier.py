import skimage.io
import skimage.color
import os
import numpy as np
from skimage.util import img_as_float, img_as_uint, img_as_ubyte
from skimage.exposure import rescale_intensity
import imutils
import cv2
import time
import statistics
from scipy.stats import moment
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from skimage import data
from multiprocessing import Pool


def loadImageNr(id, show=False):
    img_list = os.listdir("./all/images")
    if (id >= len(img_list)):
        print("> bad img num, max: " + str(len(img_list) - 1))
        return
    exp_list = os.listdir("./all/manual1")
    fov_list = os.listdir("./all/mask")
    img = img_as_float(skimage.io.imread("./all/images/" + str(img_list[id])))
    exp = img_as_float(skimage.io.imread("./all/manual1/" + str(exp_list[id])))
    fov = skimage.io.imread("./all/mask/" + str(fov_list[id]), as_gray=True)
    img = img[:, :, 1]
    if (show):
        skimage.io.imshow(np.hstack([img, exp, fov]))
        skimage.io.show()
    return (img, exp, fov)


if __name__ == '__main__':
    (img, exp, fov) = loadImageNr(44, show=False)
    resize = False
    target_width = 200
    if (resize):
        img = imutils.resize(img, width=target_width)
        exp = imutils.resize(exp, width=target_width)
        fov = imutils.resize(fov, width=target_width)


def sliding_window(image, fov, exp, processed_data, stepSize, windowSize):
    dim = image.shape
    dim_x = dim[1]
    dim_y = dim[0]
    for y in range(0, dim_y, stepSize):
        if y + windowSize >= dim_y:  # bottom border - discard
            continue
        for x in range(0, dim_x, stepSize):
            if x + windowSize >= dim_x:  # right border - discard
                continue
            if fov[y, x]>0.5 and fov[y + windowSize, x + windowSize]>0.5 and fov[y, x + windowSize]>0.5 and fov[y + windowSize, x]>0.5:
                # all corners inside FOV
                xc = x + windowSize // 2
                yc = y + windowSize // 2
                yield (xc, yc, exp[yc, xc], processed_data[yc, xc])


def some_func(region):
    v = np.var(region.flatten())
    m = moment(region.flatten(), 2)
    return [v, m]


if __name__ == '__main__':
    input_img = img.copy()
    img_size = input_img.shape
    metrics_length = 2
    window_size = 5  # window size i.e. here is 5x5 window
    shape = [input_img.shape[0] - window_size + 1, input_img.shape[1] - window_size + 1, window_size, window_size]
    strides = 2 * input_img.strides
    patches = np.lib.stride_tricks.as_strided(input_img, shape=shape, strides=strides)
    patches = patches.reshape(-1, window_size, window_size)
    print(len(patches))
    print("calculating metrics for all pixels...")
    with Pool(processes=16) as pool:
        temp_res = pool.map(some_func, patches, 25000)
        processed_data_unshaped = np.array(temp_res)
    processed_data = processed_data_unshaped.reshape(shape[0:2] + [metrics_length])
    processed_data = np.pad(processed_data,
                            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)),
                            'constant')

    print("preparing data for learning")
    correct_answers = []
    inputs = []
    for (xc, yc, ans, metrics) in sliding_window(img, fov, exp, processed_data, 1, window_size):
        correct_answer = ans > 0
        correct_answers.append(correct_answer)
        inputs.append(metrics)

    print("samples: ", len(inputs))
    print("learning (fitting)...")

    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(inputs, correct_answers)

    # predicting
    print("preparing data to predict...")
    predicted_image = np.zeros(fov.shape,dtype=np.float64)
    inputs_to_predict = []
    cordinates = []

    for (xc, yc, ans, metrics) in sliding_window(img, fov, exp, processed_data, 1, window_size):
        inputs_to_predict.append(metrics)
        cordinates.append((xc, yc))

    print("predicting...")
    answers = clf.predict(inputs_to_predict)
    ans_cord = zip(answers, cordinates)
    print(len(answers))
    print(answers,end=' ')
    print("writing results...")

    print("Positive answers:",len(answers[answers]))
    print("Negative answers:", len(answers[~answers]))
    print("answers",len(answers))
    for answer,cordinates in ans_cord:
        xc, yc = cordinates
        if answer:
            predicted_image[yc, xc] = 1.0
        else:
            predicted_image[yc, xc] = 0.0



    print("done...")
    skimage.io.imshow(predicted_image,cmap='gray')
    skimage.io.show()
    skimage.io.imshow(exp, cmap='gray')
    skimage.io.show()
    cv2.imshow("XD", imutils.resize(predicted_image, width=900))
    cv2.waitKey(0)
