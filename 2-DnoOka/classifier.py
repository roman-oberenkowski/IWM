import skimage.io
import skimage.color
import os
import numpy as np
from skimage.util import img_as_float, img_as_uint, img_as_ubyte, img_as_bool
from skimage.exposure import rescale_intensity
import skimage.morphology
import imutils
import cv2
import time
import statistics
from scipy.stats import moment
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from skimage import data
from multiprocessing import Pool
import imblearn


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
            if fov[y, x] > 0.5 and fov[y + windowSize, x + windowSize] > 0.5 and fov[y, x + windowSize] > 0.5 and fov[
                y + windowSize, x] > 0.5:
                # all corners inside FOV
                xc = x + windowSize // 2
                yc = y + windowSize // 2
                yield (xc, yc, exp[yc, xc], processed_data[yc, xc])


def calculate_metrics(region):
    v = np.var(region.flatten())
    m = np.mean(region.flatten())
    mom = moment(region, 2).flatten()
    res = [v, m]
    res.extend(mom)
    return res


class WTF_classifer():
    def __init__(self):
        self.window_size = 5
        self.shape_before_patches = None
        self.img = None
        self.exp = None
        self.fov = None
        self.clf = None

    def load_data(self, input_data):
        self.img = input_data[0]
        self.exp = input_data[1]
        self.fov = input_data[2]

    def cut_into_patches(self):
        input_img = self.img
        window_size = self.window_size
        self.shape_before_patches = [input_img.shape[0] - window_size + 1, input_img.shape[1] - window_size + 1,
                                     window_size,
                                     window_size]
        strides = 2 * input_img.strides
        patches = np.lib.stride_tricks.as_strided(input_img, shape=self.shape_before_patches, strides=strides)
        patches = patches.reshape(-1, window_size, window_size)
        print("patches len: ", len(patches))
        return patches

    def cut_into_patches_learning(self):
        input_img = self.img
        window_size = self.window_size
        self.shape_before_patches = [input_img.shape[0] - window_size + 1, input_img.shape[1] - window_size + 1,
                                     window_size,
                                     window_size]
        strides = 2 * input_img.strides
        patches = np.lib.stride_tricks.as_strided(input_img, shape=self.shape_before_patches, strides=strides)
        # patches = patches.reshape(-1, window_size, window_size)
        print("patches shape: ", patches.shape)
        return patches

    def filter_patches(self, patches):
        half_win = self.window_size // 2
        kernel = np.ones((3, 3), np.uint8)
        fov_smaller = self.fov.copy()
        fov_smaller = cv2.erode(fov_smaller, kernel)
        fov_smaller = fov_smaller[half_win:-half_win, half_win:-half_win]
        exp_smaller = self.exp.copy()
        exp_smaller = exp_smaller[half_win:-half_win, half_win:-half_win]
        patches = patches[fov_smaller > 0.5]
        indices = np.array([i for i in range(patches.shape[0])])
        indices = indices.reshape((-1, 1))
        exp_smaller = exp_smaller[fov_smaller > 0.5]
        print("fov filtered_patches shape ", patches.shape)
        undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='auto')
        indices, y_under = undersample.fit_resample(indices, exp_smaller > 0.5)
        patches_from_indices = np.array([patches[i[0]] for i in indices])
        print("undersamlped shape ", patches_from_indices.shape)
        return patches_from_indices, y_under

    def calculate_patches_metrics(self, patches_local):
        print("calculating metrics for all pixels...")
        with Pool() as pool:
            processed_data_unshaped = np.array(pool.map(calculate_metrics, patches_local, 25000))
        processed_data_local = processed_data_unshaped.reshape(
            self.shape_before_patches[0:2] + [len(processed_data_unshaped[0])])
        del processed_data_unshaped
        processed_data_local = np.pad(processed_data_local, (
            (self.window_size // 2, self.window_size // 2), (self.window_size // 2, self.window_size // 2), (0, 0)),
                                      'constant')
        return processed_data_local

    def calculate_patches_metrics_learning(self, patches_local):
        print("calculating metrics for some pixels (learning mode)...")
        with Pool() as pool:
            processed_data_unshaped = np.array(pool.map(calculate_metrics, patches_local, 25000))
        return processed_data_unshaped

    def learn(self, inputs, correct_answers):
        print("learning (fitting)...")
        self.clf = RandomForestClassifier(n_jobs=-1,max_depth=25)
        self.clf.fit(inputs, correct_answers)

    def prepare_data_to_predict(self, processed_data):
        print("preparing data to predict...")
        inputs_to_predict = []
        cordinates = []
        for (xc, yc, ans, metrics) in sliding_window(self.img, self.fov, self.exp, processed_data, 1, self.window_size):
            inputs_to_predict.append(metrics)
            cordinates.append((xc, yc))
        return cordinates, inputs_to_predict

    def predict(self, coordinates, inputs_to_predict):
        print("predicting...")
        answers = self.clf.predict_proba(inputs_to_predict)
        ans_cord = zip(answers, coordinates)
        print("writing results...")
        return ans_cord

    def produce_predicted_image(self, ans_cord):
        predicted_image = np.zeros(self.img.shape, dtype=bool)
        for answer, coordinates in ans_cord:
            xc, yc = coordinates
            if (answer[1] > 0.6):
                val = True
            else:
                val = False
            predicted_image[yc, xc] = val
        return predicted_image

    def postprocess_and_display_image(self, predicted_image):
        print("done...")
        plt.imshow(self.exp, cmap='gray')
        plt.show()
        plt.imshow(predicted_image, cmap='gray')
        plt.show()
        predicted_image = skimage.morphology.remove_small_objects(predicted_image, min_size=64, connectivity=2,
                                                                  in_place=False)
        plt.imshow(predicted_image, cmap='gray')
        plt.show()
        cv2.imshow("XD", imutils.resize(img_as_float(predicted_image), width=900))
        cv2.waitKey(0)


if __name__ == '__main__':
    # (img, exp, fov) = loadImageNr(0, show=False)
    (img_a, exp_a, fov_a) = loadImageNr(0, show=False)
    # resize = False
    # target_width = 1000
    # if resize:
    #     img = imutils.resize(img, width=target_width)
    #     exp = imutils.resize(exp, width=target_width)
    #     fov = imutils.resize(fov, width=target_width)
    #     img_a = imutils.resize(img_a, width=target_width)
    #     exp_a = imutils.resize(exp_a, width=target_width)
    #     fov_a = imutils.resize(fov_a, width=target_width)

    classifier = WTF_classifer()
    # learn
    X=[]
    y=[]
    for img_nr in [3,23,9]:
        (img, exp, fov) = loadImageNr(img_nr, show=False)
        classifier.load_data((img, exp, fov))
        patches, y_part = classifier.filter_patches(classifier.cut_into_patches_learning())
        X_part = classifier.calculate_patches_metrics_learning(patches)
        X.extend(X_part)
        y.extend(y_part)
    classifier.learn(X, y)
    del patches, y
    del img, exp, fov

    # predict
    classifier.load_data((img_a, exp_a, fov_a))
    processed_data = classifier.calculate_patches_metrics(classifier.cut_into_patches())
    cords_inputs = classifier.prepare_data_to_predict(processed_data)
    del processed_data
    ans_cord = classifier.predict(*cords_inputs)
    del cords_inputs
    predicted_image = classifier.produce_predicted_image(ans_cord)
    del ans_cord
    classifier.postprocess_and_display_image(predicted_image)
