import numpy as np
from skimage.measure import moments


def calculate_metrics(region):
    # first version with metrics is commented
    # v = np.var(region.flatten())
    # m = np.mean(region.flatten())
    # mom = moments(region, 2).flatten()
    # res = [v, m]
    # res.extend(mom)
    # res.append(region[2, 2])
    return region.flatten()


if __name__ == "__main__":
    import skimage.io
    import skimage.color
    import os
    from skimage.util import img_as_float, img_as_uint, img_as_ubyte, img_as_bool
    from skimage.exposure import rescale_intensity
    import skimage.morphology
    import imutils
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    from multiprocessing import Pool
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.metrics import sensitivity_score, specificity_score
    from sklearn.metrics import accuracy_score, confusion_matrix
    from joblib import dump, load


    def getAccuracySensitivitySpecificity(producedImg, trueImg):
        trueImg = trueImg.flatten()
        producedImg = producedImg.flatten()
        accuracy = accuracy_score(trueImg, producedImg)
        sensitivity = sensitivity_score(trueImg, producedImg)
        specifivity = specificity_score(trueImg, producedImg)
        temp_mat = confusion_matrix(trueImg, producedImg)
        tn, fp, fn, tp = np.array(temp_mat).flatten()
        info = "TN: " + str(tn) + " FP: " + str(fp) + " FN: " + str(fn) + " TP: " + (str(tp))
        return (accuracy, sensitivity, specifivity, info)


    def statsToString(stats_in):
        stats = list(map(lambda x: round(x, 4), stats_in[:-1]))
        return "Accuracy: \t\t" + str(stats[0]) + \
               "\nSensitivity: \t" + str(stats[1]) + \
               "\nSpecificity: \t" + str(stats[2]) + "\t " + stats_in[-1]


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
                if fov[y, x] > 0.5 and fov[y + windowSize, x + windowSize] > 0.5 and fov[y, x + windowSize] > 0.5 and \
                        fov[
                            y + windowSize, x] > 0.5:
                    # all corners inside FOV
                    xc = x + windowSize // 2
                    yc = y + windowSize // 2
                    yield (xc, yc, exp[yc, xc], processed_data[yc, xc])


    class worst_classifer():
        def __init__(self):
            self.window_size = 11
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
            undersample = RandomUnderSampler(sampling_strategy='auto')
            indices, y_under = undersample.fit_resample(indices, exp_smaller > 0.5)
            patches_from_indices = np.array([patches[i[0]] for i in indices])
            print("undersamlped shape ", patches_from_indices.shape)
            return patches_from_indices, y_under

        def calculate_patches_metrics(self, patches_local):
            print("calculating metrics for all pixels...")
            # with Pool(processes=8) as pool:
            processed_data_unshaped = np.array(list(map(calculate_metrics, patches_local)))
            processed_data_local = processed_data_unshaped.reshape(
                self.shape_before_patches[0:2] + [len(processed_data_unshaped[0])])
            del processed_data_unshaped
            processed_data_local = np.pad(processed_data_local, (
                (self.window_size // 2, self.window_size // 2), (self.window_size // 2, self.window_size // 2), (0, 0)),
                                          'constant')
            print("calculated metrics for all pixels")
            return processed_data_local

        def calculate_patches_metrics_learning(self, patches_local):
            print("calculating metrics for some pixels (learning mode)...")
            # with Pool(processes=8) as pool:
            processed_data_unshaped = np.array(list(map(calculate_metrics, patches_local)))
            return processed_data_unshaped

        def learn(self, inputs, correct_answers):
            print("learning (fitting)...")
            self.clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, max_depth=7)
            temp_params = {"verbose": 1}
            self.clf.set_params(**temp_params)
            self.clf.fit(inputs, correct_answers)

        def save_model(self, filename):
            temp_params = {"verbose": 0}
            self.clf.set_params(**temp_params)
            dump(self.clf, filename)

        def load_model(self, filename):
            self.clf = load(filename)

        def prepare_data_to_predict(self, processed_data):
            print("preparing data to predict...")
            inputs_to_predict = []
            cordinates = []
            for (xc, yc, ans, metrics) in sliding_window(self.img, self.fov, self.exp, processed_data, 1,
                                                         self.window_size):
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


    def postprocess_and_display_image(predicted_image, exp):
        print("done...")
        # plt.imshow(exp, cmap='gray')
        # plt.show()
        # plt.imshow(predicted_image, cmap='gray')
        # plt.show()
        predicted_image = skimage.morphology.remove_small_objects(predicted_image, min_size=16, connectivity=2,
                                                                  in_place=False)
        # plt.imshow(predicted_image, cmap='gray')
        # plt.show()
        # cv2.imshow("XD", imutils.resize(img_as_float(predicted_image), width=900))
        # cv2.waitKey(0)
        return predicted_image


    def learn_and_save_model():
        classifier = worst_classifer()
        X = []
        y = []
        for img_nr in range(1, 16, 1):
            (img, exp, fov) = loadImageNr(img_nr, show=False)
            classifier.load_data((img, exp, fov))
            patches, y_part = classifier.filter_patches(classifier.cut_into_patches_learning())
            X_part = classifier.calculate_patches_metrics_learning(patches)
            X.extend(X_part)
            y.extend(y_part)
        classifier.learn(X, y)
        classifier.save_model("worstModelEver")
        del patches, y
        del img, exp, fov


    def load_model_and_predict(img_a, exp_a, fov_a):
        classifier = worst_classifer()
        classifier.load_model("worstModelEver")
        classifier.load_data((img_a, exp_a, fov_a))
        processed_data = classifier.calculate_patches_metrics(classifier.cut_into_patches())
        cords_inputs = classifier.prepare_data_to_predict(processed_data)
        del processed_data
        ans_cord = classifier.predict(*cords_inputs)
        del cords_inputs
        predicted_image = classifier.produce_predicted_image(ans_cord)
        del ans_cord
        print("Finished")
        return predicted_image

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        learn_and_save_model()
        exit(0)
    import UnetAndClassic as uc
    import streamlit as st


    @st.cache
    def classic(img, exp, fov):
        classic_img = uc.classicProcessing((img, exp, fov))
        # print("CLASSIC:\n" + uc.statsToString(uc.getAccuracySensitivitySpecificity(classic_img, exp)))
        return classic_img, uc.statsToString(uc.getAccuracySensitivitySpecificity(classic_img, exp))


    @st.cache
    def unet(img, exp, fov):
        unet_img = uc.unetPreditct((img, exp, fov))
        # print("UNET:\n" + uc.statsToString(uc.getAccuracySensitivitySpecificity(unet_img, exp)))
        return unet_img, uc.statsToString(uc.getAccuracySensitivitySpecificity(unet_img, exp))


    @st.cache
    def classifier_predict(img, exp, fov):
        classifier_img = load_model_and_predict(img, exp, fov)
        return classifier_img, statsToString(
            getAccuracySensitivitySpecificity(classifier_img, exp))


    @st.cache
    def loadImageNrCached(id, show=False):
        from skimage.util import img_as_float, img_as_uint, img_as_ubyte, img_as_bool
        import os
        import skimage
        import numpy as np
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


    def main_function():
        img_number = st.sidebar.number_input("Image number", 0, 44, 0, 1)
        img, exp, fov = loadImageNrCached(img_number, show=False)
        st.write("# Eye blood vessel detection RO KL")
        classic_button = st.sidebar.button("Classic")
        unet_button = st.sidebar.button("Unet")
        classifier_button = st.sidebar.button("Classifier")
        show_image_button = st.sidebar.button("Show input image")
        show_exp_button = st.sidebar.button("Show Expert")

        if (unet_button):
            st.write("Unet")
            img, stats = unet(img, exp, fov)
            st.image(img)
            st.write(stats)
            st.write("Expert: ")
            st.image(exp)

        if (classic_button):
            st.write("Classic")
            img, stats = classic(img, exp, fov)
            st.image(img)
            st.write(stats)
            st.write("Expert: ")
            st.image(exp)

        if (classifier_button):
            st.write("Classifer")
            img, stats = classifier_predict(img, exp, fov)
            st.image(img_as_float(img))
            st.write(stats)
            st.write("Expert: ")
            st.image(exp)

        if (show_image_button):
            st.write("Input")
            st.image(img)

        if (show_exp_button):
            st.write("Expert:")
            st.image(exp)

    main_function()
