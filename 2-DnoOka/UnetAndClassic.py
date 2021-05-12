from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.metrics import sensitivity_score, specificity_score
import skimage.io
import skimage.color
import skimage.filters
import skimage.morphology
import skimage.transform
import os
import numpy as np
from skimage.util import img_as_float,img_as_bool
from joblib import dump, load
from keras_unet.models import custom_unet,vanilla_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded

imgShape = (512 ,512)
model_filename = 'eye_model_v5.h5'
#link do modelu https://drive.google.com/file/d/1QrSLZsx_EVEqfh2x5-dI1buSEod8DDoz/view?usp=sharing
def getAccuracySensitivitySpecificity (producedImg , trueImg):
    trueImg = trueImg.flatten()
    producedImg = producedImg.flatten()
    accuracy = accuracy_score(trueImg , producedImg)
    sensitivity = sensitivity_score(trueImg , producedImg)
    specifivity = specificity_score(trueImg, producedImg)
    return (accuracy , sensitivity , specifivity)

def classicProcessing(data , show=False):
    sourceImg = data[0]
    fov = img_as_bool(data[2])
    fov = skimage.morphology.binary_erosion(fov)
    res = skimage.filters.frangi(sourceImg)
    treshold = np.mean(res)
    res[res > treshold] = 1
    res[res <= treshold] = 0
    res[fov==0] = 0

    if (show):
        skimage.io.imshow(res , cmap = "gray")
        skimage.io.show()
    return res

def statsToString(stats):
    return  ("accuracy: "+ str(stats[0]) + "\nsensitivity: " + str(stats[1]) + "\nspecifivity: " + str(stats[2]))

def saveModel(model , filename):
    dump(model , filename)

def loadModel(filename):
    return load(filename)

def loadImageNr(id,show=False):

    img_list=os.listdir("./all/images")
    if(id>=len(img_list)):
        print("> bad img num, max: "+str(len(img_list)-1))
        return
    exp_list=os.listdir("./all/manual1")
    fov_list=os.listdir("./all/mask")
    img=img_as_float(skimage.io.imread("./all/images/"+str(img_list[id])))
    exp=img_as_float(skimage.io.imread("./all/manual1/"+str(exp_list[id])))
    fov=skimage.io.imread("./all/mask/"+str(fov_list[id]),as_gray=True)
    img=img[:, :, 1]
    if(show):
        skimage.io.imshow(np.hstack([img,exp,fov]))
        skimage.io.show()
    return (img,exp,fov)

def unetlearn():
    imgs = []
    expertMasks = []
    for i in range(1,41):
        temp = loadImageNr(i)
        imgs.append(skimage.transform.resize(temp[0] , imgShape ))
        expertMasks.append(img_as_bool(skimage.transform.resize(temp[1], imgShape)))

    imgs = np.asarray(imgs)
    expertMasks = np.asarray(expertMasks)

    imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)
    expertMasks = expertMasks.reshape(expertMasks.shape[0], expertMasks.shape[1], expertMasks.shape[2], 1)
    #plot_imgs(org_imgs=imgs, mask_imgs=expertMasks, nm_img_to_plot=10, figsize=6)
    #print( imgs.shape , expertMasks.shape , imgs.dtype , expertMasks.dtype)
    img_train, img_val, masks_train, masks_val = train_test_split(imgs, expertMasks, test_size=0.25, random_state=0)
    print("x_train: ", img_train.shape)
    print("y_train: ", masks_train.shape)
    print("x_val: ", img_val.shape)
    print("y_val: ", masks_val.shape)


    input_shape = img_train[0].shape

    model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.5,
    output_activation='sigmoid'
)



    callback_checkpoint = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )

    model.compile(
        #optimizer=Adam(),
        optimizer=SGD(lr=0.01, momentum=0.99),
        loss='binary_crossentropy',
        run_eagerly=True,
        # loss=jaccard_distance,
        metrics=[iou, iou_thresholded]
    )


    model.fit(img_train,masks_train,
              epochs= 120,
              callbacks=[callback_checkpoint],
              batch_size=1,
              validation_data=(img_val, masks_val),
              shuffle=False
              )

def unetPreditct(data):
    img = skimage.transform.resize(data[0] , imgShape)
    img  = img.reshape(1,img.shape[0], img.shape[1], 1)
    print(img.shape)

    input_shape = img[0].shape
    model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.5,
    output_activation='sigmoid'
)
    model.load_weights(model_filename)
    predImg = model.predict(img)


    predImg2 = predImg[0].copy()

    predImg2 = skimage.transform.resize(predImg2, data[1].shape)
    t = skimage.filters.threshold_li(predImg2)
    predImg2[predImg2 > t] = True
    predImg2[predImg2 <= t] = False
    #nie usuwam małych obiektów, bo usunęłoby to też część poprawnie wyznaczonych naczyń krwionośnych

    return predImg2


data = loadImageNr(44,show=False)


producedImg = classicProcessing(data)
stats = getAccuracySensitivitySpecificity(producedImg , data[1])
print(statsToString(stats))

#unetlearn()
predImg = unetPreditct(data)
pom = data[1].reshape(data[1].shape[0] , data[1].shape[1], 1)
skimage.io.imshow(np.hstack([predImg , pom]))
skimage.io.show()
stats = getAccuracySensitivitySpecificity(predImg , pom)
print(statsToString(stats))







