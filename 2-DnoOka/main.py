import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.metrics import sensitivity_score, specificity_score
import skimage.io
import skimage.color
import skimage.filters
import skimage.morphology
import os
import numpy as np
from skimage.util import img_as_float,img_as_bool
from joblib import dump, load
def getAccuracySensitivitySpecificity (producedImg , trueImg):
    trueImg = trueImg.flatten()
    producedImg = producedImg.flatten()
    accuracy = accuracy_score(trueImg , producedImg)
    sensitivity = sensitivity_score(trueImg , producedImg)
    specifivity = specificity_score(trueImg, producedImg)
    return (accuracy , sensitivity , specifivity)

def classicProcessing(data , show=True):
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

data = loadImageNr(0,show=False)
producedImg = classicProcessing(data)
stats = getAccuracySensitivitySpecificity(producedImg , data[1])
print(statsToString(stats))

