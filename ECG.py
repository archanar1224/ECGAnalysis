import numpy as np
import pywt
from PIL import Image
import cv2
from sklearn import svm
from os import listdir
from os.path import isfile, join
from statsmodels.tsa.ar_model import AR

# Returns features of both wavelet and AR model for each image img
def features(img, mode='haar', level=1):

    # Handling gif images
    if "gif" in img :
        img = Image.open(img).convert('RGB')
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        imArray = open_cv_image

    # Other images (png,jpg etc)
    else:
        imArray = cv2.imread(img)

    #Data pre-processing 
    # Conversion to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_BGR2GRAY )

    # Binary and OTSU thresholding
    ret2,th2 = cv2.threshold(imArray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Canny edge detection
    edges = cv2.Canny(th2,100,200)
    kernel = np.ones((3,3),np.uint8)

    # Dilation
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    # Conversion to float
    imArray =  np.float32(img_dilation)   
    imArray /= 255;

    # Extract coefficients from 2D discrete wavelet decomposition for given mode and decomposition level
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    #Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;  

    # Reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    # Proceesing the image for AR model
    # Float conversion
    imArray =  np.float32(img_dilation)   
    imArray /= 255;

    #Flattening the array
    arArray = imArray.flatten()

    # Extracting and fitting the AR model for given dimensions
    dimensions = 4
    ar_model = AR(arArray)
    ar_res = ar_model.fit(dimensions)

    # Display result
    # cv2.imshow('image',imArray_H)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return arr,ar_res.params.tolist()


# Training dataset
mypath='training_data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
trainData = []
labels = []
# For each image in training set, extract features
for i in range(0, len(onlyfiles)):
    parameters = []
    path = join(mypath,onlyfiles[i])
    filename = onlyfiles[i]
    arr, ar_params = features(path,'db1',7)

    # Apply statistics on wavelet coefficients array
    parameters.append(np.mean(arr))
    parameters.append(np.median(arr))
    parameters.append(np.amax(arr))
    parameters.append(np.amin(arr))
    parameters.append(np.ptp(arr))
    parameters.append(np.std(arr))
    parameters.append(np.var(arr))

    # Concatenate AR features to this
    parameters = parameters + ar_params
    trainData.append(parameters)

    # Label 0 : Class Normal 
    if "Normal" in filename: 
        labels.append(0)

    # Label 1 : Class MI
    else:
        labels.append(1)
    

# SVM classifier. Train the model
clf = svm.SVC(kernel='rbf')
clf.fit(trainData, labels)

# Test dataset
mypath='testing_data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
count = 0
miCount  = 0
normalCount = 0 
wrongList = []
totalNormalCount = 0
totalMICount = 0

for i in range(0, len(onlyfiles)):
    parameters = []
    path = join(mypath,onlyfiles[i])
    filename = onlyfiles[i]

    # Extract features
    arr, ar_params = features(path,'db1',7)

    # Statistics
    parameters.append(np.mean(arr))
    parameters.append(np.median(arr))
    parameters.append(np.amax(arr))
    parameters.append(np.amin(arr))
    parameters.append(np.ptp(arr))
    parameters.append(np.std(arr))
    parameters.append(np.var(arr))
    
    # AR features concat
    parameters = parameters + ar_params
    predict = clf.predict([parameters])

    count = count + 1
    
    # Total class counts
    if "Normal" in filename: 
        totalNormalCount = totalNormalCount + 1
    else:
        totalMICount = totalMICount + 1
    
    # Normal images predicted correctly
    if predict == 0 and "Normal" in filename:
        normalCount = normalCount + 1

    # MI images predicted correctly
    elif  predict == 1 and "Normal" not in filename:
        miCount = miCount + 1
    
    # Misclassified images
    else:
        wrongList.append(onlyfiles[i]);

print("=====================================================")
print("Total test data count : ",count)
print("Misclassified images:")
for i in range(len(wrongList)):
    print(wrongList[i])
print("=====================================================")
print("Accuracy", round( (  (miCount+normalCount)*100/count),2 ))
print("MI Accuracy", round((miCount/totalMICount*100),2))
print("=====================================================")


