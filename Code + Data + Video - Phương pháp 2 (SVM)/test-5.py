import numpy as np
import cv2
import time
import glob
from skimage.feature import hog

# Định nghĩa hàm trích đặc trưng cho từng ảnh
def get_hog_features(img, orient=8, pix_per_cell=16, cell_per_block=4,vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),                                  
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec,multichannel=True)
        return features, hog_image    
    else: # Otherwise call with one output     
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, visualize=vis, feature_vector=feature_vec,
                       multichannel=True)
        return features

def fd_histogram(image,bins=16, mask=None):
    # convert the image to HSV color-space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    #hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [256], [0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
    #return(hist)

#D:\New folder (5)\202111\xu-ly-anh\New folder (5)\data\tomatoChin
const = 70
def find_contours_ov2640(image):
    global const
    image = image[0:640, const:480]
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])#cân bằng sáng
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    ret, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(blurred, 30, 100)
    #cv2.imshow("Image", edged)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(edged, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #coins = image.copy()
    #cv2.drawContours(coins, contours, -1, (0, 255, 0), 1)
    #cv2.imshow("Coins", coins)
    #cv2.waitKey(0)
    return contours

xData=[]
yLabel=[]

def rotate(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def load_data(path, label):
    global xData, yLabel, const
    for filename in glob.glob(path+'\\*.png'):
        imageOrigin = cv2.imread(filename)
        #resized = np.reshape(imageOrigin, (1, 32*32*3))
        #resized=resized[0]
        #resized = np.hstack((resized, np.array((1))))
        xData.append(fd_histogram(imageOrigin))
        yLabel.append(label)
        
def load_data_old(path, label):
    global xData, yLabel, const
    for filename in glob.glob(path+'\\*.png'):
        imageOrigin = cv2.imread(filename)
        image = imageOrigin.copy()
        contours = find_contours_ov2640(image)

        # Loop over each contour
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h<3000:
                continue
            
            coin = imageOrigin[y:y + h, x+const:x+const + w]
            #print("diện tích boundingRect: ", w*h)
            #cv2.imshow(str(i)+' jpg', coin)
            #cv2.waitKey(0)

            resized = cv2.resize(coin, [32, 32], interpolation = cv2.INTER_AREA)
            #print(resized.shape)
            resized = np.reshape(resized, (1, resized.shape[0]*resized.shape[1]*3))
            resized=resized[0]
            resized = np.hstack((resized, np.array((1))))
            #print(resized.shape)
            xData.append(fd_histogram(resized))
            yLabel.append(label)            

def test_model(path, model):
    imageOrigin = cv2.imread(path)
    image = imageOrigin.copy()
    contours = find_contours_ov2640(image)

    # Loop over each contour
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h<3000:
            continue
            
        coin = imageOrigin[y:y + h, x+const:x+const + w]
        cv2.imshow(str(i)+' jpg', coin)
        cv2.waitKey(0)
        resized = cv2.resize(coin, [32, 32], interpolation = cv2.INTER_AREA)
        
        #print(resized.shape)
        #resized = np.reshape(resized, (1, resized.shape[0]*resized.shape[1]*3))
        #resized=resized[0]
        #resized = np.hstack((resized, np.array((1))))
        #print(resized.shape)
        predicted = model.predict([fd_histogram(resized)])
        print(predicted)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle


load_data('.\\data-origin\\quaChin', '0')
load_data('.\\data-origin\\quaXanh', '1')
xData = np.array(xData)
le = LabelEncoder()
yLabel = le.fit_transform(yLabel)
x_train, x_test, y_train, y_test = train_test_split(xData, yLabel, test_size = 0.10, random_state = 101)
svc_model = SVC(kernel="linear", C=1.0)
svc_model.fit(x_train, y_train)
svc_predicted = svc_model.predict(x_test)
print('SVM',accuracy_score(y_test, svc_predicted))

import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(svc_model, file)
'''

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#1 xanh, 0 chin
test_model('image_8.png', model)'''
