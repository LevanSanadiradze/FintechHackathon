try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

def show_img(img, cmap = None):
    
    if cmap:
        plt.imshow(img, cmap = cmap)
    else:     
        plt.imshow(img)
   
    plt.show()

def preprocess_edges(img, display = False):
    
    if display:
        show_img(img)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    blur = cv2.GaussianBlur(img, (5,5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    
    gray = cv2.bilateralFilter(gray, 11, 15, 15)
    
    return gray

def preprocess(img):
    
    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
	
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
    
    ret,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
    return threshold_img

def cleanPlate(plate):
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	#thresh= cv2.dilate(gray, kernel, iterations=1)

	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	im1,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
         areas = [cv2.contourArea(c) for c in contours]
         max_index = np.argmax(areas)
         
         max_cnt = contours[max_index]
         max_cntArea = areas[max_index]
         x,y,w,h = cv2.boundingRect(max_cnt)
         
         if not ratioCheck(max_cntArea,w,h):
             return plate,None
         
         cleaned_final = thresh[y:y+h, x:x+w]
         return cleaned_final,[x,y,w,h]
    
	else:
         return plate, None
 
def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
        
    aspect = 4.7272
    min = 15*aspect*15  # minimum area
    max = 125*aspect*125  # maximum area
    
    rmin = 3
    rmax = 6
    
    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
	
    return True

def isMaxWhite(plate):
    avg = np.mean(plate)
    
    if(avg>=115):
        return True
    else:
        return False

def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect
    
    if(width>height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle
        
    if angle>15:
        return False
    
    if height == 0 or width == 0:
        return False
    area = height*width
    
    if not ratioCheck(area,width,height):
        return False
    else:
        return True

def contours(img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    
    im2,contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    return contours

def choose_contours(cnts):

    numberPlate = [] 
    
    for c in cnts:
            peri = cv2.arcLength(c, True)
            
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                numberPlate.append(approx)
                
    return numberPlate


def get_rects(img,contours):
    
    rects = []
    
    for i,cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        
        if validateRotationAndRatio(min_rect):
            
            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]
            
            if(isMaxWhite(plate_img)):
                #count+=1
                clean_plate, rect = cleanPlate(plate_img)
                
                if rect:
                    x1,y1,w1,h1 = rect
                    x,y,w,h = x+x1,y+y1,w1,h1
                    
                    plate_img = img[y:y+h,x:x+w]
            
                    rects.append(plate_img)
    
    return rects

def get_text(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(threshold)
    
    return text
    

image = cv2.imread('test3.jpg')

img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

plt.imshow(img[:,:,0])
plt.imshow(img[:,:,1])
plt.imshow(img[:,:,2], cmap = 'gray')

threshold_img = preprocess(image)

contours= contours(threshold_img)

rects = get_rects(image, contours)

text = get_text(rects[0])