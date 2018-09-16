import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'



def process_img_3(img):
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morphKernel)
    
    # binarize
    _, b_grad = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # connect horizontally oriented regions
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(b_grad, cv2.MORPH_CLOSE, morphKernel)
    
    # find contours
    _, contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    texts = []
    rect_img = np.copy(img)
    # filter contours
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        
        # ratio of non-zero pixels in the filled region
        r = cv2.contourArea(contour)/(w*h)
        
        if r > 0.3 and h > 20  and w > 80 and h < 200 and w < 200 and w > h:
            patch = img[y:y+h, x:x+w]
            patches.append(patch)
            
            text = extract_text(patch)
            
            if len(text) != 0:
                texts.append(text)
                
            cv2.rectangle(rect_img, (x,y), (x+w,y+h), (0, 255, 0), 1)
            
    return texts, rect_img, patches

def process_img_4(img):
    
    img = cv2.pyrDown(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
    
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    
    # connect horizontally oriented regions
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(bw.shape, np.uint8)
    # find contours
    
    _, contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    
    patches = []
    texts = []
    rect_img = np.copy(img)
    
    for idx in range(0, len(hierarchy[0])):
        x, y, w, h = cv2.boundingRect(contours[idx])
        # fill the contour
        mask = cv2.drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
        
        r = float(cv2.countNonZero(mask)) / (w * h)
        
        if r > 0.45 and w > 5 and h > 5:
            patch = img[y:y+h, x:x+w]
            patches.append(patch)
            
            text = extract_text(patch)
            
            if len(text) != 0:
                texts.append(text)
                
            cv2.rectangle(rect_img, (x, y+h), (x+w, y), (0,255,0), 1)
    
    return texts, rect_img, patches


def erode_test(img):
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (400,400))
    
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, morph_kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, morph_kernel)
    
    _, bw = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    plt.imshow(bw)
    
    return bw

def extract_text(img):
    text_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, text_img = cv2.threshold(text_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(text_img)
    
    return text
    

def extrat_text_2(img):
    
    text_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #text_img = cv2.GaussianBlur(text_img, (3,3), 0)
    text_img = cv2.bitwise_not(text_img)
    
    texts = []
    for i in range(0,1):
        
        _, text_img1 = cv2.threshold(text_img, i, 255, cv2.THRESH_BINARY)
        #plt.imshow(text_img)
        asd = pytesseract.image_to_string(text_img1)
        if len(asd) > 0:
            print(i)
            texts.append(asd)
    
   # return texts
    
    text_img = erode_test(text_img)
    
    return pytesseract.image_to_string(text_img)

    _, text_img = cv2.threshold(text_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    coords = np.column_stack(np.where(text_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
     
    if angle < -45:
        	angle = -(90 + angle)
    
    else:
        angle = -angle
        
    h, w = text_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    text_img = cv2.warpAffine(text_img, M, (w, h),
	  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    """ 
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = np.ones((5,5),np.uint8)
    text_img = cv2.dilate(img, morph_kernel, kernel)
    """
    
    text = pytesseract.image_to_string(text_img)
    
    return text
"""
img = cv2.imread('sub_parts/1.jpg')

asd = extrat_text(img)
"""
org_image = cv2.imread('1.png')

texts, img, patches = process_img_3(org_image)

cv2.imwrite('out1.jpg', img)

if len(texts) != 0:
    for i,im in enumerate(patches):
        cv2.imwrite('sub_parts/' + str(i)+'.jpg', im)
    
    with io.open('output.txt', 'w', encoding = 'utf8') as f:
        for j, text in enumerate(texts):
            text_write = str(j+1) + ') ' + text + "\n"
            f.write(text_write)
#print(pytesseract.image_to_string(img, lang = 'kat'))
       
