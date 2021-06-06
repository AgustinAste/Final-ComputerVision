import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import draw_lines, draw_lines_polar
from scipy import ndimage
from cvum.utils import imread_rgb
# #Load Image
# highway_image = imread_rgb('images/football.png')/255


# #Greyscale Image
# greyscale_highway_image = cv2.cvtColor(highway_image,cv2.COLOR_RGB2GRAY)

# H,W = greyscale_highway_image.shape

# #Blur Image with Gaussian Filter
# blurred_highway_image = cv2.GaussianBlur(greyscale_highway_image,(7,7),2)

# #Do edge detection with canny edge detection
# highway_edges = cv2.Canny(blurred_highway_image,150,250).astype(np.uint16)
# #print(np.max(highway_edges))
# #Canny returns binary Image, convert to int16
# #highway_edges = (255*highway_edges).astype(np.uint16)
# #print(np.max(highway_edges))
# #Apply filter for edges in a particular direction edge1=left_edge edge2 =
# #right_edge

# left_edge=cv2.filter2D(highway_edges,-1,np.array([[0,1],[1,0]]))
# right_edge=cv2.filter2D(highway_edges,-1,np.array([[1,0],[0,1]]))

# #Threshold the edge image.
# left_edge[left_edge < 510] = 0
# right_edge[right_edge < 510] = 0

# plt.imshow(np.concatenate([highway_edges,left_edge,right_edge], axis=1), cmap='gray')
# plt.show()

import math

# Recibe imagen en grayscale, le aplica Canny y por Hough obtiene lineas
# Retorna imagen con lineas trazadas
# Fuentes: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html - Practico 2
def detectar_lineas(gray): 
    edges = cv2.Canny(gray,0,10,apertureSize = 7)

    lines = cv2.HoughLines(edges,1,np.pi/180,90,10,0,0)
    print ('lineas encontradas',lines.size)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    return img

img = cv2.imread('football.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detectar_lineas(gray)

plt.imshow(img)
plt.axis('off')
plt.show()
