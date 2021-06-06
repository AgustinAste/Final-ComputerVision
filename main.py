import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import draw_lines, draw_lines_polar
from scipy import ndimage
from cvum.utils import imread_rgb

import math

def limpiar_y_bordear_imagen(image):

    kernel = np.ones((3,3),np.float32)/25
    # Aplicamos filtro de eliminacion de ruido a la imagen
    dst = cv2.filter2D(image,-1,kernel)
    # **** DETECCION DE BORDES ****

    # BORDES VERTICALES
    filter_edge = np.array([-1,0,1])
    filter_edge = filter_edge.reshape(1,3)
    #convertir el filtro a 2D
    img_edge_v = cv2.filter2D(dst,-1, filter_edge)

    # BORDES HORIZONTALES
    filter_edge = np.array([-1,0,1])
    filter_edge = filter_edge.reshape(1,3).T
    #convertir el filtro a 2D
    img_edge_h = cv2.filter2D(dst,-1, filter_edge)
    
    #  CONVOLUCION Y CORRELACION 
    edge_conv  = ndimage.convolve(dst, filter_edge, mode='constant', cval=0)
    edge_corr = cv2.filter2D(dst,-1,filter_edge, borderType=cv2.BORDER_CONSTANT)
    
    # plt.imshow(np.concatenate([edge_conv,edge_corr], axis=1), cmap='gray', vmin=0, vmax=1)
    intensitity_image = np.sqrt(np.power(img_edge_h, 2) + np.power(img_edge_v,2))
    
    return intensitity_image

# Recibe imagen en grayscale y imagen original , por Hough obtiene lineas 
# Retorna imagen con lineas trazadas
# Fuentes: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html - Practico 2
def detectar_lineas(gray,img_original): 
    edges = (gray * 255).astype(np.uint8)
    
    # Combiene blurearla para que no encuentre n lineas donde hay solo 1 
    edges_blur = cv2.blur(edges, (1,1))
    lines = cv2.HoughLines(edges_blur,1,np.pi/180,160,10,0,0)
    print('lineas encontradas ',len(lines))
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img_original, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    return img_original


input_image = 'images/football.png'

image = imread_rgb(input_image)
img = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
img_prep = limpiar_y_bordear_imagen(img/255)
img_prep = (img_prep * 255).astype(np.uint8)
img_t = cv2.Canny(img_prep,175,200)


retorno = detectar_lineas(img_prep,image)
plt.imshow(retorno)
plt.title('IMG retorno')
plt.axis('off')
plt.show()