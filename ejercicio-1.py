import numpy as np
import cv2
import matplotlib.pyplot as plt

def limpiar_y_bordear_imagen(image):

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blureamos la imagen para no detectar n rectas donde solo debe haber 1.
    img_blureada = cv2.GaussianBlur(grayscale, (5, 5), 1.5)

    # filtro canny edges
    canny_edges = cv2.Canny(img_blureada, 100, 200)
    return canny_edges

# Funcion simple que retorna los indices a los maximos 
# Fuente: https://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output / https://sbme-tutorials.github.io/2019/cv/notes/4_week4.html#algorithm
def indices_maximos(H, num_peaks):
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T

# Dibuja la cantidad de lineas del primer parametro
#  formato xcosTHETHA + ysinTHETHA = rho
def hough_lines_draw(img, qtyLines, rhos, thetas):
    for i in range(len(qtyLines)):
        rho = rhos[qtyLines[i][0]]
        theta = thetas[qtyLines[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        #Fixme: no se ven pero estan
        # se estiran las lineas para que se vean, sino no se vven
        x1 = int(x0 + 3500*(-b))
        y1 = int(y0 + 3500*(a))
        x2 = int(x0 - 3500*(-b))
        y2 = int(y0 - 3500*(a))

        ancho = 4
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), ancho) 


# Fuente https://www.youtube.com/watch?v=rZtyFZ8mPwE / https://sbme-tutorials.github.io/2019/cv/notes/4_week4.html
# PRECOONDICION , img se sugiere que sea una imagen debordes.
# * Las rectas se expresan de forma polar. xcosTHETHA + ysinTHETHA = rho
# El proposito de esta funcion es encontrar para cada (x,y) pertenecientes a un borde en todos los posibles angulos (theta)
# obteniendo los rho y theta que maximizan el array (rho,theta) obtenemos punto y pendiente para la recta detectada.
# El arreglo menncionado lo devolvemos (arr).
def detector_hough(img ,):
    
    img_height, img_width = img.shape
    
    # Pitagoras para obtener la diagonal
    img_diagonal = np.ceil(np.sqrt(img_height**2 + img_width**2)) 

    rhos = np.arange(-img_diagonal, img_diagonal + 1, 1)
    
    # FIXME: establecer estos angulos para que detecte las rectas
    thetas = np.deg2rad(np.arange(-90, 90, 1))

    # creamos el acumulador
    arr = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # tomamos todos los puntros pertenecientos a los bordes, y loopeamos entre ellos
    y_idxs, x_idxs = np.nonzero(img) 
    for i in range(len(x_idxs)): 
        x = x_idxs[i]
        y = y_idxs[i]
        # dentro de cada pixel de borde , para cada theta calculamosrho =  x cos(theta) + y sin(theta)
        # e incrementamos el acumulador en rho, theta
        for j in range(len(thetas)): 
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            arr[rho, j] += 1

    return arr, rhos, thetas


# Para detectar las rectas perimetrales de la cancha utilizamos la tecnica de la 
# transformada de Hough.
# RECTAS A DETECTAR, reemplazar este valor segun convenga
qtyLines= 5
image = cv2.imread('images/football.png')

# Leemos la imagen, la convertimos a grayscale y la "limpiamos" , la dejamos pronta para procesar.
cleanimg = limpiar_y_bordear_imagen(image)

# Corremos hough_detector en la imagen limpia
acumulador, rhos, thetas = detector_hough(cleanimg)

indicies = indices_maximos(acumulador, qtyLines) # Tomamos las 5 rectas de mayor 

# Le pasamos la imagen original para que se vea bien, con las lineas dibujadas, ya que la otra fue filtrada
hough_lines_draw(image, indicies, rhos, thetas)

# # Mostramos la imagen original, con las qtyLines detectadas
cv2.imshow('Deteccion de lineas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


