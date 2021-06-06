import cv2
import numpy as np 

def imread_rgb(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def add_snp_noise(image, amount=0.005):

    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
             for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
             for i in image.shape]
    out[tuple(coords)] = 0
    
    #print(coords.shape)
    
    return out

def correlationdot_2D(image, kernel):

    '''   Two-dimensional cross-correlation.
       result = CORRELATIONDOT2D(im,temp) computes the cross-correlation of
       matrices IM and TEMP with a dot product. The matrix IM must be larger than the matrix
       TEMP for the normalization to be meaningful. The resulting matrix RESULTS
       contains correlation coefficients and its values may range from -1.0 to 1.0.

     im : grayscale image
     temp : grayscale image
    '''
    hi,wi = image.shape
    ht,wt = kernel.shape

    if image.dtype == np.uint8:
        image = image / 255. 
    
    m = hi - ht + 1; 
    n = wi - wt + 1; 
    result = np.zeros([m,n])
    for y in range(0,m):
        for x in range(0,n):
         im_patch = image[y:y+ht, x:x+wt]
         result[y,x] = im_patch.ravel().T.dot(kernel.ravel())

    return result

def calculate_image_gradient(image_grayscale):
    '''
    Calculate image gradient for grayscale image, returns two matrices the same size as the input,
    containig Gradient Magnitude and Gradient Direction.
    '''

    sobel_x = cv2.Sobel(image_grayscale,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(image_grayscale,cv2.CV_64F,0,1,ksize=5)

    g_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    g_direction = np.arctan(np.divide(sobel_y,sobel_x))

    return g_magnitude, g_direction

