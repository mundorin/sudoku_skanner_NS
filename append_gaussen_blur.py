import glob
import cv2
import numpy as np
ml=0
sigma=60
for i in range(1400):
    gaussenoise = np.random.normal(ml, sigma, size=[28, 28])
    path=glob.glob('shot-'+str(i)+'-#?#.png',root_dir='C:\python 3.9 programs\sudoku_solver\learn_set')
#    print(path)
    img=cv2.imread('learn_set\\'+path[0],cv2.IMREAD_GRAYSCALE)+gaussenoise
#    cv2.imshow('Result', img)
    cv2.imwrite('learn_set_gaussen_noize/' + path[0], img)
#    cv2.waitKey(100)
#for i in range(383):
#    file=cv2.imread('learn_set/shot-0-#9#.png')