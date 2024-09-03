import cv2
import numpy as np
import pyautogui as pag
import copy
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd=r'C:\tesseract\tesseract.exe'
def sort_box(mas):
    for i in range(row*colm-1):
        for j in range(row*colm-1):
            if mas[j][0][1] < mas[j + 1][0][1]:
                k = copy.deepcopy(mas[j])
                mas[j] = mas[j + 1]
                mas[j+1] = k

    for l in range(row):
        for i in range(colm-1):
            for j in range(colm-1):
                if mas[l*colm+j][0][0] < mas[l*colm+j+1][0][0]:
                    k = copy.deepcopy(mas[l*colm+j])
                    mas[l*colm+j] = mas[l*colm+j + 1]
                    mas[l*colm+j + 1] = k

def box_processing(box):

    minx = box[0][0]
    miny = box[0][1]
    for i in box[1:-1]:
        if i[0]<minx:
            minx=i[0]
        if i[1]<miny:
            miny=i[1]

    for i in range(len(box)):
        if box[i][0]==minx and box[i][1]==miny and i!=0:
            k=copy.deepcopy(box[0])
            box[0]=box[i]
            box[i]=k
        elif box[i][0]!=minx and box[i][1]!=miny and i!=2:
            k=copy.deepcopy(box[2])
            box[2]=box[i]
            box[i]=k

    if minx % 1 != 0 and miny % 1 != 0:
        box[0][0]+=1
        box[0][1]+=1
        box[1][1]+=1
        box[3][0]+=1
    elif minx % 1 != 0:
        box[0][0]+=1
        box[3][0]+=1
    elif miny % 1 != 0:
        box[0][1]+=1
        box[1][1]+=1
    return np.int0(box)

#def classification_cell(imgn):
    img=copy.deepcopy(imgn)
    dsize=(28,28)
    output = cv2.resize(img,dsize)
    #output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)[1]
    if 255 in output[8:20][8:20]:
        cv2.imwrite('scrshotsnorm/shotnorm'+str(i)+'.png', output)
        pred = model.predict(output.reshape(1,28,28))
        pred = np.argmax(pred, axis=1)
        print(output.reshape(1,28,28))
        return pred[0]
    cv2.imwrite('scrshotsnorm/shotnorm' + str(i) + '.png', output)
    return 0

def its_zero(img):
    for i in range(18):
        if img[12][6+i]!=0:
            return False
        if img[6+i][12]!=0:
            return False
    return True

def find_contours(img):
    img_canny = cv2.Canny(img, 1, 1)

    kernel = np.ones((3, 3), np.uint8)
    img_canny = cv2.dilate(img_canny, kernel, iterations=1)

    kernel2 = np.ones((2, 2), np.uint8)
    img_canny = cv2.erode(img_canny, kernel2, iterations=1)

    contours0, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mas = []
    for i in contours0:
        if len(i) > 180 and len(i) < 280:
            rect = cv2.minAreaRect(i)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника

            box = box_processing(box)

            mas.append((box[0], box[2]))
    sort_box(mas)
    return mas

def translate_to_matrics(img,mas):
    matrics = []
    model = load_model('LNS/inttranslated_printable_blur.h5')
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)[1]
    for i in range(len(mas)):
        width = mas[i][1][1] - mas[i][0][1] - 1
        cropped_img = img[mas[i][0][1] + 1:(mas[i][0][1] + width), mas[i][0][0] + 1:(mas[i][0][0] + width)]
        dsize = (28, 28)
        cropped_img = cv2.resize(cropped_img, dsize)
        if its_zero(cropped_img):
            intstr = 0
        else:
            cropped_img = np.expand_dims(cropped_img, axis=0)
            intstr = np.argmax(model(cropped_img, training=False), axis=1)[0]
        matrics.append(intstr)
    return matrics

def print_matrics(matrics):
    for i in range(row):
        for j in range(colm):
            print(matrics[colm * row - 1 - (i * colm + j)], end="  ")
        print()

def exit_from_pycharm():
    x, y = pag.locateCenterOnScreen('images/close_button_pycharm.png')
    pag.click(x, y)
    pag.sleep(0.2)

def new_game(new_game_button,cords_last_cell,difficult=0):
    pag.click(pag.center(new_game_button))
    pag.sleep(0.3)
    if difficult==0:
        easy_difficult = pag.locateCenterOnScreen('images/easy_difficulty.png')
        pag.click(*easy_difficult)
    pag.sleep(0.4)
    pag.click(*cords_last_cell)
    pag.moveTo(cords_last_cell[0]-75,cords_last_cell[1])
    pag.sleep(0.1)

def new_game_end(cords_last_cell,difficult=0):
    pag.click(pag.center(pag.locateOnScreen('images/new_game_end_button.png')))
    pag.sleep(0.3)
    if difficult == 0:
        easy_difficult = pag.locateCenterOnScreen('images/easy_difficulty_end.png')
        pag.click(*easy_difficult)
    pag.sleep(0.35)
    pag.click(*cords_last_cell)
    pag.moveTo(cords_last_cell[0] - 75, cords_last_cell[1])
    pag.sleep(0.1)
def find_objects():
    new_game_button = pag.locateOnScreen('images/new_game_button.png')
    remove_button = pag.locateOnScreen('images/remove_button.png')
    level_label = pag.locateOnScreen('images/level_label.png')
    mistakes_label = pag.locateOnScreen('images/mistakes_label_0.png')
    return new_game_button,remove_button,level_label,mistakes_label

def count_mistakes(mistakes_area):
    img = pag.locateOnScreen('images/mistakes_label_0.png',region=mistakes_area)
    if img!=None:
        return 0
    img = pag.locateOnScreen('images/mistakes_label_1.png',region=mistakes_area)
    if img!=None:
        return 1
    img = pag.locateOnScreen('images/mistakes_label_2.png',region=mistakes_area)
    if img!=None:
        return 2
    img = pag.locateOnScreen('images/mistakes_label_3.png',region=mistakes_area)
    if img!=None:
        return 3
row=9
colm=row
difficult=0

exit_from_pycharm()
new_game_button,remove_button,level_label,mistakes_label = find_objects()
mistakes_area=(mistakes_label[0]-10,mistakes_label[1]-10,mistakes_label[2]+20,mistakes_label[3]+20)
area_left_top=(level_label[0],level_label[1]+level_label[3]+20)
area_right_botom=(new_game_button[0]-20,new_game_button[1]+new_game_button[3]+5)
screen = pag.screenshot(region=(area_left_top[0],area_left_top[1],
                                area_right_botom[0]-area_left_top[0], area_right_botom[1]-area_left_top[1]))
#screen.show()

#img = cv2.imread('images/img_0.png',cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(np.asarray(screen), cv2.COLOR_BGR2GRAY)
contours=find_contours(img)
center_cords_cells = list(map(lambda x: pag.center((x[0][0]+area_left_top[0],x[0][1]+area_left_top[1]
                                                      ,x[1][0]-x[0][0],x[1][1]-x[0][1])) , contours))


new_game(new_game_button,center_cords_cells[80],difficult)

mistakes=0
while True:
    screen = pag.screenshot(region=(area_left_top[0],area_left_top[1],
                                    area_right_botom[0]-area_left_top[0], area_right_botom[1]-area_left_top[1]))
    img = cv2.cvtColor(np.asarray(screen), cv2.COLOR_BGR2GRAY)
    matrics=translate_to_matrics(img,contours)
    print_matrics(matrics)
    print('введите координаты и значение следущей выбраной клетки')
    massege=input().split()
    exit_from_pycharm()
    pag.click(center_cords_cells[int(massege[0])*colm+int(massege[1])])
    pag.press(massege[2])
    pag.moveTo(center_cords_cells[80][0]-75,center_cords_cells[80][1])
    pag.sleep(0.05)
    new_mistakes = count_mistakes(mistakes_area)
    if mistakes<new_mistakes:
        mistakes=new_mistakes
        pag.click(pag.center(remove_button))
        pag.sleep(0.05)
        print('wrong! =',mistakes)
        if mistakes==3:
            pag.sleep(0.3)
            mistakes = 0
            new_game_end(center_cords_cells[80],difficult)
