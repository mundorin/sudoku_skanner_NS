import cv2
import numpy as np
import pyautogui as pag
import copy
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
import time

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt

def sort_box(mas):
    for i in range(row * colm - 1):
        for j in range(row * colm - 1):
            if mas[j][0][1] < mas[j + 1][0][1]:
                k = copy.deepcopy(mas[j])
                mas[j] = mas[j + 1]
                mas[j + 1] = k

    for l in range(row):
        for i in range(colm - 1):
            for j in range(colm - 1):
                if mas[l * colm + j][0][0] < mas[l * colm + j + 1][0][0]:
                    k = copy.deepcopy(mas[l * colm + j])
                    mas[l * colm + j] = mas[l * colm + j + 1]
                    mas[l * colm + j + 1] = k


def box_processing(box):
    minx = box[0][0]
    miny = box[0][1]
    for i in box[1:-1]:
        if i[0] < minx:
            minx = i[0]
        if i[1] < miny:
            miny = i[1]

    for i in range(len(box)):
        if box[i][0] == minx and box[i][1] == miny and i != 0:
            k = copy.deepcopy(box[0])
            box[0] = box[i]
            box[i] = k
        elif box[i][0] != minx and box[i][1] != miny and i != 2:
            k = copy.deepcopy(box[2])
            box[2] = box[i]
            box[i] = k

    if minx % 1 != 0 and miny % 1 != 0:
        box[0][0] += 1
        box[0][1] += 1
        box[1][1] += 1
        box[3][0] += 1
    elif minx % 1 != 0:
        box[0][0] += 1
        box[3][0] += 1
    elif miny % 1 != 0:
        box[0][1] += 1
        box[1][1] += 1
    return np.int0(box)

    # def classification_cell(imgn):
    img = copy.deepcopy(imgn)
    dsize = (28, 28)
    output = cv2.resize(img, dsize)
    # output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)[1]
    if 255 in output[8:20][8:20]:
        cv2.imwrite('scrshotsnorm/shotnorm' + str(i) + '.png', output)
        pred = model.predict(output.reshape(1, 28, 28))
        pred = np.argmax(pred, axis=1)
        print(output.reshape(1, 28, 28))
        return pred[0]
    cv2.imwrite('scrshotsnorm/shotnorm' + str(i) + '.png', output)
    return 0


def its_zero(img):
    for i in range(18):
        if img[12][6 + i] != 0:
            return False
        if img[6 + i][12] != 0:
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


def translate_to_matrics(img, mas):
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


def new_game(new_game_button, cords_last_cell, difficult=0):
    pag.click(pag.center(new_game_button))
    pag.sleep(0.3)
    if difficult == 0:
        easy_difficult = pag.locateCenterOnScreen('images/easy_difficulty.png')
        pag.click(*easy_difficult)
    pag.sleep(0.4)
    pag.click(*cords_last_cell)
    pag.moveTo(cords_last_cell[0] - 75, cords_last_cell[1])
    pag.sleep(0.1)


def new_game_end(cords_last_cell, difficult=0):
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
    return new_game_button, remove_button, level_label, mistakes_label


def count_mistakes(mistakes_area):
    img = pag.locateOnScreen('images/mistakes_label_0.png', region=mistakes_area)
    if img != None:
        return 0
    img = pag.locateOnScreen('images/mistakes_label_1.png', region=mistakes_area)
    if img != None:
        return 1
    img = pag.locateOnScreen('images/mistakes_label_2.png', region=mistakes_area)
    if img != None:
        return 2
    img = pag.locateOnScreen('images/mistakes_label_3.png', region=mistakes_area)
    if img != None:
        return 3


def normalisation_matrics(matrics,row,colm,translate_dict):
    for i in range(row*colm):
        matrics[i]=translate_dict[matrics[i]]
    return np.array(matrics).reshape(1,810,1)

model = keras.Sequential([
    Flatten(input_shape=(810,1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(27, activation='softmax')
])

def getScore(individual):
    individual=np.array(individual)
    individual = [individual[:layar1_shape[0] * layar1_shape[1]].reshape(layar1_shape),
                   individual[
                   layar1_shape[0] * layar1_shape[1]:layar1_shape[0] * layar1_shape[1] + layar1_shape[1]],
                   individual[
                   layar1_shape[0] * layar1_shape[1] + layar1_shape[1]:layar1_shape[0] * layar1_shape[1] + layar1_shape[
                       1] + layar2_shape[0] * layar2_shape[1]].reshape(layar2_shape),
                   individual[
                   layar1_shape[0] * layar1_shape[1] + layar1_shape[1] + layar2_shape[0] * layar2_shape[1]:layar1_shape[
                                                                                                               0] *
                                                                                                           layar1_shape[
                                                                                                               1] +
                                                                                                           layar1_shape[
                                                                                                               1] +
                                                                                                           layar2_shape[
                                                                                                               0] *
                                                                                                           layar2_shape[
                                                                                                               1] +
                                                                                                           layar2_shape[
                                                                                                               1]],
                   individual[
                   layar1_shape[0] * layar1_shape[1] + layar1_shape[1] + layar2_shape[0] * layar2_shape[1] +
                   layar2_shape[1]:layar1_shape[0] * layar1_shape[1] + layar1_shape[1] + layar2_shape[0] * layar2_shape[
                       1] + layar2_shape[1] + layar3_shape[0] * layar3_shape[1]].reshape(layar3_shape),
                   individual[
                   layar1_shape[0] * layar1_shape[1] + layar1_shape[1] + layar2_shape[0] * layar2_shape[1] +
                   layar2_shape[1] + layar3_shape[0] * layar3_shape[1]:layar1_shape[0] * layar1_shape[1] + layar1_shape[
                       1] + layar2_shape[0] * layar2_shape[1] + layar2_shape[1] + layar3_shape[0] * layar3_shape[1] +
                                                                       layar3_shape[1]]
                   ]
    model.set_weights(individual)

    mistakes = 0
    totalReward = 0

    done = False
    while True:
        screen = pag.screenshot(region=(area_left_top[0], area_left_top[1],
                                        area_right_botom[0] - area_left_top[0], area_right_botom[1] - area_left_top[1]))
        img = cv2.cvtColor(np.asarray(screen), cv2.COLOR_BGR2GRAY)
        matrics = translate_to_matrics(img, contours)
        norm_matrics = normalisation_matrics(matrics.copy(), row, colm, translate_dict)
        res=model.predict(norm_matrics).reshape(27)
        res=(np.argmax(res[:9]),np.argmax(res[9:18]),np.argmax(res[18:27])+1)
        pag.click(center_cords_cells[res[0] * colm + res[1]])
        pag.press(str(res[2]))
        pag.moveTo(center_cords_cells[80][0] - 75, center_cords_cells[80][1])
        pag.sleep(0.05)
        new_mistakes = count_mistakes(mistakes_area)
        if new_mistakes>0 or matrics[res[0] * colm + res[1]]!=0:
            mistakes += 1
            if new_mistakes>0:
                totalReward -= 1
            if matrics[res[0] * colm + res[1]]!=0:
                totalReward -= 4.5
            new_game(new_game_button, center_cords_cells[80], difficult)
            if mistakes>=5:
                break
        else:
            totalReward+=5

    return totalReward,

print(model.summary())
weights = model.get_weights()
layar1_shape=np.shape(weights[0])
layar2_shape=np.shape(weights[2])
layar3_shape=np.shape(weights[4])
weights[0]=weights[0].reshape(-1)
weights[2]=weights[2].reshape(-1)
weights[4]=weights[4].reshape(-1)
normal_weights=np.concatenate([weights[0], weights[1],weights[2],weights[3],weights[4],weights[5]])

# константы генетического алгоритма
POPULATION_SIZE = 3   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1        # вероятность мутации индивидуума
MAX_GENERATIONS = 1    # максимальное количество поколений
HALL_OF_FAME_SIZE = 2
LENGTH_CHROM=len(normal_weights)
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
LOW = -1.0
UP = 1.0
ETA = 20
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

row = 9
colm = row
difficult = 0
translate_dict = {
    0: (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    1: (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    2: (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    3: (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    4: (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    5: (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    6: (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    7: (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    8: (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    9: (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
}

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

exit_from_pycharm()
new_game_button, remove_button, level_label, mistakes_label = find_objects()
mistakes_area = (mistakes_label[0] - 10, mistakes_label[1] - 10, mistakes_label[2] + 20, mistakes_label[3] + 20)
area_left_top = (level_label[0], level_label[1] + level_label[3] + 20)
area_right_botom = (new_game_button[0] - 20, new_game_button[1] + new_game_button[3] + 5)
screen = pag.screenshot(region=(area_left_top[0], area_left_top[1],
                                area_right_botom[0] - area_left_top[0], area_right_botom[1] - area_left_top[1]))
img = cv2.cvtColor(np.asarray(screen), cv2.COLOR_BGR2GRAY)
contours = find_contours(img)
center_cords_cells = list(map(lambda x: pag.center((x[0][0] + area_left_top[0], x[0][1] + area_left_top[1]
                                                    , x[1][0] - x[0][0], x[1][1] - x[0][1])), contours))
new_game(new_game_button, center_cords_cells[80], difficult)


toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)


#algelitism.eaSimpleElitism
#algorithms.eaSimple
population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

best = hof.items[0]
#

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

#observation = env.reset()
#action = int(model.predict(observation[0].reshape(1, -1)))
#
#while True:
#    env.render()
#    observation, reward, done, info = env.step(action)[:4]
#
#    if done:
#        break
#
#    time.sleep(0.03)
#    action = int(model.predict(observation.reshape(1, -1)))
#
#env.close()
