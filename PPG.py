import time

import cv2
import numpy as np


def get_green_component(cell_green, cell_other):
    '''
    :param cell_green: ячейка размера 5 на 5 c зелёными компонентами
    :param cell_other: ячейка размера 5 на 5 c красными или синими компонентами
    :return: зелёная компонента центральной ячейки
    '''
    N = np.abs(cell_other[2][2] - cell_other[0][2]) * 2 + np.abs(cell_green[3][2] - cell_green[1][2])
    E = np.abs(cell_other[2][2] - cell_other[2][4]) * 2 + np.abs(cell_green[2][1] - cell_green[2][3])
    W = np.abs(cell_other[2][2] - cell_other[2][0]) * 2 + np.abs(cell_green[2][1] - cell_green[2][3])
    S = np.abs(cell_other[2][2] - cell_other[4][2]) * 2 + np.abs(cell_green[3][2] - cell_green[1][2])
    min_grad_idx = np.argmin([N, E, W, S])
    if min_grad_idx == 0:
        green_component = (cell_green[1][2] * 3 + cell_green[3][2] + cell_other[2][2] - cell_other[0][2]) / 4
    elif min_grad_idx == 1:
        green_component = (cell_green[2][3] * 3 + cell_green[2][1] + cell_other[2][2] - cell_other[2][4]) / 4
    elif min_grad_idx == 2:
        green_component = (cell_green[2][1] * 3 + cell_green[2][3] + cell_other[2][2] - cell_other[2][0]) / 4
    else:
        green_component = (cell_green[3][2] * 3 + cell_green[1][2] + cell_other[2][2] - cell_other[4][2]) / 4
    return np.min([np.max([np.round(green_component), 0]), 255])


def get_rb_component(cell_green, cell_target, cell_other):
    '''
    :param cell_green: ячейка размера 5 на 5 c зелёными компонентами
    :param cell_target: ячейка размера 5 на 5 c целевой компонентой
    :param cell_other: ячейка размера 5 на 5 c нецелевой компонентой
    :return: целевая компонента центральной ячейки
    '''
    NE = np.abs(cell_target[1, 3] - cell_target[3, 1]) + np.abs(cell_other[2, 2] - cell_other[0, 4]) + np.abs(
        cell_other[2, 2] - cell_other[4, 0]) + np.abs(cell_green[2, 2] - cell_green[1, 3]) + np.abs(
        cell_green[2, 2] - cell_green[3, 1])
    NW = np.abs(cell_target[1, 1] - cell_target[3, 3]) + np.abs(cell_other[2, 2] - cell_other[0, 0]) + np.abs(
        cell_other[2, 2] - cell_other[4, 4]) + np.abs(cell_green[2, 2] - cell_green[1, 1]) + np.abs(
        cell_green[2, 2] - cell_green[3, 3])
    if NE < NW:
        target_component = hue_transit(cell_green[1, 3], cell_green[2, 2], cell_green[3, 1], cell_target[1, 3],
                                       cell_target[3, 1])
    else:
        target_component = hue_transit(cell_green[1, 1], cell_green[2, 2], cell_green[3, 3], cell_target[1, 1],
                                       cell_target[3, 3])
    return np.min([np.max([np.round(target_component), 0]), 255])


def calculate_image_green_component(image):
    '''
    Вычисляет зелёную компоненту для всего изображения
    :param image: исходное полутоновое изображение с 3 каналами
    '''
    height, width, _ = image.shape
    for row in range(2, height - 2):
        if row % 2 == 0:
            start_col = 2
            component_num = 2
        else:
            start_col = 3
            component_num = 0
        for col in range(start_col, width - 2, 2):
            cell = image[row - 2:row + 3, col - 2:col + 3]
            image[row, col, 1] = get_green_component(cell[:, :, 1], cell[:, :, component_num])


def hue_transit(L1, L2, L3, V1, V3):
    '''
    Функция пытается восстановить значение сигнала V в точке V2 по соседним V1 и V3,
    перенося «форму» из сигнала L.
    :return: восстановленное значение V2
    '''
    if (L1 < L2 < L3) or (L1 > L2 > L3):
        value = V1 + (V3 - V1) * (L2 - L1) / (L3 - L1)
    else:
        value = (V1 + V3) / 2 + (L2 - (L1 + L3) / 2) / 2
    return np.min([np.max([np.round(value)])])


def calculate_image_rb_component(image):
    '''
    Вычисляет красную и синюю компоненту для всего изображения
    :param image: исходное полутоновое изображение с 3 каналами
    '''
    height, width, _ = image.shape
    for row in range(1, height - 1):
        if row % 2 == 0:
            start_col = 1
            vertical_component = 0
            horizontal_component = 2
        else:
            start_col = 2
            vertical_component = 2
            horizontal_component = 0
        for col in range(start_col, width - 1, 2):
            cell = image[row - 1:row + 2, col - 1:col + 2]
            image[row, col, vertical_component] = hue_transit(cell[0, 1, 1], cell[1, 1, 1], cell[2, 1, 1],
                                                              cell[0, 1, vertical_component],
                                                              cell[2, 1, vertical_component])
            image[row, col, horizontal_component] = hue_transit(cell[1, 0, 1], cell[1, 1, 1], cell[1, 2, 1],
                                                                cell[1, 0, horizontal_component],
                                                                cell[1, 2, horizontal_component])

    for row in range(2, height - 2):
        if row % 2 == 0:
            start_col = 2
            component_num = 0
        else:
            start_col = 3
            component_num = 2
        for col in range(start_col, width - 2, 2):
            cell = image[row - 2:row + 3, col - 2:col + 3]
            image[row, col, component_num] = get_rb_component(cell[:, :, 1], cell[:, :, component_num],
                                                              cell[:, :, 2 - component_num])


def PPG(image):
    '''
    :param image: исходное полутоновое изображение с 3 каналами
    :return: восстановленное цветное изображение
    '''
    start_time = time.time()
    height, width, _ = image.shape
    target_image = image.copy()

    # первый шаг
    calculate_image_green_component(target_image)
    # второй шаг
    calculate_image_rb_component(target_image)

    target_image = target_image[2:height - 2, 2:width - 2]

    algorithm_time = time.time() - start_time
    print(f'Время работы алгоритма: {algorithm_time:.3f} секунд')
    print(f'Время работы на одном мегапикселе: {algorithm_time * 3e6 / target_image.size:.3f} секунд')

    return target_image


if __name__ == '__main__':
    cfa_image = cv2.imread('RGB_CFA.bmp').astype('float')
    image = PPG(cfa_image)
    cv2.imwrite('Recovered.bmp', image)
