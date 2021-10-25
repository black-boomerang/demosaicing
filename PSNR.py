import cv2
import numpy as np


def get_psnr(original_image, recovered_image):
    '''
    :param original_image: оригинальное изображение
    :param recovered_image: восстановленное изображение
    :return: значение PSNR
    '''
    Y = lambda image: 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    original_Y = Y(original_image)
    recovered_Y = Y(recovered_image)
    mse = ((original_Y - recovered_Y) ** 2).mean()
    cv2.imwrite('OriginalLuminance.bmp', original_Y)
    cv2.imwrite('RecoveredLuminance.bmp', recovered_Y)
    return 10 * np.log10(255 ** 2 / mse)


if __name__ == '__main__':
    original_image = cv2.imread('Original.bmp')
    recovered_image = cv2.imread('Recovered.bmp')
    print(f'PSNR = {get_psnr(original_image[2:-2, 2:-2], recovered_image):.3f}')
