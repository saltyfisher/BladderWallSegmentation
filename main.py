import os
import string

import numpy as np
import pydicom
import pywt
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from PIL import Image
from scipy.spatial import distance

from IWSegmentationClass import IWSegmentationClass


if __name__ == '__main__':
    """
        算法流程：
        1.找到最大的联通区，确定操作窗口为最大联通区的矩形框
        3.先对最大联通区所在区域做抠图操作
        4.以最大联通区前景为蒙版
        5.对于其他各帧，其最大联通区与蒙版的交集为前景，蒙版外为背景，最大联通区在蒙版处的补集是未知区域
        6.进行抠图操作，找到IW（Inner Wall）
        7.对IW做后处理
    """

    t2w_folder_name = 't2'
    dce_folder_name = 't1'
    dwi_folder_name = 'ep2d'
    adc_folder_name = 'ADC'

    cancer_level = '1'
    patient_id = '1'

    if os.path.exists('./results/V%sP%s/' % (cancer_level, patient_id)) is not True:
        os.mkdir('./results/V%sP%s/' % (cancer_level, patient_id))
    IWS = IWSegmentationClass(cancer_level, patient_id)
    IWS.enhancement()
    # IWS.generate_trimap()
    # IWS.segment_IW()

    # IW_filenme = [f for f in os.listdir('./results/V%sP%s/' % (cancer_level, patient_id)) if '.png' in f and 'IW' in f]
    # for filename in IW_filenme:
    #     IW_trimap = cv.imread('./results/V%sP%s/' % (cancer_level, patient_id) + '/' + filename, 0)
    #     IW_frame = int(filename.split('I')[0])
    #     num_connected, labels, stats, centroids = cv.connectedComponentsWithStats(IW_trimap)
    #     areas = [stats[j][-1] for j in range(1, stats.shape[0])]
    #     max_area = np.max(areas)
    #     max_idx = np.argmax(areas)
    #     trimap = (labels == (max_idx + 1))
    #     img = normalized_slices[IW_frame - 1] * (trimap != 1)
    #     # img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    #     gaussian_blurred = cv.GaussianBlur(img, (3, 3), 0, 0)
    #     x = cv.Sobel(gaussian_blurred, cv.FILTER_SCHARR, 1, 0)
    #     y = cv.Sobel(gaussian_blurred, cv.FILTER_SCHARR, 0, 1)
    #     x = cv.convertScaleAbs(x)
    #     y = cv.convertScaleAbs(y)
    #     sobeled = cv.addWeighted(x, 0.5, y, 0.5, 0)
    #     plt.subplot(131);plt.imshow(img, cmap='gray')
    #     plt.subplot(132);plt.imshow(gaussian_blurred, cmap='gray')
    #     plt.subplot(133);plt.imshow(sobeled, cmap='gray')
    #     plt.show()
pass