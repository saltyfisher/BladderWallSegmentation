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

from utils import *

t2w_folder_name = 't2'
dce_folder_name = 't1'
dwi_folder_name = 'ep2d'
adc_folder_name = 'ADC'

class IWSegmentationClass:
    def __init__(self, cancer_level, patient_id):
        self.cancer_level = cancer_level
        self.patient_id = patient_id

        patient_folder_name = './data/' + 'VIRADS' + cancer_level + '/P' + patient_id + '/'
        imgs_folder_name = [f for f in os.listdir(patient_folder_name) if t2w_folder_name in f.casefold()][0]
        imgs_folder_name = patient_folder_name + imgs_folder_name

        # 读取序列
        self.slices = [pydicom.read_file(imgs_folder_name + '/' + f) for f in os.listdir(imgs_folder_name)]
        # slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        raw_slices = [self.slices[i].pixel_array for i in range(len(self.slices))]
        self.normalized_slices = [(raw_slices[i] / np.max(raw_slices[i]) * 255).astype(np.uint8) for i in
                             range(len(raw_slices))]

    def enhancement(self):
        if os.path.exists('./results/V%sP%s/raw/' % (self.cancer_level, self.patient_id)) is not True:
            os.mkdir('./results/V%sP%s/raw/' % (self.cancer_level, self.patient_id))
        if os.path.exists('./results/V%sP%s/enhanced/' % (self.cancer_level, self.patient_id)) is not True:
            os.mkdir('./results/V%sP%s/enhanced/' % (self.cancer_level, self.patient_id))
        for i in range(len(self.normalized_slices)):
            if i > len(self.normalized_slices) * 2 / 3:
                break
            slice = self.normalized_slices[i].copy()
            cv.imwrite('./results/V%sP%s/raw/%d.png' % (self.cancer_level, self.patient_id, (i + 1)), slice)
            slice = cv.medianBlur(slice, 5)
            slice = cv.equalizeHist(slice)
            cv.imwrite('./results/V%sP%s/enhanced/%d.png' % (self.cancer_level, self.patient_id, (i + 1)), slice)
            # if i != 0:
            #     if np.abs(np.max(slice * mask) - I_max_pre) > 50:
            #         break
            # slice[slice < I_min] = I_min
            # slice[slice > I_max] = I_max
            # idx = (slice >= I_min) & (slice <= I_max)
            # slice[idx] = slice[idx] / (I_max - I_min) * I_max
            # plt.imshow(slice, cmap='gray');plt.show()
            # slice = cv.medianBlur(slice, 5)
            # plt.imshow(slice, cmap='gray');plt.show()

    def generate_foreground(self):
        self.largest_connected = {
            'frame': 0,
            'mask': 0,
            'area': 0,
            'centroid': 0,
            'stats': 0
        }
        threshold_slices = []
        self.fg_slices = []
        I_min = 100
        I_max = 200
        mask = np.ones_like(self.normalized_slices[0])
        I_max_pre = 0
        if os.path.exists('./results/V%sP%s/raw/' % (self.cancer_level, self.patient_id)) is not True:
            os.mkdir('./results/V%sP%s/raw/' % (self.cancer_level, self.patient_id))
        if os.path.exists('./results/V%sP%s/coarse/' % (self.cancer_level, self.patient_id)) is not True:
            os.mkdir('./results/V%sP%s/coarse/' % (self.cancer_level, self.patient_id))
        for i in range(len(self.normalized_slices)):
            if i > len(self.normalized_slices) * 2 / 3:
                break
            slice = self.normalized_slices[i].copy()
            # coeff = pywt.wavedec2(slice, 'haar', level=2)
            # coeff[0] = np.zeros_like(coeff[1][0])
            # rec = pywt.waverec2(coeff, 'haar')
            # rec = np.round(rec * 255)
            # rec[rec < 0] = 0
            # plt.imshow(rec, cmap='gray');plt.show()
            cv.imwrite('./results/V%sP%s/raw/%d.png' % (self.cancer_level, self.patient_id, (i + 1)), slice)
            # if i != 0:
            #     if np.abs(np.max(slice * mask) - I_max_pre) > 50:
            #         break
            # slice[slice < I_min] = I_min
            # slice[slice > I_max] = I_max
            # idx = (slice >= I_min) & (slice <= I_max)
            # slice[idx] = slice[idx] / (I_max - I_min) * I_max
            # plt.imshow(slice, cmap='gray');plt.show()
            # slice = cv.medianBlur(slice, 5)
            # plt.imshow(slice, cmap='gray');plt.show()
            threshold, threshold_slice = cv.threshold(slice, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            _, slice = cv.threshold(slice, threshold, 255, cv.THRESH_TOZERO)
            threshold_slices.append(threshold_slice)
            num_connected, labels, stats, centroids = cv.connectedComponentsWithStats(threshold_slice)
            areas = [stats[j][-1] for j in range(1, stats.shape[0])]
            max_area = np.max(areas)
            max_idx = np.argmax(areas)
            self.fg_slices.append((labels == (max_idx + 1)).astype(np.uint8))
            mask = self.fg_slices[i]
            I_max_pre = np.max(mask * self.normalized_slices[i])
            slice = slice * mask
            cv.imwrite('./results/V%sP%s/coarse/%dIW.png' % (self.cancer_level, self.patient_id, (i + 1)), slice)
            # plt.imshow(fg_slices[i], cmap='gray');plt.show()
            if max_area > self.largest_connected['area']:
                self.largest_connected['frame'] = i + 1
                self.largest_connected['mask'] = labels == (max_idx + 1)
                self.largest_connected['area'] = np.max(areas)
                self.largest_connected['centroid'] = centroids[max_idx + 1, :]
                self.largest_connected['stats'] = stats[max_idx + 1, :]

    def generate_trimap(self):
        self.generate_foreground()
        trimap_file_path = './results/V%sP%s/trimap/' % (self.cancer_level, self.patient_id)
        largest_component = self.normalized_slices[self.largest_connected['frame']] * self.largest_connected['mask']
        threshold, threshold_slice = cv.threshold(largest_component, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, slice = cv.threshold(threshold_slice, threshold, 255, cv.THRESH_TOZERO)
        num_connected, labels, stats, centroids = cv.connectedComponentsWithStats(threshold_slice)
        areas = [stats[j][-1] for j in range(1, stats.shape[0])]
        max_idx = np.argmax(areas)
        self.largest_connected['mask'] = (labels == max_idx + 1)
        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        largest_trimap = np.zeros_like(self.normalized_slices[0])
        mask = (self.largest_connected['mask'] * 255).astype(np.uint8)
        u = cv.dilate(mask, morph_kernel, iterations=10)
        fg = cv.erode(mask, morph_kernel, iterations=10)
        largest_trimap[fg == 255] = 255
        largest_trimap[(u - fg) == 255] = 128
        if os.path.exists(trimap_file_path) is not True:
            os.mkdir(trimap_file_path)
        cv.imwrite(trimap_file_path + '%dIW.png' % (self.largest_connected['frame']),
                   largest_trimap)
        for i in range(len(self.fg_slices)):
            slice = self.normalized_slices[i]
            # if i + 1 == self.largest_connected['frame']:
            #     continue
            trimap = np.zeros_like(slice)
            # fg = self.fg_slices[i]
            fg = cv.erode(self.fg_slices[i], morph_kernel, iterations=10)
            if np.sum(fg) < 100:
                continue
            # u = (largest_trimap == 255) ^ fg
            u = cv.dilate(self.fg_slices[i], morph_kernel, iterations=10)
            trimap[fg == 1] = 255
            trimap[(u - fg) == 1] = 128
            cv.imwrite(trimap_file_path + '%dIW.png' % (i+1), trimap)

    def segment_IW(self):
        trimap_file_path = './results/V%sP%s/trimap/' % (self.cancer_level, self.patient_id)
        result_file_path = './results/V%sP%s/results/' % (self.cancer_level, self.patient_id)
        if os.path.exists(result_file_path) is not True:
            os.mkdir(result_file_path)
        for trimap_file in os.listdir(trimap_file_path):
            trimap = cv.imread(trimap_file_path + trimap_file, 0)
            f_num = int(trimap_file.split('I')[0])

            u = (trimap == 128)
            bbox = cv.boundingRect(u.astype(np.uint8))

            # slice = cv.GaussianBlur(self.normalized_slices[f_num], (3, 3), 0, 0)
            slice = self.normalized_slices[f_num - 1]
            slice_patch = slice[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            trimap_slice = trimap[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

            trimap[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = generate_trimap_local(slice_patch, trimap_slice)

            cv.imwrite(result_file_path + '%dIW.png' % (f_num), trimap)


