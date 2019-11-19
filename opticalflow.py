import cv2
import numpy as np
import torch


# input is numpy image array
def opticalflow(img1, img2):
    img1 = np.squeeze(np.transpose(img1, (0, 2, 3, 1)))
    img2 = np.squeeze(np.transpose(img2, (0, 2, 3, 1)))
    prvs_r = img1[:, :, 0]
    prvs_g = img1[:, :, 1]
    prvs_b = img1[:, :, 2]

    next_r = img2[:, :, 0]
    next_g = img2[:, :, 1]
    next_b = img2[:, :, 2]

    flow_r = cv2.calcOpticalFlowFarneback(prvs_r, next_r, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_g = cv2.calcOpticalFlowFarneback(prvs_g, next_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_b = cv2.calcOpticalFlowFarneback(prvs_b, next_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    real_flow_r = np.sqrt(np.power(flow_r[..., 0], 2) + np.power(flow_r[..., 1], 2))
    real_flow_g = np.sqrt(np.power(flow_g[..., 0], 2) + np.power(flow_g[..., 1], 2))
    real_flow_b = np.sqrt(np.power(flow_b[..., 0], 2) + np.power(flow_b[..., 1], 2))

    real_flow_rgb = np.asarray([real_flow_r, real_flow_g, real_flow_b])

    real_flow = real_flow_r * 0.299 + real_flow_g * 0.587 + real_flow_b * 0.114

    real_flow = cv2.normalize(real_flow, None, 0, 1, cv2.NORM_MINMAX)
    # real_flow_rgb_save = cv2.normalize(real_flow_rgb_save, None, 0, 255, cv2.NORM_MINMAX)


    rgb = torch.from_numpy(np.expand_dims(real_flow_rgb, 0)).contiguous()
    gray = torch.from_numpy(np.expand_dims(np.expand_dims(real_flow,0),0)).contiguous()
    return rgb, gray