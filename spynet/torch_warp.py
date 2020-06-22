import pickle
import numpy as np
import gzip
import torch
import cv2
import math
import random
import matplotlib.pyplot as plt


def get_warp_label(flow1, flow2, label1): #H,W,C  h*w*2
    flow_shape = flow1.shape
    label_shape = label1.shape
    height, width = flow1.shape[0], flow1.shape[1]

    label2 = torch.zeros(label_shape)   #label2 = np.zeros_like(label1, dtype=label1.dtype)
    flow_t = torch.zeros(flow_shape)    #flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    #grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    h_grid = torch.arange(0, height)
    w_grid = torch.arange(0, width)
    h_grid = h_grid.repeat(width, 1).permute(1,0) #.unsqueeze(0)
    w_grid = w_grid.repeat(height,1)              #.unsqueeze(0)
    grid = torch.stack((h_grid,w_grid),0).permute(1,2,0) #float3
    #grid = torch.cat([h_grid, w_grid],0).permute(1,2,0)

    dx = grid[:, :, 0] + flow2[:, :, 1].long()
    dy = grid[:, :, 1] + flow2[:, :, 0].long()
    sx = torch.floor(dx.float()) #float32 #sx = np.floor(dx).astype(int)
    sy = torch.floor(dy.float())

    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1) #H* W 512 x 512 uint8

    # sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    # sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    # sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
    #                   (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    sx_mat = torch.stack((sx, sx + 1, sx, sx + 1), dim=2).clamp(0, height - 1)  #torch.float32
    sy_mat = torch.stack((sy, sy, sy + 1, sy + 1), dim=2).clamp(0, width - 1)
    sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1, 2, 0))) * (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1, 2, 0))))

    for i in range(4):
        flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1,2,0) * flow1.long()[sx_mat.long()[:, :, i], sy_mat.long()[:, :, i], :]

    valid = valid & (torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(),dy.float()),dim=2) - grid.float(), dim=2, keepdim=True).squeeze(2) < 100)

    flow_t = (flow2.float() - flow_t.float()) / 2.0
    dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
    dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)

    label2[valid, :] = label1.float()[dx[valid].long(), dy[valid].long(), :]

    return label2 #HW3


def warp_torch(flow1, flow2, img):  # H,W,C  h*w*2

    flow_shape = flow1.shape
    label_shape = img.shape
    height, width = flow1.shape[0], flow1.shape[1]

    output = torch.zeros(label_shape)  # output = np.zeros_like(img, dtype=img.dtype)
    flow_t = torch.zeros(flow_shape)  # flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    # grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    h_grid = torch.arange(0, height)
    w_grid = torch.arange(0, width)
    h_grid = h_grid.repeat(width, 1).permute(1, 0)  # .unsqueeze(0)
    w_grid = w_grid.repeat(height, 1)  # .unsqueeze(0)
    grid = torch.stack((h_grid, w_grid), 0).permute(1, 2, 0)  # float3
    # grid = torch.cat([h_grid, w_grid],0).permute(1,2,0)

    dx = grid[:, :, 0] + flow2[:, :, 1].long()
    dy = grid[:, :, 1] + flow2[:, :, 0].long()
    sx = torch.floor(dx.float())  # float32 #sx = np.floor(dx).astype(int)
    sy = torch.floor(dy.float())

    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)  # H* W 512 x 512 uint8

    # sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    # sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    # sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
    #                   (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    sx_mat = torch.stack((sx, sx + 1, sx, sx + 1), dim=2).clamp(0, height - 1)  # torch.float32
    sy_mat = torch.stack((sy, sy, sy + 1, sy + 1), dim=2).clamp(0, width - 1)
    sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1, 2, 0))) *
                         (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1, 2, 0))))

    for i in range(4):
        flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1, 2, 0) * flow1.long()[
                                                                                          sx_mat.long()[:, :, i],
                                                                                          sy_mat.long()[:, :, i], :]

    valid = valid & (
                torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(), dy.float()), dim=2) - grid.float(),
                           dim=2, keepdim=True).squeeze(2) < 100)

    flow_t = (flow2.float() - flow_t.float()) / 2.0
    dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
    dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)

    output[valid, :] = img.float()[dx[valid].long(), dy[valid].long(), :]

    return output  # HW3

def warp(flow1, flow2, img):

    output = np.zeros_like(img, dtype=img.dtype)
    height = flow1.shape[0]
    width = flow1.shape[1]
    flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow2[:, :, 1]
    dy = grid[:, :, 1] + flow2[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
                      (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    for i in range(4):
        flow_t = flow_t + sxsy_mat[:, :, i][:, :, np.
                                            newaxis] * flow1[sx_mat[:, :, i],
                                                             sy_mat[:, :, i], :]

    valid = valid & (np.linalg.norm(
        flow_t[:, :, [1, 0]] + np.dstack((dx, dy)) - grid, axis=2) < 100)

    flow_t = (flow2 - flow_t) / 2.0
    dx = grid[:, :, 0] + flow_t[:, :, 1]
    dy = grid[:, :, 1] + flow_t[:, :, 0]

    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)
    output[valid, :] = img[dx[valid].round().astype(int), dy[valid].round()
                              .astype(int), :]
    return output


def main():
    root_dir = '/media/rodrigo/ubuntu/temporal_loss_with_optical_flow'
    prev = cv2.imread(root_dir + '/image/org/0000.png')
    cur = cv2.imread(root_dir + '/image/org/0001.png')
    h, w = prev.shape[:2]
    flow1 = pickle.loads(gzip.GzipFile(root_dir + '/image/flowpkl/forward_0_1.pkl', 'rb').read())  # 'forward_0_5.pkl'
    flow2 = pickle.loads(gzip.GzipFile(root_dir + '/image/flowpkl/backward_1_0.pkl', 'rb').read())  # 'backward_5_0.pkl'

    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(flow1[:, :, 0] / 255.0)
    ax[0, 1].imshow(flow1[:, :, 1] / 255.0)

    ax[1, 0].imshow(flow2[:, :, 0] / 255.0)
    ax[1, 1].imshow(flow2[:, :, 1] / 255.0)

    print("read flow and image")
    print(flow1.dtype)
    print(prev.shape)

    tf1 = torch.from_numpy(flow1)
    tf2 = torch.from_numpy(flow2)
    tcur = torch.from_numpy(cur)
    tprev = torch.from_numpy(prev)

    w0 = warp_torch(tf1, tf2, tprev).numpy()  # 0->1
    w1 = warp_torch(tf2, tf1, tcur).numpy()  # 1->0
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(w0[:, :, ::-1]/255.0)
    ax[1].imshow(w1[:, :, ::-1]/255.0)

    fx = 0.3  # fx * 1280(shape[1])
    fy = 0.6  # fy * 720(shape[0])
    prev_resize = cv2.resize(prev, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    cur_resize = cv2.resize(cur, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(prev_resize[:, :, ::-1] / 255.0)
    ax[1].imshow(cur_resize[:, :, ::-1] / 255.0)

    flow1_resize = cv2.resize(flow1, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    flow2_resize = cv2.resize(flow2, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    rcur = torch.from_numpy(cur_resize)
    rprev = torch.from_numpy(prev_resize)
    rf1 = torch.from_numpy(flow1_resize).float()
    rf2 = torch.from_numpy(flow2_resize).float()
    rf1[:, :, 1] *= fy
    rf2[:, :, 1] *= fy
    rf1[:, :, 0] *= fx
    rf2[:, :, 0] *= fx

    r0 = get_warp_label(rf1, rf2, rprev).numpy()  # 0->1
    r1 = get_warp_label(rf2, rf1, rcur).numpy()  # 1->0

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(r0[:, :, ::-1] / 255.0)
    ax[1].imshow(r1[:, :, ::-1] / 255.0)

    print(prev_resize.shape)

    print('Finish')

if __name__ == '__main__':
    main()

