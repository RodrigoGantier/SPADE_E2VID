import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from scipy.stats import t
import torchvision
from spynet import run as spynet
import math
import torch.nn.functional as F
from collections import deque
import numpy as np


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class LossFn:
    def __init__(self, to_cuda):
        self.vgg = VGG19().to(to_cuda).eval()
        self.criterion = nn.L1Loss(size_average=True)
        self.criterionMSE = nn.MSELoss()
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.flownet = spynet.Network().to(to_cuda).eval()
        self.alpha = 50

    def tem_loss(self, pred0, pred1, img0, img1):
        with torch.no_grad():
            flow = self.flownet(img1.detach(), img0.detach())  # Backward optical flow
            img0_warp = spynet.backwarp(img0.detach(), flow)
            pred0_warp = spynet.backwarp(pred0, flow)
            noc_mask = torch.exp(-self.alpha * torch.sum(img1.detach() - img0_warp, dim=1).pow(2)).unsqueeze(1)

        temp_loss = self.criterion(pred1 * noc_mask, pred0_warp * noc_mask)

        return temp_loss

    def loss3(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss

        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.criterion(pred, y)
        # -------features and style loss------
        y = (y * 2) - 1
        pred = (pred * 2) - 1
        features_loss, style_loss = self.featStyleLoss(pred, y.detach())

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def loss(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss

        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.criterion(pred, y)
        # -------features and style loss------
        y = (y * 2) - 1
        pred = (pred * 2) - 1
        features_loss, style_loss = self.featStyleLoss(pred, y)

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def loss2(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss
        with torch.no_grad():
            y = (y * 2) - 1
        # -------features and style loss------
        features_loss, style_loss = self.featStyleLoss(pred, y.detach())
        # -------SSIM loss------
        y = (y + 1) / 2
        pred = (pred + 1) / 2
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.criterion(pred, y)

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (h * w)
        return gram

    def featStyleLoss(self, pred, y):
        x_vgg, y_vgg = self.vgg(pred), self.vgg(y.detach())
        f_loss = 0
        s_loss = 0
        for i in range(len(x_vgg)):
            f_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            s_loss += self.weights[i] * self.criterion(self.gram_matrix(x_vgg[i]),
                                                       self.gram_matrix(y_vgg[i].detach()))
        return f_loss, s_loss

    def pixelwise(self, pred, y):
        bs = pred.shape[0]
        pixel_loss = torch.pow(pred.view(bs, -1) - y.view(bs, -1), 2)
        pixel_loss = pixel_loss.mean(1)
        # pixel_loss = torch.sqrt(pixel_loss)
        pixel_loss = pixel_loss.mean()
        return pixel_loss

    def warping(self, x, flo):

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flo

        ## scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = torch.nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.ones(x.size()).cuda()
        mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=False)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


def events_to_voxel_grid_pytorch(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].long()
        ys = events[:, 2].long()
        pols = events[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=torch.clamp(xs[valid_indices] + ys[valid_indices] * width + (tis_long[valid_indices] + 1) * width * height, min=0, max=voxel_grid.shape[0]-1),
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def hotpix_torch(evs):
        nonzero_ev = (evs != 0)
        num_nonzeros = nonzero_ev.sum()
        mean = evs.sum() / num_nonzeros
        stddev = torch.sqrt((evs ** 2).sum() / num_nonzeros - mean ** 2)
        h = stddev * t.ppf((1 + 0.99) / 2, num_nonzeros.item() - 1)
        clip_v = mean + h
        clip_mat = evs >= clip_v
        clip_mat = clip_mat.float()
        evs -= clip_mat * evs
        return evs


def norm(events):
    with torch.no_grad():
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            events = mask * (events - mean) / (stddev + 1e-8)

    return events


def detach_tensor(h):
    """detach tensors from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """
    def __init__(self):
        self.auto_hdr = True
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = 10
        self.Imin = 0
        self.Imax = 1

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:

            Imin = torch.min(img).item()
            Imax = torch.max(img).item()
            # ensure that the range is at least 0.1
            Imin = np.clip(Imin, 0.0, 0.45)
            Imax = np.clip(Imax, 0.55, 1.0)
            # adjust image dynamic range (i.e. its contrast)
            if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                self.intensity_bounds.popleft()
            self.intensity_bounds.append((Imin, Imax))
            self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
            self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img.byte()  # convert to 8-bit tensor
        return img


def lr_schedule2(max_v, min_v, len_v, cicle=1, invert=False):
    flen = len_v
    len_v /= cicle

    lr_sch0 = np.cos(np.arange(np.pi, 2*np.pi, np.pi / (len_v * 0.1)))
    lr_sch1 = np.cos(np.arange(0, np.pi, np.pi / (len_v * 0.9)))

    lr_sch0 = (lr_sch0 + 1) / 2
    lr_sch0 = (max_v - min_v) * lr_sch0
    lr_sch0 += min_v

    lr_sch1 = (lr_sch1 + 1) / 2
    lr_sch1 = lr_sch1 * max_v

    lr_sch = np.concatenate([lr_sch0, lr_sch1])

    if invert:
        lr_sch = np.cos(lr_sch)

    lr_sch = np.tile(lr_sch, [cicle])
    return lr_sch[:flen]


def lr_schedule(max_v, min_v, len_v, cicle=1, invert=False):
    flen = len_v
    len_v /= cicle
    if invert:
        lr_sch = np.cos(np.arange(np.pi, 2*np.pi, np.pi / len_v))
    else:
        lr_sch = np.cos(np.arange(0, np.pi, np.pi / len_v))
    lr_sch = (lr_sch + 1) / 2
    lr_sch = (max_v - min_v) * lr_sch
    lr_sch += min_v
    lr_sch = np.tile(lr_sch, [cicle])
    return lr_sch[:flen]


def lr_finder(tr, model, criterion):

    lr_init = 1e-8
    lr_fin = 1
    max_step = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)
    loss_sig = []
    lr_sch = lr_schedule(max_v=lr_fin, min_v=lr_init, len_v=max_step, invert=True)
    mo_sch = lr_schedule(max_v=0.9, min_v=0.1, len_v=max_step, invert=False)
    plt.ion()
    _, ax = plt.subplots(1)
    for i, data in enumerate(tr):
        x = data[0].cuda()
        y = data[1].cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion.loss(y_hat, y)
        loss.backward()
        loss_sig.append(loss.item())
        optimizer.step()

        ax.cla()
        ax.plot(loss_sig)
        plt.pause(0.001)
        plt.show()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_sch[i]
            param_group['momentum'] = mo_sch[i]
        if i == max_step-1:
            break