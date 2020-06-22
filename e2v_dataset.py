import torch
import torch.utils.data as data
import cv2
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import kornia
import itertools
from kornia.geometry.transform import imgwarp
import torch.nn.functional as F
from math import fabs


class DataSet(data.Dataset):
    def __init__(self, path, train, seq_len, abs_e=True, crop_x=256, crop_y=256, img_ch=1, norm_e=False):

        self.train = train
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.num_bins = 5
        self.abs_e = abs_e
        self.norm_e = norm_e
        self.img_ch = img_ch
        self.w = 240
        self.h = 180
        self.flip_p = 0.5
        self.ang = 10
        self.num_evs = int(0.35*self.w * self.h)
        self.t, self.x, self.y, self.p = 2, 3, 4, 1
        self.seq_len = seq_len
        self.evs_len = seq_len * self.num_evs

        if train:
            trainpath = Path(path)
            fname = [itm for itm in os.scandir(trainpath) if itm.is_file()]

            files = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.csv']
            self.files = [list(g) for _, g in itertools.groupby(sorted(files), lambda x: x[0:-9])]
            self.imgs = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.jpg']

        else:
            self.json_path = Path(path) / "event"
            self.img_path = Path(path) / "img"
            self.fnames = [o.name for o in os.scandir(self.json_path) if o.is_file()]
            self.fnames = [o for o in self.fnames if (self.img_path / (o.split('.')[0] + '.jpg')).is_file()]
            self.fnames = [o.split('.')[0] for o in self.fnames if int(o.split('_')[-1].split('.')[0]) > 2]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):

        fname = self.files[item]
        iname = [l for l in self.imgs if l.startswith(fname[0][:-9])]
        iname.sort()

        evs_stream = torch.zeros(0, 4).float()
        events = torch.zeros([self.seq_len, self.num_bins, self.crop_x, self.crop_y]).float()

        randx = np.random.randint(0, self.w - self.crop_x)
        randy = np.random.randint(0, self.h - self.crop_y)
        flipud = np.random.uniform(0, 1) > self.flip_p
        fliprl = np.random.uniform(0, 1) > self.flip_p
        angle = np.random.uniform(-self.ang, self.ang)
        args_ = [randx, randy, flipud, fliprl, angle]
        img_t = self.fix_time([int(itm[-16:-4]) for itm in iname])

        for i, fn in enumerate(fname):
            ev_ten = pd.read_csv(fn)
            ev_ten = torch.from_numpy(ev_ten.values[:, [self.t, self.x, self.y, self.p]]).float()
            evs_stream = torch.cat([evs_stream, ev_ten], 0)
            if evs_stream.shape[0] > self.evs_len:
                break

        evs_stream[:, 0] = torch.from_numpy(self.fix_time(evs_stream[:, 0].numpy()))
        evs_stream = evs_stream[-self.evs_len:]
        if self.abs_e:
            evs_stream[:, 3] = 1

        im_idx = np.searchsorted(img_t, evs_stream[-1, 0].item(), side="rigth")
        img = cv2.imread(iname[im_idx - 1], cv2.IMREAD_GRAYSCALE) / 255.0
        img = torch.from_numpy(img[None]).float()
        imgs = self.transform_koria(img, *args_)

        for i, ev_ten in enumerate(np.split(evs_stream, self.seq_len)):
            ev_ten = self.ev2grid(ev_ten, num_bins=self.num_bins, width=self.w, height=self.h)
            events[i] = self.transform_koria(ev_ten, *args_)
            if self.norm_e:
                events[i] = self.norm(events[i])

        if self.img_ch == 3:
            imgs = imgs.repeat(self.img_ch, 1, 1)

        return events, imgs

    def fix_time(self, vect):
        ref = np.ones(len(vect) - 1)
        y_hat = np.diff(vect) / ref
        starts = np.where(y_hat < 0)[0]
        vect = np.asarray(vect)
        for i in range(len(starts)):
            vect[starts[i]+1:] += vect[starts[i]]
        return vect

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def closest_element_to(self, values, req_value):
        """Returns the tuple (i, values[i], diff) such that i is the closest value to req_value,
        and diff = |values(i) - req_value|
        Note: this function assumes that values is a sorted array!"""
        assert (len(values) > 0)

        i = np.searchsorted(values, req_value, side='left')
        if i > 0 and (i == len(values) or fabs(req_value - values[i - 1]) < fabs(req_value - values[i])):
            idx = i - 1
            val = values[i - 1]
        else:
            idx = i
            val = values[i]

        diff = fabs(val - req_value)
        return (idx, val, diff)

    def ev2grid(self, events, num_bins, width, height):
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
                                  index=xs[valid_indices] + ys[valid_indices]
                                            * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                            + (tis_long[valid_indices] + 1) * width * height,
                                   source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(num_bins, height, width)

        return voxel_grid

    def transform(self, evs, img, randx, randy, flipud, fliprl, angle):
        img = img.transpose([1, 2, 0])  # channels last
        evs = evs.transpose([1, 2, 0])  # channels last
        if fliprl:
            evs = cv2.flip(evs, 1)
            img = cv2.flip(img, 1)
        if flipud:
            evs = cv2.flip(evs, 0)
            img = cv2.flip(img, 0)

        center = (img.shape[0] // 2, img.shape[1] // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        evs = cv2.warpAffine(evs, M, (evs.shape[1], evs.shape[0]))

        evs = evs.transpose([2, 0, 1])  # channels first
        img = img.transpose([2, 0, 1])  # channels first

        evs = evs[:, randy: randy + self.crop_size, randx: randx + self.crop_size]
        img = img[:, randy: randy + self.crop_size, randx: randx + self.crop_size]

        return evs, img

    def transform_koria(self, tensor, randx, randy, flipud, fliprl, angle):

        if flipud:
            tensor = torch.flip(tensor, dims=(0, 1))
        if fliprl:
            tensor = torch.flip(tensor, dims=(0, 2))

        # tensor = kornia.rotate(tensor, angle=angle, center=(tensor.shape[3], tensor.shape[2])
        center = torch.ones(1, 2)
        center[..., 0] = tensor.shape[2] / 2  # x
        center[..., 1] = tensor.shape[1] / 2  # y
        scale = torch.ones(1)
        angle = torch.ones(1) * angle

        M = kornia.get_rotation_matrix2d(center, angle, scale)
        tensor = kornia.warp_affine(tensor[None], M, dsize=(self.h, self.w))[0]

        tensor = tensor[:, randy: randy + self.crop_y, randx: randx + self.crop_x]

        return tensor

    def norm(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)

        return events


class testset:
    def __init__(self, root_dir, ev_rate, norm_e):

        self.norm_e = norm_e
        self.bs = 1
        self.h = 180
        self.w = 240
        self.img_num = 0
        self.bins = 5
        self.root_dir = root_dir + '/'
        self.events_file = 'events.txt'
        self.img_file = 'images.txt'
        self.num_events = int(ev_rate * self.h * self.w)
        self.iterator = pd.read_csv(self.root_dir + self.events_file, header=None,
                                    delimiter=' ',
                                    names=['t', 'x', 'y', 'p'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16},
                                    engine='c',
                                    index_col=False,
                                    skiprows=0, chunksize=self.num_events, nrows=None, memory_map=True)

        self.img_metadata = pd.read_csv(self.root_dir + self.img_file,
                                        delimiter=' ',
                                        header=None, names=['t', 'fname'],
                                        index_col=False)

    def __iter__(self):
        return self

    def __next__(self):

        args = [self.bins, self.w, self.h]  # num_bins, width, height
        event_tensor = self.iterator.__next__().values
        idx = np.searchsorted(self.img_metadata.t.values, event_tensor[-1, 0], side="rigth")
        img_name = self.root_dir + self.img_metadata.fname[idx-1]

        with torch.no_grad():
            event_tensor = torch.from_numpy(event_tensor)
            event_tensor[:, 3][event_tensor[:, 3] == -1] = 1
            event_tensor[:, 3][event_tensor[:, 3] == 0] = 1
            event_tensor = self.events_to_voxel_grid_pytorch(event_tensor, *args)
            if self.norm_e:
                event_tensor = self.norm(event_tensor)
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) / 255.0
            img = torch.from_numpy(img).float()

        return event_tensor, img

    def norm(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)
        return events

    def events_to_voxel_grid_pytorch(self, events, num_bins, width, height):
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
            events_torch = events

            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height,
                source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices] + ys[valid_indices] * width + (tis_long[valid_indices] + 1) * width * height,
                source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(num_bins, height, width)

        return voxel_grid
