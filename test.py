import numpy as np
import torch
import cv2
from spade_e2v import Unet6 as Unet
from benchmark import testset
import argparse
import os.path as osp
import tqdm


def draw(pred, y, d, model_path):

    y = y.repeat(1, 3, 1, 1)
    p = pred[0].detach().cpu().numpy().mean(0)
    y = y[0].detach().cpu().numpy().mean(0)

    cat_img = np.concatenate([p, y], 1).astype(np.float32)
    cat_img = cv2.resize(cat_img, (p.shape[1] * 4, p.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(d[0].split('/')[-1] + '---' + model_path.split('/')[-1].split('.')[0], cat_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        return True


def main(arg):

    device = 'cuda:0'
    torch.cuda.empty_cache()
    ev_rate = 0.35
    root_dir = arg.root_dir

    model_path = osp.join(root_dir, 'models/SPADE_E2VID.pth')
    data_dir = list()

    data_dir.append([osp.join(root_dir, 'dvs_datasets/dynamic_6dof'), 550])
    data_dir.append([osp.join(root_dir, 'dvs_datasets/boxes_6dof'), 550])
    data_dir.append([osp.join(root_dir, 'dvs_datasets/poster_6dof'), 550])
    data_dir.append([osp.join(root_dir, 'dvs_datasets/office_zigzag'), 247])
    data_dir.append([osp.join(root_dir, 'dvs_datasets/slider_depth'), 84])
    data_dir.append([osp.join(root_dir, 'dvs_datasets/calibration'), 550])

    netG = Unet().cuda().half()
    netG.load_state_dict(torch.load(model_path, map_location=device))
    d = data_dir[args.data_n]
    te = testset(d[0], ev_rate, norm_e=True, e_abs=False, num_freams=d[1])
    stats = None

    for f in tqdm.tqdm(range(550)):
        with torch.no_grad():
            x, y = te.getitem(f)
            x = x[:, :, :176].cuda().half()
            y = y[None, None, :176].cuda().half()

            if f == 0:
                pred = x[:, :3]
                pred -= pred.min()
                pred /= pred.max()

            for ii in range(x.shape[0]):

                xx = x[ii][None]
                pred, stats = netG(xx, stats, pred)
                if draw(pred, y, d, model_path):
                    break

    print('Finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/media/rodrigo/ubuntu/e2v_public')
    parser.add_argument('--data_n', type=int, default=0, help='Possible Choose 1 to 6')
    args = parser.parse_args()
    main(args)
