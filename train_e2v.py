import torch
from torch.utils.data import DataLoader
from e2v_utils import LossFn, lr_schedule2
from e2v_dataset import DataSet, testset
from spade_e2v import Unet6 as Unet
import numpy as np
from skimage.measure import compare_mse, compare_ssim
import time
import argparse
import os.path as osp


def dataset(args):

    trainpath = osp.join(args.root_dir, 'coco_dvs2_train')
    trainpath = '/media/rodrigo/Samsumg/coco_dvs2_train'
    tr = DataSet(trainpath, train=True, seq_len=args.seq_len, abs_e=args.abs_e,
                 norm_e=args.norm_e, crop_x=128, crop_y=128, img_ch=3)

    tr_loder = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=12)

    return tr_loder


def main(args):

    torch.cuda.empty_cache()
    device = 'cuda:0'
    ssim_save = 0
    tr = dataset(args)
    v_len = args.epochs * len(tr)
    lossfn = LossFn(to_cuda=device)
    netG = Unet().to(device)

    netG = netG.train()
    lossG, lpips_loss, test_lpips, ssim_e, test_ssim, mse_e, test_mse = [], [], [], [], [], [], []

    tr_param = netG.parameters()

    lr_sch = lr_schedule2(max_v=args.lr, min_v=args.lr * 0.1, len_v=v_len, cicle=1)
    optimizerG = torch.optim.Adam(tr_param, args.lr)

    for e in range(args.epochs):
        for i, (x, y) in enumerate(tr):
            x = x.to(device)
            with torch.no_grad():
                pred = x[:, 0, :3].detach().to(device)
            y = y.to(device)

            seq_len = x.shape[1]
            stats = None
            optimizerG.zero_grad()

            for ii in range(seq_len):
                pred, stats = netG(x[:, ii], stats, pred)
            lossg, lpips = lossfn.loss(pred, y)
            lossg.backward()
            optimizerG.step()

            step_num = (e * len(tr)) + i
            for param_group in optimizerG.param_groups:
                step_num = (e * len(tr)) + i
                param_group['lr'] = lr_sch[step_num]
                # param_group['momentum'] = m_sch[step_num]

            lossG.append(lossg.item())
            lpips_loss.append(lpips.item())

            pred = pred[0].mean(0).detach().cpu().numpy()
            y = y[0].mean(0).detach().cpu().numpy()

            ssim_e.append(compare_ssim(pred, y, dynamic_range=1, multichannel=False))
            mse_e.append(compare_mse(pred, y))

            if (i + 1) % 50 == 0:
                netG = netG.eval()
                with torch.no_grad():
                    stats = None
                    ttt = []
                    testpath = osp.join(args.root_dir, 'dvs_datasets/slider_depth')
                    ev_rate = 0.35
                    te = testset(testpath, ev_rate, args.norm_e)
                    for iii, (x, y) in enumerate(te):
                        x = x[None, :, :176].to(device)
                        if iii == 0:
                            pred = x[:, :3].detach().to(device)
                        y = y[None, None, :176].to(device)
                        tic = time.time()
                        pred, stats = netG(x, stats, pred)
                        ttt.append(time.time() - tic)
                        _, lpips = lossfn.loss(pred, y.repeat(1, 3, 1, 1))
                        p = pred[0].mean(0).detach().cpu().numpy()
                        y = y[0, 0].detach().cpu().numpy()

                        test_ssim.append(compare_ssim(p, y, data_range=1, multichannel=False))
                        test_mse.append(compare_mse(p, y))
                        test_lpips.append(lpips.item())
                netG = netG.train()

                print(f'Epoch: {e:2}, iter {i + 1:6}, '
                      f'step: {step_num + 1:06} / {v_len:06}, '
                      f'lossG mean: {np.mean(lossG[-100:]):3.4f}, '
                      f'Lpips: {np.mean(lpips_loss[-100:]):3.4f}, '
                      f'SISSM: {np.mean(ssim_e[-100:]):3.4f}, '
                      f'MSE error: {np.mean(mse_e[-100:]):3.4f}, '
                      f'test lpips: {np.mean(test_lpips[-100:]):3.4f}, '
                      f'test ssim: {np.mean(test_ssim[-100:]):3.4f}, '
                      f'test MSE: {np.mean(test_mse[-100:]):3.4f}, '
                      f'test time: {np.mean(ttt):3.4f}, ')
                if np.mean(test_ssim[-100:]) > ssim_save:
                    torch.save(
                        netG.state_dict(),
                        osp.join(args.root_dir, 'models/SPADE_E2VID_best.pth'))
                    ssim_save = np.mean(test_ssim[-100:])

    torch.save(
        netG.state_dict(),
        osp.join(args.root_dir, 'models/SPADE_E2VID_full.pth'))
    print('Finish')


if __name__ == '__main__':
    # bs, epochs, lr, ssim_save, seq_len, abs_e, norm_e = 1, 170, 1e-4, 0, 15, False, True
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        type=str,
                        default='/media/rodrigo/ubuntu/e2v_public',
                        help='Path to dir')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=170, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=15, help='Sequence length')
    parser.add_argument('--abs_e', type=bool, default=False, help='Use non-polarity format')
    parser.add_argument('--norm_e', type=bool, default=True, help='Normalize events')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()
    main(args)
