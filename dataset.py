from pathlib import Path
import os
import itertools
import shutil
import tqdm


def main():
    path = '/media/rodrigo/Samsumg/coco_dvs2_train'
    dist = '/media/rodrigo/Samsumg/evs_2'

    trainpath = Path(path)
    fname = [itm for itm in os.scandir(trainpath) if itm.is_file()]

    files = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.csv']
    files = [list(g) for _, g in itertools.groupby(sorted(files), lambda x: x[0:-9])][:30]
    imgs = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.jpg']

    ffiles = []

    for f in tqdm.tqdm(files):
        for ff in f:
            shutil.copy2(ff, dist)
        ffiles.append(Path(ff[:-9]).name)

    for i in tqdm.tqdm(imgs):
        if '_'.join(Path(i).name.split('_')[:3]):
            shutil.copy2(i, dist)

    print('stop')


if __name__ == '__main__':
    main()