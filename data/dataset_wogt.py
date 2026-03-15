import random
import torch.utils.data as data
import utils.utils_image as util

class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()
        print('Dataset: IVF')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])

    def __getitem__(self, index):

        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, 1)
        img_B = util.imread_uint(B_path, self.n_channels)

        if self.n_channels == 3:
            img_B = util.RGB_to_Y(img_B)

        if self.opt['phase'] == 'train':

            H, W, _ = img_A.shape

            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_A = img_A[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_B = img_B[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]

            mode = random.randint(0,7)

            patch_A, patch_B = util.augment_img(patch_A, mode=mode), util.augment_img(patch_B, mode=mode)
            img_A = util.uint2tensor3(patch_A)
            img_B = util.uint2tensor3(patch_B)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

        else:

            img_A = util.uint2single(img_A)
            img_B = util.uint2single(img_B)

            img_A = util.single2tensor3(img_A)
            img_B = util.single2tensor3(img_B)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)