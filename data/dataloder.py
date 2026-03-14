import torch.utils.data as data
import utils.utils_image as util

class Dataset(data.Dataset):

    def __init__(self, root_ir, root_vi, vi_chans):
        super(Dataset, self).__init__()

        self.paths_ir = util.get_image_paths(root_ir)
        self.paths_vi = util.get_image_paths(root_vi)
        self.vi_chans = vi_chans

    def __getitem__(self, index):

        ir_path = self.paths_ir[index]
        vi_path = self.paths_vi[index]

        img_ir = util.imread_uint(ir_path, 1)
        img_vi = util.imread_uint(vi_path, self.vi_chans)

        img_ir = util.uint2single(img_ir)
        img_vi = util.uint2single(img_vi)

        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_ir = util.single2tensor3(img_ir)
        img_vi = util.single2tensor3(img_vi)

        return {'ir': img_ir, 'vi': img_vi, 'ir_path': ir_path, 'vi_path': vi_path}

    def __len__(self):
        return len(self.paths_ir)