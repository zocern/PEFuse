import argparse
import os
import time
import sys
import torch
from tqdm import tqdm
from model.PEFuse import PEFuse as Net
from data.dataloder import Dataset
from torch.utils.data import DataLoader
from utils.utils_image import YCbCr2RGB, RGB2YCbCr, tensor2uint, imsave

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str, default='./weight/infrared_visible_fusion/model/')
    parser.add_argument('--iter_number', type=str, default='10000')
    parser.add_argument('--root_path', type=str, default='./dataset/test/', help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='MSRS', help='input test image name')
    parser.add_argument('--ir_dir', type=str, default='ir', help='input test image name')
    parser.add_argument('--vi_dir', type=str, default='vi', help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--vi_chans', type=int, default=3, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    if os.path.exists(model_path):
        print(f'Loading EMA model from {model_path} ...')
    else:
        print(f'Target model path {model_path} does not exist')
        sys.exit()

    model = define_model(args)
    model.eval()
    model = model.to(device)

    save_dir, window_size = setup(args)
    ir_dir = os.path.join(args.root_path, args.dataset, args.ir_dir)
    vi_dir = os.path.join(args.root_path, args.dataset, args.vi_dir)
    print(f"IR testing directory: {ir_dir}")
    os.makedirs(save_dir, exist_ok=True)

    test_set = Dataset(ir_dir, vi_dir, args.vi_chans)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    test_bar = tqdm(test_loader)
    for test_data in test_bar:
        imgname = test_data['ir_path'][0]
        img_ir = test_data['ir'].to(device)
        img_vi = test_data['vi'].to(device)
        start = time.time()

        with torch.no_grad():
            _, _, h_old, w_old = img_ir.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_ir = torch.cat([img_ir, torch.flip(img_ir, [2])], 2)[:, :, :h_old + h_pad, :]
            img_ir = torch.cat([img_ir, torch.flip(img_ir, [3])], 3)[:, :, :, :w_old + w_pad]
            img_vi = torch.cat([img_vi, torch.flip(img_vi, [2])], 2)[:, :, :h_old + h_pad, :]
            img_vi = torch.cat([img_vi, torch.flip(img_vi, [3])], 3)[:, :, :, :w_old + w_pad]

            if args.vi_chans == 3:
                vi_Y, vi_Cr, vi_Cb = RGB2YCbCr(img_vi)
                vi_Y = vi_Y.to(device)
                vi_Cb = vi_Cb.to(device)
                vi_Cr = vi_Cr.to(device)

                output = test(img_ir, vi_Y, model, args, window_size)
                output = torch.cat((output, vi_Cb, vi_Cr), 1)
                output = YCbCr2RGB(output)
            else:
                output = test(img_ir, img_vi, model, args, window_size)

            output = output[..., :h_old * args.scale, :w_old * args.scale]
            output = output.detach()[0].float().cpu()

        end = time.time()
        test_time = end - start
        test_bar.set_description(
            'Fusion {:s} Successfully! Processing time is {:.4f} s'.format(os.path.basename(imgname), test_time))

        output = tensor2uint(output)
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        imsave(output, save_name)


def define_model(args):
    model = Net(upscale=args.scale, img_size=128, window_size=8, img_range=1., embed_dim=60, mlp_ratio=2)
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    return model

def setup(args):
    save_dir = f'result/PEFuse_{args.dataset}'
    window_size = 8
    return save_dir, window_size

def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        output = model(img_a, img_b)
    else:
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch_a = img_a[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                in_patch_b = img_b[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

                out_patch = model(in_patch_a, in_patch_b)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()