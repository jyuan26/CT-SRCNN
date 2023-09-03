import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, calc_mse, compute_ssim

def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    max_test_item = 20
    filelist = get_list(args.image_file+"/X"+str(args.scale), ext="png")
    filelist = filelist[:max_test_item]
    psnr_list = np.zeros(len(filelist))
    mse_list = np.zeros(len(filelist))
    ssim_list = np.zeros(len(filelist))
    i = 0

    for imname in filelist:

        image = pil_image.open(imname).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        bicublic = imname.replace('2020','output/2020')
        image.save(bicublic.replace('.', '_bicubic_x{}.'.format(args.scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        psnr_list[i] = calc_psnr(y, preds)
        #mse_list[i] = calc_mse(y, preds)
        ssim_y = y.numpy().squeeze(0).squeeze(0)
        ssim_preds = preds.numpy().squeeze(0).squeeze(0)
        ssim_list[i] = compute_ssim(ssim_y, ssim_preds)
        #psnr_list[i] = calc_psnr(ssim_y, ssim_preds)
        mse_list[i] = calc_mse(y, ssim_preds)

        print("i="+str(i)+',PSNR: {:.9f}'.format(psnr_list[i]) +
              ",mse: {:.9f}".format(mse_list[i]) + ",ssie: {:.9f}".format(ssim_list[i]))


        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        #output_dir = imname + "/output/"
        imname = imname.replace('2020','output/2020')
        output.save(imname.replace('.', '_srcnn_x{}.'.format(args.scale)))
        i = i + 1
    psnr = np.mean(psnr_list)
    print("Mean PSNR: " + str(psnr))
    print("Mean MSE: " + str(np.mean(mse_list)))
    print("Mean SSIM: " + str(np.mean(ssim_list)))