import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import numpy as np  # noqa: F401
import copy
import time
import torch
import torch.nn as nn
import logging
import argparse
from quant.solver import imagenet_utils
from quant.solver.recon import reconstruction
from quant.solver.fold_bn import search_fold_and_remove_bn, StraightThrough
from quant.model import load_model, specials
from quant.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all,enable_shift_value_update,disable_shift_value_update
from quant.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from quant.quantization.fake_quant import QuantizeBase
from quant.quantization.observer import ObserverBase
logger = logging.getLogger('quant')
logging.basicConfig(level=logging.INFO, format='%(message)s')

import gc
from DiT.models import DiT_models
from DiT.download import find_model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from DiT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import random

from tqdm import tqdm
from PIL import Image
import math
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while(st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            # print(name)
            if "adaLN_modulation" in name:
                # print("skip",name)
                st += 1
                continue
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    'the image input has already been in 256, set the last layer\'s input to 8-bit'
    a_list[-1].set_bit(8)
    logger.info('finish quantize model:\n{}'.format(str(model)))
    return model


def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]

def get_train_samples_DiT_cali(n, st, sample_data, custom_steps=None):
    num_samples, num_st = n, st
    if num_st == 1:
        xs = sample_data["xs"][249][:num_samples]
        xs=torch.tensor(xs)
        ts = sample_data["ts"][249][:num_samples]
        ts=torch.tensor(ts)
        conds = sample_data["cs"][249][:num_samples]
        conds=torch.tensor(conds)
        cfg_scale=sample_data["cfg_scale"]
    else:
        nsteps = len(sample_data["ts"])
        timesteps = list(range(0, nsteps, nsteps//num_st))
        print(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
        xs_lst=[]
        ts_lst=[]
        sample_number_list = random.sample(range(2048), num_samples)
        for n in sample_number_list:
            xs_lst1 = [sample_data["xs"][i][n] for i in timesteps]
            xs1 = torch.stack(xs_lst1, dim=0)
            xs_lst.append(xs1)

            ts_lst1 = [sample_data["ts"][i][n] for i in timesteps]
            ts1 = torch.stack(ts_lst1, dim=0)
            ts_lst.append(ts1)

        if True:
            conds_lst=[]
            for n in sample_number_list:
                conds_lst1 = [sample_data["cs"][i][n] for i in timesteps]
                conds1 = torch.stack(conds_lst1, dim=0)
                conds_lst.append(conds1)
            cfg_scale=sample_data["cfg_scale"]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)

        if True:
            conds = torch.cat(conds_lst,dim=0)
            return xs, ts, conds,cfg_scale 
    return xs, ts, conds,cfg_scale


def main(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = imagenet_utils.parse_config(config_path)
    imagenet_utils.set_seed(config.process.seed)
    torch.manual_seed(config.process.seed)
    'cali data'
    sample_data = torch.load(config.quant.recon.cali_path)
    cali_data = get_train_samples_DiT_cali(config.quant.recon.cali_n, config.quant.recon.cali_st, sample_data)
    del(sample_data)
    gc.collect()
    print(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape} {cali_data[3]}")

    'model'
    latent_size = config.DiT.image_size // 8
    model = DiT_models[config.DiT.model](
        input_size=latent_size,
        num_classes=config.DiT.num_classes
    ).to(device)
    state_dict = find_model(config.DiT.ckpt_path)
    model.load_state_dict(state_dict)

    if hasattr(config, 'quant'):
        model.blocks = quantize_model(model.blocks, config.quant)
    model.to(device)
    model.eval()
    fp_model = copy.deepcopy(model)  #TODO:?
    disable_all(fp_model)
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    # calibrate first
    with torch.no_grad():
        disable_shift_value_update(model)
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        model.forward(cali_data[0][: 256].to(device),cali_data[1][: 256].to(device),cali_data[2][: 256].to(device))
        mid = time.time()
        logger.info('the act calibration time is {}'.format(mid - st))
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model.forward(cali_data[0][: 2].to(device),cali_data[1][: 2].to(device),cali_data[2][: 2].to(device))
        ed = time.time()
        logger.info('the weight calibration time is {}'.format(ed - mid))
    enable_shift_value_update(model)
    if hasattr(config.quant, 'recon'):
        enable_quantization(model)
        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, config.quant.recon)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        # Start reconstruction
        recon_model(model, fp_model)
    enable_quantization(model)
    disable_shift_value_update(model)  
    diffusion = create_diffusion(str(config.DiT.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/data1/wangmeiqi/lxy/data/ema").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n_test = len(class_labels)
    z_test = torch.randn(n_test, 4, latent_size, latent_size, device=device)
    y_test = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z_test = torch.cat([z_test, z_test], 0)
    y_null_test = torch.tensor([1000] * n_test, device=device)
    y_test = torch.cat([y_test, y_null_test], 0)
    model_kwargs_test = dict(y=y_test, cfg_scale=config.DiT.cfg_scale_test)
    logger.info('finish quantize model:\n{}'.format(str(model)))
    # Sample images:
    samples_test = diffusion.p_sample_loop(
        model.forward_with_cfg, z_test.shape, z_test, clip_denoised=False, model_kwargs=model_kwargs_test, progress=True, device=device
    )
    samples_test, _ = samples_test.chunk(2, dim=0)  # Remove null class samples
    samples_test = vae.decode(samples_test / 0.18215).sample

    # Save and display images:
    save_image(samples_test, "sample_test.png", nrow=4, normalize=True, value_range=(-1, 1)) 
    del(model_kwargs_test)
    del(n_test)
    del(z_test)
    del(y_test)
    del(samples_test)
    gc.collect()   
    model_string_name = "qaunt_DiT"
    ckpt_string_name = "without3"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{config.DiT.image_size}-" \
                  f"cfg-{config.DiT.cfg_scale}-seed-{config.process.seed}"
    sample_folder_dir = f"{config.DiT.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = config.DiT.per_proc_batch_size
    global_batch_size = n 
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(config.DiT.num_fid_samples / global_batch_size) * global_batch_size)
    print(f"Total number of images that will be sampled: {total_samples}")
    samples_needed_this_gpu = int(total_samples)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar)
    total = 0
    logger.info('finish quantize model:\n{}'.format(str(model)))
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, config.DiT.num_classes, (n,), device=device)

        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=config.DiT.cfg_scale)
        sample_fn = model.forward_with_cfg
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    create_npz_from_sample_folder(sample_folder_dir, config.DiT.num_fid_samples)
    print("Done.") 
     
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='quant/solver/config_DiT.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
