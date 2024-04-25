import argparse, datetime, os, sys
import cv2
import torch
import numpy as np
from scipy import ndimage
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import torchvision
from typing import List
sys.path[0] = "/home/yx/code/virtual_tryon/TPD"
print(sys.path)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from torchvision.transforms import Resize


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
    

def crop_image(input_image, target_size):
    width = target_size[0]
    left = (input_image.width - width) // 2
    right = left + width
    cropped_image = input_image.crop((left, 0, right, input_image.height))
    return cropped_image



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def un_norm(x):
    return (x+1.0)/2.0
def un_norm_clip(x):
    x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
    x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
    x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
    return x


def blend(
    dilate_kernel: int,
    blur_kernel: int,
    foreground: np.array,
    background: np.array,
    mask: np.array,
) -> torch.tensor:
    mask = ((1 - mask) * 255).astype(np.uint8)

    blended_images = []
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        if mask_i.ndim > 2:
            mask_i = mask_i[:,:,0]
        mask_i = cv2.dilate(mask_i, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=1)
        mask_blur = cv2.blur(mask_i, (blur_kernel, blur_kernel)).astype(np.float32) / 255
        mask_blur = mask_blur[:,:,None]

        foreground_np = foreground[i]
        background_np = background[i]

        blended_image = (
            mask_blur * foreground_np + (1 - mask_blur) * background_np
        )
        blended_images.append(blended_image)

    blended_images = np.array(blended_images)

    return blended_images




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="batch_size",
    )

    parser.add_argument(
        "--predicted_mask_dilation",
        type=int,
        default=10,
        help="predicted_mask_dilation",
    )


    opt = parser.parse_args()


    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    dataset = instantiate_from_config(config.data.params.test)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)


    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    outpath = os.path.join(opt.outdir, current_time)
    os.makedirs(outpath, exist_ok=True)

    first_stage_path = os.path.join(outpath, "first_stage")
    second_stage_path = os.path.join(outpath, "second_stage")

    first_stage_grid_path = os.path.join(first_stage_path, "grid")
    first_stage_result_path = os.path.join(first_stage_path, "result")
    first_stage_middle_figure_path = os.path.join(first_stage_path, "middle_figure")

    second_stage_grid_path = os.path.join(second_stage_path, "grid")
    second_stage_result_path = os.path.join(second_stage_path, "result")
    second_stage_middle_figure_path = os.path.join(second_stage_path, "middle_figure")


    os.makedirs(first_stage_middle_figure_path, exist_ok=True)
    os.makedirs(first_stage_result_path, exist_ok=True)
    os.makedirs(first_stage_grid_path, exist_ok=True)


    os.makedirs(second_stage_middle_figure_path, exist_ok=True)
    os.makedirs(second_stage_result_path, exist_ok=True)
    os.makedirs(second_stage_grid_path, exist_ok=True)


    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


    iterator = tqdm(loader, desc='test Dataset', total=len(loader))
    precision_scope = autocast if opt.precision == "autocast" else nullcontext


    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for batch in iterator:
                    torch.cuda.empty_cache()
                    
                    image_name = batch["image_name"]

                    person = batch["GT_image"]
                    garment_mask = batch["GT_mask"]
                    bbox_inpaint_person = batch["inpaint_image"]
                    bbox_mask = batch["inpaint_mask"]
                    posemap = batch["posemap"]
                    densepose = batch["densepose"]
                    ref_list = batch["ref_list"] 


                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_image'] = bbox_inpaint_person.to(device)
                    test_model_kwargs['inpaint_mask'] = bbox_mask.to(device)
                    test_model_kwargs['posemap'] = posemap.to(device)
                    test_model_kwargs['densepose'] = densepose.to(device)

                    def VAE_Encode(*image_list):
                        latents = []
                        for image in image_list:
                            encoder_posterior = model.encode_first_stage(image) 
                            z = model.get_first_stage_encoding(encoder_posterior).detach() 
                            latents.append(z)
                        return latents
                    
                    z_inpaint_image, z_posemap, z_densepose = VAE_Encode(test_model_kwargs['inpaint_image'], test_model_kwargs['posemap'], test_model_kwargs['densepose'])
                                
                    z_shape = (z_inpaint_image.shape[-2], z_inpaint_image.shape[-1])
                    z_inpaint_mask = Resize([z_shape[0], z_shape[1]])(test_model_kwargs['inpaint_mask'])

                    test_model_kwargs['inpaint_image'] = z_inpaint_image
                    test_model_kwargs['inpaint_mask'] = z_inpaint_mask
                    test_model_kwargs['posemap'] = z_posemap
                    test_model_kwargs['densepose'] = z_densepose

                    uc = model.learnable_vector
                    uc = uc.repeat(person.size(0), 1, 1)
                    c = uc

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                    # 1st stage
                    z_result_first_stage, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=len(image_name),
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)
                    
                    result_image_first_stage = model.decode_first_stage(z_result_first_stage[:,:4,:,:])
                    result_image_first_stage = torch.clamp((result_image_first_stage + 1.0) / 2.0, min=0.0, max=1.0)
                    result_image_first_stage = result_image_first_stage.cpu()

                    garment_foreground = result_image_first_stage.permute(0, 2, 3, 1).numpy()
                    person_background = torch.clamp((person + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
                    selectRegion = np.repeat(rearrange(bbox_mask.cpu().numpy(), "b c h w -> b h w c"), 3, axis=3)
                    result_image_first_stage_blended = blend(dilate_kernel=30, blur_kernel=30,foreground=garment_foreground, background=person_background, mask=selectRegion)
                    result_image_first_stage_blended = torch.from_numpy(result_image_first_stage_blended).permute(0, 3, 1, 2).cpu()


                    if not opt.skip_save:
                        for i in range(result_image_first_stage_blended.shape[0]):
                            all_img_first_stage=[]
                            all_img_first_stage.append(un_norm(person[i]).cpu())
                            all_img_first_stage.append(un_norm(bbox_inpaint_person[i]).cpu())
                            all_img_first_stage.append(un_norm(posemap[i]).cpu())
                            all_img_first_stage.append(un_norm(densepose[i]).cpu())
                            all_img_first_stage.append(result_image_first_stage[i].cpu())
                            all_img_first_stage.append(result_image_first_stage_blended[i].cpu())
                            grid_first_stage = torch.stack(all_img_first_stage, 0)
                            grid_first_stage = make_grid(grid_first_stage)
                            grid_first_stage = 255. * rearrange(grid_first_stage, 'c h w -> h w c').cpu().numpy()
                            grid_first_stage = Image.fromarray(grid_first_stage.astype(np.uint8))
                            grid_first_stage.save(os.path.join(first_stage_grid_path, image_name[i][:-4]+'_grid.jpg'))


                            result_image_first_stage_blended_numpy = 255. * rearrange(result_image_first_stage_blended[i], 'c h w -> h w c').cpu().numpy()
                            result_image_first_stage_blended_img = Image.fromarray(result_image_first_stage_blended_numpy[:,:(opt.W//2),:].astype(np.uint8))
                            result_image_first_stage_blended_img.save(os.path.join(first_stage_result_path, image_name[i][:-4]+".jpg"))


                            person_numpy=255.*rearrange(un_norm(person[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            person_img = Image.fromarray(person_numpy.astype(np.uint8))
                            person_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_person.jpg"))
                                

                            ref_img=255.*rearrange(un_norm(ref_list[0][i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_garment.jpg"))


                            bbox_inpaint_person_numpy=255.*rearrange(un_norm(bbox_inpaint_person[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            bbox_inpaint_person_img = Image.fromarray(bbox_inpaint_person_numpy.astype(np.uint8))
                            bbox_inpaint_person_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_bbox_inpaint.jpg"))


                            bbox_mask_numpy=255.*rearrange(un_norm(bbox_mask[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            bbox_mask_numpy= cv2.cvtColor(bbox_mask_numpy,cv2.COLOR_GRAY2RGB)
                            bbox_mask_img = Image.fromarray(bbox_mask_numpy.astype(np.uint8))
                            bbox_mask_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_bbox_mask.jpg"))


                            posemap_numpy=255.*rearrange(un_norm(posemap[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            posemap_img = Image.fromarray(posemap_numpy.astype(np.uint8))
                            posemap_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_posemap.jpg"))


                            densepose_numpy=255.*rearrange(un_norm(densepose[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                            densepose_img = Image.fromarray(densepose_numpy.astype(np.uint8))
                            densepose_img.save(os.path.join(first_stage_middle_figure_path, image_name[i][:-4]+"_densepose.jpg"))



                    if (opt.C == 5):
                        torch.cuda.empty_cache()
                        
                        predicted_mask = z_result_first_stage[:,4,:,:]
                        # mask_samples_ddim = torch.clamp((mask_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        predicted_mask = Resize([512, 768])(predicted_mask)
                        predicted_mask = predicted_mask.cpu().numpy()[:,None,:,:]
                        # second_mask = second_mask.astype(np.float32)
                        predicted_mask[predicted_mask < 0.5] = 0
                        predicted_mask[predicted_mask >= 0.5] = 1


                        predicted_mask_before_dilate_tensor = torch.from_numpy(predicted_mask).to(device)
                        predicted_mask_before_dilate_inpaint_tensor = person.to(device) * predicted_mask_before_dilate_tensor

                        
                        for i in range(predicted_mask.shape[0]):
                            # Step 1: Label connected components
                            labels, num = ndimage.label(predicted_mask[i,0,:,:] == 0)

                            # Step 2: Compute the size of each component
                            sizes = np.bincount(labels.ravel())
                            if len(sizes) > 1:
                                # Step 3: Find the two largest components (excluding the background)
                                sorted_sizes = np.argsort(sizes)

                                largest = sorted_sizes[-2]  # Exclude the background

                                # Step 4: Create a new mask where only the largest components are 0
                                predicted_mask[i,0,:,:][labels != largest] = 1

                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            predicted_mask[i,0,:,:] = cv2.erode(predicted_mask[i,0,:,:], kernel, iterations=opt.predicted_mask_dilation)

                        predicted_mask_tensor = torch.from_numpy(predicted_mask).to(device)
                        predicted_mask_inpaint_tensor = person.to(device) * predicted_mask_tensor


                        predicted_mask_unioned_tensor = predicted_mask_tensor * garment_mask.to(device)
                        predicted_mask_unioned_inpaint_tensor = person.to(device)*predicted_mask_unioned_tensor


                        test_model_kwargs['inpaint_mask']=predicted_mask_unioned_tensor.to(device)
                        test_model_kwargs['inpaint_image']=predicted_mask_unioned_inpaint_tensor.to(device)


                        z_inpaint_image_second_stage = VAE_Encode(test_model_kwargs['inpaint_image'])[0]
                                    
                        z_shape_second_stage = (z_inpaint_image_second_stage.shape[-2], z_inpaint_image_second_stage.shape[-1])
                        z_inpaint_mask_second_stage = Resize([z_shape_second_stage[0], z_shape_second_stage[1]])(test_model_kwargs['inpaint_mask'])

                        test_model_kwargs['inpaint_mask']=z_inpaint_mask_second_stage
                        test_model_kwargs['inpaint_image']=z_inpaint_image_second_stage



                        z_result_second_stage, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=len(image_name),
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            test_model_kwargs=test_model_kwargs)

                        result_image_second_stage = model.decode_first_stage(z_result_second_stage[:,:4,:,:])
                        result_image_second_stage = torch.clamp((result_image_second_stage + 1.0) / 2.0, min=0.0, max=1.0)
                        result_image_second_stage = result_image_second_stage.cpu()


                        garment_foreground = result_image_second_stage.permute(0, 2, 3, 1).numpy()
                        person_background = torch.clamp((person + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
                        selectRegion = np.repeat(rearrange(predicted_mask_unioned_tensor.cpu().numpy(), "b c h w -> b h w c"), 3, axis=3)
                        result_image_second_stage_blended = blend(dilate_kernel=30, blur_kernel=30,foreground=garment_foreground, background=person_background, mask=selectRegion)
                        result_image_second_stage_blended = torch.from_numpy(result_image_second_stage_blended).permute(0, 3, 1, 2).cpu()


                        if not opt.skip_save:
                            for i in range(result_image_first_stage_blended.shape[0]):
                                all_img_second_stage=[]
                                all_img_second_stage.append(un_norm(person[i]).cpu())
                                all_img_second_stage.append(un_norm(bbox_inpaint_person[i]).cpu())
                                all_img_second_stage.append(un_norm(predicted_mask_before_dilate_inpaint_tensor[i]).cpu())
                                all_img_second_stage.append(un_norm(predicted_mask_inpaint_tensor[i]).cpu())
                                all_img_second_stage.append(un_norm(predicted_mask_unioned_inpaint_tensor[i]).cpu())
                                all_img_second_stage.append(un_norm(posemap[i].cpu()))
                                all_img_second_stage.append(un_norm(densepose[i].cpu()))
                                all_img_second_stage.append(result_image_first_stage[i].cpu())
                                all_img_second_stage.append(result_image_second_stage[i].cpu())
                                all_img_second_stage.append(result_image_first_stage_blended[i].cpu())
                                all_img_second_stage.append(result_image_second_stage_blended[i].cpu())

                                grid_second_stage = torch.stack(all_img_second_stage, 0)
                                grid_second_stage = make_grid(grid_second_stage)
                                grid_second_stage = 255. * rearrange(grid_second_stage, 'c h w -> h w c').cpu().numpy()
                                grid_second_stage = Image.fromarray(grid_second_stage.astype(np.uint8))
                                grid_second_stage.save(os.path.join(second_stage_grid_path, image_name[i][:-4]+'_grid.jpg'))


                                result_image_second_stage_blended_numpy = 255. * rearrange(result_image_second_stage_blended[i], 'c h w -> h w c').cpu().numpy()
                                result_image_second_stage_blended_img = Image.fromarray(result_image_second_stage_blended_numpy[:,:(opt.W//2),:].astype(np.uint8))
                                result_image_second_stage_blended_img.save(os.path.join(second_stage_result_path, image_name[i][:-4]+".jpg"))




                                person_numpy=255.*rearrange(un_norm(person[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                person_img = Image.fromarray(person_numpy.astype(np.uint8))
                                person_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_person.jpg"))     

                                ref_img=255.*rearrange(un_norm(ref_list[0][i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                ref_img = Image.fromarray(ref_img.astype(np.uint8))
                                ref_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_garment.jpg"))



                                bbox_inpaint_person_numpy=255.*rearrange(un_norm(bbox_inpaint_person[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                bbox_inpaint_person_img = Image.fromarray(bbox_inpaint_person_numpy.astype(np.uint8))
                                bbox_inpaint_person_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_bbox_inpaint.jpg"))

                                bbox_mask_numpy=255.*rearrange(un_norm(bbox_mask[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                bbox_mask_numpy= cv2.cvtColor(bbox_mask_numpy,cv2.COLOR_GRAY2RGB)
                                bbox_mask_img = Image.fromarray(bbox_mask_numpy.astype(np.uint8))
                                bbox_mask_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_bbox_mask.jpg"))

                                predicted_mask_before_dilate_inpaint_numpy = 255. * rearrange(un_norm(predicted_mask_before_dilate_inpaint_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_before_dilate_inpaint_img = Image.fromarray(predicted_mask_before_dilate_inpaint_numpy.astype(np.uint8))
                                predicted_mask_before_dilate_inpaint_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask_before_dilate_inpaint.jpg"))




                                predicted_mask_before_dilate_numpy=255.*rearrange(un_norm(predicted_mask_before_dilate_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_before_dilate_numpy= cv2.cvtColor(predicted_mask_before_dilate_numpy,cv2.COLOR_GRAY2RGB)
                                predicted_mask_before_dilate_img = Image.fromarray(predicted_mask_before_dilate_numpy.astype(np.uint8))
                                predicted_mask_before_dilate_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask_before_dilate.jpg"))
    

                                predicted_mask_inpaint_numpy=255.*rearrange(un_norm(predicted_mask_inpaint_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_inpaint_img = Image.fromarray(predicted_mask_inpaint_numpy.astype(np.uint8))
                                predicted_mask_inpaint_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask_inpaint.jpg"))

                                predicted_mask_numpy=255.*rearrange(un_norm(predicted_mask_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_numpy= cv2.cvtColor(predicted_mask_numpy,cv2.COLOR_GRAY2RGB)
                                predicted_mask_img = Image.fromarray(predicted_mask_numpy.astype(np.uint8))
                                predicted_mask_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask.jpg"))


                                garment_mask_numpy=255.*rearrange(un_norm(garment_mask[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                garment_mask_numpy= cv2.cvtColor(garment_mask_numpy,cv2.COLOR_GRAY2RGB)
                                garment_mask_img = Image.fromarray(garment_mask_numpy.astype(np.uint8))
                                garment_mask_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_garment_mask.jpg"))
                 

                                predicted_mask_unioned_inpaint_numpy=255.*rearrange(un_norm(predicted_mask_unioned_inpaint_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_unioned_inpaint_img = Image.fromarray(predicted_mask_unioned_inpaint_numpy.astype(np.uint8))
                                predicted_mask_unioned_inpaint_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask_unioned_inpaint.jpg"))

                                predicted_mask_unioned_numpy=255.*rearrange(un_norm(predicted_mask_unioned_tensor[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                predicted_mask_unioned_numpy= cv2.cvtColor(predicted_mask_unioned_numpy,cv2.COLOR_GRAY2RGB)
                                predicted_mask_unioned_img = Image.fromarray(predicted_mask_unioned_numpy.astype(np.uint8))
                                predicted_mask_unioned_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_predicted_mask_unioned.jpg"))
                                
                                
  

                                posemap_numpy=255.*rearrange(un_norm(posemap[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                posemap_img = Image.fromarray(posemap_numpy.astype(np.uint8))
                                posemap_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_posemap.jpg"))


                                densepose_numpy=255.*rearrange(un_norm(densepose[i]).cpu(), 'c h w -> h w c').cpu().numpy()
                                densepose_img = Image.fromarray(densepose_numpy.astype(np.uint8))
                                densepose_img.save(os.path.join(second_stage_middle_figure_path, image_name[i][:-4]+"_densepose.jpg"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()