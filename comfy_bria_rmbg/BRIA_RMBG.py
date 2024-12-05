import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from comfy.model_downloader import get_or_download, add_model_folder_path, add_known_models, KnownDownloadables
from comfy.model_downloader_types import HuggingFile
from comfy.model_management import unet_offload_device, get_torch_device, load_models_gpu
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file
from .briarmbg import BriaRMBG

FOLDER_NAME = "bria_rmbg_models"
KNOWN_BRIA_MODELS = KnownDownloadables(data=[HuggingFile("briaai/RMBG-1.4", "model.safetensors")],
                                       folder_name=FOLDER_NAME)

add_model_folder_path(FOLDER_NAME)
add_known_models(FOLDER_NAME, KNOWN_BRIA_MODELS)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BRIA_RMBG_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ("RMBGMODEL",)
    RETURN_NAMES = ("rmbgmodel",)
    FUNCTION = "load_model"
    CATEGORY = "🧹BRIA RMBG"

    def load_model(self):
        model_path = get_or_download(FOLDER_NAME, "model.safetensors", KNOWN_BRIA_MODELS)
        model_weights = load_torch_file(model_path, safe_load=True, device=unet_offload_device())

        net = BriaRMBG()
        net.load_state_dict(state_dict=model_weights)
        net.eval()

        wrapped = ModelPatcher(net, get_torch_device(), unet_offload_device())
        return wrapped,


class BRIA_RMBG_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rmbgmodel": ("RMBGMODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "remove_background"
    CATEGORY = "🧹BRIA RMBG"

    def remove_background(self, rmbgmodel: ModelPatcher, image):
        load_models_gpu([rmbgmodel])
        rmbgmodel_wrapped = rmbgmodel
        rmbgmodel: torch.nn.Module = rmbgmodel.model
        processed_images = []
        processed_masks = []

        for image in image:
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = torch.unsqueeze(im_tensor, 0)
            im_tensor = torch.divide(im_tensor, 255.0)
            im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            im_tensor = im_tensor.to(dtype=rmbgmodel_wrapped.model_dtype(), device=rmbgmodel_wrapped.current_device)

            result = rmbgmodel(im_tensor)
            result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result - mi) / (ma - mi)
            im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)
            del im_tensor

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "BRIA_RMBG_ModelLoader_Zho": BRIA_RMBG_ModelLoader_Zho,
    "BRIA_RMBG_Zho": BRIA_RMBG_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BRIA_RMBG_ModelLoader_Zho": "🧹BRIA_RMBG Model Loader",
    "BRIA_RMBG_Zho": "🧹BRIA RMBG",
}
