# src/embed.py  —— transformers 版 CLIP 适配器
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder:
    """
    统一接口：
      - encode_images(List[PIL.Image]) -> (N, D) float32 torch.Tensor（已L2归一化）
      - encode_text(str)              -> (D,)   float32 torch.Tensor（已L2归一化）
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True,).to(self.device).eval()
        self.proc  = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, pil_images, batch_size: int = 16):
        embs = []
        for i in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[i:i+batch_size]
            inputs = self.proc(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
            e = self.model.get_image_features(**inputs)                # (B, D)
            e = torch.nn.functional.normalize(e, dim=-1)               # L2 norm
            embs.append(e)
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def encode_text(self, text: str):
        inputs = self.proc(text=[text], return_tensors="pt", padding=True).to(self.device)
        e = self.model.get_text_features(**inputs).squeeze(0)          # (D,)
        e = torch.nn.functional.normalize(e, dim=-1)                   # L2 norm
        return e
