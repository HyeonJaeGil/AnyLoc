import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import numpy as np
import os
import glob
import natsort
from tqdm.auto import tqdm
from onedrivedownloader import download
from utilities import od_down_links, DinoV2ExtractFeatures, VLAD
from typing import Literal, Union


class VLADDescriptorGenerator:
    def __init__(self, in_dir: str, out_dir: str, imgs_ext: str = "jpg", max_img_size: int = 1024,
                 domain: Literal["aerial", "indoor", "urban"] = "indoor", num_c: int = 32, first_n: Union[int, None] = None):
        self.in_dir = os.path.realpath(os.path.expanduser(in_dir))
        self.out_dir = os.path.realpath(os.path.expanduser(out_dir))
        self.imgs_ext = imgs_ext
        self.max_img_size = max_img_size
        self.domain = domain
        self.num_c = num_c
        self.first_n = first_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._initialize()

    def _initialize(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", layer=31, facet="value", device=self.device)
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_vlad()

    def _load_vlad(self):
        cache_dir = os.path.realpath("./cache")
        ext_specifier = f"dinov2_vitg14/l31_value_c{self.num_c}"
        c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier, self.domain, "c_centers.pt")
        if not os.path.isfile(c_centers_file):
            raise FileNotFoundError("Vocabulary not cached! Run cache download first.")
        c_centers = torch.load(c_centers_file)
        if c_centers.shape[0] != self.num_c:
            raise ValueError("Wrong number of clusters!")
        self.vlad = VLAD(self.num_c, desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
        self.vlad.fit(None)  # Load the vocabulary

    def _resize_image(self, img_pt: torch.Tensor) -> torch.Tensor:
        c, h, w = img_pt.shape
        if max(h, w) > self.max_img_size:
            scale = self.max_img_size / max(h, w)
            h, w = int(h * scale), int(w * scale)
            img_pt = T.resize(img_pt, (h, w), interpolation=T.InterpolationMode.BICUBIC)
        return img_pt

    def generate_descriptors(self):
        if not os.path.isdir(self.in_dir):
            raise FileNotFoundError("Input directory doesn't exist!")
        
        img_fnames = glob.glob(f"{self.in_dir}/*.{self.imgs_ext}")
        img_fnames = natsort.natsorted(img_fnames)
        if self.first_n is not None:
            img_fnames = img_fnames[:self.first_n]
        
        for img_fname in tqdm(img_fnames, desc="Processing Images"):
            self._process_image(img_fname)

    def _process_image(self, img_fname: str):
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            img_pt = self.base_tf(pil_img).to(self.device)
            img_pt = self._resize_image(img_pt)
            h_new, w_new = (img_pt.shape[1] // 14) * 14, (img_pt.shape[2] // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            
            descriptor = self.extractor(img_pt).cpu().squeeze()
            gd = self.vlad.generate(descriptor)
            gd_np = gd.numpy()[np.newaxis, ...]
            np.save(os.path.join(self.out_dir, f"{os.path.basename(img_fname)}.npy"), gd_np)