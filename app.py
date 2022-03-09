# Credits to https://github.com/ZHKKKe/MODNet for the model.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageColor
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
from st_style import apply

# apply(st)

MODEL = "./models/modnet_photographic_portrait_matting.ckpt"


def change_background(image, matte, background_alpha: float=1.0, background_hex: str="#000000"):
    """ 
    image: PIL Image (RGBA)
    matte: PIL Image (grayscale, if 255 it is foreground)
    background_alpha: float
    background_hex: string
    """
    img = deepcopy(image)
    if image.mode != "RGBA":
        img = img.convert("RGBA")
        
    background_color = ImageColor.getrgb(background_hex)
    background_alpha = int(255 * background_alpha)
    background = Image.new("RGBA", img.size, color=background_color + (background_alpha,))
    background.paste(img, mask=matte)
    return background


def matte(image):
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(MODEL)
    else:
        weights = torch.load(MODEL, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # read image
    im = deepcopy(image)

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return Image.fromarray(((matte * 255).astype('uint8')), mode='L')


if __name__ == "__main__":
    st.title("AI Photo Background Removal")
    st.image(Image.open("assets/demo.png"))
    st.write(
        """
        Tidak punya waktu untuk edit foto? Dalam hitungan detik, 
        aplikasi ini akan menggunakan AI untuk menghapus latar belakang foto selfie (humanoid) yang diupload.
        Silahkan **upload foto**, kemudian klik tombol **"Hapus Background"**. Developed by **BIT student @ BINUS Bekasi**. 
        """
    )
    
    uploaded_img = st.file_uploader(label="Upload foto disini", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    
    with st.expander("Foto asli", expanded=True):
        if uploaded_img is not None:
            st.image(uploaded_img)
        else:
            st.warning("Kamu belum upload foto")
    
    in_mode = st.selectbox("Pilih warna background", ["Transparan (PNG)", "Putih", "Hitam", "Hijau"])
    in_submit = st.button("Hapus Background")

    if uploaded_img is not None and in_submit:
        up_time = str(int(time.time()))
        path_out = "./output/" + up_time + ".png"
        uploaded_name = os.path.splitext(uploaded_img.name)[0]
        pil_uploaded = Image.open(uploaded_img)
        
        with st.spinner("AI sedang menghapus background. Mohon ditunggu..."):
            hexmap = {
                "Transparan (PNG)": "#000000",
                "Putih": "#FFFFFF",
                "Hitam": "#000000",
                "Hijau": "#05EE55"
            }
            alpha = 0.0 if in_mode == "Transparan (PNG)" else 1.0
            pil_matte = matte(pil_uploaded)
            pil_bgedit = change_background(pil_uploaded, pil_matte, background_alpha=alpha, background_hex=hexmap[in_mode])
            pil_bgedit.save(path_out)
        
        with st.expander("Berhasil hapus background!", expanded=True):
            st.image(pil_bgedit)
            
            with open(path_out, "rb") as fs:
                st.download_button(
                    label="Download",
                    data=fs,
                    file_name=f'edited_{uploaded_name}.png',
                    mime='image/png',
                )