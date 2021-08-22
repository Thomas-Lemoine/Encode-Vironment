import os
import cv2
from PIL import Image
import torch

DATA_PATH = 'data'
IMAGES_PATH = 'static'

def transform_img_to_numpy(file_name):
    path = os.path.join(IMAGES_PATH, file_name)
    numpy_img = cv2.imread(path, 0)
    return numpy_img

def get_latent_repr(file_name, model):
    numpy_img = transform_img_to_numpy(file_name)
    tensor_img = torch.from_numpy(numpy_img)
    latent, _ = model.encoder(tensor_img)
    latent_np = latent.detatch().numpy().squeeze()
    return latent, latent_np

def get_recon_img(latent, model):
    recon_img = model.decoder(latent)
    recon_img_np = recon_img.detach().numpy().squeeze()

    recon_img = Image.fromnumpy(recon_img_np)
    return recon_img

# bring the image (png, jpg, jpeg) through the model and get the reconstructed image, 
# and the latent representation of the image

if __name__ == "__main__":
    # get_latent_repr
    pass