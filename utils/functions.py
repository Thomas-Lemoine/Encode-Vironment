import os
import cv2
from settings import *

def transform_img_to_numpy(file_name):
    path = os.path.join(UPLOAD_PATH, file_name)
    numpy_img = cv2.imread(path, 0)
    return numpy_img

def get_latent_repr(file_name, model):
    numpy_img = transform_img_to_numpy(file_name)
    latent, _ = model.encoder(numpy_img)
    latent_np = latent.detatch().numpy().squeeze()
    return latent, latent_np

def get_recon_img(latent, model):
    recon_img = model.decoder(latent)
    recon_img_np = recon_img.detach().numpy().squeeze()
    return recon_img_np

# bring the image (png, jpg, jpeg) through the model and get the reconstructed image, 
# and the latent representation of the image