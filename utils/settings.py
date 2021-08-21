UPLOAD_PATH = 'uploads'

# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 4   # latent_dims = 10 for non-variational auto-encoder
num_epochs = 20
batch_size = 64
capacity = 64
learning_rate = 0.0008  # 0.01
variational_beta = 1
use_gpu = True