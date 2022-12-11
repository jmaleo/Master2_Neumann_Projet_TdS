import torch
import os
import random
import sys
sys.path.append('/home/arnal/Documents/Master2/Traitement du signal/Neumann/')
import torch.nn as nn
from torchvision import transforms

import operators.blurs as blurs
from operators.operator import OperatorPlusNoise
from utils.celeba_dataloader import CelebaTrainingDatasetSubset, CelebaTestDataset, CelebaEvalDataset
from networks.u_net import UnetModel
from solvers.neumann import NeumannNet
from training import standard_training

#Importation for saving tensor to rgb imgs.
from utils.testing_utils import save_batch_as_color_imgs 
from utils.testing_utils import save_tensor_as_color_img

batch_size = 1
n_channels = 3
initial_eta = 0.0
kernel_size = 5
noise_sigma = 0.01

data_out_location = "evaluations/blur_data/"

# Determine if we use celeba or not.
celeba = False
# Determine nb iteration we want.
nb_ite = 1
# Determine if we want to add blur to the image or not.
bluring = False
# If puting on None, the image will be random.
num_image = 0

if (celeba) :
    #For CELEBA : 
    data_location = "data/img_align_celeba/"
    save_location = "result/gaussianblur_neumann_2.ckpt"
    what = "celeba"

else : 
    #For Thyroid
    data_location = "data/Thyroid/"
    save_location = "result/gaussianblur_neumann.ckpt"
    what = "thyroid"


gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!", flush=True)

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# Set up data and dataloaders
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

dataset = CelebaEvalDataset(data_location, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
)

### Set up solver and problem setting
forward_operator = blurs.GaussianBlur(sigma=5.0, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

internal_forward_operator = blurs.GaussianBlur(sigma=5.0, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)

# standard u-net
learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32)
solver = NeumannNet(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                    eta_initial_val=initial_eta)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

cpu_only = not torch.cuda.is_available()


if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')

    solver.load_state_dict(saved_dict['solver_state_dict'])


# set up loss and train
lossfunction = torch.nn.MSELoss()

# Do evaluation

pickup = random.randint(0, len(dataloader)) if num_image == None else num_image
print ("We've taken the {}'s image.".format(pickup))
sample_batch = None
for ii, s_b in enumerate(dataloader) :
    if ii == pickup :
        sample_batch = s_b
        break

bl = "blur" if bluring else "no_blur"

save_batch_as_color_imgs(sample_batch, 1, pickup, data_out_location, ["dry_"+what+"_"+bl+"_"+str(nb_ite)+"_ite"])
sample_batch = sample_batch.to(device=device)


y = None

if (bluring) :
    y = measurement_process(sample_batch)
    save_batch_as_color_imgs(y, 1, pickup, data_out_location, ["blur_"+what+"_"+bl+"_"+str(nb_ite)+"_ite"])
else :
    y = sample_batch

initial_point = y


reconstruction = solver(initial_point, iterations=nb_ite)

reconstruction = torch.clamp(reconstruction, -1, 1)
save_batch_as_color_imgs(reconstruction, 1, pickup, data_out_location, ["recon_"+what+"_"+bl+"_"+str(nb_ite)+"_ite"])

