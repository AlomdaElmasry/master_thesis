import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import utils.paths
import argparse
import thesis.data
import torch.utils.data
import models.rrdb_net
import models.adversarial_discriminators
import torch.optim

parser = argparse.ArgumentParser(description='Visualize samples from the dataset')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--device', default='cpu', help='Name of the device to use')

args = parser.parse_args()

# Get meta
gts_meta = utils.paths.DatasetPaths.get_items('got-10k', args.data_path, 'train', return_masks=False)
masks_meta = utils.paths.DatasetPaths.get_items('youtube-vos', args.data_path, 'train', return_gts=False)

# Create ContentProvider objects
gts_dataset = thesis.data.ContentProvider(args.data_path, gts_meta, None)
masks_dataset = thesis.data.ContentProvider(args.data_path, masks_meta, None)

# Create MaskedSequenceDataset object
dataset = thesis.data.MaskedSequenceDataset(
    gts_dataset=gts_dataset,
    masks_dataset=masks_dataset,
    gts_simulator=None,
    masks_simulator=None,
    image_size=[256, 256],
    frames_n=3,
    frames_spacing=2,
    frames_randomize=False,
    dilatation_filter_size=(3, 3),
    dilatation_iterations=4,
    force_resize=False,
    keep_ratio=True
)

# Created Loader object
loader = iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True))

# Create the models
netG = models.rrdb_net.RRDBNet(in_nc=4, out_nc=3)
netD = models.adversarial_discriminators.AdversarialDiscriminator(nc=3)

# Create the optimizers
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Iterate over the data
for it_data in loader:
    (x, m), y, info = it_data
    x = x.transpose(1, 2).reshape(-1, 3, 256, 256).to(args.device)
    m = m.transpose(1, 2).reshape(-1, 1, 256, 256).to(args.device)
    y = y.transpose(1, 2).reshape(-1, 3, 256, 256).to(args.device)

    # Prepare NN input
    real_input = y
    fake_input = torch.cat((x, m), dim=1)

    # Resize the data to be 64x64
    real_input = F.interpolate(real_input, (64, 64))
    fake_input = F.interpolate(fake_input, (64, 64))

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    netD.zero_grad()
    label = torch.full((real_input.size(0),), 1, device=args.device)

    # Forward pass real batch through D (the output should be close to 1)
    output = netD(real_input).view(-1)

    # Calculate loss on all-real batch
    errD_real = criterion(output, label)

    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    fake = netG(fake_input)
    label.fill_(0)

    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)

    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)

    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake

    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(1)  # fake labels are real for generator cost

    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)

    # Calculate G's loss based on this output
    errG = criterion(output, label)

    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()

    # Update G
    optimizerG.step()

    # Print iteration results
    print('errG: {} | errD: {}'.format(D_G_z1, D_G_z2))