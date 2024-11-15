from dataset_gan_v2 import SAR_Dataset, get_loader
from network import Generator, Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import os
import time
import matplotlib.pyplot as plt

##########################################################################
#                                CelebA                                  #
##########################################################################

def create_label(cls_org, cls_dim = 5, selected_attrs = None):
    
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    cls_trg_list = []
    for i in range(cls_dim):
        cls_trg = cls_org
        if i in hair_color_indices:
            cls_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    cls_trg[:, j] = 0
        else:
            cls_trg[:, i] = (cls_trg[:, i] == 0)
    
        cls_trg_list.append(cls_trg.to(device))

    return cls_trg_list

def gradient_penalty(img, img_fake, D):
    
    alpha = torch.rand(img.size(0), 1, 1, 1).to(device)
    img_hat = (alpha * img + (1 - alpha)* img_fake).requires_grad_(True)
    out_src, _ = D(img_hat)
    
    weight = torch.ones(out_src.size()).to(device)
    dydx = torch.autograd.grad(outputs = out_src,
                               inputs = img_hat,
                               grad_outputs = weight,
                               retain_graph = True,
                               create_graph = True,
                               only_inputs = True)[0]
    
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim = 1))
    return torch.mean((dydx_l2norm - 1)**2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Parameters
    path = './Weight/StarGAN/StarGAN_Test_WGAN_5iter'
    celeba_image_dir = './Model/StarGAN/data/celeba/images'
    attr_path = './Model/StarGAN/data/celeba/list_attr_celeba.txt'
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young', 'Eyeglasses']
    lr = 1e-4
    num_epoch = 50
    lambda_cls = 10
    lambda_gp = 10
    lambda_rec = 10

    os.makedirs(path, exist_ok = True)
    writer = SummaryWriter(path + '/run')

    # DataLoader
    celeba_loader = get_loader(celeba_image_dir, attr_path, selected_attrs)

    G = Generator(in_channel = (3 + len(selected_attrs)), out_channel = 3).to(device)
    D = Discriminator(WGAN = True).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr = lr, betas = [0.5, 0.999])
    opt_D = torch.optim.Adam(D.parameters(), lr = lr, betas = [0.5, 0.999])
    sche_G = optim.lr_scheduler.StepLR(opt_G, step_size = 40, gamma = 0.5)
    sche_D = optim.lr_scheduler.StepLR(opt_D, step_size = 40, gamma = 0.5)

    gan_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    cls_loss = nn.BCEWithLogitsLoss()

    data_iter = iter(celeba_loader)
    img_fixed, cls_org = next(data_iter)
    img_fixed = img_fixed.to(device)
    c_fixed_list = create_label(cls_org, cls_dim = 6, selected_attrs = selected_attrs)

    G.train()
    D.train()

    print('Train Start')
    start_time = time.time()
    iter_count = 0

    for epoch in range(num_epoch):
        for img, cls in celeba_loader:

            iter_count += 1
            rand_idx = torch.randperm(cls.size(0))
            trg = cls[rand_idx]
            label_cls = cls.clone()
            label_trg = trg.clone()

            img = img.to(device)
            cls = cls.to(device)
            trg = trg.to(device)
            label_cls = label_cls.to(device)
            label_trg = label_trg.to(device)
            
            # Discriminator #
            D.zero_grad()

            # Real Images
            out_src, out_cls = D(img)
            # d_loss_real = gan_loss(out_src, torch.ones(out_src.shape, requires_grad = False).to(device))
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = cls_loss(out_cls, label_cls)

            # Fake Images
            img_fake = G(img, trg)
            out_src, out_cls = D(img_fake)
            # d_loss_fake = gan_loss(out_src, torch.zeros(out_src.shape, requires_grad = False).to(device))
            d_loss_fake = torch.mean(out_src)

            d_loss_gp = gradient_penalty(img, img_fake, D)

            d_loss = d_loss_real + d_loss_fake + d_loss_cls * lambda_cls + d_loss_gp * lambda_gp
            d_loss.backward()
            opt_D.step()

            if iter_count % 5 == 0:
                # Generator #
                G.zero_grad()

                # Original to Target
                img_fake = G(img, trg)
                out_src, out_cls = D(img_fake)
                # g_loss_fake = gan_loss(out_src, torch.ones(out_src.shape, requires_grad = False).to(device))
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = cls_loss(out_cls, label_trg)

                # Target to Original
                img_recon = G(img_fake, cls)
                g_loss_rec = l1_loss(img, img_recon)

                g_loss = g_loss_fake + g_loss_rec * lambda_rec + g_loss_cls * lambda_cls
                g_loss.backward()
                opt_G.step()

                # Tensorboard
                writer.add_scalar("G_Loss", g_loss, iter_count)
            writer.add_scalar("D_Loss", d_loss, iter_count)

            if (iter_count + 1) % 100 == 0:
                print('Epoch: {}, G_Loss: {:.2f}, D_Loss: {:.2f}, Time: {:.2f}min'.format(epoch, g_loss, d_loss, (time.time() - start_time) / 60))

        with torch.no_grad():
            img_fake_list = [img_fixed]
            for c_fixed in c_fixed_list:
                img_fake_list.append(G(img_fixed, c_fixed))
            img_concat = torch.cat(img_fake_list, dim = 3)
            plot_result = make_grid(img_concat * 0.5 + 0.5, nrow = 2, padding = 20, pad_value = 0.5)
            writer.add_image("Generated Image", plot_result, epoch)

        sche_G.step()
        sche_D.step()
    writer.close()

    # Save Weight (*.pt)
    os.makedirs(path, exist_ok = True)
    path2pt = os.path.join(path, 'weights_gen_5iter.pt')
    torch.save(G.state_dict(), path2pt)