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
#                                SAMPLE                                  #
##########################################################################

def initialize_weight(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv2d') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)

if __name__ == '__main__':
    
    # Parameter
    mu = 0.5
    std = 0.5
    batch_size = 4
    lr = 1e-4
    num_epoch = 500

    lambda_cls = 10
    lambda_rec = 50
    lambda_con = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mu, std)])
    
    train_ds = SAR_Dataset('QPM', transform = transform, train = True)

    path = './Weight/TestGAN'
    os.makedirs(path, exist_ok = True)
    writer = SummaryWriter(path + '/run')

    # img, cls = train_ds[1255]
    # plt.imshow(img.squeeze() * std + mu, cmap = 'gray')
    # plt.axis('off')
    # plt.title(cls)
    # plt.colorbar()
    # plt.show()
    # plt.savefig('test.png')

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle =True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    G.apply(initialize_weight)
    D.apply(initialize_weight)

    gan_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    cls_loss = nn.BCELoss()

    opt_G = optim.Adam(G.parameters(), lr = lr, betas = (0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr = lr, betas = (0.5, 0.999))
    sche_G = optim.lr_scheduler.StepLR(opt_G, step_size = 40, gamma = 0.5)
    sche_D = optim.lr_scheduler.StepLR(opt_D, step_size = 40, gamma = 0.5)

    G.train()
    D.train()

    start_time = time.time()
    iter_count = 0
    print('Start Train')

    for epoch in range(num_epoch):
        for img, cls in train_dl:

            iter_count += 1
            img = img.to(device)
            cls = cls.to(device)
            trg = 1 - cls

            # Discriminator #
            D.zero_grad()

            # Fake Loss
            fake_img = G(img, trg)
            out_dis_fake, _ = D(fake_img.detach()) # out_trg를 stargan에서는 사용하지 않는다.
            fake_label = torch.zeros(out_dis_fake.shape, requires_grad = False).to(device)
            fake_loss = gan_loss(out_dis_fake, fake_label)
            # d_fake_cls_loss = cls_loss(out_trg, trg)

            # Real Loss
            out_dis_real, out_cls = D(img)
            real_label = torch.ones(out_dis_real.shape, requires_grad = False).to(device)
            real_loss = gan_loss(out_dis_real, real_label)
            d_real_cls_loss = cls_loss(out_cls, cls)

            # Total Loss
            d_loss = fake_loss + real_loss + d_real_cls_loss * lambda_cls

            d_loss.backward()
            opt_D.step()

            # Generator #
            G.zero_grad()

            # Original to Target # 여기에 fake_img - img로 l1 loss 추가해보는거 어떨까
            out_dis, out_trg = D(fake_img)
            g_fake_loss = gan_loss(out_dis, torch.ones(out_dis.shape, requires_grad = False).to(device))
            g_fake_cls_loss = cls_loss(out_trg, trg)
            con_loss = l1_loss(fake_img, img)

            # Target to Original
            recon_img = G(fake_img, cls)
            recon_loss = l1_loss(recon_img, img)

            # Total Loss
            g_loss = g_fake_loss + g_fake_cls_loss * lambda_cls + recon_loss * lambda_rec + con_loss * lambda_con

            g_loss.backward()
            opt_G.step()

            # Tensorboard
            writer.add_scalar("G_Loss", g_loss, iter_count)
            writer.add_scalar("D_Loss", d_loss, iter_count)
        
        print('Epoch: {}, G_Loss: {:.2f}, D_Loss: {:.2f}, Time: {:.2f}min'.format(epoch, g_loss, d_loss, (time.time() - start_time) / 60))
        plot_result = make_grid(fake_img * 0.5 + 0.5, nrow = 2, padding = 20, pad_value = 0.5)
        writer.add_image("Generated Image", plot_result, iter_count)

        sche_G.step()
        sche_D.step()
    writer.close()

    # Save Weight (*.pt)
    os.makedirs(path, exist_ok = True)
    path2pt = os.path.join(path, 'weights_gen.pt')
    torch.save(G.state_dict(), path2pt)