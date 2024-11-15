import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from dataset_gan_v2 import SAR_Dataset
from network import Generator
import os

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    
    test_ds = SAR_Dataset('QPM', train = False, transform = transform)
    test_dl = DataLoader(test_ds, batch_size = 32, shuffle = False)

    G = Generator().to(device)

    path2model = './Weight/TestGAN/weights_gen.pt'
    path2save = './Data/SAR Data/SAMPLE/TestGAN'

    weight = torch.load(path2model)
    G.load_state_dict(weight)
    G.eval()

    with torch.no_grad():
        for img, cls, label, name in test_dl:
            trg = cls
            fake_real = G(img.to(device), trg.to(device)).detach().cpu()

            for ii, fake_img in enumerate(fake_real):
                path = os.path.join(path2save, label[ii])
                os.makedirs(path, exist_ok = True)
                save_image(0.5 * img + 0.5, os.path.join(path, name[ii]))
                # save_image(0.5 * fake_img + 0.5, os.path.join(path,'fake'+ name[ii]))