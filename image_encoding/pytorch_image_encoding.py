import os, sys
sys.path.append("/nitthilan/source_code/multiview_rendering/neural_sparse_voxel_field/") # Adds higher directory to python modules path.

print(sys.path)
from opt_image import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from datasets import dataset_dict

# models
# from models.nerf import Embedding, NeRF
# from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *
# import utils
# from .. import utils

# losses
# from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

from torch import nn

# Signal makers


class RandomSignal(Dataset):
    def __init__(self, image_path, split='train', img_wh=(512, 512), resize_factor=2):
        img = Image.open(image_path)
        self.define_transforms()
        self.img_wh = img_wh
        self.image_path = image_path
        self.split = split

        img = img.resize(self.img_wh, Image.LANCZOS)
        self.image_base = img
        self.resize_wh = (int(img_wh[0]/resize_factor), int(img_wh[1]/resize_factor))
        self.train_image = img.resize(self.resize_wh, Image.LANCZOS)
        # print("Image dimension ", self.image_base.size, self.train_image.size)

        self.train_data = self.transform(self.train_image) # (4, h, w)
        self.train_data = self.train_data.view(3, -1).permute(1, 0) # (h*w, 4) RGBA

        self.test_data = self.transform(self.image_base) # (4, h, w)
        self.test_data = self.test_data.view(3, -1).permute(1, 0) # (h*w, 4) RGBA


        x_lin = np.linspace(0, 1, num=img_wh[0], endpoint=False)
        y_lin = np.linspace(0, 1, num=img_wh[1], endpoint=False)
        x,y = np.meshgrid(x_lin, y_lin)
        xy_list = np.array([x.flatten(), y.flatten()]).T
        # print(xy_list[:4])
        self.test_input = torch.from_numpy(xy_list).float().to(self.test_data.device)

        x_lin += 1/(resize_factor*img_wh[0])
        y_lin += 1/(resize_factor*img_wh[1])
        x,y = np.meshgrid(x_lin[::2], y_lin[::2])
        xy_list = np.array([x.flatten(), y.flatten()]).T
        # print(xy_list[:4])
        self.train_input = torch.from_numpy(xy_list).float().to(self.test_data.device)

        # print(self.train_data.size(), self.train_input.size())
        # print(self.test_data.size(), self.test_input.size())

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        else:# self.split == 'val':
            return len(self.test_data) # only validate 8 images (to support <=8 gpus)

    def __getitem__(self, idx):
        if self.split == 'train':
            return (self.train_input[idx], self.train_data[idx])
        else:# self.split == 'val':
            return (self.test_input[idx], self.test_data[idx])

input_encoder = lambda x, a, b: torch.cat([x[...,None], a * torch.sin((2.*3.1415*x[...,None]) * b[...,None].T), 
        a * torch.cos((2.*3.1415*x[...,None]) * b[...,None].T)], axis=-1) / torch.norm(a)

class SignalEncoder(LightningModule):
    def __init__(self, num_layers, num_channels, encoding, num_inputs=2, num_outputs=3):
        """
        num_layers: Number of layers
        num_channels: Number of channels in each layer
        num_outputs: The output value to be predicted
        """
        super(SignalEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_outputs = num_outputs

        self.encoding = encoding
        N = 32 # Number of training samples
        bvals = np.float32(np.arange(1, N//2+1))
        if(self.encoding == None):
            avals = None
            num_features = 1
        elif(self.encoding == 1000):
            avals = torch.FloatTensor(np.eye(bvals.shape[0])[0])
            num_features = bvals.shape[0]*2 + 1
        else:
            avals = torch.FloatTensor(bvals**-np.float32(self.encoding))
            num_features = 1 + 2*bvals.shape[0]

        self.num_inputs = num_inputs*num_features

        self.register_buffer('bvals', torch.FloatTensor(bvals))
        self.register_buffer('avals', avals)

        layer_list = []
        for i in range(self.num_layers):
            if(i==0):
                layer = nn.Linear(self.num_inputs, self.num_channels)
            else:
                layer = nn.Linear(self.num_channels, self.num_channels)
            setattr(self, f"linear_layer_{i}", layer)

        setattr(self, f"linear_layer_{i+1}", nn.Linear(self.num_channels, self.num_outputs))

        layer_list.append(nn.Sigmoid())

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        if(self.num_inputs != 2):
            # print(self.avals.size(), self.bvals.size(), x.size(), x[:,0,None].size())
            # output = self.avals * torch.sin((2.*3.1415*x[:,0,None])* self.bvals[...,None].T)
            # print(output.size())
            x1 = input_encoder(x[:,0], self.avals, self.bvals)
            x2 = input_encoder(x[:,1], self.avals, self.bvals)
            x = torch.cat([x1, x2], axis=-1)
            # print("Input dimension ", x.size(), x1.size(), x2.size())

        for i in range(self.num_layers):
            # print("Layer id ", i)
            x = getattr(self, f"linear_layer_{i}")(x)
            x = nn.ReLU(True)(x)
        x = getattr(self, f"linear_layer_{i+1}")(x)
        out = nn.Sigmoid()(x)
        return out





class EncoderSystem(LightningModule):
    def __init__(self, hparams):
        super(EncoderSystem, self).__init__()
        self.hparams = hparams
        self.signal_encoder = SignalEncoder(hparams.num_layers, hparams.num_channels,
            hparams.encoding)
        self.models = [self.signal_encoder]
        self.loss = nn.MSELoss(reduction='mean')
        return

    def forward(self, x):
        # print(x)
        return self.models[0](x)

    def prepare_data(self):
        image_path = "./data/image_embedding/scenery.jpg"
        self.train_dataset = RandomSignal(image_path, split='train')
        self.val_dataset = RandomSignal(image_path, split='val')
        return
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=4*self.hparams.batch_size, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        x, y = batch
        x = x.view(self.hparams.batch_size, -1)
        # print(x.size())
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        with torch.no_grad():
            psnr_ = psnr(y_pred, y)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        # log = {'val_loss': self.loss(results, rgbs, valid_mask)}
        log = {'val_loss': self.loss(y_pred, y)}
        log['val_psnr'] = psnr(y_pred, y)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]


if __name__ == '__main__':
    hparams = get_opts()
    system = EncoderSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='train/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)