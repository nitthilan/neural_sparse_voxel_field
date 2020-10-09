import os, sys
sys.path.append("/nitthilan/source_code/multiview_rendering/neural_sparse_voxel_field/") # Adds higher directory to python modules path.

print(sys.path)
from opt import get_opts
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
from utils.metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

from torch import nn

# Signal makers


class RandomSignal(Dataset):
    def __init__(self, split='train'):
        N_train = 32
        data_power = 1
        # Signal
        M = 8
        N = N_train
        x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
        x_train = x_test[::M]

        test_mask = np.ones(len(x_test), np.bool)
        test_mask[np.arange(0,x_test.shape[0],M)] = 0

        s = self.sample_random_powerlaw(N*M, data_power) 
        s = (s-s.min()) / (s.max()-s.min()) - .5
        self.split = split
        self.define_transforms()
        self.x_test = torch.FloatTensor(x_test)
        self.x_train = torch.FloatTensor(x_train)
        self.y_train = torch.FloatTensor(s[::M])
        self.y_test = torch.FloatTensor(s)
        self.define_transforms()

    def sample_random_signal(self, decay_vec):
      key = np.random.seed(0)
      N = decay_vec.shape[0]
      raw = np.random.normal(size=[N, 2]) @ np.array([1, 1j])
      signal_f = raw * decay_vec
      signal = np.real(np.fft.ifft(signal_f))
      return signal

    def sample_random_powerlaw(self, N, power):
      coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
      decay_vec = coords ** -power
      decay_vec = np.array(decay_vec)
      decay_vec[N//4:] = 0
      return self.sample_random_signal(decay_vec)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            # print("Total length ", len(self.y_train))
            return len(self.y_train)
        else:# self.split == 'val':
            return len(self.y_test) # only validate 8 images (to support <=8 gpus)

    def __getitem__(self, idx):
        if self.split == 'train':
            return (self.x_train[idx], self.y_train[idx])
        else:# self.split == 'val':
            return (self.x_test[idx], self.y_test[idx])

input_encoder = lambda x, a, b: torch.cat([x.unsqueeze(0), 
        a * torch.sin((2.*3.1415*x[...,None]) * b), 
        a * torch.cos((2.*3.1415*x[...,None]) * b)], axis=-1) / torch.norm(a)

class EncodingStyle1(nn.Module):
    def __init__(self, N_freqs=16, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(EncodingStyle1, self).__init__()
        self.N_freqs = N_freqs
        # self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        # self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        self.scale_factor = len(self.funcs)*N_freqs+1

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def get_scale_factor(self):
        return self.scale_factor

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs: x: (B, self.in_channels)
        Outputs: out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class EncodingStyle2(nn.Module):
    def __init__(self, N_freqs=16, encoding=None):
        """
        Defines a function that embeds x to (x, a*sin(2*pi*b*x), a*cos(2*pi*b*x), ...)
        """
        super(EncodingStyle2, self).__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        self.encoding = encoding
        bvals = np.float32(np.arange(1, N_freqs+1))
        if(self.encoding == None):
            avals = None
            self.scale_factor = 0
        elif(self.encoding == 1000):
            avals = torch.FloatTensor(np.eye(bvals.shape[0])[0])
            self.scale_factor = 2+1
        else:
            avals = torch.FloatTensor(bvals**-np.float32(self.encoding))
            self.scale_factor = 2*bvals.shape[0] + 1
        self.register_buffer('bvals', torch.FloatTensor(bvals))
        self.register_buffer('avals', avals)

    def get_scale_factor(self):
        return self.scale_factor

    def forward(self, x):
        """
        Inputs: x: (B, self.in_channels)
        Outputs: out: (B, self.out_channels=self.scale_factor*self.in_channels)
        """
        out = [x]
        for a,b in zip(self.avals, self.bvals):
            for func in self.funcs:
                out += [a*func(2*3.1415*b*x)]
        return torch.cat(out, -1)

class SignalEncoder(LightningModule):
    def __init__(self, num_layers, num_channels, num_inputs=1, num_outputs=1):
        """
        num_layers: Number of layers
        num_channels: Number of channels in each layer
        num_outputs: The output value to be predicted
        """
        super(SignalEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs

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
        """
        Encodes input signal x ranging between 0-1 to a functional value F
        Inputs: x: It can be 1, 2, 3 dimension based on signal, image or 3d models
        Outputs: out: if function it a 1D, RGB (3D), 1D 
        """
        # if(self.num_embedding):
        #     # print("Embedding ", x, self.num_embedding)            
        #     x = self.num_embedding*x
        #     x = self.embed(x.long()).squeeze()
        #     # print("Embedding 1 ", x, self.num_embedding)
        # if(self.num_inputs != 1):
        #     x = input_encoder(x, self.avals, self.bvals)
        #     print("self.num_inputs ", x, self.num_inputs)
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
        num_inputs = 1

        self.num_embedding = hparams.num_embedding
        self.embedding_dim = hparams.embedding_dim
        self.embed = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.models = [self.embed]
        if(self.num_embedding):
            num_inputs = self.embedding_dim*num_inputs
        print("Num_inputs ", num_inputs, self.num_embedding)

        if(hparams.encoding_type == "style1"):
            self.encoder = EncodingStyle1(hparams.N_freqs, hparams.logscale)
        else:
            self.encoder = EncodingStyle2(hparams.N_freqs, hparams.encoding)
        self.models += [self.encoder]
        num_inputs = num_inputs*self.encoder.get_scale_factor()

        self.signal_encoder = SignalEncoder(hparams.num_layers, hparams.num_channels, 
            num_inputs=num_inputs)
        self.models += [self.signal_encoder]

        self.register_buffer('num_embedding_tensor', torch.FloatTensor(self.num_embedding))

        self.loss = nn.MSELoss(reduction='mean')
        return

    def forward(self, x):
        if(self.num_embedding):
            # print("Embedding ", x, self.num_embedding)
            x_in = x
            # x = self.num_embedding_tensor*x
            x = torch.mul(x, 32.0)
            x_scale = x.clone() + 0.5
            w1 = x_scale - x_scale.long()
            w2 = 1 - w1
            # print(w1, w2)
            # print(x_in, x, x_scale, x.long(), w1, w2, self.num_embedding)
            # print(x_in.type(), x.type(), x_scale.type(), x.long().type(), w1.type(), w2.type())

            x1 = self.embed(x.long()).squeeze()
            x2 = self.embed((x.long()+1).long()).squeeze()
            x = x1*w2 + x2*w1
        x = self.encoder(x)
        x = self.signal_encoder(x)
        return x

    def prepare_data(self):
        self.train_dataset = RandomSignal(split='train')
        self.val_dataset = RandomSignal(split='val')
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
                          batch_size=1, # validate one image (H*W rays) at a time
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

        if(self.current_epoch % 20 == 0):
            print(log['val_psnr'], log['val_loss'])
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