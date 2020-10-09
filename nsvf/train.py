import os, sys
sys.path.append("/nitthilan/source_code/multiview_rendering/neural_sparse_voxel_field/") # Adds higher directory to python modules path.

from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from data_loader import dataset_dict
# import data_loader.shapenet_dl
# import data_loader.zbrush_dl

# models
from model import Embedding, NeRF, GridEmbedding
from rendering import render_rays, prune_embedding

# optimizer, scheduler, visualization
from utils import *

# losses
# from losses import loss_dict

# metrics
from utils.metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.bceloss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, valid_mask=None):
        
        # valid_mask = torch.squeeze(valid_mask)
        # print("Shape of output ", valid_mask.shape, inputs['opacity_coarse'].unsqueeze(0).shape,
        #   inputs['opacity_fine'].unsqueeze(0).shape)
        loss = self.loss(inputs['rgb_coarse'], targets)
        if valid_mask != None:
            bceloss = self.bceloss(inputs['opacity_coarse'].unsqueeze(0), valid_mask.float  ())
        # if 'rgb_fine' in inputs:
        #     loss += self.loss(inputs['rgb_fine'], targets)
        #     if valid_mask != None:
        #         bceloss += self.bceloss(inputs['opacity_fine'].unsqueeze(0), valid_mask.float())

        # print("Output losses ", bceloss, loss)
        return loss # bceloss+loss
               

# loss_dict = {'mse': MSELoss}

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        self.dataset = dataset_dict[self.hparams.dataset_name]
        train_dataset = self.dataset(split='train', **kwargs)


        self.loss = MSELoss() # loss_dict[hparams.loss_type]()

        self.grid_size = (10,10,10)

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        # bounding_box, grid_size, embedding_dim
        self.grid_embedding = GridEmbedding(self.grid_size, 32, train_dataset.get_bounding_box()) # grid_size, embedding_dim
        self.embeddings = [self.embedding_xyz, self.embedding_dir, self.grid_embedding]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        # valid_mask = batch['valid_mask']
        # return rays, rgbs, valid_mask
        return rays, rgbs

    def forward(self, rays):
        # sample_per_voxel = (8, 8, 8)
        # bounding_box = self.train_dataset.get_bounding_box()
        # prune_embedding(self.models, self.embeddings, sample_per_voxel,
        #     self.grid_size, bounding_box)
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
    # def setup(self, stage):
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = self.dataset(split='train', **kwargs)
        self.val_dataset = self.dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

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
        # rays, rgbs, valid_mask = self.decode_batch(batch)
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        # log['train/loss'] = loss = self.loss(results, rgbs, valid_mask)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log}

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        # rays, rgbs, valid_mask = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        # log = {'val_loss': self.loss(results, rgbs, valid_mask)}
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            if(self.hparams.dataset_name == "zbrush"):
                W, H = int(W*0.5), int(H*0.5)
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
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