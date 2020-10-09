import torch
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class GridEmbedding(nn.Module):
    def __init__(self, grid_size, embedding_dim, bounding_box):
        """ """
        super(GridEmbedding, self).__init__()
        # self.grid_mapping = 
        self.num_embedding =  grid_size[0]*grid_size[1]*grid_size[2] # 3000000 #
        self.embedding_dim = embedding_dim
        grid_to_embedding_map = torch.range(0, self.num_embedding-1, dtype=torch.int64)
        self.register_buffer(name='grid_to_embedding_map', tensor=grid_to_embedding_map)

        # print("total number of embedding ", self.num_embedding)
        grid_embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        setattr(self, f"grid_embedding", grid_embedding)
        box_min, box_max = bounding_box
        self.register_buffer(name='grid_size', tensor=torch.LongTensor(grid_size))
        self.register_buffer(name='box_min', tensor=torch.FloatTensor(box_min))
        self.register_buffer(name='box_max', tensor=torch.FloatTensor(box_max))
        scale_factor = self.grid_size / (self.box_max - self.box_min)
        self.register_buffer(name='scale_factor', tensor=scale_factor)

    def forward(self, x):
        """ """
        # subtract min box from all points to offset to box origin
        # print(x.size(), self.box_min.device, x.device, self.scale_factor)
        xsf = x - self.box_min
        xr = (x - self.box_min) * self.scale_factor
        x_idx = torch.floor(xr)
        xw = xr - x_idx

        # x_embed = getattr(self, f"grid_embedding")(x_idx_val)
        # print(x[:2,:2], x_idx[:2,:2], xw[:2,:2], x_idx_val[:2,:2], x_embed[:2,:2])
        # # xw = x-

        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # Let x, y, z => (0, 1, 2)
        c000_idx = x_idx[:,0]+x_idx[:,1]*self.grid_size[0]+x_idx[:,2]*self.grid_size[0]*self.grid_size[1]
        c000_idx = c000_idx.type(torch.cuda.LongTensor)
        # Extracting the index of eight values
        c001_idx = c000_idx + 1
        c010_idx = c000_idx + self.grid_size[0]
        c100_idx = c000_idx + self.grid_size[0]*self.grid_size[1]
        c011_idx = c000_idx + self.grid_size[0] + 1
        c101_idx = c000_idx + self.grid_size[0]*self.grid_size[1] + 1
        c110_idx = c000_idx + self.grid_size[0]*self.grid_size[1] + self.grid_size[0]
        c111_idx = c000_idx + self.grid_size[0]*self.grid_size[1] + self.grid_size[0] + 1

        c000_idx = torch.where(c000_idx >= self.num_embedding, (self.num_embedding-1)*torch.ones_like(c001_idx), c000_idx)

        c001_idx = torch.where(c001_idx >= self.num_embedding, c000_idx, c001_idx)
        c010_idx = torch.where(c010_idx >= self.num_embedding, c000_idx, c010_idx)
        c100_idx = torch.where(c100_idx >= self.num_embedding, c000_idx, c100_idx)
        c011_idx = torch.where(c011_idx >= self.num_embedding, c000_idx, c011_idx)
        c101_idx = torch.where(c101_idx >= self.num_embedding, c000_idx, c101_idx)
        c110_idx = torch.where(c110_idx >= self.num_embedding, c000_idx, c110_idx)
        c111_idx = torch.where(c111_idx >= self.num_embedding, c000_idx, c111_idx)
        
        all_idx = torch.cat((c000_idx, c001_idx, c010_idx, c100_idx, c011_idx, c101_idx, c110_idx, c111_idx), -1)
        # print("All index size ", all_idx.size(), c000_idx.size(), all_idx[-10:])
        if(all_idx < 0).any():
            print("Index less than zero", all_idx[(all_idx < 0).nonzero()])
            exit()
        if(all_idx >= self.num_embedding).any():
            print("Index greater than 1000", all_idx[(all_idx >= self.num_embedding).nonzero()])
            exit()

        # Get the embedding values
        c000_val = self.grid_embedding(self.grid_to_embedding_map[c000_idx])
        c001_val = self.grid_embedding(self.grid_to_embedding_map[c001_idx])
        c010_val = self.grid_embedding(self.grid_to_embedding_map[c010_idx])
        c100_val = self.grid_embedding(self.grid_to_embedding_map[c100_idx])
        c011_val = self.grid_embedding(self.grid_to_embedding_map[c011_idx])
        c101_val = self.grid_embedding(self.grid_to_embedding_map[c101_idx])
        c110_val = self.grid_embedding(self.grid_to_embedding_map[c110_idx])
        c111_val = self.grid_embedding(self.grid_to_embedding_map[c111_idx])
        # Interpolation Level 1
        xw = torch.unsqueeze(xw, -1)
        # print(c000_val.size(), xw[:, :, 0].size())
        c00 = c000_val*(1 - xw[:, 0])+c100_val*(xw[:, 0])
        c01 = c001_val*(1 - xw[:, 0])+c101_val*(xw[:, 0])
        c10 = c010_val*(1 - xw[:, 0])+c110_val*(xw[:, 0])
        c11 = c011_val*(1 - xw[:, 0])+c111_val*(xw[:, 0])
        # Level 2
        c0 = c00*(1 - xw[:, 1]) + c10*(xw[:, 1])
        c1 = c01*(1 - xw[:, 1]) + c11*(xw[:, 1])
        # Level 3
        c = c0*(1 - xw[:, 2]) + c1*(xw[:, 2])

        return c



class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=672, in_channels_dir=27, #63
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out