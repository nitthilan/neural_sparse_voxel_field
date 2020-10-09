# from .blender import BlenderDataset
# from .llff import LLFFDataset
# from .shapenet_crop import ShapeNet
# from .shapenet_mask import ShapeNet
# # from .shapenet import ShapeNet
# from .shapenet_seg import ShapeNet
# from .turbosquid import TurboSquid

from .shapenet import ShapeNet
from .zbrush import Zbrush

dataset_dict = {'shapenet':ShapeNet,
                'zbrush':Zbrush}