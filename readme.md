
Working on Neural scene representation for fast and high-quality free-viewpoint rendering. Extending the ideas of Neural Sparse Voxel Fields (NSVF, Liu et. al. 2020) and Neural Radiance Fields (NeRF, Mildenhall et. al. 2020) to a) Multi-resolution coarse to fine approach of learning b) Single image to scene representation c) Animate 3D objects by adding a time dimension. 


List of experiments:
experiment with the number of input samples - is there a nyquist sampling frequency for approximation?

Can different kernals be used liike Gaussian or wavelet instead of sin/cos

Differential learning rate for different frequency components? So can we have a NN which has approximation initially and then refines it for further inputs



Compression dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/


List of conditions to take care:
- Not worry about transparent objects
- Assume object is at the center of the multiple views??

Three variations of multi view generation:
- Given a partial embedded 3D volume can we generate the remaining 3D volume using GAN or auto generation? - Can embedding be completed based on partial information?
	- Once this is generated using the GAN, we use the neural network trained with the available partial 3D volume and try generating the rest of the 3D volume information?
- Use a fully trained embedded 3D volume using large number of 3D shapes. Then try training a small MLP to generate the 3D shape with as minimal number of images as possible?
	- What is the minimum number of images to reconstruct 3D image? Ideally one?
	- Based on symetrical objects
- what is the minimum number of images even for creating a 3D volume embedding (same as NSVF) but what is the minimum number of images?
- 
- find a intersecting volume based on provided min-max depth and trapezium
	- intersecting volume between two trapezoidal prism, find the overlapping area of n trapezoids


NeRF in the Wild: bringing in ligting and situation variation and allowing transient
- https://arxiv.org/pdf/2008.02268.pdf
- https://nerf-w.github.io/
- photo tourism dataset

HyperNetworks:
- https://arxiv.org/pdf/1609.09106.pdf
- https://www.youtube.com/watch?v=KY9DoutzH6k&ab_channel=TAUVOD - Hypernetworks: a versatile and powerful tool
- hypernetworks continual learning: 
	- https://arxiv.org/pdf/2006.12109.pdf
	- https://arxiv.org/pdf/1906.00695.pdf

Dataset: https://arxiv.org/pdf/1911.10127.pdf - BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks

Neural Volumes: Learning Dynamic Renderable Volumes from Images : https://arxiv.org/pdf/1906.07751.pdf
Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations : 	https://vsitzmann.github.io/srns/, 	https://papers.nips.cc/paper/8396-scene-representation-networks-continuous-3d-structure-aware-neural-scene-representations.pdf



Meta-learning: 
https://www.youtube.com/watch?v=A0a1M61gjgI&feature=youtu.be&ab_channel=virtualmlss2020 - Meta Learning, part 1 - Yee Whye Teh - MLSS 2020, Tübingen

Toonify:
- https://toonify.justinpinkney.com/
- https://www.justinpinkney.com/making-toonify/

Search engine: https://www.semion.io/Home/Help

AABB algorithm:
- hierarchical octree structure AABB to optimize the number of ray to pixel interaction
- https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
- https://github.com/stackgl/ray-aabb-intersection, aabb ray voxel intersection
https://aabbtree.readthedocs.io/en/latest/#example
- pytorch ray aabb intersection
- https://github.com/francisengelmann/fast_voxel_traversal

Ray triangle intersection: Möller-Trumbore intersection algorithm

trilinear interpolation calculator: trilinear interpolation pytorch
F.upsample(x, size=(3, 60, 100), mode='trilinear') # 5 6 3 60 100

- https://github.com/tedyhabtegebrial/PyTorch-Trilinear-Interpolation/blob/master/interpolation.py

- Ray intersect bounding box: https://pyrr.readthedocs.io/en/latest/api_geometric_tests.html
	- https://python.hotexamples.com/examples/pyrr.geometric_tests/-/ray_intersect_aabb/python-ray_intersect_aabb-function-examples.html

Trilinear interpolation:
- https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation

Things to implement:
- Not taken care of cases where we are not doing 360 degree capture of the central object

Photogrammetry:
- https://www.reddit.com/r/photogrammetry/
- https://www.reddit.com/r/photogrammetry/comments/a1ok6r/terrible_results_with_turntable_images/
- photogrammetry turntable

- https://github.com/weihaox/awesome-neural-rendering

Turntable camera parameters: http://pixologic.com/turntable/instructions/



Things to try out:
- Try reducing the number of images to learn an object and check the performance
- Try changing the learning rate across number of epochs
- Explore         self.white_back = True
- Try not scaling the image and use the original dimension but reduce the number of images for training
- Why does it not work with the http://www.pixologic01.com/turntable/turntables3/eric_keller/ztg/3/image01.jpg
- Try this color image : http://www.pixologic01.com/turntable/turntables3/pornsin_sooksanguan/ztg/1/image01.jpg
- Is there a voxel embedding compared to word2vec embedding?