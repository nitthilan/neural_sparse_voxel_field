import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
import tqdm as tqdm
from matplotlib.lines import Line2D

from jax import random, grad, jit, vmap


import time

import numpy as np
# import random


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))

# Signal makers

def sample_random_signal(key, decay_vec):
  N = decay_vec.shape[0]
  raw = random.normal(key, [N, 2]) @ np.array([1, 1j])
  signal_f = raw * decay_vec
  signal = np.real(np.fft.ifft(signal_f))
  return signal

def sample_random_powerlaw(key, N, power):
  coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
  decay_vec = coords ** -power
  return sample_random_signal(key, decay_vec) # * 100

# Encoding 


input_encoder = lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x[...,None]) * b), 
                                                a * np.cos((2.*np.pi*x[...,None]) * b)], axis=-1) / np.linalg.norm(a) #* np.sqrt(a.shape[0])

### complicated things


def train_model_lite(rand_key, network_size, lr, iters, 
                train_input, test_input, optimizer, ab):
    init_fn, apply_fn, kernel_fn = make_network(*network_size)
    kernel_fn = jit(kernel_fn)

    run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))
    model_loss = jit(lambda params, ab, x, y: .5 * np.sum((run_model(params, ab, x) - y) ** 2))
    model_psnr = jit(lambda params, ab, x, y: -10 * np.log10(np.mean((run_model(params, ab, x) - y) ** 2)))
    model_grad_loss = jit(lambda params, ab, x, y: jax.grad(model_loss)(params, ab, x, y))

    opt_init, opt_update, get_params = optimizer(lr)
    opt_update = jit(opt_update)

    _, params = init_fn(rand_key, (-1, input_encoder(train_input[0], *ab).shape[-1]))
    opt_state = opt_init(params)

    pred0 = run_model(get_params(opt_state), ab, test_input[0])
    pred0_f = np.fft.fft(pred0)
    
    for i in (range(iters)):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), ab, *train_input), opt_state)
    
    train_psnr = model_psnr(get_params(opt_state), ab, *train_input)
    test_psnr = model_psnr(get_params(opt_state), ab, *test_input)
    # theory = predict_psnr(kernel_fn, np.fft.fft(test_input[1]), pred0_f, ab, i * lr)
            
    return get_params(opt_state), train_psnr, test_psnr #, theory


N_train = 1024
data_powers = [.5, 1.0, 1.5]
# data_powers = [1.0]
N_test_signals = 4

N_embed = 16

network_size = (4, 256)

# learning_rate = 5e-3
learning_rate = 2e-3
sgd_iters = 1000

rand_key = random.PRNGKey(0)


def data_maker(rand_key, N_pts, N_signals, p):
  rand_key, *ensemble = random.split(rand_key, 1 + N_signals)
  data = np.stack([sample_random_powerlaw(ensemble[i], N_pts, p) for i in range(N_signals)])
  # data = (data - data.min(-1, keepdims=True)) / (data.max(-1, keepdims=True) - data.min(-1, keepdims=True))  - .5
  data = (data - data.min()) / (data.max() - data.min())  - .5
  return data, rand_key


# Signal
M = 2 ## Dont change
N = N_train
x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
x_train = x_test[::M]


search_vals = 2. ** np.linspace(-5., 4., 1*8+1)

bval_generators = {
    'gaussian' : (32, lambda key, sc, N : random.normal(key, [N]) * sc),
    'unif' : (64, lambda key, sc, N : random.uniform(key, [N]) * sc),
    'power1' : (80, lambda key, sc, N : (sc ** random.uniform(key, [N]))),
    'laplace' : (20, lambda key, sc, N : random.laplace(key, [N]) * sc),
}

names = list(bval_generators.keys())
train_fn = lambda s, key, ab : train_model_lite(key, network_size, learning_rate, sgd_iters,
                    (x_train, s[::2]), (x_test[1::2], s[1::2]), optimizers.adam, ab)


best_powers = [.4, .75, 1.5]
outputs_meta = []
dense_meta = []
s_lists = []
# print(tqdm(zip(data_powers, best_powers)))

for p, bp in zip(data_powers, best_powers):
  s_list, rand_key = data_maker(rand_key, N*M, N_test_signals, p)
  s_lists.append(s_list)

  b = np.float32(np.arange(1, N//2+1))
  ab_dense = (b ** -bp, b)
  rand_key, *ensemble_key = random.split(rand_key, 1+s_list.shape[0])
  ensemble_key = np.array(ensemble_key)
  ab_samples = np.array([ab_dense] * s_list.shape[0])
  # outputs_dense = vmap(train_fn, in_axes=(0, 0, 0))(s_list, ensemble_key, ab_samples)
  dense_meta.append(outputs_dense)

  outputs = []
  
  for sc in search_vals:
    outputs.append([])

    for k in names:
      
      rand_key, *ensemble_key = random.split(rand_key, 1+s_list.shape[0])
      ensemble_key = np.array(ensemble_key)

      factor, b_fn = bval_generators[k]
      ab_samples = np.array([(np.ones([N_embed]), b_fn(ensemble_key[i], factor * sc, N_embed)) for i in range(s_list.shape[0])])
      rand_key, *ensemble_key = random.split(rand_key, 1+s_list.shape[0])
      ensemble_key = np.array(ensemble_key)
      
      # z = vmap(train_fn, in_axes=(0, 0, 0))(s_list, ensemble_key, ab_samples)
      outputs[-1].append(list(z) + [ab_samples])
      

  outputs_meta.append(outputs)
