#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train
from tools.Config import Config

# save config for later possible reference/check
config = Config(
    L_training = 100,          # training batch size
    L_testing = 10000,          # testing batch size
    SNR=15,                     # signal-to-noise ration in dB
    pnz=1.,                     # probability of non-zero
    delta = 0.5,                # measurement ratio m/N
    N = 500,                    # unknown signal dimension
    untied = False,             # B matrices are shared acrossed layers or not
    T = 2,                      # number of layers
    m_matrix = 'Gauss',         # measurement matrix. Possible values: 'Gauss', 'sparse Gauss', 'sparse pm1'
    kappa = None,               # measurement matrix parameter
    col_normalized = False,     # measurement matrix parameter
    shrink = 'gm4',             # thresholding function. Possible values: binary, 'ternary', 'gm4', 'soft'
    problem_name = 'dd',        # source distribution. Possible values: 'dm' for a discrete distribution or 'bg' for a Bernoulli-Gaussian mixture
    B_init = 'A^T',             # the way the weight matrix B is initialized. Possible values: 'A^T scaled', 'A^T'
    maxit = 1,               # maximum number of training iterations. Suggested 1000000
    better_wait = 1,          # number of iterations before stepping out of training in case of no improvement. Suggested 5000
    trinit = 1e-3,              # learning rate
    refinements = (.5, .1, .01),# learning rates for refinements
    final_refine = None,        # do final refine
    ivl = 10,                   # parameter for printing validation results
    tf_floattype=tf.float64)    # simulation number format. TF tends to be much smaller when operating with tf.float64

config.M =  int(np.floor(config.delta * config.N))
config.name_prefix = 'networks/LAMP_A=' + config.m_matrix + ';_SNR=' + str(config.SNR) + 'dB;_pnz=' + str(config.pnz) + ';_delta=' + str(config.delta) + ';_N=' + str(config.N) + ';_problem=' + config.problem_name + ';_shrink=' + config.shrink + ';_mi=' + str(config.maxit) +';_bw=' + str(config.better_wait)  +';_L_testing=' + str(config.L_testing)  +';_L_training=' + str(config.L_training)
config.file_name = config.name_prefix + '.npz'
config.config_file_name = config.name_prefix + '.pkl'
# config.save_to_file()

file_name = 'LAMP_A=Gauss;_SNR=15dB;_pnz=1.0;_delta=0.65;_N=500;_problem=dd;_shrink=gm2;_mi=1000;_bw=20;_L_testing=10000;_L_training=100.pkl'
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, file_name)
config = Config().read_from_file(file_path)
config.T=2


# TODO generation of test and validation samples can be simplified
# Create the basic problem structure.
prob = problems.get_trial(config)

# build a L-GM-AMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LGMAMP(prob)
print('Building layers ... done')

# plan the learning
training_stages = train.setup_training(layers, prob)
print('Planing learning ... done')

# do the learning (takes a while)
sess = train.do_LGMAMP_training(training_stages, prob, config)
print('Training done')

print('Evaluating network on test data ...')
train.evaluate_nmse(sess, training_stages, prob, config)
