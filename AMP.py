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
from tools import problems,networks
from tools.Config import Config


config = Config(
    L_training = 1000,          # actual number of samples used to calculate test nmse. Problems involving discrete distributions should be evaluated with L_training = 1000 samples. Fro a dd problem with pnz=1., and N = 500, use 10x training samples
    L_testing = 10000,          # testing batch size
    SNR=15,                     # signal-to-noise ration in dB
    pnz=1.,                     # probability of non-zero
    delta = 0.65,               # measurement ratio m/N
    N = 5000,                    # unknown signal dimension
    untied = False,             # B matrices are shared acrossed layers or not
    T = 10,                      # number of layers
    m_matrix = 'Gauss',         # measurement matrix. Possible values: 'Gauss', 'sparse Gauss', 'sparse pm1'
    kappa = None,               # measurement matrix parameter
    col_normalized = False,     # measurement matrix parameter
    shrink = 'binary',              # thresholding function. Possible values: binary, 'ternary', 'gm4', 'soft'
    B_init = 'A^T',             # the way the weight matrix B is initialized. Possible values: 'A^T scaled', 'A^T'
    problem_name = 'dd',        # source distribution. Possible values: 'dd' for a discrete distribution or 'bg' for a Bernoulli-Gaussian mixture
    maxit = 1,                  # maximum number of training iterations. Suggested 1000000
    better_wait = 1,            # number of iterations before stepping out of training in case of no improvement. Suggested 5000
    trinit = 1e-3,              # learning rate
    refinements = (.5, .1, .01),# learning rates for refinements
    final_refine = None,        # do final refine
    ivl = 10,                   # parameter for printing validation results
    tf_floattype=tf.float64)    # simulation number format. TF tends to be much smaller when operating with tf.float64

config.M =  int(np.floor(config.delta * config.N))
config.name_prefix = 'LAMP_bg_giid_SNR=' + str(config.SNR) + 'dB_' + str(config.pnz) + '-pnz_' + config.problem_name + '-problem_' + config.shrink + '-shrink'
config.file_name = config.name_prefix + '.npz'
config.config_file_name = config.name_prefix + '.pkl'

## no need to save the config to a file when evaluating the networks
## config.save_to_file()

# Create the basic problem structure.
prob = problems.get_trial(config)
print(prob.name + ' problem created ...')
print('A is:')
print(prob.A)

# build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LGMAMP(prob)
print('Building layers ... done')


nmse_arrray = np.array([])
mse_arrray = np.array([])
ser_arrray = np.array([])

# Evaluating classical AMP in TensorFlow
sess = tf.Session()
y,x = prob(sess)


sess.run(tf.global_variables_initializer())

for name, xhat_, var_list in layers:

    if "non-linear T=" in name:

        nmse_denom_ = tf.nn.l2_loss(prob.x_)
        nmse_ = tf.nn.l2_loss( xhat_ - prob.x_) / nmse_denom_
        mse_ = 2* tf.nn.l2_loss(xhat_ - prob.x_) / (config.L_training*config.N)

        x_hat, nmse, mse = sess.run([xhat_, nmse_, mse_], feed_dict={prob.y_: y, prob.x_: x})

        nmse_arrray = np.append(nmse_arrray,nmse)
        mse_arrray= np.append(mse_arrray,mse)

        if prob.name.startswith('Bernoulli-Bernoulli') or prob.name.startswith('Binary'):
            x_hat_hard = np.rint(x_hat)
            ser = np.sum(np.abs(x - x_hat_hard) > 0.5) / (config.N * config.L_training)  # symbol error rate

            print(name + '\tnmse={nmse:.4f} nmse_dB={nmse_dB:.6f} dB   \tser={ser:.5f}'.format(nmse=nmse, nmse_dB=10 * np.log10(nmse), ser=ser))
            ser_arrray = np.append(ser_arrray, ser)
        else:
            print(name + '\tnmse={nmse:.4f} nmse_dB={nmse_dB:.6f} dB '.format(nmse=nmse, nmse_dB=10*np.log10(nmse)))

sess.close()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('nmse/dB=', 10*np.log10(nmse_arrray))
print('mse/dB=', 10*np.log10(mse_arrray))


if prob.name == 'dm':
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print('ser=', ser_arrray,';')