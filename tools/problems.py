#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math, sys
import tensorflow as tf

class Generator(object):
    def __init__(self,A,tf_floattype,**kwargs):
        self.A = A
        self.tf_floattype = tf_floattype
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf_floattype,(N,None),name='x' )
        self.y_ = tf.placeholder( tf_floattype,(M,None),name='y' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ) )

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)

def get_trial(config):
    if config.problem_name == 'bg':
        prob = bernoulli_gaussian_trial(
            config)  # a Bernoulli-Gaussian x, noisily observed through a random matrix
    elif config.problem_name == 'dd':
        prob = discrete_distribution_trial(
            config)  # a Bernoulli-Bernoulli x, noisily observed through a random matrix
    else:
        print('Unknown problem name')
        sys.exit()

    return prob

def get_measurement_matrix(config):
    # read problem parameters from config
    M = config.M
    N = config.N
    kappa = config.kappa
    col_normalized = config.col_normalized

    Gauss_base = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)

    if config.m_matrix == 'Gauss':
        A = Gauss_base
    elif config.m_matrix == 'sparse Gauss':
        A_pnz = 0.1
        A_sparse = ((np.random.uniform(0, 1, (M, N)) < Gauss_base) * Gauss_base / math.sqrt(A_pnz)).astype(np.float32)
        A = A_sparse
    elif config.m_matrix == 'sparse pm1':
        A_0pm1_pnz = 0.1
        A_0pm1 = ((np.random.uniform(0, 1, (M, N)) < A_0pm1_pnz) * np.sign(Gauss_base) / math.sqrt(
            M * A_0pm1_pnz)).astype(np.float32)
        A = A_0pm1
    else:
        print('Something is wrong with problem parameters: Unknown m_matrix name')
        sys.exit()

    if col_normalized:
        A = A / np.sqrt(np.sum(np.square(A), axis=0, keepdims=True))

    if not(kappa is None):
        if kappa >= 1:
            # create a random operator with a specific condition number
            U,_,V = la.svd(A,full_matrices=False)
            s = np.logspace( 0, np.log10( 1/kappa),M)
            A = np.dot( U*(s*np.sqrt(N)/la.norm(s)),V).astype(np.float32)

    return A

def bernoulli_gaussian_trial(config):

    M = config.M
    N = config.N
    L = config.L_training
    pnz = config.pnz
    SNR = config.SNR
    kappa = config.kappa
    tf_floattype = config.tf_floattype


    A = get_measurement_matrix(config)
    A_ = tf.constant(A, name='A', dtype=tf_floattype)
    prob = TFGenerator(A=A, tf_floattype=tf_floattype,A_=A_,pnz=pnz,kappa=kappa,SNR=SNR)
    prob.name = 'Bernoulli-Gaussian, random A'

    # training samples
    bernoulli_ = tf.to_float( tf.random_uniform( (N,L) ) < pnz)
    xgen_ = bernoulli_ * tf.random_normal( (N,L) )
    xgen_ = tf.dtypes.cast(xgen_, dtype=tf_floattype)

    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    noise_ = tf.random_normal( (M,L),stddev=math.sqrt( noise_var ), dtype=tf_floattype )

    ygen_ = tf.matmul( A_,xgen_) + noise_

    # validation samples
    prob.xval = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yval = np.matmul(A,prob.xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

    # test samples
    prob.xtest = ((np.random.uniform( 0,1,(N,config.L_testing))<pnz) * np.random.normal(0,1,(N,config.L_testing))).astype(np.float32)
    prob.ytest = np.matmul(A,prob.xtest) + np.random.normal(0,math.sqrt( noise_var ),(M,config.L_testing))

    # init values - probably not being used
    prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yinit = np.matmul(A,prob.xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

    # store necessary variables to prob
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var
    prob.config = config

    return prob

# This function generates discrete distribution trial for two types of symmetric distributions: binary with aplhabet {-1, 1}
# and ternary wtih alphabet {-1, 0, 1}. Binary is obtained by setting config.pnz = 0
def discrete_distribution_trial(config):
    # This function returns an object called prob which contains:
    # the measurement matrix, both numpy array A and TensorFlow constant A_,
    # Tensors xgen, ygen_ which can be used in TensorFlow to generate new training data,
    # numpy arrays xval and yval which are used to evaluate the learned network
    # numpy arrays xinit and yinit, which I am not sure are used at all ???
    # and a scalar noise_var

    # read problem parameters from config
    M = config.M
    N = config.N
    L = config.L_training
    pnz = config.pnz
    SNR = config.SNR
    kappa = config.kappa
    tf_floattype = config.tf_floattype

    A = get_measurement_matrix(config)

    A_ = tf.constant(A,name='A', dtype=tf_floattype)
    prob = TFGenerator(A=A, tf_floattype=tf_floattype, A_=A_,pnz=pnz,kappa=kappa,SNR=SNR)
    prob.name = 'Bernoulli-Bernoulli, random A'

    # training samples
    bernoulli_ = tf.to_float( tf.random_uniform( (N,L) ) < pnz)
    xgen_ = bernoulli_ * tf.sign(tf.random_normal( (N,L) ))
    xgen_ = tf.dtypes.cast(xgen_, dtype=tf_floattype)

    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    noise_ = tf.random_normal( (M,L),stddev=math.sqrt( noise_var ), dtype=tf_floattype )

    ygen_ = tf.matmul( A_,xgen_) + noise_

    # validation samples
    L_validation = config.L_testing
    prob.xval = ((np.random.uniform( 0,1,(N,L_validation))<pnz) * np.sign(np.random.normal(0,1,(N,L_validation)))).astype(np.float32)
    prob.yval = np.matmul(A,prob.xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L_validation))

    # test samples
    prob.xtest = ((np.random.uniform( 0,1,(N,config.L_testing))<pnz) * np.sign(np.random.normal(0,1,(N,config.L_testing)))).astype(np.float32)
    prob.ytest = np.matmul(A,prob.xtest) + np.random.normal(0,math.sqrt( noise_var ),(M,config.L_testing))

    # init values - probably not being used
    prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.sign(np.random.normal(0,1,(N,L)))).astype(np.float32)
    prob.yinit = np.matmul(A,prob.xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

    # store necessary variables to prob
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var
    prob.config = config

    return prob


def random_access_problem(which=1):
    from tools import raputil as ru
    if which == 1:
        opts = ru.Problem.scenario1()
    else:
        opts = ru.Problem.scenario2()

    p = ru.Problem(**opts)
    x1 = p.genX(1)
    y1 = p.fwd(x1)
    A = p.S
    M,N = A.shape
    nbatches = int(math.ceil(1000 /x1.shape[1]))
    prob = NumpyGenerator(p=p,nbatches=nbatches,A=A,opts=opts,iid=(which==1))
    if which==2:
        prob.maskX_ = tf.expand_dims( tf.constant( (np.arange(N) % (N//2) < opts['Nu']).astype(np.float32) ) , 1)

    _,prob.noise_var = p.add_noise(y1)

    unused = p.genYX(nbatches) # for legacy reasons -- want to compare against a previous run
    (prob.yval, prob.xval) = p.genYX(nbatches)
    (prob.yinit, prob.xinit) = p.genYX(nbatches)
    import multiprocessing as mp
    prob.nsubprocs = mp.cpu_count()
    return prob
