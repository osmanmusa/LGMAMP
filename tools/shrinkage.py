#!/usr/bin/python
import numpy as np
import sys
import re
import math
import tensorflow as tf
import tensorflow_probability as tfp

__doc__ = """
This file contains various separable shrinkage functions for use in TensorFlow.
All functions perform shrinkage toward zero on each elements of an input vector
    r = x + w, where x is sparse and w is iid Gaussian noise of a known variance rvar

All shrink_* functions are called with signature

    xhat,dxdr = func(r,rvar,theta)

Hyperparameters are supplied via theta (which has length ranging from 1 to 5)
    shrink_soft_threshold : 1 or 2 parameters
    shrink_bgest : 2 parameters
    shrink_expo : 3 parameters
    shrink_spline : 3 parameters
    shrink_piecwise_linear : 5 parameters

A note about dxdr:
    dxdr is the per-column average derivative of xhat with respect to r.
    So if r is in Real^(NxL),
    then xhat is in Real^(NxL)
    and dxdr is in Real^L
"""

def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def auto_gradients(xhat , r ):
    """Return the per-column average gradient of xhat xhat with respect to r.
    """
    dxdr = tf.gradients(xhat,r)[0]
    dxdr = tf.reduce_mean(dxdr,0)
    minVal=.5/int(r.get_shape()[0])
    dxdr = tf.maximum( dxdr, minVal)
    return dxdr

def shrink_soft_threshold(r,rvar,theta):
    """
    soft threshold function
        y=sign(x)*max(0,abs(x)-theta[0]*sqrt(rvar) )*scaling
    where scaling is theta[1] (default=1)
    in other words, if theta is len(1), then the standard
    """
    if len(theta.get_shape())>0 and theta.get_shape() != (1,):
        lam = theta[0] * tf.sqrt(rvar)
        scale=theta[1]
    else:
        lam  = theta * tf.sqrt(rvar)
        scale = None
    lam = tf.maximum(lam,0)
    arml = tf.abs(r) - lam
    xhat = tf.sign(r) * tf.maximum(arml,0)
    dxdr = tf.reduce_mean(tf.to_float(arml>0),0)
    if scale is not None:
        xhat = xhat*scale
        dxdr = dxdr*scale
    return (xhat, dxdr, 0)

def shrink_bgest(r,rvar,theta):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    xvar1 = abs(theta[...,0])
    loglam = theta[...,1] # log(1/lambda - 1)
    beta = 1/(1+rvar/xvar1)
    r2scale = r*r*beta/rvar
    rho = tf.exp(loglam - .5*r2scale ) * tf.sqrt(1 +xvar1/rvar)
    rho1 = rho+1
    xhat = beta*r/rho1
    dxdr = beta*((1+rho*(1+r2scale) ) / tf.square( rho1 ))
    dxdr = tf.reduce_mean(dxdr,0)
    return (xhat, dxdr, 0)

def shrink_piecwise_linear(r,rvar,theta):
    """Implement the piecewise linear shrinkage function.
        With minor modifications and variance normalization.
        theta[...,0] : abscissa of first vertex, scaled by sqrt(rvar)
        theta[...,1] : abscissa of second vertex, scaled by sqrt(rvar)
        theta[...,2] : slope from origin to first vertex
        theta[''',3] : slope from first vertex to second vertex
        theta[...,4] : slope after second vertex
    """
    ab0 = theta[...,0]
    ab1 = theta[...,1]
    sl0 = theta[...,2]
    sl1 = theta[...,3]
    sl2 = theta[...,4]

    # scale each column by sqrt(rvar)
    scale_out = tf.sqrt(rvar)
    scale_in = 1/scale_out
    rs = tf.sign(r*scale_in)
    ra = tf.abs(r*scale_in)

    # split the piecewise linear function into regions
    rgn0 = tf.to_float( ra<ab0)
    rgn1 = tf.to_float( ra<ab1) - rgn0
    rgn2 = tf.to_float( ra>=ab1)
    xhat = scale_out * rs*(
            rgn0*sl0*ra +
            rgn1*(sl1*(ra - ab0) + sl0*ab0 ) +
            rgn2*(sl2*(ra - ab1) +  sl0*ab0 + sl1*(ab1-ab0) )
            )
    dxdr =  sl0*rgn0 + sl1*rgn1 + sl2*rgn2
    dxdr = tf.reduce_mean(dxdr,0)
    return (xhat,dxdr)

def pwlin_grid(r_,rvar_,theta_,dtheta = .75):
    """piecewise linear with noise-adaptive grid spacing.
    returns xhat,dxdr
    where
        q = r/dtheta/sqrt(rvar)
        xhat = r * interp(q,theta)

    all but the  last dimensions of theta must broadcast to r_
    e.g. r.shape = (500,1000) is compatible with theta.shape=(500,1,7)
    """
    ntheta = int(theta_.get_shape()[-1])
    scale_ = dtheta / tf.sqrt(rvar_)
    ars_ = tf.clip_by_value( tf.expand_dims( tf.abs(r_)*scale_,-1),0.0, ntheta-1.0 )
    centers_ = tf.constant( np.arange(ntheta),dtype=tf.float32 )
    outer_distance_ = tf.maximum(0., 1.0-tf.abs(ars_ - centers_) ) # new dimension for distance to closest bin centers (or center)
    gain_ = tf.reduce_sum( theta_ * outer_distance_,axis=-1) # apply the gain (learnable)
    xhat_ = gain_ * r_
    dxdr_ = tf.gradients(xhat_,r_)[0]
    return (xhat_,dxdr_)

def shrink_expo(r,rvar,theta):
    """ Exponential shrinkage function
        xhat = r*(theta[1] + theta[2]*exp( - r^2/(2*theta[0]^2*rvar ) ) )
    """
    r2 = tf.square(r)
    den = -1/(2*tf.square(theta[0])*rvar)
    rho = tf.exp( r2 * den)
    xhat = r*( theta[1] + theta[2] * rho )
    return (xhat,auto_gradients(xhat,r) )

def shrink_spline(r,rvar,theta):
    """ Spline-based shrinkage function
    """
    scale = theta[0]*tf.sqrt(rvar)
    rs = tf.sign(r)
    ar = tf.abs(r/scale)
    ar2 = tf.square(ar)
    ar3 = ar*ar2
    reg1 = tf.to_float(ar<1)
    reg2 = tf.to_float(ar<2)-reg1
    ar_m2 = 2-ar
    ar_m2_p2 = tf.square(ar_m2)
    ar_m2_p3 = ar_m2*ar_m2_p2
    beta3 = ( (2./3 - ar2  + .5*ar3)*reg1 + (1./6*(ar_m2_p3))*reg2 )
    xhat = r*(theta[1] + theta[2]*beta3)
    return (xhat,auto_gradients(xhat,r))

def shrink_gm2(r, rvar, theta):

    print('In shrink_gm2 ...')

    w1 = theta[0]
    w2 = 1 - w1

    # logw12_ratio = theta[0]

    mul1 = theta[1]
    sigmal1 = tf.abs(theta[2])

    mul2 = theta[3]
    sigmal2 = tf.abs(theta[4])

    eps = 0.0
    r = tf.real(r)

    tfd = tfp.distributions
    N1 = tfd.Normal(loc=mul1, scale=tf.sqrt(sigmal1 + rvar + eps))
    N2 = tfd.Normal(loc=mul2, scale=tf.sqrt(sigmal2 + rvar + eps))

    beta1 = w1 * N1.prob(r)
    beta2 = w2 * N2.prob(r)

    beta1_bar = beta1 / (beta1 + beta2)
    beta2_bar = beta2 / (beta1 + beta2)

    # beta12_ratio = w1 / w2 * tf.sqrt((sigmal2 + rvar) / (sigmal1 + rvar)) * \
    #                tf.exp(-(r - mul1) ** 2 / (2 * (sigmal1 + rvar))) * \
    #                tf.exp((r - mul2) ** 2 / (2 * (sigmal2 + rvar)))
    # beta21_ratio = w2 / w1 * tf.sqrt((sigmal1 + rvar) / (sigmal2 + rvar)) * \
    #                tf.exp(-(r - mul2) ** 2 / (2 * (sigmal2 + rvar))) * \
    #                tf.exp((r - mul1) ** 2 / (2 * (sigmal1 + rvar)))

    # beta12_ratio = tf.sqrt((sigmal2 + rvar) / (sigmal1 + rvar)) * \
    #                tf.exp(-(r - mul1) ** 2 / (2 * (sigmal1 + rvar)) + logw12_ratio + (r - mul2) ** 2 / (2 * (sigmal2 + rvar)))
    # beta21_ratio = tf.sqrt((sigmal1 + rvar) / (sigmal2 + rvar)) * \
    #                tf.exp(-(r - mul2) ** 2 / (2 * (sigmal2 + rvar)) - logw12_ratio + (r - mul1) ** 2 / (2 * (sigmal1 + rvar)))
    #
    # beta1_bar = 1/(1 + beta21_ratio)
    # beta2_bar = 1/(1 + beta12_ratio)

    gamma1 = (mul1 * rvar + r * sigmal1) / (sigmal1 + rvar + eps)
    gamma2 = (mul2 * rvar + r * sigmal2) / (sigmal2 + rvar + eps)

    nu1 = (rvar * sigmal1) / (sigmal1 + rvar + eps)
    nu2 = (rvar * sigmal2) / (sigmal2 + rvar + eps)

    expectation = beta1_bar * gamma1 + beta2_bar * gamma2
    power = beta1_bar * (nu1 + tf.square(gamma1)) + beta2_bar * (nu2 + tf.square(gamma2))

    var = power - tf.square(expectation)

    xhat = expectation

    dxdr1 = -beta1_bar*beta2_bar*((r-mul1)/(sigmal1 + rvar) - (r-mul2)/(sigmal2 + rvar))*(gamma1 - gamma2)
    dxdr2 = beta1_bar* sigmal1 / (sigmal1 + rvar) + beta2_bar* sigmal2 / (sigmal2 + rvar)
    dxdr = dxdr1 + dxdr2
    return (xhat, dxdr, var)

def shrink_gm3(r, rvar, theta):
    # (0.5, -.5, 1, 0.2, 0.2, 0.2, .5, 1)),
    print('In shrink_gm3 ...')
    w1 = theta[0]
    mul1 = theta[1]
    sigmal1 = theta[2]

    w2 = theta[3]
    mul2 = theta[4]
    sigmal2 = theta[5]

    w3 = 1 - w1 - w2
    mul3 = theta[6]
    sigmal3 = theta[7]

    tfd = tfp.distributions
    N1 = tfd.Normal(loc=mul1, scale=tf.sqrt(sigmal1 + rvar))
    N2 = tfd.Normal(loc=mul2, scale=tf.sqrt(sigmal2 + rvar))
    N3 = tfd.Normal(loc=mul3, scale=tf.sqrt(sigmal3 + rvar))

    beta1 = w1 * N1.prob(r)
    beta2 = w2 * N2.prob(r)
    beta3 = w3 * N3.prob(r)

    beta1_bar = beta1 / (beta1 + beta2 + beta3)
    beta2_bar = beta2 / (beta1 + beta2 + beta3)
    beta3_bar = beta3 / (beta1 + beta2 + beta3)

    gamma1 = (mul1 * rvar + r * sigmal1) / (sigmal1 + rvar)
    gamma2 = (mul2 * rvar + r * sigmal2) / (sigmal2 + rvar)
    gamma3 = (mul3 * rvar + r * sigmal3) / (sigmal3 + rvar)

    nu1 = (rvar * sigmal1) / (sigmal1 + rvar)
    nu2 = (rvar * sigmal2) / (sigmal2 + rvar)
    nu3 = (rvar * sigmal3) / (sigmal3 + rvar)

    expectation = beta1_bar * gamma1 + beta2_bar * gamma2 + beta3_bar * gamma3
    power = beta1_bar * (nu1 + tf.square(gamma1)) + beta2_bar * (nu2 + tf.square(gamma2)) + beta3_bar * (nu3 + tf.square(gamma3))

    var = power - tf.square(expectation)

    xhat = expectation

    rmmos21 = (r - mul1) / (sigmal1 + rvar)
    rmmos22 = (r - mul2) / (sigmal2 + rvar)
    rmmos23 = (r - mul3) / (sigmal3 + rvar)

    dxdr112 = beta1_bar * beta2_bar * (rmmos22 - rmmos21) * (gamma1 - gamma2)
    dxdr123 = beta2_bar * beta3_bar * (rmmos23 - rmmos22) * (gamma2 - gamma3)
    dxdr131 = beta3_bar * beta1_bar * (rmmos21 - rmmos23) * (gamma3 - gamma1)

    dxdr21 = beta1_bar * sigmal1 / (sigmal1 + rvar)
    dxdr22 = beta2_bar * sigmal2 / (sigmal2 + rvar)
    dxdr23 = beta3_bar * sigmal3 / (sigmal3 + rvar)

    dxdr = dxdr112 + dxdr123 + dxdr131 + dxdr21 + dxdr22 + dxdr23

    return (xhat, dxdr, var)


def shrink_gm4(r, rvar, theta):
    print('In shrink_gm4 ...')

    w1 = tf.abs(theta[0])
    mul1 = theta[1]
    sigmal1 = tf.abs(theta[2])

    w2 = tf.abs(theta[3])
    mul2 = theta[4]
    sigmal2 = tf.abs(theta[5])

    w3 = tf.abs(theta[6])
    mul3 = theta[7]
    sigmal3 = tf.abs(theta[8])

    w4 = tf.abs(theta[9])
    mul4 = theta[10]
    sigmal4 = tf.abs(theta[11])

    sum_w = w1 + w2 + w3 + w4

    w1 = w1 / sum_w
    w2 = w2 / sum_w
    w3 = w3 / sum_w
    w4 = w4 / sum_w



    tfd = tfp.distributions
    N1 = tfd.Normal(loc=mul1, scale=tf.sqrt(sigmal1 + rvar))
    N2 = tfd.Normal(loc=mul2, scale=tf.sqrt(sigmal2 + rvar))
    N3 = tfd.Normal(loc=mul3, scale=tf.sqrt(sigmal3 + rvar))
    N4 = tfd.Normal(loc=mul4, scale=tf.sqrt(sigmal4 + rvar))

    beta1 = w1 * N1.prob(r)
    beta2 = w2 * N2.prob(r)
    beta3 = w3 * N3.prob(r)
    beta4 = w4 * N4.prob(r)

    beta1_bar = beta1 / (beta1 + beta2 + beta3 + beta4)
    beta2_bar = beta2 / (beta1 + beta2 + beta3 + beta4)
    beta3_bar = beta3 / (beta1 + beta2 + beta3 + beta4)
    beta4_bar = beta4 / (beta1 + beta2 + beta3 + beta4)

    gamma1 = (mul1 * rvar + r * sigmal1) / (sigmal1 + rvar)
    gamma2 = (mul2 * rvar + r * sigmal2) / (sigmal2 + rvar)
    gamma3 = (mul3 * rvar + r * sigmal3) / (sigmal3 + rvar)
    gamma4 = (mul4 * rvar + r * sigmal4) / (sigmal4 + rvar)

    nu1 = (rvar * sigmal1) / (sigmal1 + rvar)
    nu2 = (rvar * sigmal2) / (sigmal2 + rvar)
    nu3 = (rvar * sigmal3) / (sigmal3 + rvar)
    nu4 = (rvar * sigmal4) / (sigmal4 + rvar)

    expectation = beta1_bar * gamma1 + beta2_bar * gamma2 + beta3_bar * gamma3 + beta4_bar * gamma4
    power = beta1_bar * (nu1 + tf.square(gamma1)) + beta2_bar * (nu2 + tf.square(gamma2)) + beta3_bar * (nu3 + tf.square(gamma3)) + beta4_bar * (nu4 + tf.square(gamma4))

    var = power - tf.square(expectation)

    xhat = expectation

    rmmos21 = (r - mul1) / (sigmal1 + rvar)
    rmmos22 = (r - mul2) / (sigmal2 + rvar)
    rmmos23 = (r - mul3) / (sigmal3 + rvar)
    rmmos24 = (r - mul4) / (sigmal4 + rvar)

    dxdr1_12 = beta1_bar * beta2_bar * (rmmos22 - rmmos21) * (gamma1 - gamma2)
    dxdr1_13 = beta1_bar * beta3_bar * (rmmos23 - rmmos21) * (gamma1 - gamma3)
    dxdr1_14 = beta1_bar * beta4_bar * (rmmos24 - rmmos21) * (gamma1 - gamma4)
    dxdr1_23 = beta2_bar * beta3_bar * (rmmos23 - rmmos22) * (gamma2 - gamma3)
    dxdr1_24 = beta2_bar * beta4_bar * (rmmos24 - rmmos22) * (gamma2 - gamma4)
    dxdr1_34 = beta3_bar * beta4_bar * (rmmos24 - rmmos23) * (gamma3 - gamma4)
    dxdr1 = dxdr1_12 + dxdr1_13 + dxdr1_14 + dxdr1_23 + dxdr1_24 + dxdr1_34

    dxdr2_1 = beta1_bar * sigmal1 / (sigmal1 + rvar)
    dxdr2_2 = beta2_bar * sigmal2 / (sigmal2 + rvar)
    dxdr2_3 = beta3_bar * sigmal3 / (sigmal3 + rvar)
    dxdr2_4 = beta4_bar * sigmal4 / (sigmal4 + rvar)
    dxdr2 = dxdr2_1 + dxdr2_2 + dxdr2_3 + dxdr2_4

    dxdr = dxdr1 + dxdr2

    return (xhat, dxdr, var)

def shrink_ternary(r, rvar, theta):
    print('In shrink_ternary ...')

    omega = (0.05, 0.05, 0.9)
    mu = (-1.0, 1.0, 0.0)
    return shrink_dd(r, rvar, theta, omega, mu)


def shrink_binary(r, rvar, theta):
    print('In shrink_binary ...')

    omega = (0.5, 0.5, 0.0)
    mu = (-1.0, 1.0, 0.0)
    return shrink_dd(r, rvar, theta, omega, mu)

def shrink_dd(r, rvar, theta, omega, mu):

    print('In shrink_dd ...')

    if len(theta.get_shape())>0 and theta.get_shape() != (1,):
        # sigma = theta[0] * tf.sqrt(rvar)
        sigma = 1.0 * tf.sqrt(rvar)
        scale = theta[1]
    else:
        sigma = theta * tf.sqrt(rvar)
        scale = None

    w1 = omega[0]
    w2 = omega[1]
    w3 = omega[2]

    mu1 = mu[0]
    mu2 = mu[1]
    mu3 = mu[2]

    tfd = tfp.distributions
    N1 = tfd.Normal(loc=mu1, scale=sigma)
    N2 = tfd.Normal(loc=mu2, scale=sigma)
    N3 = tfd.Normal(loc=mu3, scale=sigma)

    beta1 = w1 * N1.prob(r)
    beta2 = w2 * N2.prob(r)
    beta3 = w3 * N3.prob(r)

    denom = beta1 + beta2 + beta3
    beta1_bar = beta1 / denom
    beta2_bar = beta2 / denom
    beta3_bar = beta3 / denom

    xhat = mu1*beta1_bar + mu2*beta2_bar + mu3*beta3_bar

    dxdr1 = (xhat - mu1) * (r - mu1) * beta1_bar
    dxdr2 = (xhat - mu2) * (r - mu2) * beta2_bar
    dxdr3 = (xhat - mu3) * (r - mu3) * beta3_bar
    dxdr = 1/tf.square(sigma)*(dxdr1 + dxdr2 + dxdr3)

    if scale is not None:
        xhat = xhat*scale
        dxdr = dxdr*scale

    var = 0 # this needs to be implemented
    return (xhat, dxdr, var)


def get_shrinkage_function(name, prob):
	"retrieve a shrinkage function and some (probably awful) default parameter values"
	try:
		return {
			'soft':(shrink_soft_threshold,(1.,1.) ),
			'bg':(shrink_bgest, (1,math.log(1/prob.config.pnz-1+sys.float_info.epsilon)) ),
			'pwlin':(shrink_piecwise_linear, (2,4,0.1,1.5,.95) ),
			'pwgrid':(pwlin_grid, np.linspace(.1,1,15).astype(np.float32)  ),
			'expo':(shrink_expo, (2.5,.9,-1) ),
			'spline':(shrink_spline, (3.7,.9,-1.5)),
            'ternary': (shrink_ternary, (1., 1.)),
            'binary': (shrink_binary, (1., 1.)),
            'gm2': (shrink_gm2, (0.5, -0.5, 3.36e-01, 0.5, 1)),
            'gm3': (shrink_gm3, (0.5, -.5, 1, 0.2, 0.2, 0.2, .5, 1)),
            'gm4': (shrink_gm4, (5.0e-02, -1.00e+00, -2.43e-07, 2.95e+00, 1.32e-07, 2.67e-08, -5.00e-02, 1.00e+00, -6.54e-07, 3.87e-04, 6.28e+00, 6.72e-01))
		}[name]
	except KeyError:
		raise ValueError('unrecognized shrink function %s' % name)
		sys.exit(1)

def tfcf(v):
    " return a tensorflow constant float version of v"
    return tf.constant(v,dtype=tf.float32)

def tfvar(v):
    " return a tensorflow variable float version of v"
    return tf.Variable(v,dtype=tf.float32)

def nmse(x1,x2):
    "return the normalized mean squared error between 2 numpy arrays"
    xdif=x1-x2
    return 2*(xdif*xdif).sum() / ( (x1*x1).sum() + (x2*x2).sum())

def test_func(shrink_func,theta,**kwargs):
    # repeat the same experiment
    tf.reset_default_graph()
    tf.set_random_seed(kwargs.get('seed',1) )

    N = kwargs.get('N',200)
    L = kwargs.get('L',400)
    tol = kwargs.get('tol',1e-6)
    step = kwargs.get('step',1e-4)
    shape = (N,L)
    xvar_ = tfcf(kwargs.get('xvar1',1))
    pnz_ = tfcf(kwargs.get('pnz',.1))
    rvar = np.ones(L)*kwargs.get('rvar',.1)
    rvar_ = tfcf(rvar)
    gx = tf.to_float(tf.random_uniform(shape ) < pnz_) * tf.random_normal(shape, stddev=tf.sqrt(xvar_), dtype=tf.float32)
    gr = gx + tf.random_normal(shape,stddev=tf.sqrt(rvar_), dtype=tf.float32)
    x_ = tf.placeholder(gx.dtype,gx.get_shape())
    r_ = tf.placeholder(gr.dtype,gr.get_shape())

    theta_ = tfvar(theta)

    xhat_,dxdr_ = shrink_func(r_,rvar_ ,theta_)
    loss = tf.nn.l2_loss(xhat_-x_)
    optimize_theta = tf.train.AdamOptimizer(step).minimize(loss,var_list=[theta_])

    # calculate an empirical gradient for comparison
    dr_ = tfcf(1e-4)
    dxdre_ = tf.reduce_mean( (shrink_func(r_+.5*dr_,rvar_ ,theta_)[0] - shrink_func(r_-.5*dr_,rvar_ ,theta_)[0]) / dr_ ,0)

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        (x,r) = sess.run((gx,gr))
        fd = {x_:x,r_:r}
        loss_prev = float('inf')
        for i in range(500):
            for j in range(50):
                sess.run(optimize_theta,fd)
            loss_cur,theta_cur = sess.run((loss,theta_),fd)
            #print 'loss=%s, theta=%s' % (str(loss_cur),str(theta_cur))
            if (1-loss_cur/loss_prev) < tol:
                break
            loss_prev = loss_cur
        xhat,dxdr,theta,dxdre = sess.run( (xhat_,dxdr_,theta_,dxdre_),fd)

    assert xhat.shape==(N,L)
    assert dxdr.shape==(L,) # MMV-specific -- we assume one average gradient per column
    assert nmse(dxdr,dxdre) < tol

    tf.reset_default_graph()
    estname = re.sub('.*shrink_([^ ]*).*','\\1', repr(shrink_func) )
    print('####  ' +  repr(estname) + ' loss=' + repr(loss_cur) + ' \ttheta=' % + repr(theta))
    if False:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(r.reshape(-1),xhat.reshape(-1),'b.')
        plt.plot(r,xhat,'.')
        plt.show()
    return (x,r,xhat,rvar)


def show_shrinkage(shrink_func,theta,**kwargs):
    tf.reset_default_graph()
    tf.set_random_seed(kwargs.get('seed',1) )

    N = kwargs.get('N',500)
    L = kwargs.get('L',4)
    nsigmas = kwargs.get('sigmas',10)
    shape = (N,L)
    rvar = 1e-4
    r = np.reshape( np.linspace(0,nsigmas,N*L)*math.sqrt(rvar),shape)
    r_ = tfcf(r)
    rvar_ = tfcf(np.ones(L)*rvar)

    xhat_,dxdr_ = shrink_func(r_,rvar_ ,tfcf(theta))

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        xhat = sess.run(xhat_)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(r.reshape(-1),r.reshape(-1),'y')
    plt.plot(r.reshape(-1),xhat.reshape(-1),'b')
    if kwargs.has_key('title'):
        plt.suptitle(kwargs['title'])
    plt.show()


if __name__ == "__main__":
    import sys
    import getopt
    usage = """
    -h : help
    -p file : load problem definition parameters from npz file
    -f function : use the named shrinkage function, one of {soft,bg,pwlin,expo,spline}
    """
    try:
        opts,args = getopt.getopt(sys.argv[1:] , 'hp:s:f:')
        opts = dict(opts)
    except getopt.GetoptError:
        opts={'-h':True}
    if opts.has_key('-h'):
        sys.stderr.write(usage)
        sys.exit()

    shrinkage_name = opts.get('-f','soft')
    f,theta = get_shrinkage_function( shrinkage_name )
    if opts.has_key('-s'):
        D=dict(np.load(opts['-s']).items())
        t=0
        while D.has_key('theta_%d'% t):
            theta_t = D['theta_%d' % t]
            show_shrinkage(f,theta_t,title='shrinkage=%s, theta_%d=%s' % (shrinkage_name,t, theta_t))
            t += 1
    else:
        show_shrinkage(f,theta)

    """
    test_func(shrink_bgest, (1,math.log(1/.1-1)) ,**parms)
    test_func(shrink_soft_threshold,(1.7,1.2) ,**parms)
    test_func(shrink_piecwise_linear, (2,4,0.1,1.5,.95) ,**parms)
    test_func(shrink_expo, (2.5,.9,-1) ,**parms)
    test_func(shrink_spline, (3.7,.9,-1.5) ,**parms)
    """
