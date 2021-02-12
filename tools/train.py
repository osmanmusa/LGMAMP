#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
import time

def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,prob):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """

    trinit = prob.config.trinit
    refinements = prob.config.refinements
    final_refine = prob.config.final_refine

    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    maskX_ = getattr(prob,'maskX_',1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ *maskX_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,var_list in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        nmse_  = tf.nn.l2_loss( (xhat_ - prob.x_)*maskX_) / nmse_denom_
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,train2_,()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,()) )

    return training_stages

def do_LGMAMP_training(training_stages, prob, config):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, config.file_name)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))

    theta_current = None

    for name, xhat_, loss_, nmse_, train_, var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue

        if len(var_list) == 1:
            if str.startswith(var_list[0].name, 'theta_'):
                theta_current = var_list[0]

        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])

            for v in var_list:
                # if str.startswith(v.name, 'theta_') and not str.startswith(v.name, 'theta_0'):
                if str.startswith(v.name, 'theta_') and not str.endswith(v.name, '0:0'):
                    v_value_init = sess.run(v)
                    for trainable_var in tf.trainable_variables():
                        layer_index = int(v.name[6:str.find(v.name,':')])
                        name_last = 'theta_' + str(layer_index - 1)
                        if str.startswith(trainable_var.name, name_last):
                            theta_last = trainable_var
                            break

                    assign_op = v.assign(theta_last)
                    sess.run(assign_op)

                    v_value_new = sess.run(v)
                    print('\n' + v.name + 'Initial value and the new one are : ')
                    print(['{:.2e}'.format(f) for f in v_value_init])
                    print(['{:.2e}'.format(f) for f in v_value_new])
                    break
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])

        print('\n' + name + ' ' + describe_var_list)
        nmse_history = []
        for i in range(config.maxit + 1):

            if i % config.ivl == 0:
                nmse = sess.run(nmse_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
                nmse_history = np.append(nmse_history, nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(nmse_history.min())

                if np.isnan(nmse):
                    for trainable_var in tf.trainable_variables():
                        if str.startswith(trainable_var.name,'theta_'):
                            print(trainable_var.name + ': ')
                            print(sess.run(trainable_var))

                    raise RuntimeError('nmse is NaN')

                if theta_current is not None:
                    v_value = sess.run(theta_current)
                    v_value_str = ['{:.2e}'.format(f) for f in v_value]
                    sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})  \t'.format(i=i, nmse=nmse_dB,
                                        best=nmsebest_dB) + theta_current.name + ': ' + str(v_value_str))
                else:
                    sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i, nmse=nmse_dB, best=nmsebest_dB))

                sys.stdout.flush()
                if i % (100 * config.ivl) == 0:

                    if theta_current is not None:
                        v_value = sess.run(theta_current)
                        v_value_str = ['{:.2e}'.format(f) for f in v_value]
                        sys.stdout.write(
                            '\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})  \t'.format(i=i, nmse=nmse_dB,
                                                                                          best=nmsebest_dB) + theta_current.name + ': ' + str(
                                v_value_str))
                    else:
                        sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i, nmse=nmse_dB,
                                                                                                   best=nmsebest_dB))

                    sys.stdout.flush()

                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1  # how long ago was the best nmse?
                    if age_of_best * config.ivl > config.better_wait:
                        break  # if it has not improved on the best answer for quite some time, then move along

            y, x = prob(sess)
            sess.run(train_, feed_dict={prob.y_: y, prob.x_: x})

        done = np.append(done, name)
        if theta_current is not None:
            v_value = sess.run(theta_current)
            v_value_str = ['{:.2e}'.format(f) for f in v_value]
            log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations with '.format(name=name, nmse=nmse_dB, i=i) + theta_current.name + ': ' + str(v_value_str)
        else:
            log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i)

        # evaluating nmse on test data
        nmse_testing = sess.run(nmse_, feed_dict={prob.y_: prob.ytest, prob.x_: prob.xtest})
        nmse_testing_dB = 10 * np.log10(nmse_testing)
        log = log + '\n{name} nmse={nmse:.6f} dB  <---- TEST'.format(name=name, nmse=nmse_testing_dB)
        sys.stdout.write('\n{name} nmse={nmse:.6f} dB  <---- TEST'.format(name=name, nmse=nmse_testing_dB))

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess, config.file_name, **state)

    print("\nPrinting log")
    print(log)
    print("EOF printing log")

    return sess


def do_training(training_stages,prob,savefile,ivl=10,maxit=1000000,better_wait=5000):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        start = time.time()
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(100*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y,prob.x_:x} )
        done = np.append(done,name)
        
        end = time.time()
        time_log = 'Took me {totaltime:.3f} minutes, or {time_per_interation:.1f} ms per iteration'.format(totaltime = (end-start)/60, time_per_interation = (end-start)*1000/i)
        print(time_log)
        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)
    return sess

def evaluate_nmse(sess, training_stages, prob, config):
    import math

    pnz = config.pnz
    SNR = config.SNR
    L = config.L_testing # testing data set size
    A = prob.A
    M,N = A.shape

    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)

    if prob.name.startswith('Bernoulli-Gaussian'):
        xtest = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
        ytest = np.matmul(A, xtest) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    elif prob.name.startswith('Bernoulli-Bernoulli'):
        xtest = ((np.random.uniform(0, 1, (N, L)) < pnz) * np.sign(np.random.normal(0, 1, (N, L)))).astype(np.float32)
        ytest = np.matmul(A, xtest) + np.random.normal(0, math.sqrt(noise_var), (M, L))
    elif prob.name.startswith('Binary'):
        xtest = (np.random.uniform(0, 1, (N, L)) < pnz).astype(np.float32)
        ytest = np.matmul(A, xtest) + np.random.normal(0, math.sqrt(noise_var), (M, L))
    else:
        print('------- Unknown problem name -------')
        sys.exit()

    nmse_dB_arrray = np.array([])
    ser_arrray = np.array([])

    for name, xhat_, loss_, nmse_, train_, var_list in training_stages:

        if (str.find(name, 'non-linear') != -1 and str.find(name, 'trainrate='+str(config.refinements[-1])) != -1) or (name.startswith('LVAMP') and str.find(name, 'trainrate='+str(config.refinements[-1])) != -1):

            nmse = sess.run(nmse_, feed_dict={prob.y_: ytest, prob.x_: xtest})
            nmse_dB = 10 * np.log10(nmse)
            nmse_dB_arrray = np.append(nmse_dB_arrray, nmse_dB)

            if prob.name.startswith('Bernoulli-Bernoulli') or prob.name.startswith('Binary'):
                xhat = sess.run(xhat_, feed_dict={prob.y_: ytest, prob.x_: xtest})
                x_hat_hard = np.rint(xhat)
                ser = np.sum(np.abs(xtest - x_hat_hard) > 0.5) / (N * L)  # symbol error rate
                print(name +  '\tnmse/dB={nmse_dB:.6f};  ser={ser:.7f}'.format(nmse_dB=nmse_dB,ser=ser))
                ser_arrray = np.append(ser_arrray, ser)
            else:
                print(name +  '\tnmse/dB={nmse_dB:.6f}'.format(nmse_dB=nmse_dB))



    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('nmse/dB=', nmse_dB_arrray)
    if prob.name.startswith('Bernoulli-Bernoulli') or prob.name.startswith('Binary'):
        np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
        print('ser = ', ser_arrray)


    state = load_trainable_vars(sess, config.file_name)  # must load AFTER the initializer
    log = str(state.get('log', ''))

    test_nmse = ''
    log_entries = log.split('\n')

    print('Printing log entries that contain the word TEST at the end ...')
    for entry in log_entries:
        if str.find(entry, 'non-linear') != -1 and str.find(entry, 'trainrate='+str(config.refinements[-1])) != -1 and str.endswith(entry,
                                                                                                            'TEST'):
            print(entry)
            index_start = str.find(entry, 'nmse=') + str.__len__('nmse=')
            index_end = entry.find('.',index_start, len(entry)) + 4
            test_nmse += entry[index_start:index_end]
            test_nmse += ' '

    print('TEST of ' + prob.name + ' nmse/dB = [', test_nmse, '];')