#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import os, itertools, pickle
import exp_configurations as Conf
import sys, getopt

def create_configuration(names, hp_list):
    outputs_ = []
    
    for hp in hp_list:
        tmp = []
        for name in names:
            tmp.append(hp[name])
        outputs_ += list(itertools.product(*tmp))
        
    outputs = []
    for conf in outputs_:
        if conf not in outputs:
            outputs.append(conf)
    
    return outputs

def get_configurations(model):
    if model == 'PLN':
        names = Conf.PLN_names
        hp_list = [Conf.PLN_values]
    elif model == 'StackedELM':
        names = Conf.StackedELM_names
        hp_list = [Conf.StackedELM_values]
    elif model == 'PMLP':
        names = Conf.PMLP_names
        hp_list = [Conf.PMLP_values_proposal, Conf.PMLP_values_original]
    else:
        raise RuntimeError('unknown model')
        
    return create_configuration(names, hp_list)

def inspect_result(model):
    missing = []
    configurations = get_configurations(model)
    for conf in configurations:
        filename = '_'.join([str(v) for v in conf]) + '.pickle'
        filename = os.path.join(Conf.output_dir, filename)
        if not os.path.exists(filename):
            missing.append(conf)

    print('model: %s' % str(model))
    print('total configurations: %d' % len(configurations))
    print('missing configurations: %d' % len(missing))
    
    fid = open('%s_missing.pickle' % model, 'wb')
    pickle.dump({'configurations': missing}, fid)
    fid.close()
    
    return len(missing)

def main(argv):

    try:
      opts, args = getopt.getopt(argv,"h", ['model=', ])
    
    except getopt.GetoptError:
        sys.exit(2)
    
    for opt, arg in opts:             
        if opt == '--model':
            model = arg
            
    nb_jobs = inspect_result(model)
    for index in range(nb_jobs):
        os.system('python train_model_standalone.py --model %s --index %d' %(model, index))
    
if __name__ == "__main__":
    main(sys.argv[1:])
