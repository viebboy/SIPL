#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import os, sys, getopt, Runners, pickle
import exp_configurations as Configuration



def main(argv):

    try:
      opts, args = getopt.getopt(argv,"h", ['model=', 'index='])
    
    except getopt.GetoptError:
        sys.exit(2)
    
    for opt, arg in opts:             
        if opt == '--index':
            index = int(arg)
        if opt == '--model':
            model = arg
            
    assert model in ['PMLP', 'StackedELM', 'BLS', 'PLN', 'baseline'], 'Given model (%s) not supported' % model
    with open('%s_missing.pickle' % model, 'rb') as fid:
        leftover_configurations = pickle.load(fid)['configurations']
    
    conf = leftover_configurations[index]
    
    filename = '_'.join([str(v) for v in conf]) + '.pickle'
    filename = os.path.join(Configuration.output_dir, filename)
    
    if os.path.exists(filename):
        return
    
    if model == 'PMLP':
        runner = Runners.train_PMLP
    elif model == 'StackedELM':
        runner = Runners.train_StackedELM
    elif model == 'BLS':
        runner = Runners.train_BLS
    elif model == 'PLN':
        runner = Runners.train_PLN
    else:
        runner = Runners.train_baseline
    print(conf) 
    outputs = runner(conf)
    
    fid = open(filename, 'wb')
    pickle.dump(outputs, fid)
    fid.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
