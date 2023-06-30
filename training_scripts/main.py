#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:55:50 2019

@author: billault
"""
from dataset import Dataset
from modelRun import NNRun, XGBoostRun, LinRegRun
import yaml
import glob
import os

# architectures to try out 
neuronslist = [50,60,70,80,100,120,150,170,200]
layerslist = [4,5,6,7]
epochslist = [70,90]

# what should be saved
save_training_curve = False
save_plots = False
save_model = True

for path in sorted(glob.glob('PWV_retrieval_11/ERA/noGeo/noSurf/noTB34/NN/')):
    
    # load config file and training parameters
    ymlpath = path+'config.yaml'
    with open(ymlpath,'r') as file:
        variables = yaml.load(file,Loader=yaml.FullLoader)
    dataset_path = variables['dataset_path']
    dataset_parameters = variables['dataset_parameters']
    input_features = variables['input_features']
    target = variables['target']
        
    # some stations are problematic and should be removed
    frefpb = '/home/billault/Documents/LWPretrieval/code_propre/consistency_check/ref_pb.txt'
    with open(frefpb, 'r') as f:
        refpb = f.readlines()
    refpb = [r[:-1] for r in refpb]
    
    # create dataset instance
    dataset = Dataset(input_features,dataset_parameters,target=target,path=dataset_path,auto=True,normalize=True,savemean=path,addNoiseToTB=True,refpb=refpb)

    # loop over possible architectures
    for neurons in neuronslist:
        for layers in layerslist:
            for epochs in epochslist:
                modelpath =path+'nn%d_nl%d_ep%d'%(neurons,layers,epochs)
                if os.path.exists(modelpath):
                    continue
                model_parameters = {'epochs':epochs,
                         'neurons':neurons,
                         'layers':layers,
                         'loss':'mse',
                         'batch_size':512,
                         'activation':'relu',
                         'use_bias':0,
                         'optimizer':'rmsprop'} 

                model_run = NNRun(dataset,model_parameters)
                model_run.build()
                model_run.train()

                if save_training_curve:
                    model_run.plotTrainingCurve(savepath=path)
                if save_plots:
                    model_run.plots(savepath=path)
                if save_model:
                    model_run.model.save(modelpath)
