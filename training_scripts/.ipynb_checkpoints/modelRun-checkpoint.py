#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:21:07 2019

@author: billault
"""
import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ModelRun:
    
    def __init__(self,dataset,ml_type,model_parameters):
        [self.training_input,self.training_target] = dataset.splitdata['training']
        [self.validation_input,self.validation_target] = dataset.splitdata['validation']
        [self.testing_input,self.testing_target] = dataset.splitdata['testing']
        print('Training on: '+str(len(self.training_target)))
        print('Validation on: '+str(len(self.validation_target)))
        print('Testing on: '+str(len(self.testing_target)))
        self.dataset = dataset
        self.ml_type = ml_type
        self.parameters = model_parameters
        self.model = []
        self.annotation = ''
        self.specs = ''
        self.target=dataset.target
    
    def plots(self,plotset='val',savepath=''):
        if savepath == 'None':
            self.plotPredTargBias(plotset,savepath='None')
            self.plotPredTargDistrib(plotset,savepath='None')
            self.plotPredTargRMSE(plotset,savepath='None')
        else:
            self.plotPredTargBias(plotset,savepath=savepath+'pred_vs_targ/')
            self.plotPredTargDistrib(plotset,savepath=savepath+'pred_vs_targ/')
            self.plotPredTargRMSE(plotset,savepath=savepath+'pred_vs_targ/')
        
        
    def plotPredTargDistrib(self,plotset='val',savepath=''):
        if plotset=='val':
            pred = self.model.predict(self.validation_input).reshape(-1)
            targ = self.validation_target
        elif plotset == 'test':
            pred = self.model.predict(self.testing_input).reshape(-1)
            targ = self.testing_target   
        elif plotset == 'train':
            pred = self.model.predict(self.training_input).reshape(-1)
            targ = self.training_target   

        xx = [0,self.dataset.parameters['upper_thres']]
        nbins = 20
        means,tmp,tmp = stats.binned_statistic(targ,pred,'mean',nbins)
        stds,tmp,tmp = stats.binned_statistic(targ,pred,'std',nbins)
        meanst,tmp,tmp = stats.binned_statistic(targ,targ,'mean',nbins)
        mse,tmp,tmp = stats.binned_statistic(targ,(pred-targ)**2,'mean',nbins)
        
        fig = plt.figure()
        plt.fill_between(meanst,means-stds,means+stds,color='lightgray',alpha=0.7,label='std')
        plt.plot(meanst,means,'-k',label='mean')
        plt.plot(meanst,means-stds,color='gray',linewidth=1)
        plt.plot(meanst,means+stds,color='gray',linewidth=1)    
        plt.plot(xx,xx,'--',linewidth=1,label='y=x')
        plt.legend(loc='lower right')
        if self.target == 'LWP':
            plt.xlim(0,1000)
            plt.ylim(0,1000)
            plt.title('Distribution of predicted LWP',fontsize=14)
            plt.xlabel('target LWP (g/m²)')
            plt.ylabel('predicted LWP (g/m²)')
        elif self.target == 'PWV':
            plt.xlim(0,85)
            plt.ylim(0,85)
            plt.title('Distribution of predicted PWV',fontsize=14)
            plt.xlabel('target PWV (kg/m²)')
            plt.ylabel('predicted PWV (kg/m²)')
        plt.annotate(self.annotation,(0.02,0.78),xycoords='axes fraction',fontsize=8)
        if savepath != 'None':
            name = savepath + 'Pred_vs_targ_distrib_'+plotset+'_'+self.specs+'.png'
            fig.savefig(name,dpi=150,bbox_inches='tight')
            plt.close()


    def plotPredTargRMSE(self,plotset='val',savepath=''):
        if plotset=='val':
            pred = self.model.predict(self.validation_input).reshape(-1)
            targ = self.validation_target
        elif plotset == 'test':
            pred = self.model.predict(self.testing_input).reshape(-1)
            targ = self.testing_target   
        elif plotset == 'train':
            pred = self.model.predict(self.training_input).reshape(-1)
            targ = self.training_target   
        nbins = 20
        mse,tmp,tmp = stats.binned_statistic(targ,(pred-targ)**2,'mean',nbins)
        meanst,tmp,tmp = stats.binned_statistic(targ,targ,'mean',nbins)

        fig = plt.figure()
        plt.plot(meanst,mse**.5,'-k')
        if self.target == 'LWP':
            plt.title('RMSE vs. LWP',fontsize=14)
            plt.xlim(0,1000)
            plt.xlabel('target LWP (g/m²)',fontsize=12)
            plt.ylim(0,200)
            plt.ylabel('rmse (g/m²)',fontsize=12)
        elif self.target == 'PWV':
            plt.title('RMSE vs. PWV',fontsize=12)
            plt.xlim(0,85)
            plt.ylim(0,50)
            plt.xlabel('target PWV (kg/m²)')
            plt.ylabel('rmse (kg/m²)')
        plt.annotate(self.annotation,(0.02,0.78),xycoords='axes fraction',fontsize=8)
        if savepath != 'None':
            name = savepath + 'Pred_vs_targ_rmse_'+plotset+'_'+self.specs+'.png'
            fig.savefig(name,dpi=150,bbox_inches='tight')
            plt.close(fig)


    def plotPredTargBias(self,plotset='val',savepath=''):
        if plotset=='val':
            pred = self.model.predict(self.validation_input).reshape(-1)
            targ = self.validation_target
        elif plotset == 'test':
            pred = self.model.predict(self.testing_input).reshape(-1)
            targ = self.testing_target   
        elif plotset == 'train':
            pred = self.model.predict(self.training_input).reshape(-1)
            targ = self.training_target   
        nbins = 20
        means,tmp,tmp = stats.binned_statistic(targ,pred-targ,'mean',nbins)
        meanst,tmp,tmp = stats.binned_statistic(targ,targ,'mean',nbins)

        fig = plt.figure()
        plt.plot(meanst,means,'-k')
        if self.target == 'LWP':
            plt.xlim(0,1000)
            plt.xlabel('target LWP (g/m²)')
            plt.ylabel('mean bias (LWP$_{pred}$-LWP$_{targ}$) (g/m²)')
            plt.ylim(-250,100)
            plt.title('Mean bias vs. LWP')
        elif self.target == 'PWV':
            plt.xlim(0,85)
            plt.xlabel('target LWP (g/m²)')
            plt.ylabel('mean bias (LWP$_{pred}$-LWP$_{targ}$) (g/m²)')
            plt.ylim(-100,50)
            plt.title('Mean bias vs. LWP')

        plt.annotate(self.annotation,(0.02,0.78),xycoords='axes fraction',fontsize=8)
        if savepath != 'None':
            name = savepath + 'Pred_vs_targ_bias_'+plotset+'_'+self.specs+'.png'
            fig.savefig(name,dpi=150,bbox_inches='tight')
            plt.close(fig)
  
        
class NNRun(ModelRun):
    def __init__(self,dataset,nn_parameters):
        ModelRun.__init__(self,dataset,'neural_network',nn_parameters)
        self.neurons = nn_parameters['neurons']
        self.layers = nn_parameters['layers']
        self.loss = nn_parameters['loss']
        self.epochs = nn_parameters['epochs']
        self.batch_size = nn_parameters['batch_size']
        self.activation = nn_parameters['activation']
        self.use_bias = nn_parameters['use_bias']
        self.optimizer = nn_parameters['optimizer']

        self.train_loss = np.array([])
        self.train_mse = np.array([])
        self.train_mae = np.array([])
        self.train_msle = np.array([])
        self.val_loss = np.array([])
        self.val_mse = np.array([])
        self.val_mae = np.array([])
        self.val_msle = np.array([])
        self.model = keras.models.Sequential()
    
        self.annotation = 'Loss = '+self.loss+'\n'+'# layers = %d\n# neurons = %d\nupper thres =%d g/m²\nlower thres = %d g/m²'%(self.layers,
                            self.neurons,self.dataset.parameters['upper_thres'],self.dataset.parameters['lower_thres'])
        self.specs = self.loss+'_nl%dbs%dnn%dut%dlt%dep%d'%(self.layers,self.batch_size,self.neurons,self.dataset.parameters['upper_thres'],self.dataset.parameters['lower_thres'],self.epochs)

    def build(self):
        print('building...')
        if self.layers>1:
            self.model.add(keras.layers.Dense(self.neurons,activation=self.activation,use_bias=self.use_bias,input_shape=(self.training_input.shape[1],)))
            for n in range(2,self.layers):
                self.model.add(keras.layers.Dense(self.neurons,activation=self.activation,use_bias=self.use_bias))
        self.model.add(keras.layers.Dense(1,use_bias=self.use_bias,activation=self.activation))
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=['mae','mse','msle'])
        print('OK \n')
    
    
    def train(self):
        print('training...')
        history = self.model.fit(self.training_input,self.training_target,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data = (self.validation_input,self.validation_target))#,

        self.train_loss = np.array(history.history['loss'])
        self.train_mse = np.array(history.history['mse'])
        self.train_mae = np.array(history.history['mae'])
        self.train_msle = np.array(history.history['msle'])
        self.val_loss = np.array(history.history['val_loss'])
        self.val_mse = np.array(history.history['val_mse'])
        self.val_mae = np.array(history.history['val_mae'])
        self.val_msle = np.array(history.history['val_msle'])
        
        print('OK\n')
        
    def plotTrainingCurve(self,savepath=''):
        fig,ax = plt.subplots()    
        ax.plot(range(1,self.epochs+1),self.train_loss,label='Training')
        ax.plot(range(1,self.epochs+1),self.val_loss,label='Validation')
        ax.legend(fontsize=8)
        ax.set_xlabel('Epochs',fontsize=11)
        ax.set_ylabel(self.loss,fontsize=11)
        ax.annotate('min val mae = %.1f g/m²\nmin val rmse= %.1f g/m²\nmin train mae=%.1f g/m²\nmin train rmse=%.1f g/m²'%(min(self.val_mae), 
                    (min(self.val_mse))**.5, min(self.train_mae),(min(self.train_mse))**.5),xy = (0.68,0.7),xycoords = 'axes fraction',fontsize=8)
        ax.set_title('Error during training phase',fontsize=14)
        if savepath != 'None':
            name= savepath+'TC'+self.specs+'.png'
            fig.savefig(name,dpi=150,bbox_inches='tight')
            plt.close()
    
