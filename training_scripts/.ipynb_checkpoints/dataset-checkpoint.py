#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:22:46 2020

@author: billault
"""

import yaml
import glob
import numpy as np
import datetime
from keras.models import load_model
import sys
import random 
#import xgboost

class Dataset:
    
    def __init__(self,variables,parameters,target='LWP',
                 path='/',auto=False,normalize=True,savemean=None,
                 withoutERAforLWP=False, ml_type='NN',refpb=[],addNoiseToTB=False):
        self.variables=variables # list of input variables, e.g. ['TB','latitude','longitude','altitude','date','Ts']
        self.parameters = parameters # dictionary with pre-processing parameters, e.g. {'upper_thres': 1500, 'lower_thres':15, 'lo}
        filenames = sorted(glob.glob(path))
        self.target = target
        self.data = np.array([])
        self.splitdata = np.array([])      
        self.LWP = np.array([])
        self.PWV = np.array([])
        self.ml_type = ml_type
        
        if 'latitude' in self.variables:
            self.latitude = np.array([])
        if 'longitude' in self.variables:
            self.longitude = np.array([])
        if 'altitude' in self.variables:
            self.altitude = np.array([])
        if 'date' in self.variables:
            self.date = np.array([])
        if 'Ts' in self.variables:
            self.Ts = np.array([])
        if 'Ps' in self.variables:
            self.Ps = np.array([])
        if 'RHs' in self.variables:
            self.RHs = np.array([])
        if 'PWVpred' in self.variables:
            self.PWVpred = np.array([])
        if 'LWPera' in self.variables:
            self.LWPera = np.array([])
        if 'PWVera' in self.variables:
            self.PWVera = np.array([])
        if 'TB' in self.variables:
            self.TB = np.array([])
        else:
            print('Warning: TB is not part of the input variables')
            
        
        if (len(filenames) > 0) & auto:
            self.loadVariables(filenames,refpb=refpb)
            if 'PWVpred' in self.variables:
                self.loadPWVpred(modeldirectory = self.parameters['PWVmodeldirectory'])
                print('loading specified model')
            if (('TB' in self.variables) & (addNoiseToTB)):
                self.addTBStdNoise(std=.5)
            self.preprocess()
            self.wrapup(withoutERAforLWP)
            self.splitDataset()
            if normalize:
                self.meanNormalize(savemean)
            
        
        
    ############### load input variables into dataset ###############
        
    def loadVariables(self,filenames,refpb=[]):
        print('loading variables...')
        latitude = []
        longitude = []
        altitude = []
        date = []
        Ts = []
        Ps = []
        RHs = []
        PWV = []
        LWPera = []
        PWVera = []
        TB = []
        hcb = []
        LWP = []
        
        with open('/home/billault/Documents/LWPretrieval/code_propre/stations.yaml','r') as file:
            variables = yaml.load(file,Loader=yaml.FullLoader)
        Stations = np.array(variables['stations'])
        Latitudes = np.array(variables['latitudes'])
        Longitudes = np.array(variables['longitudes'])
        Altitudes = np.array(variables['altitudes'])
        
        for path in filenames:
            with open(path,'r') as myf:
                line = myf.readline()
                header = line.split('\t')
                
                station = int(header[-4])
                
                if station in Stations:
                    lat = Latitudes[Stations==station][0]
                    long = Longitudes[Stations==station][0]
                    alt = Altitudes[Stations==station][0]
                
                else:
                    lat = float(header[-3])
                    long = float(header[-2])
                    alt = float(header[-1])
            
                if alt < 0:
                    print('altitude was changed'+str(lat)+' '+str(long))
                    alt = 386 # correction for coni-levaldigi station
                if (lat < -900) | (long < -900):
                    continue

                line = myf.readline()
                line = myf.readline()
                line = myf.readline()
                    
                while len(line)>0:
                        
                    data = line.split('\t')
                        
                    datefromstr = datetime.datetime.strptime(data[0],'%Y-%m-%d-%H')
                    yr = datefromstr.year
                    daynum = (datefromstr-datetime.datetime(yr,1,1,0)).days+1+(datefromstr-datetime.datetime(yr,1,1,0)).seconds/(24*3600)
            
                    ref = str(station)+'_'+datefromstr.strftime('%Y%m%d%H')
                    if not (ref in refpb):
                        latitude.append(lat)
                        longitude.append(long)
                        altitude.append(alt)
                        date.append(daynum)# find appropriate format for date
                        Ts.append(float(data[1]))
                        Ps.append(float(data[2]))
                        RHs.append(float(data[3]))
                        LWP.append(float(data[4]))
                        PWV.append(float(data[5]))
                        TB.append(float(data[6]))
                        if 'LWPera' in self.variables:
                            LWPera.append(float(data[7]))
                        if 'PWVera' in self.variables:
                            PWVera.append(float(data[8]))
                        
                    line = myf.readline()
        
               
        if 'latitude' in self.variables:
            self.latitude = np.array(latitude)
        if 'longitude' in self.variables :
            self.longitude = np.array(longitude)
        if 'altitude' in self.variables:
            self.altitude = np.array(altitude)
        if 'date' in self.variables:
            self.date = np.array(date)
        if 'Ts' in self.variables:
            self.Ts = np.array(Ts)
        if 'Ps' in self.variables:
            self.Ps = np.array(Ps)
        if 'RHs' in self.variables:
            self.RHs = np.array(RHs)
        if 'LWPera' in self.variables:
            self.LWPera = np.array(LWPera)
        if 'PWVera' in self.variables:
            self.PWVera = np.array(PWVera)
        if 'TB' in self.variables:
            self.TB = np.array(TB)
        
        self.LWP = np.array(LWP)
        self.PWV = np.array(PWV)
        if 'Ps' in self.variables:
            self.PsToSeaLevel(altitude)
        print('OK\n')
            
        
    ################ add offsets #################    
    def addTBOffset(self,offset):
        self.TB += offset
    
    def addTBStdNoise(self,std):
        random.seed(1)
        self.TB += np.random.normal(loc=0,scale=std,size=self.TB.shape)
        
    def addTBPercOffset(self,offset):
        self.TB += offset/100*self.TB
    
    def addTsOffset(self,offset):
        self.Ts += offset

    ############### load predicted PWV if required ###############
    
    def loadPWVpred(self,modeldirectory=''):#'/home/billault/Documents/LWPretrieval/code_propre/models/PWV/'):
        print('loading PWV prediction...')

        import yaml
        ymlpath = modeldirectory+'NN/config.yaml'
    
        with open(ymlpath,'r') as file:
            variables = yaml.load(file,Loader=yaml.FullLoader)
        variables = variables['input_features']

        mean_training_input_pwv = np.load(modeldirectory+'NN/mean.npy')
        std_training_input_pwv = np.load(modeldirectory+'NN/std.npy')
       
        variables_keys = self.__dict__.keys()-{'variables','target','parameters','data','LWP','PWV','splitdata','PWVpred','PWVtrue','PWVreal','PWVreal2','ml_type'}
        variables_keys = sorted([v for v in variables_keys])
        datalist = [self.__dict__[k].tolist() for k in variables_keys]
        
        if 'TB2' in variables:
            datalist.append(((self.TB)**2).tolist())
        if 'TB3' in variables:
            datalist.append(((self.TB)**3).tolist())
        if 'TB4' in variables:
            datalist.append(((self.TB)**4).tolist())
        
        data_for_PWV = np.array(datalist).T
        print(data_for_PWV.shape)
        if ('TB3' in variables) & ('TB4' in variables):
            data_for_PWV = (data_for_PWV-mean_training_input_pwv)/std_training_input_pwv
        else:
            data_for_PWV = (data_for_PWV-mean_training_input_pwv[:-2])/std_training_input_pwv[:-2]
        
        modelpath = sorted(glob.glob(modeldirectory+self.ml_type+'/BEST*'))[0]
        modelPWV = load_model(modelpath)
        pred = modelPWV.predict(data_for_PWV)
        self.PWVpred = np.array([p[0] for p in pred])
            
        print('OK\n')
    
    
    ############### adjust pressure to sea level ###############
    
    def PsToSeaLevel(self,altitude):
        Ts = self.Ts+273.15 # T in K for this calculation
        To = np.zeros(len(Ts))
        To[0] = Ts[0]
        To[1:] = .5*(Ts[:-1]+Ts[1:])
        Rd = 287 # J/(kg.K)
        g = 9.807 # m/s²
        H = Rd/g*To
        self.Ps = self.Ps*np.exp(altitude/H)

    
    ################ extraction routine #################
    
    def keep(self,tokeep):
        variables_keys = self.__dict__.keys()-{'variables','target','parameters','data','splitdata','ml_type'}
        for key in variables_keys:
            self.__dict__[key] = self.__dict__[key][tokeep]

        
    ################ pre-processing of data #################

    def preprocess(self):
        print('dataset preprocessing...')
        ut = self.parameters['upper_thres']
        lt = self.parameters['lower_thres']   
        tokeep1 = self.PWV==self.PWV
        self.keep(tokeep1)
        if 'Ps' in self.variables:
            tokeep = (self.LWP<ut) & (self.Ps>40000) & (self.Ts>-45) & (self.LWP>=lt) & (self.PWV<np.quantile(self.PWV,.999))
        else:
            tokeep = (self.LWP<ut) & (self.LWP>=lt) & (self.PWV<np.quantile(self.PWV,.999))
        self.keep(tokeep)
        
        if self.parameters['subsample']:
            self.subsample()
        print('OK\n')
    
    
    ################ sub-sampling routine #################

    def subsample(self,nsample=150,maxtosample=800): #before: 800
        print('subsampling...')
#        if self.target == 'LWP':
        random.seed(1)
        [hist,bin_edges] = np.histogram(self.LWP,bins=nsample)
        tokeep = []
        for i in range(len(hist)):
            ii = np.where((self.LWP>=bin_edges[i]) & (self.LWP<bin_edges[i+1]))
            ii = ii[0].tolist()
            sample = min(maxtosample,len(ii))
            ii = random.sample(ii,sample)      
            for k in ii:
                tokeep.append(k)
        self.keep(tokeep)
        print('OK\n')
        
        

    ################ wrap-up variables into data array #################
    
    def wrapup(self,withoutERAforLWP):
        print('wrapping up...')
        variables_keys = self.__dict__.keys()-{'variables','target','parameters','data','LWP','PWV','splitdata','ml_type'}
        if withoutERAforLWP:
            variables_keys = variables_keys-{'LWPera','PWVera'}
        variables_keys = sorted([v for v in variables_keys])
        datalist = [self.__dict__[k].tolist() for k in variables_keys]
        if 'TB2' in self.variables:
            datalist.append(((self.TB)**2).tolist())
        if 'TB3' in self.variables:
            datalist.append(((self.TB)**3).tolist())
        if 'TB4' in self.variables:
            datalist.append(((self.TB)**4).tolist())
        self.data = np.array(datalist)
        print('OK\n')


    ################ define training, validation, testing dataset ################
    
    def splitDataset(self,train=.7,val=.15,test=.15,seed=3):
        print('splitting dataset...')
        
        random.seed(seed)
        
        trainind = []
        valind = []
        testind = []
        print(self.data.shape[-1])
        for i in range(self.data.shape[-1]):
            x = random.random() 
            if x < train:
                 trainind.append(i)
            elif x < train+val:
                 valind.append(i)
            else:
                 testind.append(i)
        
        training = self.data[:,trainind]
        validation = self.data[:,valind]
        testing = self.data[:,testind]
        
        if self.target == 'LWP':
            training_target = self.LWP[trainind]
            validation_target = self.LWP[valind]
            testing_target = self.LWP[testind]
        elif self.target == 'PWV':
            training_target = self.PWV[trainind]
            validation_target = self.PWV[valind]
            testing_target = self.PWV[testind]
        
        
        self.splitdata = {'training':[training.T,training_target], 'validation':[validation.T,validation_target],
                'testing':[testing.T,testing_target]}
        print('OK\n')
    
    ################ mean-normalization of dataset ################
    
    def meanNormalize(self,savemean=None):
        print('mean normalizing...')
        [train_data,train_targ] = self.splitdata['training']
        [val_data,val_targ] = self.splitdata['validation']
        [test_data,test_targ] = self.splitdata['testing']
        
        mean = train_data.mean(axis = 0)
        std = train_data.std(axis = 0)
        traind = (train_data-mean)/std
        vald = (val_data-mean)/std
        testd = (test_data-mean)/std
        print(mean)
        print(std)
        if savemean != None:
            np.save(savemean+'mean',mean)
            np.save(savemean+'std',std)
        self.splitdata = {'training':[traind,train_targ], 'validation':[vald,val_targ],
                'testing':[testd,test_targ]}
        print('OK\n')


    ################ mean-normalization for implementation ################

    def meanNormalizeGeneral(self,mean,std):
        print('Data normalization...')
        mean = np.array(mean)
        std = np.array(std)
        self.data = (self.data-mean.reshape(len(mean),1))/std.reshape(len(std),1)
        print('Finished data normalization!\n')