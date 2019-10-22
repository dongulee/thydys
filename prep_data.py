'''
Copyright 2019
Dongkyu Lee
Seoul National University ECE
Petabyte In-Memory Database Laboratory

Thydys: Thyroid Dysfunction (Thyrotoxicosis and Hyperthyroidism) Prediction Model
[module] Preparation of train/test dataset
- sampling
- interpolation
- extrapolation (by 0 filling)

'''

from os import listdir
from itertools import combinations
import pandas as pd
import numpy as np
import subprocessf
import tensorflow as tf
import utils as utl
#from collections import Counter

from optparse import OptionParser
import sys
 

#import matplotlib
#%matplotlib inline
#sys.stdin.encoding

'''
Helper functions to trim sequence data using lab date (blood test date)
5 days * 24 hours * 60 mins = 7200 sequences ordered by timestamp
HRseq = [66, 67, 76, 64, 90, 114, ...]
Actseq = [0.667, 0.667, ..., 3.252, 6.554, ...] # calories
'''

#for x in range(1,2): # embedded number of id
def getDiagByPid(diagnosis, pid):
    ret = []
    idx = diagnosis['Serial#']==pid
    selected = diagnosis[:][idx]
    for index, row in selected.iterrows():
        item = {}
        for key in row.keys():
            #print(key)
            item[key] = row[key]
        ret.append(item)
    return ret

def getSeqByDate(seq, lab_date):
    # (Lab date - 5 DAY) 보다 sequence date 기록이 적은 경우, labdate 후 5일 seq 반환
    
    end_date = pd.Timestamp(lab_date)
    start_date = end_date - pd.Timedelta('5 day')
    
    try:
        seq_tmp = seq[start_date:end_date]
    except:
        return 'null', None
    
    if len(seq_tmp) == 0:
        start_date = end_date
        end_date = start_date + pd.Timedelta('5 day')
        seq_tmp = seq[start_date:end_date]
        if len(seq_tmp) == 0:
            return 'null', None
        else:
            return 'before', seq_tmp
    else:
        return 'after', seq_tmp
    
def getSeqByPid(basedir, seqlist, pid):
    print(pid)
    '''
    Return HR, Activity Sequences by Pid
    ({'Timestamp': timestamp, 'Value': value})
    '''
    #find HR file name
    candidate = [x for x in seqlist if x['pid']==pid]
        
    hrfile = basedir + '/sensors/' + candidate[0]['HR']
    actfile = basedir + '/sensors/' + candidate[0]['act']
    
    # HR: 5 seconds -> 1 minute average aggregation
    hrseq = pd.read_csv(hrfile, header=0, parse_dates=['HEART RATE DATE/TIME'], index_col = 'HEART RATE DATE/TIME', usecols = ['HEART RATE DATE/TIME', 'VALUE'])
    hrseq = hrseq.resample('15T').mean().sort_index()
    actseq = pd.read_csv(actfile, header=0, parse_dates=["ACTIVITY DATE/TIME"],index_col="ACTIVITY DATE/TIME", usecols=["ACTIVITY DATE/TIME", "CALORIES"])
    actseq = actseq.resample('15T').mean().sort_index()
    return hrseq, actseq
    
    
class hr_data:
    def __init__(self, seq, diagnosis):
        self.seq_list = seq
        self.diag = diagnosis
        
        # pid_list: 1, 2, 3, 5, 6, 7, ...
        pid_list = [x['pid'] for x in self.seq_list]
        
        self.dataset = []
        # 환자별로 시퀀스 생성
        for pid in pid_list:
            pid_int = int(pid)
            for medical_record in getDiagByPid(self.diag, pid_int):
                HR_pd, ACT_pd = getSeqByPid('data', self.seq_list, pid)
                
                # Input Data
                date = medical_record['Lab_date']
                age = medical_record['age']
                gender = medical_record['gender']
                height = medical_record['Ht']
                s_type, HRseq = getSeqByDate(HR_pd, date)
                s_type2, ACTseq = getSeqByDate(ACT_pd, date)
                
                if s_type != s_type2 or s_type == 'null' or s_type2 == 'null':
                    continue
                
                # Label
                freeT4 = medical_record['freeT4']
                categorical = [age, gender, height, s_type]
                
                x1 = HRseq[:-1]
                x2 = ACTseq[:-1]
                if len(x1) == 480 and len(x2) == 480:
                    self.dataset.append({'categorical':categorical, 'HR': x1, 'ACT': x2, 'freeT4':freeT4})
                else:
                    continue
    def fillna(self):
        for record in self.dataset:
            record['HR'] = record['HR'].fillna(0)
            record['ACT'] = record['ACT'].fillna(0)

def main():
    if (len(sys.argv) <= 1):
        print ("prep_data.py -h or --help to get guideline of input options")
        exit()
        
    use = "Usage: %prog [options] filename"
    parser = OptionParser(usage = use)
    parser.add_option("-t", "--input-type", dest="input_type", default='raw', action="store", type="string", help="raw | ckpt")
    parser.add_option("-d", "--input-dir", dest="input_dir", default='data', action="store", type="string", help="in case of raw, inputdir should have \'sensor\' and \'label\' sub-directories")
    parser.add_option("-s", "--sample-rate", dest="sample_rate", default='1T', action="store", type="string", help="1T | 5T | 15T | 30T , ...")
    parser.add_option("-o", "--output-dir", dest="output_dir", default='data/default_ckpt', action="store", type="string", help="path of the directory which will contain ckpt data")

    #TODO: argument exception handling block
    
    ###

    sample_rate = options.sample_rate
    input_type = options.input_type
    base_dir = options.input_dir
    output_dir = options.output_dir

    timesteps = 24*60*5/sample_rate


    if input_type == "raw":
        # sensor data files
        datalist = listdir(base_dir + 'sensors') 
        datalist = [x for x in datalist if x not in remove_files] #포함시키지 않을 경로명 제외하고 리스트 생성
        #print(datalist)

        pid_list = [x.split('_')[0] for x in datalist]
        pid_list = list(set(pid_list))
        pid_list.sort()
        PList = []
        for pid in pid_list:

            activity = '_activity_intraday.csv'
            HR = '_HR_intraday.csv'
            PList.append({'pid': pid[4:6], 'HR': pid+HR, 'act': pid+activity})
        
        # FIXME: patient 28 has an error
        PList.remove({'pid': '28',
          'HR': 'GDAT28_HR_intraday.csv',
          'act': 'GDAT28_activity_intraday.csv'})
        diagnosis = pd.read_csv(base_dir + 'label/diagnosis.csv')
        
        # Load raw sequence data into hr_data class
        dataset = hr_data(PList, diagnosis, sample_rate)
        dataset.fillna()
        X = [ (x['HR'].values, x['ACT'].values) for x in dataset.dataset]
        X = np.array(X, dtype=float)

        # FIXME: hardcoded input component '2'
        X = X.reshape([len(X),timesteps*2]) 
        Y = [ x['freeT4'] for x in dataset.dataset]
        Y = np.array(Y, dtype = float)

        X.tofile(output_dir + '/' + 'X_'+ timesteps +'.dat')
        Y.tofile(output_dir + '/' + 'Y_'+ timesteps +'.dat')

    elif options.input_type == "ckpt":
        pass
    else:
        print("-- input-type={0}: input type is wrong".format(options.input_type))
        exit()


        
        
    
            
if __name__ == "__main__":
    sample_rate = argv[1]
    main(sys.argv[1])
    
