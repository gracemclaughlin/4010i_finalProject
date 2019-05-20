from configuration import *
from model import Mem2Seq
import numpy as np
import logging 
from tqdm import tqdm
from dataloader import *


'''
python3 main_test.py -dec= -path= -bsz= -ds=
'''

BLEU = False

# Configure models

#hardcoded in a model, change to better one from training? 
directory1 = "save/mem2seq-BABI/1HDD100BSZ32DR0.0L3lr0.001Mem2Seq0.9991732804232805"
directory = directory1.split("/")
#directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]

train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=int(args['batch']))


model = Mem2Seq(int(HDD), max_len=max_len,
                max_r=max_r, lang=lang, path=directory1, task=task,
                lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)

acc_test = model.evaluate(test, 1e6, BLEU) 
print(acc_test)
if testOOV!=[]:
    acc_oov_test = model.evaluate(testOOV,1e6,BLEU) 
    print(acc_oov_test)


