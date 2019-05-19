from configuration import *
from model import Mem2Seq
import numpy as np
import logging 
from tqdm import tqdm
from dataloader import prepare_data_seq
#from utils_mem2seq import *

#dataloader
train,lang, max_len, max_r = prepare_data_seq(batch_size = 32)
print("sucessful loading !")
#model of Mem2Seq without DMN
# model = Mem2Seq(hidden_size= 100, max_len= max_len, 
#                 max_r= max_r, lang=lang, 
#                 path="",lr=0.001, n_layers=3, dropout=0.0)

model = Mem2Seq(hidden_size=100, max_len=max_len,
                max_r=max_r, lang=lang, path="",
                task=1, lr=0.001, n_layers=3, dropout=0.0, unk_mask=False)

# model = globals()[args['decoder']](int(args['hidden']),
#                                         max_len,max_r,lang,args['path'],args['task'],
#                                         lr=float(args['learn']),
#                                         n_layers=int(args['layer']), 
#                                         dropout=float(args['drop']),
#                                         unk_mask=bool(int(args['unk_mask']))
#                                     )
print("loaded model!")
#train
avg_best = 0
for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar: 
        model.train_batch(input_batches=data[0], 
                          input_lengths=data[1], 
                          target_batches=data[2], 
                          target_lengths=data[3], 
                          target_index=data[4], 
                          target_gate=data[5],
                          batch_size=len(data[1]),
                          clip= 10.0,
                          teacher_forcing_ratio=0.5,
                          reset=0)

        pbar.set_description(model.print_loss())

# #evaluate bleu number
#     if((epoch+1) % 1 == 0):    
#         bleu = model.evaluate(train,avg_best)
#         model.scheduler.step(bleu)
#         if(bleu >= avg_best):
#             avg_best = bleu
#             cnt=0
#         else:
#             cnt+=1

#         if(cnt == 5): break
