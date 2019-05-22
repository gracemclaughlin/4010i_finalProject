from configuration import *
#from model import Mem2Seq
import numpy as np
import logging 
from tqdm import tqdm
from dataloader import *
from nmt_model import *
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0 
#dataloader
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)
print("sucessful data loading !")

# model = Mem2Seq(hidden_size=100, max_len=max_len,
#                 max_r=max_r, lang=lang, path="", task=args['task'],
#                 lr=0.001, n_layers=3, dropout=0.0, unk_mask=False)
model = Mem2Seq(hidden_size=100, max_len=max_len,
                max_r=max_r, lang=lang, path="",
                lr=0.001, n_layers=3, dropout=0.0)

print("loaded model!")
#train
avg_best = 0
#300
for epoch in range(10):
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

#evaluate bleu number
    if((epoch+1) % 1 == 0):    
        bleu, directory1 = model.evaluate(train,avg_best)
        model.scheduler.step(bleu)
        if(bleu >= avg_best):
            avg_best = bleu
            cnt=0
        else:
            cnt+=1

        if(cnt == 5):
            BLEU = False
            #hardcoded in a model, change to better one from training? 
            directory = directory1.split("/")
            #directory = args['path'].split("/")
            task = directory[2].split('HDD')[0]
            HDD = directory[2].split('HDD')[1].split('BSZ')[0]
            L = directory[2].split('L')[1].split('lr')[0]

            train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=int(args['batch']))
            model = Mem2Seq(int(HDD), max_len=max_len,
                    max_r=max_r, lang=lang, path=directory1,
                    lr=0.0, n_layers=int(L), dropout=0.0)


            # model = Mem2Seq2(int(HDD), max_len=max_len,
            #                 max_r=max_r, lang=lang, path=directory1, task=task,
            #                 lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)

            acc_test = model.evaluate(test, 1e6, BLEU) 
            print(acc_test)
            if testOOV!=[]:
                acc_oov_test = model.evaluate(testOOV,1e6,BLEU) 
                print(acc_oov_test)



        
