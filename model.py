import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from configuration import *
import random
import numpy as np
import datetime
from stats import *
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn  as sns
import nltk
import os
from sklearn.metrics import f1_score
import json
from temp import entityList
from torch.nn import functional
import tensorflow as tf
from nltk.translate import bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len ## max input
        self.max_r = max_r ## max response len        
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask
        
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
                #self.extKnow = torch.load(str(path)+'/enc_kb.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        #self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,mode='max',factor=0.5,patience=1,min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr =  self.loss_ptr / self.print_every
        print_loss_vac =  self.loss_vac / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg,print_loss_vac,print_loss_ptr)

  
    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        directory = 'save/mem2seq-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')

    #training    
    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):  

        if reset:
            print("resetting")
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
    
        loss_Vocab,loss_Ptr= 0,0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        
        #generate tokens with inputs pointing to memory
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, input_batches.size(0)))
        all_decoder_outputs_mem = Variable(torch.zeros(max_target_length, batch_size, input_batches.size(0)))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_mem = all_decoder_outputs_mem.cuda()
            decoder_input = decoder_input.cuda()

        #if use_DMN:
            

        # Choose whether to use teacher forcing
        # use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        # if use_teacher_forcing:    
        #     # Run through decoder one time step at a time
        #     for t in range(max_target_length):
        #         decoder_ptr, decoder_vocab, decoder_hidden  = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
        #         all_decoder_outputs_vocab[t] = decoder_vocab
        #         all_decoder_outputs_ptr[t] = decoder_ptr
        #         all_decoder_outputs_mem[t] = decoder_hidden
        #         decoder_input = target_batches[t]# Chosen word is next input
        #         if USE_CUDA: decoder_input = decoder_input.cuda()            
        # else:
        
        #periodically update decoder using memory 
        for t in range(max_target_length):
            decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            _, toppi = decoder_ptr.data.topk(1)
            _, topvi = decoder_vocab.data.topk(1)
            all_decoder_outputs_vocab[t] = decoder_vocab
            all_decoder_outputs_ptr[t] = decoder_ptr
            all_decoder_outputs_mem[t] = decoder_hidden
            ## get the correspective word in input
            #toppi is max probability  
            top_ptr_i = torch.gather(input_batches[:,:,0], 0, Variable(toppi.view(1, -1))).transpose(0,1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()
        
        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()
        
    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches, target_lengths, target_index,target_gate,src_plain):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        all_decoder_outputs_mem = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        #all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_mem = all_decoder_outputs_mem.cuda()
            #all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()
        
        p = []
        for elm in src_plain:
            elm_temp = [ word_triple[0] for word_triple in elm ]
            p.append(elm_temp) 
        
        self.from_whichs = []
        acc_gate,acc_ptr,acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr,decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topvi = decoder_vocab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:,:,0], 0, Variable(toppi.view(1, -1))).transpose(0,1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if(toppi[i].item() < len(p[i])-1 ):
                    temp.append(p[i][toppi[i].item()])
                    from_which.append('p')
                else:
                    ind = topvi[i].item()
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # indices = torch.LongTensor(range(target_gate.size(0)))
        # if USE_CUDA: indices = indices.cuda()

        # ## acc pointer
        # y_ptr_hat = all_decoder_outputs_ptr.topk(1)[1].squeeze()
        # y_ptr_hat = torch.index_select(y_ptr_hat, 0, indices)
        # y_ptr = target_index       
        # acc_ptr = y_ptr.eq(y_ptr_hat).sum()
        # acc_ptr = acc_ptr.data[0]/(y_ptr_hat.size(0)*y_ptr_hat.size(1))
        # ## acc vocab
        # y_vac_hat = all_decoder_outputs_vocab.topk(1)[1].squeeze()
        # y_vac_hat = torch.index_select(y_vac_hat, 0, indices)        
        # y_vac = target_batches       
        # acc_vac = y_vac.eq(y_vac_hat).sum()
        # acc_vac = acc_vac.data[0]/(y_vac_hat.size(0)*y_vac_hat.size(1))

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words #, acc_ptr, acc_vac


    def evaluate(self,dev,avg_best,BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED,microF1_PRED_cal,microF1_PRED_nav,microF1_PRED_wet = 0, 0, 0, 0
        microF1_TRUE,microF1_TRUE_cal,microF1_TRUE_nav,microF1_TRUE_wet = 0, 0, 0, 0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}
        if int(args["task"])!=6:
            global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt',int(args["task"]))
        else:
            global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt',int(args["task"]))

        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
                    data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6])          

            acc=0
            w = 0 
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e== '<EOS>': break
                    else: st+= e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]  
                ### compute F1 SCORE  
                st = st.lstrip().rstrip()
                correct = correct.lstrip().rstrip()
        
                if int(args["task"])==6:
                    f1_true,count = self.compute_prf(data_dev[10][i], st.split(), global_entity_list, data_dev[12][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count

                if data_dev[11][i] not in dialog_acc_dict.keys():
                    dialog_acc_dict[data_dev[11][i]] = []
                if (correct == st):
                    acc+=1
                    dialog_acc_dict[data_dev[11][i]].append(1)
                else:
                    dialog_acc_dict[data_dev[11][i]].append(0)
 
                w += wer(correct,st)
                ref.append(str(correct))
                hyp.append(str(st))
                ref_s+=str(correct)+ "\n"
                hyp_s+=str(st) + "\n"

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))            
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
                                                                    wer_avg/float(len(dev))))

        # dialog accuracy
        #if args['dataset']=='babi':
        dia_acc = 0
        for k in dialog_acc_dict.keys():
            if len(dialog_acc_dict[k])==sum(dialog_acc_dict[k]):
                dia_acc += 1
        logging.info("Dialog Accuracy:\t"+str(dia_acc*1.0/len(dialog_acc_dict.keys())))

        if int(args["task"])==6:
            logging.info("F1 SCORE:\t{}".format(microF1_TRUE/float(microF1_PRED)))
              
        hyp_tokens = [line.split(" ") for line in np.array(hyp)]
        ref_tokens = [line.split(" ") for line in np.array(ref)]
        smooth = SmoothingFunction().method4
        bleu_score = corpus_bleu(ref_tokens, hyp_tokens, smoothing_function= smooth)
       
        logging.info("BLEU SCORE:"+str(bleu_score))     
        if (BLEU):                                                               
            if (int(bleu_score) >= avg_best):
                self.save_model(str(self.name)+str(bleu_score))
                logging.info("MODEL SAVED")  
            return bleu_score
        else:
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name)+str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        
    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, story):
        story = story.transpose(0,1)
        story_size = story.size() # b * m * 3 
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
                story = story*a.long()
        u = [self.get_state(story.size(0))]
        #print(u)
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            #print("forward embed")
            #print(embed_A)
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))  
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0,1)   
        return u_k

class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim,1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        

    def load_memory(self, story):
        story_size = story.size() # b * m * 3 
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
                story = story*a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            m_A = embed_A    
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query) # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        # ques_rep = self.gru(embed_q, last_hidden)
        # ques_rep = ques_rep[self.input_size-1]
        # ques_rep = tf.reshape(ques_rep, [self.bsz, 1, self.hidden_size])
        # episodic_memory = tf.identity(ques_rep)
        # encoded_input = tf.transpose(encoded_input, [1,0,2])
        u = [hidden[0].squeeze()]   
        episodic_memory = tf.identity(enc_query)

        #encoded input reshape?? 
        # https://github.com/JRC1995/Dynamic-Memory-Network-Plus/blob/master/DMN%2B.ipynb 
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if(len(list(u[-1].size()))==1): u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
    

            prob_lg = torch.sum(m_A*u_temp, 2)

            prob_   = self.softmax(prob_lg)

            #prob = tf.transpose(prob, [2,0,1])
            #context_vec = attention_based_GRU(tf.transpose(u))

            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            if (hop==0):
                p_vocab = self.W1(torch.cat((u[0], o_k),1))
            u_k = u[-1] + o_k
            u.append(u_k)

        p_ptr = prob_lg 
        return p_ptr, p_vocab, hidden

    def episodic_update(ct, prev_m, q, wt, b):
        m = T.nn.relu(wt[T.concatenate([prev_m, ct, q])]+b)
    return m 

    def episode_attend(self, x, g, h):
        #r = sigmoid
        #_h = tanh
        ht = g*_h +(1. -g)*h
    return ht

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))    

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat,dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))  
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
