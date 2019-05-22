from __future__ import print_function
from __future__ import division

impos sys
import time
from copy import deepcopy
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from configuration import *
import random
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
from nltk.translate import bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import tensorflow as tf
from attention_gru_cell import AttentionGRUCell 
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops 
 
import babi_input
from stats import *
from model import *
from temp import *
from utils_mem2seq import *

tf_out = tf.placeholder(tf.float32, [None, None, word_vec_dim])

def DMN(object):

    def load_data():

    
    def add_placehodler(self):
        self.q_holder = tf.placeholder(tf.int32, shape=(self.config.bsz, self.max_response))
        self.in_holder = tf.placeholder(tf.int32, shape=(self.bsz, self.max_sent, self.max_sen_len))
        