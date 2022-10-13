import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertModel, RobertaModel


class Bert_Model(nn.Module):
    def __init__(self, config, args):
        super(Bert_Model, self).__init__()
        self.args = args
        self.hidden_size = config.hidden_size

        """
        
        定义bert神经网络
        
        """
        self.bert = BertModel(config,'E:/AI-Task2-2/NLP-Part2/bert-base-uncased')
        self.bert.to(args.device)

        self.cls_dropout = nn.Dropout(args.dropout_rate)  ## 自行查询资料，学习这个层的作用以及用法。
        self.classifier = nn.Linear(self.hidden_size,self.args.num_labels)

    def forward(self, input_ids,input_mask,segment_ids,concat_prog_ids,multi_prog_ids):    
            pooled_output= self.bert(input_ids,input_mask)[0]
            pooled_output = self.cls_dropout(pooled_output)

            logits = self.classifier(pooled_output)
            return logits 




        # """

        # 实现神经网络各层之间的联系，也就是前向传播。

        # 注：
        # 1. 该神经网络需要先经过bert层的嵌入，然后取last_hidden_state。
        # 2. bert的嵌入是为了让机器读懂自然语言同时学习文字之间的关系，嵌入之后之后就是分类，那分类应该如何实现？
        # 3. 定义的dropout层有什么作用？
        # 4. 前向传播输入的量可以自行修改，具体研究trainer.py文件中的训练相关代码

        # """
        # #pass
