
from utils import * 


import torch
from torch import nn
import torch.nn.functional as F

import numpy as  np
import random
import math



class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads = 8, mask = False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask
        

        self.toKeys = nn.Linear(emb, emb * heads, bias = False)
        self.toQueries = nn.Linear(emb, emb * heads, bias = False)
        self.toValues = nn.Linear(emb, emb * heads, bias = False)

        self.unifyHeads = nn.Linear(emb * heads, emb)

    
    def forward(self, x):

        batchSize, numVec, embs = x.size() # batch size, number of vectors, embedding vector size
        heads = self.heads
        assert embs == self.emb, "input embedding dim should match layer embedding dim"

        keys = self.toKeys(x).view(batchSize, numVec, heads, embs)
        queries = self.toQueries(x).view(batchSize, numVec, heads, embs)
        values = self.toValues(x).view(batchSize, numVec, heads, embs)

        keys = keys.transpose(1, 2).contiguous().view( batchSize*heads, numVec, embs)
        queries = queries.transpose(1, 2).contiguous().view( batchSize*heads, numVec, embs)
        values = values.transpose(1, 2).contiguous().view( batchSize*heads, numVec, embs)



        ############## Scaling queries and keys before softmax() to solve gradient vanishing
        queries = queries / (embs**(1/4))
        keys = keys / (embs**(1/4))
        #############################################

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (batchSize*heads, numVec, numVec)

        if self.mask:
            mask_(dot, maskval = float('-inf'), mask_diagonal = False)
        
        dot = F.softmax(dot, dim = 2) # softmax() per rows to convert weights to probabilities 

        out = torch.bmm(dot, values).view(batchSize, heads, numVec, embs)
        out = out.transpose(1, 2).contiguous().view(batchSize, numVec, heads*embs)

        return self.unifyHeads(out) 




class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads = 8, mask = False):

        super().__init__()

        assert emb % heads == 0, "Embedding dimension should be divisible by number of heads"

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads # break embeddings into {heads} chunks and feed each chunk to a different attention

        self.toKeys = nn.Linear(s, s, bias= False)
        self.toQueries = nn.Linear(s, s, bias = False)
        self.toValues = nn.Linear(s, s, bias = False)

        self.unifyHeads = nn.Linear(heads*s, emb)

    
    def forward(self, x):

        batchSize, numVec, embs = x.size()
        heads = self.heads 
        assert embs == self.emb, "input embedding dim should match layer embedding dim"

        s = embs // heads
        x = x.view(batchSize, numVec, heads, s)

        keys = self.toKeys(x)
        queries = self.toQueries(x)
        values = self.toValues(x)

        assert keys.size() == (batchSize, numVec, heads, s)
        assert queries.size() == (batchSize, numVec, heads, s)
        assert values.size() == (batchSize, numVec, heads, s)

        keys = keys.transpose(1, 2).contiguous().view(batchSize*heads, numVec, s)
        queries = queries.transpose(1, 2).contiguous().view(batchSize*heads, numVec, s)
        values = values.transpose(1, 2).contiguous().view(batchSize*heads, numVec, s)

        ############## Scaling queries and keys before softmax() to solve gradient vanishing
        queries = queries / (embs**(1/4))
        keys = keys / (embs**(1/4))
        #############################################

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (batchSize*heads, numVec, numVec)

        if self.mask:
            mask_(dot, maskval = float('-inf'), mask_diagonal = False)
        
        dot = F.softmax(dot, dim = 2) # softmax() per rows to convert weights to probabilities 

        out = torch.bmm(dot, values).view(batchSize, heads, numVec, s)
        out = out.transpose(1, 2).contiguous().view(batchSize, numVec, heads * s)

        return self.unifyHeads(out)




class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        if config['wide'] : self.attention = SelfAttentionWide(config['emb'], config['heads'], config['mask'])
        else : self.attention = SelfAttentionNarrow(config['emb'], config['heads'], config['mask'])

        self.norm1 = nn.LayerNorm(config['emb'])
        self.norm2 = nn.LayerNorm(config['emb'])

        self.ff = nn.Sequential(
            nn.Linear(config['emb'], config['ff_hidden_mult']*config['emb']),
            nn.ReLU(),
            nn.Linear(config['ff_hidden_mult']*config['emb'], config['emb'])
        )

        self.do = nn.Dropout(config['dropout'])
    

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1( attended + x )

        x = self.do(x)

        fedForward = self.ff(x)

        x = self.norm2( fedForward + x )

        x = self.do(x)

        return x 




if __name__ == "__main__":

    saw = SelfAttentionWide(emb = 256)
    san = SelfAttentionNarrow(emb = 256)

    x = torch.ones((1, 10, 256), dtype = torch.float)
    y = san(x)

    print("done")