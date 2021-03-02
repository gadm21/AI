

import torch
from torch import nn
import torch.nn.functional as F

from modules import *
from utils import *



class GTransformer(nn.Module):

    '''
    Transformer for generating text (character based)
    '''

    def __init__(self, config):
        '''
        config has emb, heads, depth, seq_length, num_tokens
        '''

        super().__init__()

        self.num_tokens = config['num_tokens']
        self.token_embedding = nn.Embedding(embedding_dim = config['emb'], num_embeddings = self.num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim = config['emb'], num_embeddings= config['seq_length'])

        self.unify_embeddings = nn.Linear(2*config['emb'], config['emb'])

        tBlocks = []
        for i in range(config['depth']):
            tBlocks.append(TransformerBlock(config)) 

        self.tBlocks = nn.Sequential(*tBlocks)
        self.toPorbs = nn.Linear(config['emb'], self.num_tokens)
    

    def forward(self, x):
        
        '''
        x is a batch of seq_length vectors of token indices
        '''
        tokens = self.token_embedding(x) 
        batchSize, seq_len, emb = tokens.size() 

        positions = self.pos_embedding(torch.arange(seq_len))[None, :, :].expand(batchSize, seq_len, emb)
        x = self.unify_embeddings(torch.cat((tokens, positions), dim =2).view(-1, 2*emb)).view(batchSize, seq_len, emb)

        x = self.tBlocks(x)
        x = self.toPorbs(x.view(batchSize*seq_len, emb)).view(batchSize, seq_len, self.num_tokens)

        return F.log_softmax(x, dim = 2)





