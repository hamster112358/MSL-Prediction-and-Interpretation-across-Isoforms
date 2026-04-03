import os
import cv2 
import numpy as np
import pickle
import glob
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image 
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import copy
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)
        nn.init.dirac_(self.to_attn_logits.weight)
        with torch.no_grad():
                self.to_attn_logits.weight.mul_(2)
    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)
class stem_block(nn.Module):
    def __init__(self):
        super(stem_block,self).__init__()
        self.model=nn.ModuleList(
                [
                nn.Conv1d(4,16,kernel_size=1,padding=1//2 ,bias=False),
                nn.Conv1d(4,16,kernel_size=3,padding=3//2 ,bias=False),
                nn.Conv1d(4,16,kernel_size=5,padding=5//2 ,bias=False),
                nn.Conv1d(4,16,kernel_size=7,padding=7//2 ,bias=False),
                ]
            )
        self.conv=nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.relu=nn.ReLU()
 
        self.norm=nn.LayerNorm (64)
        self.pool=AttentionPool(64,pool_size=2)
    def forward(self,x):
        x = rearrange(x,'b s h -> b h s' )
        
        x = torch.cat([func(x) for func in self.model],dim=-2)
        
        x = rearrange(x,'b h s -> b s h' )
        x = self.norm(x)
        x = rearrange(x,'b s h -> b h s' )
        out = self.relu(x)
        
        out = self.conv(out)
        out = self.relu(out + x)
        
        out =self.pool(out)
        return out
class resnet_module(nn.Module):
    def __init__(self,input_dim,embedding_dim,num):
        super(resnet_module,self).__init__()
        self.embedding_dim = embedding_dim
        self.num = num
        self.input_dim=input_dim
        self.block1=nn.Sequential(
            nn.Conv1d(self.input_dim,self.embedding_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            
            nn.Conv1d(self.embedding_dim,self.embedding_dim*4,kernel_size=1,bias=False)
        )
        self.block2=nn.Sequential(
            nn.Conv1d(self.embedding_dim*4,self.embedding_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            
            
            nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            
            nn.Conv1d(self.embedding_dim,self.embedding_dim*4,kernel_size=1,bias=False)
        )
        
                
        # First, define a small helper function
        def make_block(dim):
            return nn.Sequential(
                nn.Conv1d(dim*4, dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Conv1d(dim, dim*4, kernel_size=1, bias=False)
            )

        # Then in __init__, use it to create unique blocks:
        self.block2_total = nn.ModuleList([make_block(self.embedding_dim) for _ in range(num-1)])
        
        self.norm=nn.LayerNorm(self.embedding_dim*4)
        self.conv = nn.Conv1d(self.input_dim,self.embedding_dim*4,kernel_size=1,bias=False)
        self.relu = nn.ReLU()
        self.attention=AttentionPool(self.embedding_dim*4,pool_size=2)
    def forward(self,x):
        out1 = self.block1(x)
        out1 = rearrange(out1,'b h s -> b s h' )
        out1 = self.norm(out1)
        out1 = rearrange(out1,'b s h -> b h s' )
        out1_1=self.conv(x)
        out1_1 = rearrange(out1_1,'b h s -> b s h' )
        out1_1 = self.norm(out1_1)
        out1_1 = rearrange(out1_1,'b s h -> b h s' )
        out = self.relu(out1+out1_1)
        for item in range(self.num-1):
            out_ = self.block2_total[item](out)
            out_ = rearrange(out_,'b h s -> b s h' )
            out_ = self.norm(out_)
            out_ = rearrange(out_,'b s h -> b h s' )
            
            out = self.relu(out_ + out)

        out=self.attention(out)
        return out
class Resnet_block(nn.Module):
    
    def __init__(self,num,embedding_dim):
        super(Resnet_block,self).__init__()
        self.num=num
        self.embedding_dim=embedding_dim
        self.input_dim=64
        self.model=nn.Sequential(*[resnet_module(self.input_dim*2**(item if item==0 else item+1),self.embedding_dim*2**(item),self.num[item]) for item in range(len(self.num))])
    
    
    def forward(self,x):
        x = self.model(x)
        return x
class MHA_block(nn.Module):
    def attention(self,q,k,v,mask=None):
        qk=torch.matmul(q,k)/math.sqrt(q.size(-1))
        if mask is not None:
            qk=qk.masked_fill(mask==0,-1e9)
        qk=F.softmax(qk,dim=-1)
        result=torch.matmul(qk,v)
        return result 
        
    def layer_norm(self,x,eps=1e-6):
        a_2 = nn.Parameter(torch.ones(x.size(-1))).to(x.device)
        b_2 = nn.Parameter(torch.zeros(x.size(-1))).to(x.device)
        mean = x.mean(-1, keepdim=True)# mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)
        return a_2 * (x - mean) / (std + eps) + b_2
        
    def rms_norm(self,x,device,eps=1e-6):
        weight = nn.Parameter(torch.ones(x.size(-1))).to(x.device)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    
    def add_postion(self,q, k):
        #[b,n,s,h]
        batch_size, num_heads_q, seq_len, dim = q.size()
        num_heads_k = k.size()[1]
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)).to(q.device)
        pos_emb = position * div_term
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(1)
        pos_emb_q = pos_emb.expand(batch_size, num_heads_q, -1, -1)
        pos_emb_k = pos_emb.expand(batch_size, num_heads_k, -1, -1)
        cos_emb_q = pos_emb_q[..., 1::2].repeat_interleave(2, dim=-1)
        sin_emb_q = pos_emb_q[..., ::2].repeat_interleave(2, dim=-1)
        cos_emb_k = pos_emb_k[..., 1::2].repeat_interleave(2, dim=-1)
        sin_emb_k = pos_emb_k[..., ::2].repeat_interleave(2, dim=-1)
        q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
        k_alternate = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.size())
        q_fix = q * cos_emb_q + q_alternate * sin_emb_q
        k_fix = k * cos_emb_k + k_alternate * sin_emb_k
        return q_fix, k_fix
    
    def __init__(self,input_dim,qk_dim,v_dim,q_head,n_kv_head,device,dropout_ratio):
        super(MHA_block,self).__init__()
        self.input_dim=input_dim
        self.qk_dim=qk_dim
        self.v_dim=v_dim
        self.q_head=q_head
        self.n_kv_head=n_kv_head
        self.device=device
        self.linearq=nn.Linear(self.input_dim,self.q_head*self.qk_dim,bias=False)
        self.lineark=nn.Linear(self.input_dim,self.n_kv_head*self.qk_dim,bias=False)
        self.linearv=nn.Linear(self.input_dim,self.n_kv_head*self.v_dim,bias=False)
        self.linearfc=nn.Linear(self.n_kv_head*self.v_dim,self.input_dim,bias=False)
        self.dropout_ratio=dropout_ratio
        self.dropout=nn.Dropout(self.dropout_ratio)
    def forward(self,x ,mask,norm):
        #input [b,sq,h]
        out = x
        if mask is not None:
            mask=mask.unsqueeze(1)
        if norm == "layer_norm":
            x = self.layer_norm(x)
        elif norm =="rms_norm":
            x = self.rms_norm(x)
        else:
            x = x
        
        new_q,new_k,new_v = [func(x) for func in [self.linearq,self.lineark,self.linearv]]
        
        new_q = rearrange(new_q,'b s (n h) -> b n s h', n=self.q_head )
        new_k = rearrange(new_k,'b s (n h) -> b n s h', n=self.n_kv_head )
        new_q,new_k = self.add_postion(new_q,new_k)
        new_k = rearrange(new_k,'b n s h -> b n h s')
        new_v = rearrange(new_v,'b s (n h) -> b n s h', n=self.n_kv_head )


        result = self.attention(new_q,new_k,new_v,mask)
        result = rearrange(result,'b n s h -> b s (n h)')
        result = self.linearfc(result)
        result = self.dropout(result)
        out = out + result
   
        return out
class feed_forward(nn.Module):

    def __init__(self,embedding_dim,dropout_ratio):
        super(feed_forward,self).__init__()
        self.embedding_dim=embedding_dim
        self.dropout_ratio=dropout_ratio
        self.device=device
        self.model=nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.Dropout(self.dropout_ratio),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Dropout(self.dropout_ratio)
        )
        
    def forward(self,x,norm=None):
        out = self.model(x)
        x = x + out
        return x         
class resnet_module_1(nn.Module):
    def __init__(self,input_dim,num):
        super(resnet_module_1,self).__init__()
        self.num = num
        self.input_dim=input_dim
        self.block1=nn.Sequential(
            nn.Conv1d(self.input_dim,self.input_dim//2,kernel_size=1,bias=False),
            nn.BatchNorm1d(self.input_dim//2),
            nn.ReLU(),
            nn.Conv1d(self.input_dim//2,self.input_dim//2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm1d(self.input_dim//2),
            nn.ReLU(),
            
            nn.Conv1d(self.input_dim//2,self.input_dim//4,kernel_size=1,bias=False)
        )
        self.block2=nn.Sequential(
            nn.Conv1d(self.input_dim//4,self.input_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            
            
            nn.Conv1d(self.input_dim,self.input_dim,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            
            nn.Conv1d(self.input_dim,self.input_dim//4,kernel_size=1,bias=False)
        )
        
        self.block2_total=nn.ModuleList([ self.block2 for item in range(self.num-1)])
        
        self.conv = nn.Conv1d(self.input_dim,self.input_dim//4,kernel_size=1,bias=False)
        self.relu = nn.ReLU()
    def forward(self,x):
        out1 = self.block1(x)
        out1_1=self.conv(x)
        out = self.relu(out1+out1_1)
        for item in range(self.num-1):
            out_ = self.block2_total[item](out)
            out = self.relu(out_ + out)
        return out
class Resnet_block_1(nn.Module):
    
    def __init__(self,num,input_dim):
        super(Resnet_block_1,self).__init__()
        self.num=num
        self.input_dim=input_dim
        self.model=nn.Sequential(*[resnet_module_1(self.input_dim//4**item,self.num[item]) for item in range(len(self.num))])
    def forward(self,x):
        x = self.model(x)
        return x
class Encode(nn.Module):
    def crop(self,x,len_):
        return x[:,:,len_:-len_]
    def __init__(self,resnet_dim,MHA_num,input_dim,qk_dim,v_dim,q_head,n_kv_head,len_,max_len,device,dropout_ratio=0):
        super(Encode,self).__init__()
        self.device = device
        self.MHA_num = MHA_num
        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.q_head = q_head
        self.n_kv_head = n_kv_head
        self.dropout_ratio = dropout_ratio
        self.resnet_dim = resnet_dim
        self.max_len = max_len
        self.len_ = len_
        self.stem_ = stem_block()
        self.Resnet_ = Resnet_block([3,4,6,3],self.resnet_dim)
        self.MHA_ = nn.ModuleList([ MHA_block(self.input_dim,self.qk_dim,self.v_dim,self.q_head,self.n_kv_head,self.device,self.dropout_ratio)
                                                                for item in range(self.MHA_num) ])
        self.feedward_ = nn.ModuleList([ feed_forward(self.input_dim,self.dropout_ratio)
                                                                         for item in range(self.MHA_num) ])

        self.Conv1d_1 = nn.Sequential(
                *[
      
                nn.Conv1d(self.input_dim,256,kernel_size=1,bias=False),
                nn.GELU(),
				nn.Dropout(0.05),
				nn.Conv1d(256,1,kernel_size=1,bias=False),
                nn.GELU(),
				nn.Dropout(0.05),
				
                ]
            )  
        self.linear_1 = nn.Sequential(
               *[
                nn.Linear(self.max_len//2**5 - self.len_*2,128,bias=False),
                nn.GELU(),
				nn.Dropout(0.05),
                nn.Linear(128 , 11,bias=False),
				nn.Sigmoid(),
                ]    
              )  
        
    def forward(self,x,norm=None,mask=None):
        x = self.stem_(x)
        
        x = self.Resnet_(x)
        
        x = rearrange(x,'b h s -> b s h')

        if self.MHA_num == 1:
            x = self.MHA_[0](x,mask,norm)
            x = self.feedward_[0](x,norm)
        else:
            for item in range(self.MHA_num):
                x = self.MHA_[item](x,mask,norm)
                x = self.feedward_[item](x,norm)
        
        x = rearrange(x,'b s h -> b h s')
        x = self.crop(x,self.len_)
        x = self.Conv1d_1(x)
        x = rearrange(x,'b s h -> b (s h)')
        x = self.linear_1(x)
        return x

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)

class Mydataset(Dataset):
    def dict_tran(self,x):
        label_tran={"chromatin--(nucleus)":0,
                    "cytoplasm":1,
                    "cytosol--(cytoplasm)":2,
                    "endoplasmic reticulum--(cytoplasm)":3,
                    "extracellular region--(nucleus)":4,
                    "membrane":5,
                    "mitochondrion--(cytoplasm)":6,
                    "nucleolus--(nucleus)":7   ,
                    "nucleoplasm--(nucleus)":8     ,
                    "nucleus":9     ,
                    "ribosome--(cytoplasm)":10       
                        }
        result = torch.zeros(11)
        for i in x:
            result[label_tran[i]] = 1
        return result
    
    def read_seq(self,path,max_len):
        result=[]
        line=open(path,"r").readlines()
        label_init=torch.zeros(11)
        for line_ in line:
            seq_,label_=line_.split(",")
            seq_=seq_.lower()
            seq_result = []
            for seq__ in seq_ :
                if seq__=="u":
                    seq__="t"
                elif seq__ not in ["a","t","c","g"]:
                    seq__="o"
                seq_result.append(seq__)
            if len(seq_result) <= max_len:
                label_=label_.strip().split(";")
                result.append([seq_result,self.dict_tran(label_)])
        return result


    def seq2one_hot(self,seq):
        mapping={'a': 0, 't': 1, 'c': 2, 'g': 3 , 'o': 4}
        onehot_matrix = np.vstack((np.eye(4),np.zeros(4)))
        seq=[mapping[seq_]    for seq_ in seq ]
        return onehot_matrix[seq]

    def __init__(self,path,max_len=64*2**7,device=None):
        super(Mydataset,self).__init__()
        self.path=path
        self.max_len=max_len
        self.seqlabel=self.read_seq(self.path,self.max_len)
        
    def __getitem__(self,index):
        seq,seq_label = self.seqlabel[index]
        seq_result = self.seq2one_hot(seq)
        if len(seq) < self.max_len:
            len1 = (self.max_len-len(seq))//2
            len2 = (self.max_len-len(seq))-len1
            seq_result = np.pad(seq_result,((len1,len2),(0,0)),'constant',constant_values = (0,0))
            
        return    torch.from_numpy(seq_result).type(torch.float).to(device),seq_label.type(torch.float).to(device)
    def __len__(self):
        return len(self.seqlabel)
