import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim:int = 4096
    vocab_size:int = -1 
    n_layers:int = 32  #no.of encoders
    n_heads:int = 32
    n_kv_heads:Optional[int] = None
    multiple_of:int = 256
    ffn_dim_multipler:Optional[float] = None
    norm_eps:float = 1e-5

    max_batch_size:int = 32
    max_seq_len:int = 2048

    device:str = None


def precompute_pos_freqs(head_dim:int,seq_len:int,device:str,theta:float = 10000.0):

    assert head_dim % 2 == 0,"dim must div by 2"

    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):

    def __init__(self,dim:int,eps:float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x:torch.Tensor):
        #rsqrt = 1 / (sqrt)
        return x *  torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

    def forward(self,x):
        output = self._norm(x.float().type_as(x))
        return self.weight * output

def repeat_kv(x:torch.Tensor,n_rep:int):

    bs,slen,n_kv_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return(
        x[:,:,:,None,:].expand(bs,slen,n_kv_heads,n_rep,head_dim).reshape(bs,slen,n_kv_heads*n_rep,head_dim)
    )

class SelfAttention(nn.Module):

    def __init__(self,args:ModelArgs):
        super().__init__()

        # No.of heads for key and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is not None else args.n_kv_heads

        #No.of heads for queirs
        self.n_heads_q = args.n_heads

        # how many time the heads of key and value should be repeted to match the head of the queries
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Dimension of each head
        self.head_dim = args.dim // args.n_heads


        self.wq = nn.Linear(args.dim,args.n_heads * self.head_dim , bias= False)
        self.wk = nn.Linear(args.dim,args.n_kv_heads * self.head_dim , bias= False)
        self.wv = nn.Linear(args.dim,args.n_kv_heads * self.head_dim , bias= False)
        self.wo = nn.Linear(args.n_heads * self.head_dim,args.dim,bias=False)

        self.cache_k = torch.zeros((args.max_batch_size , args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size , args.max_seq_len,self.n_kv_heads,self.head_dim))

    def forward(self,x :torch.Tensor,start_pos:int,freqs_complex : torch.Tensor):
        batch_size,seq_len,_ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        xk = xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv = xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        xq = apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk = apply_rotary_embeddings(xk,freqs_complex,device=x.device)

        self.cache_k[:batch_size,start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len] = xv

        keys = self.cache_k[:batch_size,:start_pos+seq_len]
        value = self.cache_v[:batch_size,:start_pos+seq_len]
        
        keys = repeat_kv(keys,self.n_rep)
        value = repeat_kv(value,self.n_rep)

        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = value.transpose(1,2)

        scores = torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim=-1).type_as(xq)

        output = torch.matmul(scores,values)

        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output)


class Feedforward(nn.Module):

    def __init__(self,args:ModelArgs):
        super().__init__()

        hidden_dim = 4 *  args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multipler is not None:
            hidden_dim = int(args.ffn_dim_multipler * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,args.dim,bias=False)
        self.w3 = nn.Linear(args.dim,hidden_dim,bias=False)

    def forward(self,x:torch.Tensor):

        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x
    


class EncoderBlock(nn.Module):
    
    def __init__(self,args:ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = Feedforward(args)

        self.attention_norm = RMSNorm(args.dim,eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim,eps=args.norm_eps)

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):

        h = x + self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    


class Transformer(nn.Module):

    def __init__(self,args:ModelArgs):
        super().__init__()

        assert args.vocab_size != -1 ,"this must change"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.tok_embeddings = nn.Emdedding(self.vocab_size,self.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(self.dim,eps=args.norm_eps)
        self.output = nn.Linear(self.dim,self.vocab_size,bias=False)

        self.freqs_complex = precompute_pos_freqs(self.args.dim // self.args.n_heads, self.args.max_seq_len *2,device = self.args.device)


    
    def forward(self,tokens:torch.Tensor,start_pos:int):
        # (batch,seq_len)

        batch_size,seq_len = tokens.shape
        assert seq_len == 1 ,"only one token at atime"

        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex(start_pos:start_pos+seq_len)

        for layers in self.layers:
            h = layers(h,start_pos,freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output




