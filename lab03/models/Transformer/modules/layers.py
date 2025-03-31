import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.d_k = dim // num_heads
        self.d_v = dim // num_heads

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

 
        self.linear_out = nn.Linear(dim, dim)

        self.attn_drop = attn_drop
        self.dropout = nn.Dropout(p=attn_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1. / math.sqrt(self.d_k)
        

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, dim = x.size()
        assert dim == self.dim, f"Expected input dimension {self.dim}, but got {dim}"
        query = self.linear_q(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Reshape to (batch_size, num_heads, num_image_tokens, head_dim)
        #      key^T (batch_size, num_heads, head_dim, num_image_tokens)
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_scores = self.softmax(attn_scores)
        attn_scores = self.dropout(attn_scores) # (batch_size, num_heads, num_image_tokens, num_image_tokens)
        # Compute attention output
        attn_output = torch.matmul(attn_scores, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_image_tokens, self.dim)
        attn_output = self.linear_out(attn_output)
        return attn_output


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=drop_rate)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    