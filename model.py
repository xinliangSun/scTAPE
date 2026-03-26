import torch
import math

from torch import nn
from torch.nn import functional as F
from abc import abstractmethod


class BaseAE(nn.Module):
    def __init__(self):
        super(BaseAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size, current_device, **kwargs):
        raise RuntimeWarning()

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # the sine function is used to represent the odd-numbered sub-vectors
        pe[:, 0::2] = torch.sin(position * div_term)
        # the cosine function is used to represent the even-numbered sub-vectors
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class Enformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dop):
        super(Enformer, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dop
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, 
                                       dim_feedforward=self.dim_feedforward, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
 
        self.positionalEncoding = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)
    
    def forward(self, src):
        output = src.permute(1, 0, 2)
        output = self.positionalEncoding(output)

        for encoder in self.encoder:
            output = encoder(output)
        output = output.transpose(0, 1)
        output = output.mean(dim=1)
        return output


class DSNFormer(BaseAE):     
    def __init__(self, shared_encoder, decoder, p_encoder, alpha, norm_flag):
        super(DSNFormer, self).__init__()

        self.alpha = alpha
        self.norm_flag = norm_flag

        # build encoder
        self.shared_encoder = shared_encoder
        self.private_encoder = p_encoder
        self.decoder = decoder

    def p_encode(self, input):
        latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input):
        latent_code = self.shared_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def encode(self, input):
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)

        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z):
        outputs = self.decoder(z)

        return outputs

    def forward(self, input):
        input_v, input_i = input[0], input[1]
        z = self.encode(input_v)
        return [input_i, self.decode(z), z]

    def loss_function(self, *args):
        input = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]

        recons_loss = F.mse_loss(input, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        diff_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

        loss = recons_loss + self.alpha * diff_loss

        return {'loss': loss, 'recons_loss': recons_loss, 'diff_loss': diff_loss}

    def generate(self, x):
        return self.forward(x)[1]
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dop, act_fn=nn.ReLU, out_fn=None):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dop = dop

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.output_dim))
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.output_dim),
                out_fn())

    def forward(self, input):
        embed = self.module(input)
        output = self.output_layer(embed)

        return output
    

class Classify(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dop=0.1):
        super(Classify, self).__init__() 

        self.lin1 = nn.Linear(input_dim, hidden_dims[0])
        self.lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lin4 = nn.Linear(hidden_dims[1], output_dim)

        self.relu = nn.SELU()
        self.dropout = nn.Dropout(p=dop)

    def forward(self, input):
        embed = self.relu(self.lin1(input))
        embed = self.dropout(embed)

        embed = self.lin2(embed)

        output = self.lin4(embed)

        return output


class CellClassify(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dop=0.1):
        super(CellClassify, self).__init__() 

        self.lin1 = nn.Linear(input_dim, hidden_dims[0])
        self.lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lin4 = nn.Linear(hidden_dims[1], output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dop)

    def forward(self, input):
        embed = self.relu(self.lin1(input))
        embed = self.dropout(embed)

        embed = self.relu(self.lin2(embed))
        embed = self.dropout(embed)
        output = self.lin4(embed)

        return output


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, normalize_flag=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag

    def forward(self, input):
        encoded_input = self.encode(input)
        if self.normalize_flag:
            encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)

        output = self.decoder(encoded_input)
        return output

    def encode(self, input):
        return self.encoder(input)

    def decode(self, z):
        return self.decoder(z)