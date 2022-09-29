import numpy as np
import random
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import matplotlib.pyplot as plt


class VectorQuantizer(nn.Module):
    def __init__(
                self,
                embedding_num   = 10,
                embedding_dim   = 3,
                commitment_beta = 0.25,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                ):
        super(VectorQuantizer, self).__init__()
        self.embedding_num   = embedding_num
        self.embedding_dim   = embedding_dim
        self.commitment_beta = commitment_beta
        self.device    = device
        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.embedding_num, 1/self.embedding_num)

    def compute_loss(
                    self,
                    z_e,
                    z_q
                    ):
        codebook_loss   = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        return codebook_loss + self.commitment_beta*commitment_loss

    def forward(
                self, 
                z = torch.randn(1, 15)
                ):
        z_dim = z.shape[1]
        z = z.reshape(-1, int(z_dim/self.embedding_dim), self.embedding_dim)
        z_e = z.view(-1, self.embedding_dim)
        distances = torch.sum(z_e**2, dim=1, keepdim=True)\
                    + torch.sum(self.embedding.weight**2, dim=1, keepdim=False)\
                    - 2*torch.matmul(z_e, self.embedding.weight.t())
        q_x = torch.argmin(distances, dim=1).unsqueeze(1)
        q_x_one_hot = torch.zeros(q_x.shape[0], self.embedding_num).to(self.device)
        q_x_one_hot.scatter_(1, q_x, 1)
        z_q  = torch.matmul(q_x_one_hot, self.embedding.weight).view(z.shape)
        loss = self.compute_loss(z, z_q)
        z_q = z + (z_q-z).detach()
        return z_q.reshape(-1, z_dim), loss

class VectorQuantizedVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        name     = 'VQVAE',              
        x_dim    = 784,              # input dimension
        c_dim    = 10,               # condition dimension
        z_dim    = 15,               # latent dimension
        h_dims   = [64,32],          # hidden dimensions of encoder (and decoder)
        embedding_num   = 10,        # For VQ parameters
        embedding_dim   = 3,         # For VQ parameters
        commitment_beta = 0.25,      # For VQ parameters
        actv_enc = nn.ReLU(),        # encoder activation
        actv_dec = nn.ReLU(),        # decoder activation
        actv_q   = nn.Softplus(),    # q activation
        actv_out = None,             # output activation
        device   = 'cpu'
        ):
        """
            Initialize
        """
        super(VectorQuantizedVariationalAutoEncoder, self).__init__()
        self.name   = name
        self.x_dim  = x_dim
        self.c_dim  = c_dim
        self.z_dim  = z_dim
        self.h_dims = h_dims
        self.embedding_num   = embedding_num
        self.embedding_dim   = embedding_dim
        self.commitment_beta = commitment_beta
        self.actv_enc  = actv_enc
        self.actv_dec  = actv_dec
        self.actv_q    = actv_q
        self.actv_out  = actv_out
        self.device    = device
        # Initialize VQ class
        self.VQ = VectorQuantizer(self.embedding_num, self.embedding_dim, self.commitment_beta, self.device).to(self.device)
        # Initialize layers
        self.init_layers()
        self.init_params()
                
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        
        # Encoder part
        h_dim_prev = self.x_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['enc_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['enc_%02d_actv'%(h_idx)] = \
                self.actv_enc
            h_dim_prev = h_dim
        self.layers['z_lin']  = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        
        # Decoder part
        h_dim_prev = self.z_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['dec_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['dec_%02d_actv'%(h_idx)] = \
                self.actv_dec
            h_dim_prev = h_dim
        self.layers['out_lin'] = nn.Linear(h_dim_prev,self.x_dim,bias=True)
        
        # Append parameters
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.vqvae_parameters = nn.ParameterDict(self.param_dict)
        
    def xc_to_z(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x to z
        """
        if c is not None:
            net = torch.cat((x,c), dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        z = self.layers['z_lin'](net)
        return z
    
    def zc_to_x_recon(
        self,
        z = torch.randn(2,16),
        c = torch.randn(2,10)
        ):
        """
            z and c to x_recon
        """
        net, _ = self.VQ(z)
        if c is not None:
            net = torch.cat((net,c),dim=1)
        else:
            pass
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon

    def z_q_to_x_recon(
        self,
        z_q,
        c
        ):
        """
            z and c to x_recon
        """
        net = torch.cat((z_q,c),dim=1)
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon

    def xc_to_x_recon(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x to x_recon
        """
        z = self.xc_to_z(x=x, c=c)
        x_recon = self.zc_to_x_recon(z=z, c=c)
        return x_recon

    def sample_x(
        self,
        c = torch.randn(2,10),
        n_sample = 1
        ):
        """
            sample x from codebook
        """
        random_integers  = np.random.permutation(self.embedding_num)[:n_sample]
        random_embedding = self.VQ.embedding.weight.data[random_integers, :]
        x_sample = self.z_q_to_x_recon(z_q=random_embedding, c=c).detach().cpu().numpy()
        return x_sample

    def init_params(self,seed=0):
        """
            Initialize parameters
        """
        # Fix random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Init
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def loss_recon(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q               = torch.ones(2),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0
        ):
        """
            Recon loss
        """
        x_recon = self.xc_to_x_recon(x=x, c=c)
        if (LOSS_TYPE == 'L1') or (LOSS_TYPE == 'MAE'):
            errs = torch.mean(torch.abs(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L2') or (LOSS_TYPE == 'MSE'):
            errs = torch.mean(torch.square(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L1+L2') or (LOSS_TYPE == 'EN'):
            errs = torch.mean(
                0.5*(torch.abs(x-x_recon)+torch.square(x-x_recon)),axis=1)
        else:
            raise Exception("VAE:[%s] Unknown loss_type:[%s]"%
                            (self.name,LOSS_TYPE))
        if self.actv_q is not None: 
            q = self.actv_q(q)
        errs = errs*q
        return recon_loss_gain*torch.mean(errs)
        
    def loss_total(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q               = torch.ones(2),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0
        ):
        """
            Total loss
        """
        loss_recon_out = self.loss_recon(
            x               = x,
            c               = c,
            q               = q,
            LOSS_TYPE       = LOSS_TYPE,
            recon_loss_gain = recon_loss_gain
        )
        z = self.xc_to_z(x, c)
        _, loss_vq = self.VQ(z)
        loss_total_out = loss_recon_out + loss_vq
        info           = {'loss_total_out' : loss_total_out,
                          'loss_recon_out' : loss_recon_out,
                          'loss_vq'        : loss_vq}
        return loss_total_out, info

    def update(
        self,
        x  = torch.randn(2,784),
        c  = torch.randn(2,10),
        q  = torch.ones(2),
        lr = 0.001,
        recon_loss_gain = 1,
        max_iter   = 100,
        batch_size = 100
        ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-4)
        loss_sum  = 0
        n_x       = x.shape[0]
        for n_iter in range(max_iter):
            self.train()
            rand_idx   = np.random.permutation(n_x)[:batch_size]
            x_batch    = torch.FloatTensor(x[rand_idx, :]).to(self.device)
            c_batch    = torch.FloatTensor(c[rand_idx, :]).to(self.device)
            q_batch    = torch.FloatTensor(q[rand_idx]).to(self.device)
            total_loss, _ = self.loss_total(x=x_batch, c=c_batch, q=q_batch, LOSS_TYPE='L2', recon_loss_gain=recon_loss_gain)
            loss_sum += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return loss_sum / max_iter