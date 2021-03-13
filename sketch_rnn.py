import os

import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import argparse

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

###################################### hyperparameters
class HParams():
    def __init__(self):
        self.data_location = 'cat.npz'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 1
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.02
        self.max_seq_length = 200

hp = HParams()

################################# load and prepare data
def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

dataset = np.load(hp.data_location, allow_pickle=True, encoding='latin1')
data = dataset['train']
data = purify(data)
data = normalize(data)
print('train data size: ', len(data))
Nmax = max_size(data)
print('train max len: ', Nmax)

test_data = dataset['test']
test_data = purify(test_data)
test_data = normalize(test_data)
print('test data size: ', len(test_data))
test_Nmax = max_size(test_data)
print('test max len: ', test_Nmax)

############################## function to generate a batch:
def make_batch(batch_size):
    batch_idx = np.random.choice(len(data),batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        lengths.append(len(seq[:,0]))
        strokes.append(new_seq)
        indice += 1

    if use_cuda:
        #batch = Variable(torch.from_numpy(np.stack(strokes,1)).cuda().float())
        batch = torch.from_numpy(np.stack(strokes,1)).cuda().float()
    else:
        #batch = Variable(torch.from_numpy(np.stack(strokes,1)).float())
        batch = torch.from_numpy(np.stack(strokes,1)).float()
    return batch, lengths

def make_test_batch(batch_idx):
    #batch_idx = np.random.choice(len(data),batch_size)
    if type(batch_idx) == int:
        batch_idx = [batch_idx]
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        lengths.append(len(seq[:,0]))
        strokes.append(new_seq)
        indice += 1

    if use_cuda:
        #batch = Variable(torch.from_numpy(np.stack(strokes,1)).cuda().float())
        batch = torch.from_numpy(np.stack(strokes,1)).cuda().float()
    else:
        #batch = Variable(torch.from_numpy(np.stack(strokes,1)).float())
        batch = torch.from_numpy(np.stack(strokes,1)).float()
    return batch, lengths
################################ adaptive lr
def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

################################# encoder and decoder modules
class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        # bidirectional lstm:
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, \
            dropout=hp.dropout, bidirectional=True)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            if use_cuda:
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        # N ~ N(0,1)
        z_size = mu.size()
        if use_cuda:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size))
        z = mu + sigma*N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat, N

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size,6*hp.M+3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            #hidden,cell = torch.split(F.tanh(self.fc_hc(z)),hp.dec_hidden_size,1)
            hidden,cell = torch.split(torch.tanh(self.fc_hc(z)),hp.dec_hidden_size,1)

            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params:
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        # preprocess params::
        if self.training:
            len_out = Nmax+1
        else:
            len_out = 1
                                   
        #pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        pi = F.softmax(pi.transpose(0,1).squeeze(), dim=-1).view(len_out,-1,hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,hp.M)
        #q = F.softmax(params_pen).view(len_out,-1,3)
        q = F.softmax(params_pen, dim=1).view(len_out,-1,3)
        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,q,hidden,cell

class Model():
    def __init__(self, args, writer):
        if use_cuda:
            self.encoder = EncoderRNN().cuda()
            self.decoder = DecoderRNN().cuda()
        else:
            self.encoder = EncoderRNN()
            self.decoder = DecoderRNN()
        self.writer = writer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

        self.args = args

    def make_target(self, batch, lengths):
        if use_cuda:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(Nmax+1, batch.size()[1])
        for indice,length in enumerate(lengths):
            mask[:length,indice] = 1
        if use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:,:,0]]*hp.M,2)
        dy = torch.stack([batch.data[:,:,1]]*hp.M,2)
        p1 = batch.data[:,:,2]
        p2 = batch.data[:,:,3]
        p3 = batch.data[:,:,4]
        p = torch.stack([p1,p2,p3],2)
        return mask,dx,dy,p


    @staticmethod
    def gaussian_kernel(a, b):
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)
        b_core = b.expand(dim1_1, dim1_2, depth)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        return torch.exp(-numerator)


    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()


    def mmd_loss(self, latent):
        return self.MMD(torch.randn(200, hp.Nz, requires_grad = False).cuda(), latent)


    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = make_batch(hp.batch_size)
        # encode:
        z, self.mu, self.sigma, _ = self.encoder(batch, hp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch],0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z]*(Nmax+1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack],2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask,dx,dy,p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1-(1-hp.eta_min)*hp.R
        # compute losses:
        LR = self.reconstruction_loss(mask,dx,dy,p,epoch)
        if args.model == 'MMD':
            LMMD = args.dist_lambda * self.mmd_loss(z)
            loss = LR + LMMD
        else:
            LKL = args.dist_lambda * self.kullback_leibler_loss()
            loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch%1==0:
            if args.model == 'MMD':
                print('epoch',epoch,'loss',loss.data.item(),'LR',LR.data.item(),'LMMD',LMMD.data.item())
            else:
                print('epoch',epoch,'loss',loss.data.item(),'LR',LR.data.item(),'LKL',LKL.data.item())

            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        if epoch%10000==0:
            self.save(epoch, self.args)
            self.conditional_generation(epoch)
            self.conditional_generation(epoch, test=True)
            self.interpolate(epoch)
            self.get_pca(epoch)

        if args.model == 'MMD':
            return loss.data.item(), LR.data.item(), LMMD.data.item()
        else:
            return loss.data.item(), LR.data.item(), LKL.data.item()

    def cal_test_loss(self, epoch):
        with torch.no_grad():
            self.encoder.train()
            self.decoder.train()
            batch_idx = np.random.choice(len(test_data),hp.batch_size)
            batch, lengths = make_test_batch(batch_idx)
            # encode:
            z, self.mu, self.sigma, _ = self.encoder(batch, hp.batch_size)
            # create start of sequence:
            if use_cuda:
                sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).cuda().unsqueeze(0)
            else:
                sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).unsqueeze(0)
            # had sos at the begining of the batch:
            batch_init = torch.cat([sos, batch],0)
            # expend z to be ready to concatenate with inputs:
            z_stack = torch.stack([z]*(Nmax+1))
            # inputs is concatenation of z and batch_inputs
            inputs = torch.cat([batch_init, z_stack],2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
            # prepare targets:
            mask,dx,dy,p = self.make_target(batch, lengths)
            # compute losses:
            LR = self.reconstruction_loss(mask,dx,dy,p,epoch)
            if args.model == 'MMD':
                LMMD = args.dist_lambda * self.mmd_loss(z)
                loss = LR + LMMD
            else:
                LKL = args.dist_lambda * self.kullback_leibler_loss()
                loss = LR + LKL

            if args.model == 'MMD':
                print('[TEST] epoch',epoch,'loss',loss.data.item(),'LR',LR.data.item(),'LMMD',LMMD.data.item())
                return loss.data.item(), LR.data.item(), LMMD.data.item()
            else:
                print('[TEST] epoch',epoch,'loss',loss.data.item(),'LR',LR.data.item(),'LKL',LKL.data.item())
                return loss.data.item(), LR.data.item(), LKL.data.item()


    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask*torch.log(1e-5+torch.sum(self.pi * pdf, 2)))\
            /float(Nmax*hp.batch_size)
        LP = -torch.sum(p*torch.log(self.q))/float(Nmax*hp.batch_size)
        return LS+LP

    def kullback_leibler_loss(self):
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            /float(hp.Nz*hp.batch_size)
        if use_cuda:
            #KL_min = Variable(torch.Tensor([hp.KL_min]).cuda()).detach()
            KL_min = torch.Tensor([hp.KL_min]).cuda().detach()
        else:
            #KL_min = Variable(torch.Tensor([hp.KL_min])).detach()
            KL_min = torch.Tensor([hp.KL_min]).detach()
        return hp.wKL*self.eta_step * torch.max(LKL,KL_min)

    def save(self, epoch, args):
        sel = np.random.rand()
        enc_name = os.path.join(args.out_dir, 'encoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        dec_name = os.path.join(args.out_dir, 'decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.encoder.state_dict(), \
                enc_name)
        torch.save(self.decoder.state_dict(), \
                dec_name)

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def get_pca(self, epoch):
        Mus = []
        input_seqs = []
        self.encoder.train(False)
        self.decoder.train(False)

        for idx in range(200):
            batch,lengths = make_batch(1)
            input_seq = make_image_from_batch(batch, epoch, self.args)

            z, M, _, _ = self.encoder(batch, 1)
            Mus.append(np.array(z.squeeze().detach().cpu()))
            input_seqs.append(input_seq)

        Mus = np.array(Mus)
        make_image_pca(input_seqs, Mus, epoch, self.args, name='_pca_200')

    def interpolate(self, epoch):
        pairs = [[1,3],[1,5],[1,9],[1,13],[4,13],[5,7]]
        for idx, pair in enumerate(pairs): 
            batch_1, length_1 = make_test_batch(pair[0])
            batch_2, length_2 = make_test_batch(pair[1])

            seqs = []

            self.encoder.train(False)
            self.decoder.train(False)

            input_seq_1 = make_image_from_batch(batch_1, epoch, self.args)
            input_seq_2 = make_image_from_batch(batch_2, epoch, self.args)

            _, M_1, _, _ = self.encoder(batch_1, 1)
            _, M_2, _, _ = self.encoder(batch_2, 1)

            seqs.append(input_seq_1)

            for i in range(9):

                z = M_1 * (9-i)/10. + M_2 * (i+1)/10.
                if use_cuda:
                    sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda()
                else:
                    sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1)
                s = sos
                seq_x = []
                seq_y = []
                seq_z = []
                hidden_cell = None
                for i in range(Nmax):
                    input = torch.cat([s,z.unsqueeze(0)],2)
                    # decode:
                    self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                        self.rho_xy, self.q, hidden, cell = \
                            self.decoder(input, z, hidden_cell)
                    hidden_cell = (hidden, cell)
                    # sample from parameters:
                    s, dx, dy, pen_down, eos = self.sample_next_state()
                    #------
                    seq_x.append(dx)
                    seq_y.append(dy)
                    seq_z.append(pen_down)
                    if eos:
                        #print(i)
                        break
                # visualize result:
                x_sample = np.cumsum(seq_x, 0)
                y_sample = np.cumsum(seq_y, 0)
                z_sample = np.array(seq_z)
                sequence = np.stack([x_sample,y_sample,z_sample]).T
                seqs.append(sequence)

            seqs.append(input_seq_2)

            fig, ax1 = plt.subplots(1,11, figsize=(11, 1.2))

            for _ax in ax1:
                _ax.set_xticks([])
                _ax.set_yticks([])

            ax1[0].set_title('input 1')
            ax1[10].set_title('input 2')
            #canvas = plt.get_current_fig_manager().canvas
            #canvas.draw()

            # draw images
            for i, seq in enumerate(seqs):
                strokes = np.split(seq, np.where(seq[:,2]>0)[0]+1)
                col = i
                _len = 0
                for s in strokes:
                    ax1[col].plot(s[:,0],-s[:,1])
                canvas = plt.get_current_fig_manager().canvas
                canvas.draw()

            # save img
            pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                         canvas.tostring_rgb())

            _dir = os.path.join(args.out_dir, str(epoch))
            if not os.path.exists(_dir):
                os.mkdir(_dir)
            name = os.path.join(_dir, 'interpolation_'+str(idx)+'.jpg')
            pil_image.save(name,"JPEG")
            plt.close("all")

    def conditional_generation(self, epoch, test=False):
        Mus = []
        input_seqs = []
        for idx in range(20):
            if test:
                batch, lengths = make_test_batch(idx)
            else:
                batch,lengths = make_batch(1)
            input_seq = make_image_from_batch(batch, epoch, self.args)

            # should remove dropouts:
            self.encoder.train(False)
            self.decoder.train(False)

            seqs = []

            z, M, _, _ = self.encoder(batch, 1)
            Mus.append(np.array(z.squeeze().detach().cpu()))
            input_seqs.append(input_seq)

            for i in range(9):
                # encode:
                z, _, _, N = self.encoder(batch, 1)

                if use_cuda:
                    #sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda())
                    sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda()
                else:
                    #sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1))
                    sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1)
                s = sos
                seq_x = []
                seq_y = []
                seq_z = []
                hidden_cell = None
                for i in range(Nmax):
                    input = torch.cat([s,z.unsqueeze(0)],2)
                    # decode:
                    self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                        self.rho_xy, self.q, hidden, cell = \
                            self.decoder(input, z, hidden_cell)
                    hidden_cell = (hidden, cell)
                    # sample from parameters:
                    s, dx, dy, pen_down, eos = self.sample_next_state()
                    #------
                    seq_x.append(dx)
                    seq_y.append(dy)
                    seq_z.append(pen_down)
                    if eos:
                        #print(i)
                        break
                # visualize result:
                x_sample = np.cumsum(seq_x, 0)
                y_sample = np.cumsum(seq_y, 0)
                z_sample = np.array(seq_z)
                sequence = np.stack([x_sample,y_sample,z_sample]).T
                seqs.append(sequence)

            make_images(input_seq, seqs, epoch, self.args, idx, test)
        #make_image(sequence, epoch, self.args)

        Mus = np.array(Mus)
        if test:
            make_image_pca(input_seqs, Mus, epoch, self.args, name='_pca_test_')
        else:
            make_image_pca(input_seqs, Mus, epoch, self.args)

    def sample_next_state(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        # get pen state:
        q = self.q.data[0,0,:].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx].item()
        mu_y = self.mu_y.data[0,0,pi_idx].item()
        sigma_x = self.sigma_x.data[0,0,pi_idx].item()
        sigma_y = self.sigma_y.data[0,0,pi_idx].item()
        rho_xy = self.rho_xy.data[0,0,pi_idx].item()
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        if use_cuda:
            #return Variable(next_state.cuda()).view(1,1,-1),x,y,q_idx==1,q_idx==2
            return next_state.cuda().view(1,1,-1),x,y,q_idx==1,q_idx==2
        else:
            #return Variable(next_state).view(1,1,-1),x,y,q_idx==1,q_idx==2
            return next_state.view(1,1,-1),x,y,q_idx==1,q_idx==2

def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
    # inputs must be floats
    if greedy:
      return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def make_image_from_batch(batch, epoch, args):
    batch = batch.squeeze()
    seq_x = []
    seq_y = []
    seq_z = []
    for i in range(batch.size()[0]):
        dx, dy, p1, p2, eos = batch[i]
        seq_x.append(dx)
        seq_y.append(dy)
        pendown = 1 if p2 else 0
        seq_z.append(pendown)
        if eos:
            break

    x_sample = np.cumsum(seq_x, 0)
    y_sample = np.cumsum(seq_y, 0)
    z_sample = np.array(seq_z)
    sequence = np.stack([x_sample,y_sample,z_sample]).T
    return sequence
    #make_image(sequence, epoch, args, 'input')

def make_image(sequence, epoch, args, name='_output_'):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())
    name = os.path.join(args.out_dir, str(epoch)+name+'.jpg')
    pil_image.save(name,"JPEG")
    plt.close("all")

def make_images(input_seq, seqs, epoch, args, idx=0, test=False):
    #fig = plt.figure()
    #ax1 = fig.add_subplot(251)
    fig, ax1 = plt.subplots(2,5, figsize=(10, 4))

    for _ax in ax1:
        for __ax in _ax:
            __ax.set_xticks([])
            __ax.set_yticks([])

    # input img
    strokes = np.split(input_seq, np.where(input_seq[:,2]>0)[0]+1)
    _len = 0
    for s in strokes:
        ax1[0, 0].plot(s[:,0],-s[:,1])
        _len += len(s[:,0])
    ax1[0, 0].set_title('input, len: ' + str(_len+1))
    #canvas = plt.get_current_fig_manager().canvas
    #canvas.draw()

    # output img
    for i, seq in enumerate(seqs):
        strokes = np.split(seq, np.where(seq[:,2]>0)[0]+1)
        row = (i+1) // 5
        col = (i+1) % 5
        _len = 0
        for s in strokes:
            ax1[row, col].plot(s[:,0],-s[:,1])
            _len += len(s[:,0])
        ax1[row, col].set_title('len: ' +str(_len+1))
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

    # save img
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())

    _dir = os.path.join(args.out_dir, str(epoch))
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    name = os.path.join(_dir, '_test_'+str(idx)+'.jpg') if test else os.path.join(_dir, '_'+str(idx)+'.jpg')
    pil_image.save(name,"JPEG")
    plt.close("all")


def make_image_pca(input_seqs, Mus, epoch, args, name='_pca_'):
    fig, ax1 = plt.subplots(2,20, figsize=(40, 4))

    '''
    for _ax in ax1:
        for __ax in _ax:
            __ax.set_xticks([])
            __ax.set_yticks([])
    '''

    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(Mus)

    for i in range(20):
        ax1[1, i].scatter(pca_res[:,0], pca_res[:,1],s=0.3,c='b')
        ax1[1, i].scatter(pca_res[i,0], pca_res[i,1],c='r')

        ax1[0, i].set_xticks([])
        ax1[0, i].set_yticks([])
        strokes = np.split(input_seqs[i], np.where(input_seqs[i][:,2]>0)[0]+1)
        for s in strokes:
            ax1[0, i].plot(s[:,0],-s[:,1])
    

    # save img
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())

    _dir = os.path.join(args.out_dir, str(epoch))
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    name = os.path.join(_dir, name +'.jpg')
    pil_image.save(name,"JPEG")
    plt.close("all")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--model', default='default', type=str)
    parser.add_argument('--dist_lambda', default=1., type=float)
    parser.add_argument('--load', default=False, type=bool)

    parser.add_argument('--encoder_path', default='', type=str)
    parser.add_argument('--decoder_path', default='', type=str)

    args = parser.parse_args()
    print('args.name: ', args.name)
    print('args.model: ', args.model)
    print('args.dst_lambda: ', args.dist_lambda)
    args.out_dir = os.path.join('out', args.name)

    cwd = os.getcwd()
    _out_dir = os.path.join(cwd, 'out')
    out_dir = os.path.join(_out_dir, args.name)
    if not os.path.exists(_out_dir):
        os.mkdir(_out_dir)
        print('created ', _out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('created ', out_dir)

    writer = SummaryWriter(logdir=out_dir + '/scalar')

    model = Model(args, writer)

    if not args.load:
        print('====start training====')
        for epoch in range(50001):
            L, LR, LD = model.train(epoch)
            L_te, LR_te, LD_te = model.cal_test_loss(epoch)

            writer.add_scalars('loss', {'L_train': L, 'LR_train': LR, 'LD_train': LD, 'L_test': L_te, 'LR_test': LR_te, 'LD_test': LD_te}, epoch)

        writer.close()

    else:

        print('====load and infer only====')
        decoder_path = args.decoder_path
        encoder_path = args.encoder_path

        epoch = int(encoder_path.split('_')[-1][:-4])
        model.load(encoder_path,decoder_path)
        model.interpolate(epoch)
        model.conditional_generation(epoch)

