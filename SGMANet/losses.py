import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, weight):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        cc_weighted = cc * weight

        return -torch.mean(cc_weighted)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, weight):
        mse = (y_true - y_pred) ** 2
        mse_weighted = mse * weight
        return torch.mean(mse_weighted)


class Dice:
    """
    N-D dice for segmentation
    """
    def __init__(self, inshape):
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))

    def loss(self, y_true, y_pred, distance_map=None, balance=0):
        top = 2 * (y_true * y_pred).sum(dim=self.vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=self.vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1-dice


# class Grad:
#     """
#     N-D gradient loss.
#     """

#     def __init__(self, inshape, penalty='l1', loss_mult=None):
#         self.penalty = penalty
#         self.loss_mult = loss_mult
#         self.ndims = len(inshape)

#     def _diffs(self, y):

#         df = [None] * self.ndims
#         for i in range(self.ndims):
#             d = i + 2
#             # permute dimensions
#             r = [d, *range(0, d), *range(d + 1, self.ndims + 2)]
#             y = y.permute(r)
#             dfi = y[1:, ...] - y[:-1, ...]

#             # permute back
#             # note: this might not be necessary for this loss specifically,
#             # since the results are just summed over anyway.
#             r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, self.ndims + 2)]
#             df[i] = dfi.permute(r)

#         return df

#     def loss(self, _, y_pred):
#         if self.penalty == 'l1':
#             dif = [torch.abs(f) for f in self._diffs(y_pred)]
#         else:
#             assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
#             dif = [f * f for f in self._diffs(y_pred)]

#         df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
#         grad = sum(df) / len(df)

#         if self.loss_mult is not None:
#             grad *= self.loss_mult

#         return grad.mean()

'''
Regularizers
'''
class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self,  y_true, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class mind_ssc():
    def __init__(self, win=None, radius=2, dilation=2):
        super(mind_ssc, self).__init__()

        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        
        self.mshift1 = mshift1
        self.mshift2 = mshift2
        self.rpad1 = nn.ReplicationPad3d(dilation)
        self.rpad2 = nn.ReplicationPad3d(radius)

        self.win = win
        self.dilation = dilation
        self.radius = radius
        self.kernel_size = kernel_size

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img):
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2(
            (F.conv3d(self.rpad1(img), self.mshift1, dilation=self.dilation) 
             - F.conv3d(self.rpad1(img), self.mshift2, dilation=self.dilation)) ** 2),
                           self.kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def loss(self, y_true, y_pred, weight):
        describe_pred = self.MINDSSC(y_pred)
        describe_true = self.MINDSSC(y_true)
        describe_error = (describe_pred - describe_true) ** 2
        describe_error_weighted = describe_error * weight
        return torch.mean(describe_error_weighted)


class boundary_loss():
    def __init__(self, inshape):
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))

    def loss(self, y_true, y_pred, distance_map, balance):
 
        multiplied = (distance_map * y_pred).sum(dim=self.vol_axes)
 
        loss = multiplied.mean()
 
        return loss
    
class dice_blundary_comb():
    def __init__(self, inshape):
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))

    def loss(self,  y_true, y_pred, distance_map, balance):

        if balance != 1 :
            top = 2 * (y_true * y_pred).sum(dim=self.vol_axes)
            bottom = torch.clamp((y_true + y_pred).sum(dim=self.vol_axes), min=1e-5)
            dice = 1 - torch.mean(top / bottom)
        else:
            dice = 0

        if balance != 0 :
            multiplied = (distance_map * y_pred).sum(dim=self.vol_axes)
            multiplied = multiplied.mean()
        else:
            multiplied = 0
 
        loss = (1 - balance) * dice + balance * multiplied
 
        return loss


class MutualInformation(nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred, weight):
        return -self.mi(y_true, y_pred)
    
class MINE(nn.Module):
    def __init__(self, hidden_input = 30, hidden_output = 30, shuffle_mode = "global"):
        super().__init__()

        self.T = nn.Sequential(

            nn.Conv3d(2, hidden_input, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_input, hidden_output, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_output, 1, 1, 1, 0),
        )
        self.shuffle_mode = shuffle_mode

    def forward(self, y_true, y_pred, weight=None,  mask=None):
        if self.shuffle_mode == "global":
            batch_size = y_true.shape[0]
            idx = torch.randperm(y_true[0].nelement())
            target_shuffle = y_true.view(batch_size, -1)
            target_shuffle = target_shuffle[:, idx]
            target_shuffle = target_shuffle.view(y_true.size())
        elif self.shuffle_mode == "mask":
            B, C, D, H, W = y_true.size()
            assert((B==1) and (C == 1))

            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            mask = y_true if mask is None else mask.flatten()
            y_true = y_true[mask != 0]
            y_pred = y_pred[mask != 0]

            idx = torch.randperm(y_true.nelement())
            target_shuffle = y_true[idx]

            y_true = y_true.unsqueeze(0).unsqueeze(1)
            y_pred = y_pred.unsqueeze(0).unsqueeze(1)
            target_shuffle = target_shuffle.unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError()


        t = self.T(torch.cat((y_pred, y_true), dim=1))
        t_marg = self.T(torch.cat((y_pred, target_shuffle), dim=1))

        second_term = torch.exp(t_marg - 1)
            
        mi_result = t - second_term

        if mi_result.ndim == 5:
            mi_result = mi_result.mean((1,2,3,4))
        else:
            mi_result = mi_result.mean((1,2,3))

        return mi_result
    
    def loss(self, y_true, y_pred, weight=None):
        mi = self(y_true, y_pred)

        return mi.mean() * (-1)

    # def weights_init(self):
    #     for m in self.modules():
    #         classname = m.__class__.__name__
    #         if classname.find('Conv') != -1:
    #             if not m.weight is None:
    #                 nn.init.xavier_normal_(m.weight.data)
    #             if not m.bias is None:
    #                 m.bias.data.zero_()


