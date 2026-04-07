#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

def boundary_condition_2D_u(values_u, values_uu, ub):
    ny = values_u.shape[2]
    nx = values_u.shape[3]
    nny = values_uu.shape[2]
    nnx = values_uu.shape[3]

    y_idx = torch.arange(ny, dtype=values_u.dtype, device=values_u.device)
    beta = 0.25
    # U_x = 1.5 * [1 - (|1 - 2*beta*y|)^2],  y in [0, H/D = 1/beta]
    y = y_idx / (beta * (ny - 1))  # map grid indices [0, ny-1] to y in [0, 1/beta]
    u_profile = ub * (1.0 - (torch.abs(1.0 - 2.0 * beta * y)) ** 2)  # shape: (ny,)
    values_uu[0,0,1:nny-1,1:nnx-1] = values_u[0,0,:,:]
    values_uu[0,0,1:nny-1,0] = u_profile   # left ghost column only
    values_uu[0,0,:,nx+1] = values_uu[0,0,:,nx]      # outlet: zero-gradient
    values_uu[0,0,0,:].fill_(0.0)                    # top wall
    values_uu[0,0,ny+1,:].fill_(0.0)                 # bottom wall
    return values_uu
    
def boundary_condition_2D_v(values_v, values_vv, ub):
    ny = values_v.shape[2]
    nx = values_v.shape[3]
    nny = values_vv.shape[2]
    nnx = values_vv.shape[3]

    values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
    values_vv[0,0,:,0].fill_(0.0)
    values_vv[0,0,:,nx+1].fill_(0.0)
    values_vv[0,0,0,:].fill_(0.0)
    values_vv[0,0,ny+1,:].fill_(0.0)
    return values_vv

def boundary_condition_2D_p(values_p, values_pp):
    ny = values_p.shape[2]
    nx = values_p.shape[3]
    nny = values_pp.shape[2]
    nnx = values_pp.shape[3]

    values_pp[0,0,1:nny-1,1:nnx-1] = values_p[0,0,:,:]
    values_pp[0,0,:,0] =  values_pp[0,0,:,1] 
    values_pp[0,0,:,nx+1] = 0.0
    values_pp[0,0,0,:] = values_pp[0,0,1,:]
    values_pp[0,0,ny+1,:] = values_pp[0,0,ny,:]
    return values_pp

def boundary_condition_2D_cw(w):
    ny = w.shape[2]
    nx = w.shape[3]
    ww = F.pad(w, (1, 1, 1, 1), mode='constant', value=0)
    ww[0,0,:,0] =  ww[0,0,:,1]*0
    ww[0,0,:,nx+1] = ww[0,0,:,nx]*0
    ww[0,0,0,:] = ww[0,0,1,:]*0
    ww[0,0,ny+1,:] = ww[0,0,ny,:]*0
    return ww

def boundary_condition_2D_T(values_T, values_TT):
    ny = values_T.shape[2]
    nx = values_T.shape[3]

    values_TT[0,0,1:-1,1:-1] = values_T[0,0,:,:]

    # inlet: cold fluid
    values_TT[0,0,:,0] = 0.0

    # outlet: zero gradient
    values_TT[0,0,:,nx+1] = values_TT[0,0,:,nx]

    # walls: adiabatic
    values_TT[0,0,0,:] = values_TT[0,0,1,:]
    values_TT[0,0,ny+1,:] = values_TT[0,0,ny,:]

    return values_TT



def boundary_condition_3D_u(values_u, values_uu, ub):
    nz = values_u.shape[2]
    ny = values_u.shape[3]
    nx = values_u.shape[4]
    nnz = values_uu.shape[2]
    nny = values_uu.shape[3]
    nnx = values_uu.shape[4]

    values_uu[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_u[0,0,:,:,:]
    values_uu[0,0,:,:,0].fill_(ub)
    values_uu[0,0,:,:,nx+1].fill_(ub)
    values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:]
    values_uu[0,0,:,ny+1,:] = values_uu[0,0,:,ny,:]
    values_uu[0,0,0,:,:].fill_(0.0)
    values_uu[0,0,nz+1,:,:] = values_uu[0,0,nz,:,:]
    return values_uu
    
def boundary_condition_3D_v(values_v, values_vv, ub):
    nz = values_v.shape[2]
    ny = values_v.shape[3]
    nx = values_v.shape[4]
    nnz = values_vv.shape[2]
    nny = values_vv.shape[3]
    nnx = values_vv.shape[4]

    values_vv[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v[0,0,:,:,:]
    values_vv[0,0,:,:,0].fill_(0.0)
    values_vv[0,0,:,:,nx+1].fill_(0.0)
    values_vv[0,0,:,0,:].fill_(0.0)
    values_vv[0,0,:,ny+1,:].fill_(0.0)
    values_vv[0,0,0,:,:].fill_(0.0)
    values_vv[0,0,nz+1,:,:] = values_vv[0,0,nz,:,:]
    return values_vv

def boundary_condition_3D_w(values_w, values_ww, ub):
    nz = values_w.shape[2]
    ny = values_w.shape[3]
    nx = values_w.shape[4]
    nnz = values_ww.shape[2]
    nny = values_ww.shape[3]
    nnx = values_ww.shape[4]

    values_ww[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_w[0,0,:,:,:]
    values_ww[0,0,:,:,0].fill_(0.0)
    values_ww[0,0,:,:,nx+1].fill_(0.0)
    values_ww[0,0,:,0,:].fill_(0.0)
    values_ww[0,0,:,ny+1,:].fill_(0.0)
    values_ww[0,0,0,:,:].fill_(0.0)
    values_ww[0,0,nz+1,:,:] = values_ww[0,0,nz,:,:]
    return values_ww

def boundary_condition_3D_p(values_p, values_pp):
    nz = values_p.shape[2]
    ny = values_p.shape[3]
    nx = values_p.shape[4]
    nnz = values_pp.shape[2]
    nny = values_pp.shape[3]
    nnx = values_pp.shape[4]

    values_pp[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_p[0,0,:,:,:]
    values_pp[0,0,:,:,0] =  values_pp[0,0,:,:,1] 
    values_pp[0,0,:,:,nx+1].fill_(0.0)
    values_pp[0,0,:,0,:] = values_pp[0,0,:,1,:]
    values_pp[0,0,:,ny+1,:] = values_pp[0,0,:,ny,:]
    values_pp[0,0,0,:,:] = values_pp[0,0,1,:,:]
    values_pp[0,0,nz+1,:,:] = values_pp[0,0,nz,:,:]
    return values_pp

def boundary_condition_3D_k(k_u):
    k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    return k_uu

def boundary_condition_3D_cw(w):
    ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
    return ww