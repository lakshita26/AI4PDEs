#!/usr/bin/env python
import numpy as np
import torch

def create_tensors_2D(nx, ny):
    input_shape = (1, 1, ny, nx)
    input_shape_pad = (1, 1, ny + 2, nx + 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    values_u = torch.zeros(input_shape, device=device)
    values_v = torch.zeros(input_shape, device=device)
    values_p = torch.zeros(input_shape, device=device)
    
    values_uu = torch.zeros(input_shape_pad, device=device)
    values_vv = torch.zeros(input_shape_pad, device=device)
    values_pp = torch.zeros(input_shape_pad, device=device)
    
    b_uu = torch.zeros(input_shape_pad, device=device)
    b_vv = torch.zeros(input_shape_pad, device=device)
    
    print('All the required 2D tensors have been created successfully!')
    return values_u, values_v, values_p, values_uu, values_vv, values_pp, b_uu, b_vv

def get_weights_linear_2D(dx):
    w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2], [1/3/dx**2]],
                        [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                        [[1/3/dx**2], [1/3/dx**2], [1/3/dx**2]]]])

    w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
                        [[1/(3*dx)], [0.0], [-1/(3*dx)]],
                        [[1/(12*dx)], [0.0], [-1/(12*dx)]]]])

    w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
                        [[0.0], [0.0], [0.0]],
                        [[1/(12*dx)], [1/(3*dx)], [1/(12*dx)]]]])

    wA = torch.tensor([[[[-1/3/dx**2], [-1/3/dx**2], [-1/3/dx**2]],
                        [[-1/3/dx**2], [8/3/dx**2], [-1/3/dx**2]],
                        [[-1/3/dx**2], [-1/3/dx**2], [-1/3/dx**2]]]])

    w1 = torch.reshape(w1, (1,1,3,3))
    w2 = torch.reshape(w2, (1,1,3,3))
    w3 = torch.reshape(w3, (1,1,3,3))
    wA = torch.reshape(wA, (1,1,3,3)) 
    
    w_res = torch.zeros([1,1,2,2]) 
    w_res[0,0,:,:] = 0.25
    diag = np.array(wA)[0,0,1,1] 
    
    print('All the required 2D filters have been created successfully!')
    return w1, w2, w3, wA, w_res, diag

def create_circular_body_2D(nx, ny, cor_x, cor_y, radius):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma = torch.zeros((1,1,ny,nx), device=device)

    y = torch.arange(0, ny, device=device).view(-1,1)
    x = torch.arange(0, nx, device=device).view(1,-1)

    dist = (x - cor_x)**2 + (y - cor_y)**2

    mask = dist <= radius**2

    # ✅ IMPORTANT CHANGE
    sigma[0,0,mask] = 1.0   # <-- NOT 1e8 anymore

    print('Circular cylinder created!')
    return sigma