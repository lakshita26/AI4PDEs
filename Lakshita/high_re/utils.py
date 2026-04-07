import torch
import torch.nn as nn
import torch.nn.functional as F

def create_tensors_2D(nx, ny):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 1, ny, nx)
    input_shape_pad = (1, 1, ny + 2, nx + 2)
    
    values_u = torch.zeros(input_shape, device=device)
    values_v = torch.zeros(input_shape, device=device)
    values_p = torch.zeros(input_shape, device=device)
    
    values_uu = torch.zeros(input_shape_pad, device=device)
    values_vv = torch.zeros(input_shape_pad, device=device)
    values_pp = torch.zeros(input_shape_pad, device=device)
    
    return values_u, values_v, values_p, values_uu, values_vv, values_pp

def create_solid_body_2D(nx, ny, center_x, center_y, radius):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = torch.zeros((1, 1, ny, nx), device=device)
    Y, X = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    
    dist = (X - center_x)**2 + (Y - center_y)**2
    mask = dist <= radius**2
    sigma[0, 0, mask] = 100.0
    return sigma


def create_solid_body_3D(nx, ny, nz, center_x, center_y, radius):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = torch.zeros((1, 1, nz, ny, nx), device=device)
    # Z is spanwise, Y is vertical, X is streamwise
    Z, Y, X = torch.meshgrid(torch.arange(nz, device=device), 
                             torch.arange(ny, device=device), 
                             torch.arange(nx, device=device), indexing='ij')
    # Cylinder is aligned with the Z-axis
    dist = (X - center_x)**2 + (Y - center_y)**2
    mask = dist <= radius**2
    sigma[0, 0, mask] = 100.0
    return sigma