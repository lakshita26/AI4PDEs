import torch

# ===============================
# U VELOCITY BC
# ===============================
def boundary_condition_2D_u(u, uu, U, sigma=None):
    """
    Applies boundary conditions for the horizontal velocity (u).
    """
    uu[:,:,1:-1,1:-1] = u

    # 1. INLET: Uniform flow at infinity (u = U) [cite: 119]
    uu[:,:,:,0] = U

    # 2. OUTLET: Zero-gradient (standard for Cartesian downstream approximation)
    uu[:,:,:,-1] = uu[:,:,:,-2]

    # 3. TOP & BOTTOM: Periodic condition [cite: 122]
    # "A periodic condition is used on the upper and lower horizontal boundaries" [cite: 122]
    uu[:,:,0,:] = uu[:,:,-2,:]
    uu[:,:,-1,:] = uu[:,:,1,:]

    # 4. BODY SURFACE: No-slip condition (u = 0) 
    if sigma is not None:
        interior = uu[0,0,1:-1,1:-1]
        interior[sigma[0,0] == 1] = 0.0

    return uu


# ===============================
# V VELOCITY BC
# ===============================
def boundary_condition_2D_v(v, vv, U=None, sigma=None):
    """
    Applies boundary conditions for the vertical velocity (v).
    """
    vv[:,:,1:-1,1:-1] = v

    # 1. INLET: Uniform flow at infinity (v = 0) [cite: 119]
    vv[:,:,:,0] = 0.0
    
    # 2. OUTLET: Zero-gradient downstream
    vv[:,:,:,-1] = vv[:,:,:,-2]

    # 3. TOP & BOTTOM: Periodic condition [cite: 122]
    vv[:,:,0,:] = vv[:,:,-2,:]
    vv[:,:,-1,:] = vv[:,:,1,:]

    # 4. BODY SURFACE: No-slip condition (v = 0) 
    if sigma is not None:
        interior = vv[0,0,1:-1,1:-1]
        interior[sigma[0,0] == 1] = 0.0

    return vv


# ===============================
# PRESSURE BC
# ===============================
def boundary_condition_2D_p(p, pp):
    """
    Applies boundary conditions for the pressure field.
    """
    pp[:,:,1:-1,1:-1] = p

    # 1. INLET: Zero normal gradient
    pp[:,:,:,0] = pp[:,:,:,1]

    # 2. OUTLET: Reference pressure (p = 0)
    pp[:,:,:,-1] = 0.0

    # 3. TOP & BOTTOM: Periodic condition [cite: 122]
    pp[:,:,0,:] = pp[:,:,-2,:]
    pp[:,:,-1,:] = pp[:,:,1,:]
    
    # Note: The paper states pressure at the body can be obtained by extrapolation[cite: 121].
    # In this Cartesian fractional-step method, the pressure field inherently resolves 
    # around the penalty term (sigma) applied to the velocity field, so explicit 
    # Neumann wall boundaries for pressure are handled implicitly by the Poisson equation.

    return pp

# ===============================
# PRESSURE CORRECTION WEIGHT (MG) BC
# ===============================
def boundary_condition_2D_cw(w):
    """
    Applies homogeneous boundary conditions for the multigrid pressure correction (w).
    This was missing from your original file but is required by the solver.
    """
    ww = torch.zeros((w.shape[0], w.shape[1], w.shape[2]+2, w.shape[3]+2), device=w.device)
    ww[:,:,1:-1,1:-1] = w

    # Inlet: Homogeneous Neumann
    ww[:,:,:,0] = ww[:,:,:,1]
    
    # Outlet: Homogeneous Dirichlet
    ww[:,:,:,-1] = 0.0
    
    # Top & Bottom: Periodic [cite: 122]
    ww[:,:,0,:] = ww[:,:,-2,:]
    ww[:,:,-1,:] = ww[:,:,1,:]
    
    return ww