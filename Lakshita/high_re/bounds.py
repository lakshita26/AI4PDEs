#!/usr/bin/env python
import torch
import torch.nn.functional as F

# =========================================================
# ======================= 2D ===============================
# =========================================================

def boundary_condition_2D_u(values_u, values_uu, values_u_old, ub, dt, dx, Uc):
    ny, nx = values_u.shape[2], values_u.shape[3]
    values_uu[0, 0, 1:ny+1, 1:nx+1] = values_u[0, 0, :, :]

    # Inlet: Fixed free-stream velocity
    values_uu[0, 0, 1:ny+1, 0] = ub

    # Outlet: Convective BC (Non-reflective) 
    # We use nx-1 as the interior reference point for better stability
    values_uu[0, 0, 1:ny+1, nx+1] = (
        values_u_old[0, 0, :, nx-1]
        - Uc * dt/dx * (values_u_old[0, 0, :, nx-1] - values_u_old[0, 0, :, nx-2])
    )

    # Far-field (top/bottom): Slip Wall / Neumann [cite: 117]
    values_uu[0, 0, 0, 1:nx+1] = values_uu[0, 0, 1, 1:nx+1]
    values_uu[0, 0, ny+1, 1:nx+1] = values_uu[0, 0, ny, 1:nx+1]

    return values_uu

def boundary_condition_2D_v(values_v, values_vv, values_v_old, ub, dt, dx, Uc):
    ny, nx = values_v.shape[2], values_v.shape[3]
    values_vv[0, 0, 1:ny+1, 1:nx+1] = values_v[0, 0, :, :]

    # Inlet: v = 0
    values_vv[0, 0, 1:ny+1, 0] = 0.0

    # Outlet: Convective BC [cite: 117]
    values_vv[0, 0, 1:ny+1, nx+1] = (
        values_v_old[0, 0, :, nx-1]
        - Uc * dt/dx * (values_v_old[0, 0, :, nx-1] - values_v_old[0, 0, :, nx-2])
    )

    # Far-field
    values_vv[0, 0, 0, 1:nx+1] = values_vv[0, 0, 1, 1:nx+1]
    values_vv[0, 0, ny+1, 1:nx+1] = values_vv[0, 0, ny, 1:nx+1]

    return values_vv

def boundary_condition_2D_p(values_p, values_pp):
    ny, nx = values_p.shape[2], values_p.shape[3]
    values_pp[0, 0, 1:ny+1, 1:nx+1] = values_p[0, 0, :, :]

    # Inlet: Neumann (dp/dx = 0)
    values_pp[0, 0, 1:ny+1, 0] = values_pp[0, 0, 1:ny+1, 1]

    # Outlet: Dirichlet (Reference pressure = 0)
    values_pp[0, 0, 1:ny+1, nx+1] = 0.0

    # Far-field: Neumann
    values_pp[0, 0, 0, 1:nx+1] = values_pp[0, 0, 1, 1:nx+1]
    values_pp[0, 0, ny+1, 1:nx+1] = values_pp[0, 0, ny, 1:nx+1]

    return values_pp

# =========================================================
# WALL STRESS MODEL (Equation 2 & 3 from Paper)
# =========================================================

def boundary_condition_wall_model(u_mag, distance_to_wall, nu):
    """
    Implements the mixing length wall model with near-wall damping 
    as described in Catalano et al. (2003) [cite: 56, 80].
    """
    # Physical constants from the paper [cite: 83]
    kappa = 0.4  # von Karman constant
    A = 19.0     # Damping constant
    
    # 1. Estimate u_tau (friction velocity)
    # We use a square-root approximation for the initial guess
    u_tau = torch.sqrt(nu * torch.abs(u_mag) / (distance_to_wall + 1e-9))
    
    # 2. Calculate y+ (wall units) [cite: 83]
    y_plus = (distance_to_wall * u_tau) / nu
    
    # 3. Mixing length eddy viscosity (nu_t) with damping [cite: 81]
    # vt/v = kappa * y+ * (1 - exp(-y+/A))^2
    damping = (1.0 - torch.exp(-y_plus / A))**2
    nu_t = nu * (kappa * y_plus * damping)
    
    # 4. Total effective viscosity [cite: 57]
    nu_eff = nu + nu_t
    
    # 5. Wall shear stress (tau_w) [cite: 72]
    # tau_w = rho * (nu + nu_t) * (du/dy)
    tau_w = nu_eff * (u_mag / (distance_to_wall + 1e-9))
    
    return tau_w

# =========================================================
# ======================= 3D ===============================
# =========================================================

def boundary_condition_3D_u(values_u, values_uu, values_u_old, ub, dt, dx, Uc):
	nz = values_u.shape[2]
	ny = values_u.shape[3]
	nx = values_u.shape[4]

	values_uu[0,0,1:nz+1,1:ny+1,1:nx+1] = values_u[0,0,:,:,:]

	# Inlet
	values_uu[0,0,:,:,0].fill_(ub)

	# Outlet (Convective BC)
	values_uu[0,0,:,:,nx+1] = (
		values_u_old[0,0,:,:,nx]
		- Uc * dt/dx * (values_u_old[0,0,:,:,nx] - values_u_old[0,0,:,:,nx-1])
	)

	# Top/Bottom
	values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:]
	values_uu[0,0,:,ny+1,:] = values_uu[0,0,:,ny,:]

	# Periodic (spanwise)
	values_uu[0,0,0,:,:] = values_uu[0,0,nz,:,:]
	values_uu[0,0,nz+1,:,:] = values_uu[0,0,1,:,:]

	return values_uu


def boundary_condition_3D_v(values_v, values_vv, values_v_old, ub, dt, dx, Uc):
	nz = values_v.shape[2]
	ny = values_v.shape[3]
	nx = values_v.shape[4]

	values_vv[0,0,1:nz+1,1:ny+1,1:nx+1] = values_v[0,0,:,:,:]

	values_vv[0,0,:,:,0].fill_(0.0)

	values_vv[0,0,:,:,nx+1] = (
		values_v_old[0,0,:,:,nx]
		- Uc * dt/dx * (values_v_old[0,0,:,:,nx] - values_v_old[0,0,:,:,nx-1])
	)

	values_vv[0,0,:,0,:] = values_vv[0,0,:,1,:]
	values_vv[0,0,:,ny+1,:] = values_vv[0,0,:,ny,:]

	values_vv[0,0,0,:,:] = values_vv[0,0,nz,:,:]
	values_vv[0,0,nz+1,:,:] = values_vv[0,0,1,:,:]

	return values_vv


def boundary_condition_3D_w(values_w, values_ww, values_w_old, ub, dt, dx, Uc):
	nz = values_w.shape[2]
	ny = values_w.shape[3]
	nx = values_w.shape[4]

	values_ww[0,0,1:nz+1,1:ny+1,1:nx+1] = values_w[0,0,:,:,:]

	values_ww[0,0,:,:,0].fill_(0.0)

	values_ww[0,0,:,:,nx+1] = (
		values_w_old[0,0,:,:,nx]
		- Uc * dt/dx * (values_w_old[0,0,:,:,nx] - values_w_old[0,0,:,:,nx-1])
	)

	values_ww[0,0,:,0,:] = values_ww[0,0,:,1,:]
	values_ww[0,0,:,ny+1,:] = values_ww[0,0,:,ny,:]

	values_ww[0,0,0,:,:] = values_ww[0,0,nz,:,:]
	values_ww[0,0,nz+1,:,:] = values_ww[0,0,1,:,:]

	return values_ww


def boundary_condition_3D_p(values_p, values_pp):
	nz = values_p.shape[2]
	ny = values_p.shape[3]
	nx = values_p.shape[4]

	values_pp[0,0,1:nz+1,1:ny+1,1:nx+1] = values_p[0,0,:,:,:]

	values_pp[0,0,:,:,0] = values_pp[0,0,:,:,1]
	values_pp[0,0,:,:,nx+1].fill_(0.0)

	values_pp[0,0,:,0,:] = values_pp[0,0,:,1,:]
	values_pp[0,0,:,ny+1,:] = values_pp[0,0,:,ny,:]

	values_pp[0,0,0,:,:] = values_pp[0,0,nz,:,:]
	values_pp[0,0,nz+1,:,:] = values_pp[0,0,1,:,:]

	return values_pp


def boundary_condition_3D_k(k_u):
	return F.pad(k_u, (1,1,1,1,1,1), mode='constant', value=0)


def boundary_condition_3D_cw(w):
	return F.pad(w, (1,1,1,1,1,1), mode='constant', value=0)