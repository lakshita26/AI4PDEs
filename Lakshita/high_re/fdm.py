#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt

from bounds import *
from utils import *

# =========================================================
#  MAIN SIMULATION LOOP
# =========================================================
def run_fd_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Parameters
    nx, ny = 200, 80
    dx = dy = 1.0 / ny
    dt = 0.0005
    # nu = 0.002           # Re = 100
    nu = 0.001
    max_iter = 3000
    ub = 1.0    
    Uc = 1.0    

    # Initialize Tensors
    values_u, values_v, values_p, values_uu, values_vv, values_pp = create_tensors_2D(nx, ny)
    u_old = torch.zeros_like(values_u)
    v_old = torch.zeros_like(values_v)

    # Create Cylinder
    cyl_x, cyl_y, cyl_r = nx // 4, ny // 2, ny // 10
    sigma = create_solid_body_2D(nx, ny, center_x=cyl_x, center_y=cyl_y, radius=cyl_r)
    mask = sigma[0, 0] > 0.0

    # --- NEW: Create a "halo" mask for the wall stress model ---
    Y, X = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    dist_from_center = torch.sqrt((X - cyl_x)**2 + (Y - cyl_y)**2)
    # Identify fluid cells immediately touching the solid cylinder
    halo_mask = (dist_from_center > cyl_r) & (dist_from_center <= cyl_r + 1.5)
    # Calculate exact distance from those cell centers to the cylinder wall
    distance_to_wall = (dist_from_center[halo_mask] - cyl_r) * dx

    print(f"Starting simulation at high Reynolds number (nu={nu}) with Wall Stress Model...")

    for step in range(max_iter):
        
        # 1. Apply BCs
        values_uu = boundary_condition_2D_u(values_u, values_uu, u_old, ub, dt, dx, Uc)
        values_vv = boundary_condition_2D_v(values_v, values_vv, v_old, ub, dt, dx, Uc)

        u_pad = values_uu[0, 0]
        v_pad = values_vv[0, 0]

        # 2. Spatial Derivatives
        u_x = (u_pad[1:-1, 2:] - u_pad[1:-1, :-2]) / (2 * dx)
        u_y = (u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]) / (2 * dy)
        v_x = (v_pad[1:-1, 2:] - v_pad[1:-1, :-2]) / (2 * dx)
        v_y = (v_pad[2:, 1:-1] - v_pad[:-2, 1:-1]) / (2 * dy)

        lap_u = (u_pad[1:-1, 2:] - 2*u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2])/dx**2 + \
                (u_pad[2:, 1:-1] - 2*u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1])/dy**2
                
        lap_v = (v_pad[1:-1, 2:] - 2*v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2])/dx**2 + \
                (v_pad[2:, 1:-1] - 2*v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1])/dy**2

        # 3. Predictor Step
        u_c = u_pad[1:-1, 1:-1]
        v_c = v_pad[1:-1, 1:-1]

        u_old.copy_(values_u)
        v_old.copy_(values_v)

        u_star = u_c + dt * (-(u_c * u_x + v_c * u_y) + nu * lap_u)
        v_star = v_c + dt * (-(u_c * v_x + v_c * v_y) + nu * lap_v)

        # --- NEW: Apply Wall Stress Model ---
        # Get fluid velocity magnitude at the halo cells (add epsilon to avoid div by zero)
        u_tangential = torch.sqrt(u_c[halo_mask]**2 + v_c[halo_mask]**2) + 1e-8
        
        # Calculate wall shear stress using the equation from bounds.py
        tau_w = boundary_condition_wall_model(u_tangential, distance_to_wall, nu)
        
        # Apply tau_w as a deceleration force opposite to flow direction
        u_star[halo_mask] -= dt * (tau_w / distance_to_wall) * (u_c[halo_mask] / u_tangential)
        v_star[halo_mask] -= dt * (tau_w / distance_to_wall) * (v_c[halo_mask] / u_tangential)

        # Enforce zero velocity inside the deep solid mask so fluid doesn't bleed through
        u_star[mask] = 0.0
        v_star[mask] = 0.0

        values_u[0, 0] = u_star
        values_v[0, 0] = v_star

        # 4. Pressure Poisson Equation
        values_uu = boundary_condition_2D_u(values_u, values_uu, u_old, ub, dt, dx, Uc)
        values_vv = boundary_condition_2D_v(values_v, values_vv, v_old, ub, dt, dx, Uc)
        
        div_u = (values_uu[0, 0, 1:-1, 2:] - values_uu[0, 0, 1:-1, :-2]) / (2 * dx) + \
                (values_vv[0, 0, 2:, 1:-1] - values_vv[0, 0, :-2, 1:-1]) / (2 * dy)
        
        rhs_p = div_u / dt

        for _ in range(40):
            values_pp = boundary_condition_2D_p(values_p, values_pp)
            p_pad = values_pp[0, 0]
            values_p[0, 0] = 0.25 * (p_pad[1:-1, 2:] + p_pad[1:-1, :-2] + 
                                     p_pad[2:, 1:-1] + p_pad[:-2, 1:-1] - 
                                     dx**2 * rhs_p)

        # 5. Corrector Step
        values_pp = boundary_condition_2D_p(values_p, values_pp)
        p_pad = values_pp[0, 0]
        
        p_x = (p_pad[1:-1, 2:] - p_pad[1:-1, :-2]) / (2 * dx)
        p_y = (p_pad[2:, 1:-1] - p_pad[:-2, 1:-1]) / (2 * dy)

        values_u[0, 0] -= dt * p_x
        values_v[0, 0] -= dt * p_y

        values_u[0, 0][mask] = 0.0
        values_v[0, 0][mask] = 0.0

        if step % 250 == 0:
            print(f"Step {step}/{max_iter} computed.")

    # =========================================================
    # 4. PLOTTING
    # =========================================================
    u_final = values_u[0, 0].cpu().numpy()
    v_final = values_v[0, 0].cpu().numpy()
    velocity_mag = np.sqrt(u_final**2 + v_final**2)

    # --- Plot 1: Velocity Magnitude ---
    plt.figure(figsize=(12, 4))
    plt.contourf(velocity_mag, 50, cmap='jet')
    
    circle1 = plt.Circle((cyl_x, cyl_y), cyl_r, color='black')
    plt.gca().add_patch(circle1)
    
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Flow Channel with Cylindrical Obstacle (Magnitude)')
    plt.savefig('flow_channel_mag.png', dpi=200)
    print("Plot saved to flow_channel_mag.png")

    # --- Plot 2: Streamlines ---
    x_grid = np.arange(nx)
    y_grid = np.arange(ny)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    plt.figure(figsize=(12, 4))
    plt.streamplot(X_grid, Y_grid, u_final, v_final, color=velocity_mag, cmap='jet', density=2.0, linewidth=1)
    
    circle2 = plt.Circle((cyl_x, cyl_y), cyl_r, color='black', zorder=3)
    plt.gca().add_patch(circle2)
    
    plt.title('Streamlines Around the Cylindrical Obstacle')
    plt.xlim(0, nx - 1)
    plt.ylim(0, ny - 1)
    plt.gca().set_aspect('equal')
    
    plt.savefig('flow_channel_streamlines.png', dpi=200)
    print("Plot saved to flow_channel_streamlines.png")

if __name__ == '__main__':
    with torch.no_grad():
        run_fd_simulation()