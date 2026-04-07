
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# FIXED VERSION: External Flow (Catalano-style BCs)
# ==========================================================

def run_cnn_fluid_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running solver on: {device}")

    # ---------------- PARAMETERS ----------------
    nx, ny = 200, 80
    dx = dy = 1.0 / ny
    dt = 0.0005
    nu = 0.001
    max_iter = 10000
    Uc = 1.0

    # ---------------- INITIALIZATION ----------------
    u = torch.zeros((1,1,ny,nx), device=device)
    v = torch.zeros((1,1,ny,nx), device=device)
    p = torch.zeros((1,1,ny,nx), device=device)

    # Cylinder
    cyl_x, cyl_y, cyl_r = nx//4, ny//2, ny//10
    Y, X = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    dist = torch.sqrt((X-cyl_x)**2 + (Y-cyl_y)**2)
    solid_mask = (dist <= cyl_r).view(1,1,ny,nx)

    # ---------------- KERNELS ----------------
    k_dx = torch.tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]], device=device, dtype=torch.float32) / dx
    k_dy = torch.tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]], device=device, dtype=torch.float32) / dy
    k_lap = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], device=device, dtype=torch.float32) / (dx**2)
    k_jac = torch.tensor([[[[0,1,0],[1,0,1],[0,1,0]]]], device=device, dtype=torch.float32)
    
    # ---------------- BC FUNCTION ----------------
    def apply_bcs(u_field, v_field):
        # Inlet
        u_field[:,:,:,0] = 1.0
        v_field[:,:,:,0] = 0.0

        # Convective outlet
        u_field[:,:,:,-1] = u_field[:,:,:,-2] - Uc*dt/dx*(u_field[:,:,:,-2] - u_field[:,:,:,-3])
        v_field[:,:,:,-1] = v_field[:,:,:,-2] - Uc*dt/dx*(v_field[:,:,:,-2] - v_field[:,:,:,-3])

        # Far-field (top/bottom = free-slip)
        u_field[:,:,0,:] = u_field[:,:,1,:]
        u_field[:,:,-1,:] = u_field[:,:,-2,:]
        v_field[:,:,0,:] = v_field[:,:,1,:]
        v_field[:,:,-1,:] = v_field[:,:,-2,:]

        # Cylinder
        u_field[solid_mask] = 0.0
        v_field[solid_mask] = 0.0

        return u_field, v_field

    print("Starting simulation...")

    for step in range(max_iter):
        u, v = apply_bcs(u, v)

        u_pad = F.pad(u,(1,1,1,1),'replicate')
        v_pad = F.pad(v,(1,1,1,1),'replicate')

        u_x = F.conv2d(u_pad, k_dx)
        u_y = F.conv2d(u_pad, k_dy)
        v_x = F.conv2d(v_pad, k_dx)
        v_y = F.conv2d(v_pad, k_dy)

        lap_u = F.conv2d(u_pad, k_lap)
        lap_v = F.conv2d(v_pad, k_lap)

        # Predictor
        u_star = u + dt * (-(u*u_x + v*u_y) + nu*lap_u)
        v_star = v + dt * (-(u*v_x + v*v_y) + nu*lap_v)

        u_star, v_star = apply_bcs(u_star, v_star)

        # Pressure Poisson
        u_pad = F.pad(u_star,(1,1,1,1),'replicate')
        v_pad = F.pad(v_star,(1,1,1,1),'replicate')

        rhs = (F.conv2d(u_pad,k_dx) + F.conv2d(v_pad,k_dy)) / dt

        for _ in range(60):
            p = 0.25*(F.conv2d(F.pad(p,(1,1,1,1),'replicate'), k_jac) - dx**2*rhs)
            p[:,:,:,-1] = 0
            p[:,:,:,0] = p[:,:,:,1]
            p[:,:,0,:] = p[:,:,1,:]
            p[:,:,-1,:] = p[:,:,-2,:]

        # Correction
        p_pad = F.pad(p,(1,1,1,1),'replicate')
        u = u_star - dt * F.conv2d(p_pad, k_dx)
        v = v_star - dt * F.conv2d(p_pad, k_dy)

        u[solid_mask] = 0.0
        v[solid_mask] = 0.0

        # Enforce BC again
        u, v = apply_bcs(u, v)

        # Disturbance (important for wake)
        if step < 2000:
            v[:,:,:,0] += 0.01 * torch.sin(torch.tensor(step*0.1, device=device))

        if step % 500 == 0:
            print(f"Step {step}")

    # ---------------- PLOTTING ----------------
    u_np = u[0,0].cpu().numpy()
    v_np = v[0,0].cpu().numpy()

    mag = np.sqrt(u_np**2 + v_np**2)

    plt.figure(figsize=(12,4))
    plt.imshow(mag, origin='lower', cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('External Flow Around Cylinder')
    plt.savefig('cnn_flow.png')

    # Streamlines
    Yg, Xg = np.mgrid[0:ny, 0:nx]
    plt.figure(figsize=(12,4))
    plt.streamplot(Xg, Yg, u_np, v_np, density=2)
    plt.title('Streamlines')
    plt.savefig('streamlines_cnn.png')

    print("Done. Flow should now develop correctly.")


if __name__ == '__main__':
    with torch.no_grad():
        run_cnn_fluid_simulation()