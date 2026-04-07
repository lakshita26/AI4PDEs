import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Catalano et al. (2003) - LES aligned implementation (improved)
# Includes:
# - Fractional step method
# - RK3 time integration
# - Smagorinsky SGS (approx dynamic)
# - Proper BCs (inlet, outlet, wall)
# - Improved wall stress model
# - Streamline + magnitude plots
# ==========================================================

def run_catalano_les():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- PARAMETERS ----------------
    nx, ny = 240, 100
    dx = dy = 1.0 / ny
    dt = 0.0025
    nu = 5e-6   # adjusted for stability
    Cs = 0.05
    delta = np.sqrt(dx * dy)
    max_iter = 15000
    ub = 1.0
    Uc = 1.0

    # ---------------- INITIALIZE ----------------
    u = torch.ones((1,1,ny,nx), device=device) * ub
    v = torch.zeros((1,1,ny,nx), device=device)
    p = torch.zeros((1,1,ny,nx), device=device)

    # ---------------- CYLINDER ----------------
    cyl_x, cyl_y, cyl_r = nx//4, ny//2, ny//10
    Y, X = torch.meshgrid(torch.arange(ny, device=device),
                          torch.arange(nx, device=device), indexing='ij')

    dist = torch.sqrt((X-cyl_x)**2 + (Y-cyl_y)**2)
    solid = (dist <= cyl_r).view(1,1,ny,nx)
    halo = (dist > cyl_r) & (dist <= cyl_r+2)

    # ---------------- KERNELS ----------------
    def k(arr): return torch.tensor([[arr]], device=device, dtype=torch.float32)

    k_dx = k([[0,0,0],[-0.5,0,0.5],[0,0,0]])/dx
    k_dy = k([[0,-0.5,0],[0,0,0],[0,0.5,0]])/dy
    k_lap = k([[0,1,0],[1,-4,1],[0,1,0]])/(dx**2)
    k_jac = k([[0,1,0],[1,0,1],[0,1,0]])

    # ---------------- WALL MODEL ----------------
    def wall_shear(u_t, dist):
        u_tau = torch.sqrt(nu * u_t / (dist + 1e-8))
        return u_tau**2

    # ---------------- RK3 STEP ----------------
    def rhs(u, v):
        u_p = F.pad(u,(1,1,1,1),'replicate')
        v_p = F.pad(v,(1,1,1,1),'replicate')

        du_dx = F.conv2d(u_p, k_dx)
        du_dy = F.conv2d(u_p, k_dy)
        dv_dx = F.conv2d(v_p, k_dx)
        dv_dy = F.conv2d(v_p, k_dy)

        S = torch.sqrt(2*(du_dx**2 + dv_dy**2 + 0.5*(du_dy+dv_dx)**2) + 1e-10)
        nu_t = (Cs*delta)**2 * S

        lap_u = F.conv2d(u_p, k_lap)
        lap_v = F.conv2d(v_p, k_lap)

        Ru = -(u*du_dx + v*du_dy) + (nu+nu_t)*lap_u
        Rv = -(u*dv_dx + v*dv_dy) + (nu+nu_t)*lap_v

        return Ru, Rv

    print("Running LES...")

    for step in range(max_iter):

        # RK3
        Ru1, Rv1 = rhs(u,v)
        u1 = u + dt*Ru1
        v1 = v + dt*Rv1

        Ru2, Rv2 = rhs(u1,v1)
        u2 = 0.75*u + 0.25*(u1 + dt*Ru2)
        v2 = 0.75*v + 0.25*(v1 + dt*Rv2)

        Ru3, Rv3 = rhs(u2,v2)
        u_star = (1/3)*u + (2/3)*(u2 + dt*Ru3)
        v_star = (1/3)*v + (2/3)*(v2 + dt*Rv3)

        # WALL MODEL
        ut = torch.sqrt(u[0,0][halo]**2 + v[0,0][halo]**2) + 1e-8
        tau = wall_shear(ut, 1.0)
        u_star[0,0][halo] -= dt*tau*(u[0,0][halo]/ut)
        v_star[0,0][halo] -= dt*tau*(v[0,0][halo]/ut)

        # SOLID
        u_star[solid] = 0
        v_star[solid] = 0

        # INLET BC
        u_star[:,:,:,0] = ub
        v_star[:,:,:,0] = 0

        # PRESSURE POISSON
        u_p = F.pad(u_star,(1,1,1,1),'replicate')
        v_p = F.pad(v_star,(1,1,1,1),'replicate')

        rhs_p = (F.conv2d(u_p,k_dx)+F.conv2d(v_p,k_dy))/dt

        for _ in range(150):
            p = 0.25*(F.conv2d(F.pad(p,(1,1,1,1),'replicate'),k_jac) - dx**2*rhs_p)
            p[:,:,:,-1] = 0

        p_p = F.pad(p,(1,1,1,1),'replicate')

        # CORRECTION
        u = u_star - dt*F.conv2d(p_p,k_dx)
        v = v_star - dt*F.conv2d(p_p,k_dy)

        # OUTLET
        u[:,:,:,-1] = u[:,:,:,-2]
        v[:,:,:,-1] = v[:,:,:,-2]

        if step % 1000 == 0:
            print(f"Step {step}")

    # ---------------- PLOTS ----------------
    u_np = u[0,0].cpu().numpy()
    v_np = v[0,0].cpu().numpy()

    mag = np.sqrt(u_np**2 + v_np**2)

    plt.figure(figsize=(12,5))
    plt.imshow(mag, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title("Velocity Magnitude")
    plt.savefig('cnn_les_flow.png')

    # Streamline
    Yg, Xg = np.mgrid[0:ny, 0:nx]

    plt.figure(figsize=(12,5))
    plt.streamplot(Xg, Yg, u_np, v_np, density=2)
    plt.title("Streamlines")
    plt.savefig('streamlines.png')

    print("Done.")


if __name__ == '__main__':
    with torch.no_grad():
        run_catalano_les()