import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ==========================================================
# PARAMETERS (IMPROVED)
# ==========================================================
nx, ny = 256, 100
dx = dy = 1.0 / ny
dt = 0.0001
nt = 15000
nit = 120

nu = 5e-5          # ↓ reduced (IMPORTANT)
ub = 1.0

# ==========================================================
# CREATE FIELDS (WITH HALO)
# ==========================================================
def create():
    return torch.zeros((1,1,ny+2,nx+2), device=device)

u = create()
v = create()
p = create()

# ==========================================================
# CYLINDER
# ==========================================================
cx, cy = nx//4, ny//2
r = ny//10

Y, X = torch.meshgrid(torch.arange(ny+2, device=device),
                      torch.arange(nx+2, device=device), indexing='ij')

mask = (X-cx)**2 + (Y-cy)**2 <= r**2

# ==========================================================
# BCs
# ==========================================================
def bc_u(u, v):
    # inlet
    u[:,:, :,0] = ub
    v[:,:, :,0] = 0.01 * torch.sin(2*np.pi*Y[:,0]/ny)  # small perturbation

    # outlet
    u[:,:, :,-1] = u[:,:, :,-2]
    v[:,:, :,-1] = v[:,:, :,-2]

    # top/bottom
    u[:,:, 0,:] = u[:,:, 1,:]
    u[:,:, -1,:] = u[:,:, -2,:]
    v[:,:, 0,:] = 0
    v[:,:, -1,:] = 0
    return u, v

def bc_p(p):
    # NO artificial pressure forcing
    p[:,:, :, -1] = p[:,:, :, -2]
    p[:,:, :, 0]  = p[:,:, :, 1]
    p[:,:, 0,:]   = p[:,:, 1,:]
    p[:,:, -1,:]  = p[:,:, -2,:]
    return p

# ==========================================================
# SOLVER
# ==========================================================
for n in range(nt):

    u, v = bc_u(u, v)

    u_pad = u[0,0]
    v_pad = v[0,0]

    uc = u_pad[1:-1,1:-1]
    vc = v_pad[1:-1,1:-1]

    # ------------------------------------------------------
    # HYBRID ADVECTION (LESS DIFFUSIVE)
    # ------------------------------------------------------
    u_central = uc*(u_pad[1:-1,2:] - u_pad[1:-1,:-2])/(2*dx) + \
                vc*(u_pad[2:,1:-1] - u_pad[:-2,1:-1])/(2*dy)

    v_central = uc*(v_pad[1:-1,2:] - v_pad[1:-1,:-2])/(2*dx) + \
                vc*(v_pad[2:,1:-1] - v_pad[:-2,1:-1])/(2*dy)

    u_up = torch.where(uc>0,
                       uc*(uc - u_pad[1:-1,:-2])/dx,
                       uc*(u_pad[1:-1,2:] - uc)/dx) + \
           torch.where(vc>0,
                       vc*(uc - u_pad[:-2,1:-1])/dy,
                       vc*(u_pad[2:,1:-1] - uc)/dy)

    v_up = torch.where(uc>0,
                       uc*(vc - v_pad[1:-1,:-2])/dx,
                       uc*(v_pad[1:-1,2:] - vc)/dx) + \
           torch.where(vc>0,
                       vc*(vc - v_pad[:-2,1:-1])/dy,
                       vc*(v_pad[2:,1:-1] - vc)/dy)

    gamma = 0.15   # ↓ less damping
    u_adv = (1-gamma)*u_central + gamma*u_up
    v_adv = (1-gamma)*v_central + gamma*v_up

    # ------------------------------------------------------
    # DIFFUSION
    # ------------------------------------------------------
    lap_u = (u_pad[1:-1,2:] - 2*uc + u_pad[1:-1,:-2])/dx**2 + \
            (u_pad[2:,1:-1] - 2*uc + u_pad[:-2,1:-1])/dy**2

    lap_v = (v_pad[1:-1,2:] - 2*vc + v_pad[1:-1,:-2])/dx**2 + \
            (v_pad[2:,1:-1] - 2*vc + v_pad[:-2,1:-1])/dy**2

    # ------------------------------------------------------
    # PREDICTOR
    # ------------------------------------------------------
    u_star = uc + dt * (-u_adv + nu * lap_u)
    v_star = vc + dt * (-v_adv + nu * lap_v)

    # safety clamp
    u_star = torch.clamp(u_star, -5, 5)
    v_star = torch.clamp(v_star, -5, 5)

    # cylinder no-slip
    u_star[mask[1:-1,1:-1]] = 0
    v_star[mask[1:-1,1:-1]] = 0

    u[0,0,1:-1,1:-1] = u_star
    v[0,0,1:-1,1:-1] = v_star

    # ------------------------------------------------------
    # PRESSURE POISSON
    # ------------------------------------------------------
    for _ in range(nit):
        p = bc_p(p)
        p_pad = p[0,0]

        div = (u[0,0,1:-1,2:] - u[0,0,1:-1,:-2])/(2*dx) + \
              (v[0,0,2:,1:-1] - v[0,0,:-2,1:-1])/(2*dy)

        p[0,0,1:-1,1:-1] = 0.25 * (
            p_pad[1:-1,2:] + p_pad[1:-1,:-2] +
            p_pad[2:,1:-1] + p_pad[:-2,1:-1]
            - dx**2 * div / dt
        )

    # ------------------------------------------------------
    # CORRECTOR
    # ------------------------------------------------------
    p_pad = p[0,0]

    u[0,0,1:-1,1:-1] -= dt*(p_pad[1:-1,2:] - p_pad[1:-1,:-2])/(2*dx)
    v[0,0,1:-1,1:-1] -= dt*(p_pad[2:,1:-1] - p_pad[:-2,1:-1])/(2*dy)

    u[0,0][mask] = 0
    v[0,0][mask] = 0

    if n % 1000 == 0:
        print(f"Step {n}, max U: {u.max().item():.4f}")

# ==========================================================
# VISUALIZATION (FINAL)
# ==========================================================
u_np = u[0,0].cpu().numpy()
v_np = v[0,0].cpu().numpy()
p_np = p[0,0].cpu().numpy()

Xg, Yg = np.meshgrid(np.arange(nx+2), np.arange(ny+2))
speed = np.sqrt(u_np**2 + v_np**2)

fig, axs = plt.subplots(2,2, figsize=(14,8))

# magnitude + streamlines
c1 = axs[0,0].contourf(speed, 50)
axs[0,0].streamplot(Xg, Yg, u_np, v_np, color='k', density=1.2)
axs[0,0].set_title("Velocity Magnitude + Streamlines")
plt.colorbar(c1, ax=axs[0,0])

# u
c2 = axs[0,1].contourf(u_np, 50)
axs[0,1].set_title("u-velocity")
plt.colorbar(c2, ax=axs[0,1])

# v
c3 = axs[1,0].contourf(v_np, 50)
axs[1,0].set_title("v-velocity")
plt.colorbar(c3, ax=axs[1,0])

# pressure
c4 = axs[1,1].contourf(p_np, 50)
axs[1,1].set_title("Pressure")
plt.colorbar(c4, ax=axs[1,1])

# cylinder
for ax in axs.flat:
    circle = plt.Circle((cx, cy), r, color='black')
    ax.add_patch(circle)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("final_flow.png", dpi=300)
plt.show()