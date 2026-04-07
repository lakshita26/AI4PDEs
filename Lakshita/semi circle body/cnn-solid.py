import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-white')

# ==================================================
# Device
# ==================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================================================
# Domain Parameters
# ==================================================
nx, ny = 256, 128
Lx, Ly = 4.0, 1.0

dx = Lx / nx
dy = Ly / ny

# dt = 5e-4
Re = 200
Pr = 1
nt = 4000

# ==================================================
# Derivative Kernels (CNN operators)
# ==================================================
def kernel_dx():
    k = torch.zeros((1,1,3,3), device=device)
    k[0,0,1,0] = -1/(2*dx)
    k[0,0,1,2] =  1/(2*dx)
    return k

def kernel_dy():
    k = torch.zeros((1,1,3,3), device=device)
    k[0,0,0,1] = -1/(2*dy)
    k[0,0,2,1] =  1/(2*dy)
    return k

def kernel_lap():
    k = torch.zeros((1,1,3,3), device=device)
    k[0,0,1,1] = -2/dx**2 - 2/dy**2
    k[0,0,1,0] = 1/dx**2
    k[0,0,1,2] = 1/dx**2
    k[0,0,0,1] = 1/dy**2
    k[0,0,2,1] = 1/dy**2
    return k

wx = kernel_dx()
wy = kernel_dy()
wL = kernel_lap()

# ==================================================
# INITIAL CONDITIONS (Velocity initialized here)
# ==================================================
u = torch.zeros((1,1,ny,nx), device=device)   # u(x,y,0)=0
v = torch.zeros_like(u)                       # v(x,y,0)=0
p = torch.zeros_like(u)
theta = torch.zeros_like(u)

# ==================================================
# Semi-circle Geometry (Attached to Bottom)
# ==================================================
xc = Lx / 2
yc = 0.0
R  = 3*Ly / 4

x = torch.linspace(0, Lx, nx, device=device)
y = torch.linspace(0, Ly, ny, device=device)

Y, X = torch.meshgrid(y, x, indexing='ij')

mask = ((X - xc)**2 + (Y - yc)**2 <= R**2)
mask = mask.unsqueeze(0).unsqueeze(0)

# ==================================================
# Plot Initial Geometry Only
# ==================================================
geometry = np.zeros((ny, nx))
geometry[mask.squeeze().cpu().numpy()] = 1

plt.figure(figsize=(10,3))
plt.imshow(geometry, origin='lower',
           extent=[0, Lx, 0, Ly],
           cmap='gray', aspect='auto')
plt.title("Initial Channel Geometry")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("initial_geometry.png", dpi=300)
plt.close()

# ==================================================
# Boundary Conditions
# ==================================================
def apply_bc(u, v, theta):

    # Inlet
    u[:,:,:,0] = 1.0
    v[:,:,:,0] = 0.0
    theta[:,:,:,0] = 0.0

    # Walls
    u[:,:,0,:] = 0
    u[:,:,-1,:] = 0
    v[:,:,0,:] = 0
    v[:,:,-1,:] = 0

    # Obstacle
    u[mask] = 0
    v[mask] = 0
    theta[mask] = 1.0

    return u, v, theta

# ==================================================
# Time Loop (Navier–Stokes + Energy)
# ==================================================
for n in range(nt):

    u, v, theta = apply_bc(u, v, theta)

    du_dx = F.conv2d(u, wx, padding=1)
    du_dy = F.conv2d(u, wy, padding=1)
    dv_dx = F.conv2d(v, wx, padding=1)
    dv_dy = F.conv2d(v, wy, padding=1)

    lap_u = F.conv2d(u, wL, padding=1)
    lap_v = F.conv2d(v, wL, padding=1)

    u_star = u - dt*(u*du_dx + v*du_dy) + dt*(1/Re)*lap_u
    v_star = v - dt*(u*dv_dx + v*dv_dy) + dt*(1/Re)*lap_v

    u_star[mask] = 0
    v_star[mask] = 0

    div_u = F.conv2d(u_star, wx, padding=1) + F.conv2d(v_star, wy, padding=1)
    rhs = div_u / dt

    for _ in range(60):
        p = 0.25 * (
            torch.roll(p, 1, 2) +
            torch.roll(p,-1, 2) +
            torch.roll(p, 1, 3) +
            torch.roll(p,-1, 3)
            - dx*dy*rhs
        )

    dp_dx = F.conv2d(p, wx, padding=1)
    dp_dy = F.conv2d(p, wy, padding=1)

    u = u_star - dt*dp_dx
    v = v_star - dt*dp_dy

    u[mask] = 0
    v[mask] = 0

    # Energy equation
    dT_dx = F.conv2d(theta, wx, padding=1)
    dT_dy = F.conv2d(theta, wy, padding=1)
    lap_T = F.conv2d(theta, wL, padding=1)

    theta = theta - dt*(u*dT_dx + v*dT_dy) + dt*(1/(Re*Pr))*lap_T
    theta[mask] = 1.0

    if n % 500 == 0:
        print("Step:", n)

# ==================================================
# Convert to NumPy
# ==================================================
u_np = u.squeeze().cpu().numpy()
v_np = v.squeeze().cpu().numpy()
theta_np = theta.squeeze().cpu().numpy()
solid = mask.squeeze().cpu().numpy()

x_np = x.cpu().numpy()
y_np = y.cpu().numpy()
X_np, Y_np = np.meshgrid(x_np, y_np)

# ==================================================
# U Velocity Plot
# ==================================================
plt.figure(figsize=(10,3))
levels_u = np.linspace(np.min(u_np), np.max(u_np), 200)
cf_u = plt.contourf(X_np, Y_np, u_np, levels=levels_u, cmap='jet')
plt.contourf(X_np, Y_np, solid, levels=[0.5,1], colors='red')
plt.colorbar(cf_u, orientation='horizontal', pad=0.2)
plt.title("u velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("u_velocity.png", dpi=300)
plt.close()

# ==================================================
# V Velocity Plot
# ==================================================
plt.figure(figsize=(10,3))
v_abs = np.max(np.abs(v_np))
levels_v = np.linspace(-v_abs, v_abs, 200)
cf_v = plt.contourf(X_np, Y_np, v_np, levels=levels_v, cmap='jet')
plt.contourf(X_np, Y_np, solid, levels=[0.5,1], colors='red')
plt.colorbar(cf_v, orientation='horizontal', pad=0.2)
plt.title("v velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("v_velocity.png", dpi=300)
plt.close()

# ==================================================
# Isotherm Plot
# ==================================================
plt.figure(figsize=(10,3))
levels_T = np.linspace(np.min(theta_np), np.max(theta_np), 200)
cf_T = plt.contourf(X_np, Y_np, theta_np, levels=levels_T, cmap='jet')
iso = plt.contour(X_np, Y_np, theta_np,
                  levels=np.linspace(0,1,15),
                  colors='black', linewidths=0.5)
plt.clabel(iso, inline=True, fontsize=7)
plt.contourf(X_np, Y_np, solid, levels=[0.5,1], colors='red')
plt.colorbar(cf_T, orientation='horizontal', pad=0.2)
plt.title("Isotherms")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("iso.png", dpi=300)
plt.close()

print("Simulation complete. All plots saved.")