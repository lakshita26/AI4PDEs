import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import your modularized physics and bounds
from AI4PDEs_utils import create_tensors_2D, get_weights_linear_2D, create_circular_body_2D
from AI4PDEs_bounds import boundary_condition_2D_u, boundary_condition_2D_v, boundary_condition_2D_p, boundary_condition_2D_cw

# ==========================================================
# 1. SETUP AND HYPERPARAMETERS
# ==========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running physics on: {device}")

# Grid parameters
nx, ny = 320, 120  
dx = dy = 1.0 / ny
dt = 0.0005  
nt = 200000  

Re = 1000.0  
nu = 1.0 / Re
rho = 1.0
ub = 1.0  # Flow is positive (left to right)

iteration = 20  
nlevel = 4      

# Base cylinder configuration (CONSTANT FOR ALL RUNS)
current_radius = ny // 20  
D = 2 * current_radius
D_phys = D * dx  

# Pre-compute static weights and sponge layer
w1, w2, w3, wA, w_res, diag = get_weights_linear_2D(dx)
w1 = w1.to(device)
w2 = w2.to(device)
w3 = w3.to(device)
wA = wA.to(device)
w_res = w_res.to(device)
diag = float(diag)

# Sponge layer on the RIGHT boundary to absorb wake turbulence
sponge_np = np.zeros((1, 1, ny, nx))
sponge_thickness = 60
sponge_np[0, 0, :, -sponge_thickness:] = np.linspace(0, 30, sponge_thickness)
sponge = torch.tensor(sponge_np, dtype=torch.float32).to(device)

# ==========================================================
# 2. NEURAL NETWORK SOLVER
# ==========================================================
class AI4CFD_Fixed(nn.Module):
    def __init__(self, w1, w2, w3, wA, w_res):
        super(AI4CFD_Fixed, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.A = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0, bias=True)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1
        self.A.weight.data = wA
        self.res.weight.data = w_res

        bias_init = torch.zeros(1).to(device)
        self.xadv.bias.data = bias_init
        self.yadv.bias.data = bias_init
        self.diff.bias.data = bias_init
        self.A.bias.data = bias_init
        self.res.bias.data = bias_init
        
    def solid_body(self, values_u, values_v, sigma, dt):
        """Penalty method for no-slip BC"""
        values_u = values_u / (1 + dt * sigma) 
        values_v = values_v / (1 + dt * sigma) 
        return values_u, values_v   
         
    def F_cycle_MG(self, values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel):
        """Multigrid pressure solver"""
        b = -(self.xadv(values_uu) + self.yadv(values_vv)) / dt
        
        for MG in range(iteration):
            r = self.A(boundary_condition_2D_p(values_p, values_pp)) - b 
            r_s = [r]  
            
            for i in range(1, nlevel):
                r = self.res(r)
                r_s.append(r)
            
            w = torch.zeros_like(r_s[-1]) 
            
            for i in reversed(range(1, nlevel)):
                ww = boundary_condition_2D_cw(w)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)         
            
            values_p = values_p - w
            values_p = values_p - self.A(boundary_condition_2D_p(values_p, values_pp)) / diag + b / diag
        
        return values_p, w, r
        
    def forward(self, values_u, values_uu, values_v, values_vv, values_p, values_pp, 
                sigma, b_uu, b_vv, dt, iteration, nlevel):      
        
        values_uu = boundary_condition_2D_u(values_u, values_uu, ub, sigma) 
        values_vv = boundary_condition_2D_v(values_v, values_vv, ub, sigma)  
        values_pp = boundary_condition_2D_p(values_p, values_pp)   
        
        # Force inlet boundary only. Periodic BCs handle top/bottom.
        values_uu[:, :, :, 0] = ub      
        values_vv[:, :, :, 0] = 0.0     
        
        Grapx_p = self.xadv(values_pp) * dt
        Grapy_p = self.yadv(values_pp) * dt 
        ADx_u = self.xadv(values_uu)
        ADy_u = self.yadv(values_uu) 
        ADx_v = self.xadv(values_vv)
        ADy_v = self.yadv(values_vv) 
        AD2_u = self.diff(values_uu)
        AD2_v = self.diff(values_vv) 
        
        b_u = (values_u + 0.5 * nu * AD2_u * dt - values_u * ADx_u * dt - values_v * ADy_u * dt - Grapx_p)
        b_v = (values_v + 0.5 * nu * AD2_v * dt - values_u * ADx_v * dt - values_v * ADy_v * dt - Grapy_p)
        b_u, b_v = self.solid_body(b_u, b_v, sigma, dt)
        
        b_uu = boundary_condition_2D_u(b_u, b_uu, ub, sigma)
        b_vv = boundary_condition_2D_v(b_v, b_vv, ub, sigma) 
        
        b_uu[:, :, :, 0] = ub
        b_vv[:, :, :, 0] = 0.0
        
        ADx_u = self.xadv(b_uu)
        ADy_u = self.yadv(b_uu) 
        ADx_v = self.xadv(b_vv)
        ADy_v = self.yadv(b_vv) 
        AD2_u = self.diff(b_uu)
        AD2_v = self.diff(b_vv) 
           
        values_u = (values_u + nu * AD2_u * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - Grapx_p)
        values_v = (values_v + nu * AD2_v * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - Grapy_p)
        values_u, values_v = self.solid_body(values_u, values_v, sigma, dt)
        
        values_uu = boundary_condition_2D_u(values_u, values_uu, ub, sigma) 
        values_vv = boundary_condition_2D_v(values_v, values_vv, ub, sigma)  
        
        values_uu[:, :, :, 0] = ub
        values_vv[:, :, :, 0] = 0.0
        
        values_p, w, r = self.F_cycle_MG(values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel)
          
        values_pp = boundary_condition_2D_p(values_p, values_pp)  
        values_u = values_u - self.xadv(values_pp) * dt
        values_v = values_v - self.yadv(values_pp) * dt 
        values_u, values_v = self.solid_body(values_u, values_v, sigma, dt)
        
        return values_u, values_v, values_p, w, r


# ==========================================================
# 3. MAIN PARAMETRIC SWEEP
# ==========================================================
# Positions 20%, 40%, 60%, 80% of domain length
positions_pct = [0.2, 0.4, 0.6, 0.8]
results_summary = {}

for step, pos in enumerate(positions_pct):
    pos_label = int(pos * 100)
    print(f"\n{'='*60}")
    print(f"STARTING CONFIGURATION: L_in = {pos_label}%")
    
    # Vertically offset by 2 cells to break symmetry naturally
    cor_x = int(pos * nx)
    cor_y = ny // 2 + 2  
    
    print(f"Position (x,y): ({cor_x}, {cor_y}) | Radius: {current_radius} cells | Diameter: {D} cells")
    
    # Initialize tensors for the current configuration
    values_u, values_v, values_p, values_uu, values_vv, values_pp, b_uu, b_vv = create_tensors_2D(nx, ny)
    
    # SYMMETRY BREAKER: A strictly divergence-free initial vertical crossflow.
    values_v += 0.1 * ub

    # Generate the raw cylinder mask 
    sigma_raw = create_circular_body_2D(nx, ny, cor_x, cor_y, current_radius).to(device)
    
    # Smooth the penalty mask slightly and scale strongly to create a stable solid wall
    pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    sigma = pool(sigma_raw) * 20000.0  

    # Initialize solver for this run
    cfd_solver = AI4CFD_Fixed(w1, w2, w3, wA, w_res).to(device)
    cfd_solver.eval()

    Cd_history, Cl_history, time_history = [], [], []

    # Run Time Steps
    with torch.no_grad(): 
        for t in range(nt):
            values_u, values_v, values_p, w, r_val = cfd_solver(
                values_u, values_uu, values_v, values_vv, values_p, values_pp, 
                sigma, b_uu, b_vv, dt, iteration, nlevel
            )

            # Apply sponge layer to outlet to absorb reflections
            values_v = values_v / (1 + dt * sponge)
            values_u = (values_u - ub) / (1 + dt * sponge) + ub
            
            # Re-enforce inlet 
            values_u[:, :, :, 0] = ub
            values_v[:, :, :, 0] = 0.0

            # --- IMMERSED BOUNDARY VOLUME FORCE INTEGRATION ---
            # Sum the penalty forces enforcing the solid wall
            sigma_np = sigma.squeeze().cpu().numpy()
            u_np = values_u.squeeze().cpu().numpy()
            v_np = values_v.squeeze().cpu().numpy()
            
            # Total force exerted by fluid on body
            F_x_total = np.sum(sigma_np * u_np) * dx * dy * rho
            F_y_total = np.sum(sigma_np * v_np) * dx * dy * rho
            
            # Non-dimensionalize forces
            denom = 0.5 * rho * (ub**2) * D_phys
            Cd = F_x_total / (denom + 1e-10)
            Cl = F_y_total / (denom + 1e-10) 
            
            Cd_history.append(Cd)
            Cl_history.append(Cl)
            time_history.append(t * dt)

            if t % 5000 == 0:  
                print(f"Step {t:06d} | t={t*dt:7.2f} | Cd={Cd:8.4f} | Cl={Cl:8.4f}")

    # Process and store metrics (evaluating the second half to ignore spin-up)
    start_idx = int(nt * 0.5)
    Cd_mean = np.mean(Cd_history[start_idx:])
    Cl_rms = np.std(Cl_history[start_idx:])
    results_summary[pos_label] = Cd_mean
    print(f"FINAL STATS L_in={pos_label}%: Mean Cd={Cd_mean:.4f}, RMS Cl={Cl_rms:.4f}")

    # ==========================================================
    # 4. POST-PROCESSING & PLOTTING FOR CURRENT RUN
    # ==========================================================
    dv_dy, dv_dx = np.gradient(v_np, dy, dx)
    du_dy, du_dx = np.gradient(u_np, dy, dx)
    vorticity = dv_dx - du_dy
    
    p_np = values_p.squeeze().cpu().numpy()

    psi = np.zeros_like(u_np)
    for i in range(nx):
        for j in range(1, ny):
            psi[j, i] = psi[j-1, i] + u_np[j, i] * dy

    x_lin = np.linspace(0, nx * dx, nx)
    y_lin = np.linspace(0, ny * dy, ny)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    c_x, c_y = cor_x * dx, cor_y * dy
    r_phys = current_radius * dx

    def draw_circle(ax):
        circle = patches.Circle((c_x, c_y), r_phys, fill=True, facecolor='lightgray', 
                               edgecolor='black', lw=2, zorder=5)
        ax.add_patch(circle)

    plot_nx = nx - sponge_thickness
    fig, axs = plt.subplots(4, 1, figsize=(14, 16))

    # (a) Stream function
    levels_psi = np.linspace(np.min(psi[:, :plot_nx]), np.max(psi[:, :plot_nx]), 50)
    cs0 = axs[0].contourf(X_grid[:, :plot_nx], Y_grid[:, :plot_nx], psi[:, :plot_nx], 
                          levels=levels_psi, cmap='viridis')
    fig.colorbar(cs0, ax=axs[0], fraction=0.015, pad=0.04)
    draw_circle(axs[0])
    axs[0].set_title(f"(a) Streamline contours (L_in={pos_label}%, D={D})", fontsize=12, fontweight='bold')
    axs[0].set_aspect('equal')

    # (b) Pressure
    p_core = p_np[:, 5:plot_nx]
    levels_p = np.linspace(np.percentile(p_core, 5), np.percentile(p_core, 95), 50)
    cs1 = axs[1].contourf(X_grid[:, :plot_nx], Y_grid[:, :plot_nx], p_np[:, :plot_nx], 
                          levels=levels_p, cmap='coolwarm')
    fig.colorbar(cs1, ax=axs[1], fraction=0.015, pad=0.04)
    draw_circle(axs[1])
    axs[1].set_title(f"(b) Pressure contours (L_in={pos_label}%)", fontsize=12, fontweight='bold')
    axs[1].set_aspect('equal')

    # (c) Vorticity
    v_core = vorticity[:, 5:plot_nx]
    vmax = np.percentile(np.abs(v_core), 98)
    if vmax < 1e-6: vmax = 1e-5
    levels_v = np.linspace(-vmax, vmax, 50)
    cs2 = axs[2].contourf(X_grid[:, :plot_nx], Y_grid[:, :plot_nx], vorticity[:, :plot_nx], 
                          levels=levels_v, cmap='RdBu_r') 
    fig.colorbar(cs2, ax=axs[2], fraction=0.015, pad=0.04)
    draw_circle(axs[2])
    axs[2].set_title(f"(c) Vorticity contours (L_in={pos_label}%)", fontsize=12, fontweight='bold')
    axs[2].set_aspect('equal')

    # (d) Cd, Cl vs time
    plot_start_time = time_history[start_idx]
    axs[3].plot(np.array(time_history[start_idx:]) - plot_start_time, Cd_history[start_idx:], 
                'k-', linewidth=1.5, label='$C_d$')
    axs[3].plot(np.array(time_history[start_idx:]) - plot_start_time, Cl_history[start_idx:], 
                'k--', linewidth=1.5, label='$C_l$')
    axs[3].set_title(f"(d) Drag & Lift Coefficients vs Time (L_in={pos_label}%)", fontsize=12, fontweight='bold')
    axs[3].set_xlabel("time")
    axs[3].set_ylabel("$C_d$, $C_l$")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"Re{int(Re)}_Lin{pos_label}_Sweep.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as Re{int(Re)}_Lin{pos_label}_Sweep.png")

    # ==========================================
    # DOMAIN INDEPENDENCE CHECK (1% CRITERIA)
    # ==========================================
    if step > 0:
        prev_pos_label = int(positions_pct[step-1] * 100)
        Cd_prev = results_summary[prev_pos_label]
        
        diff_pct = abs((Cd_prev - Cd_mean) / Cd_prev) * 100
        
        print(f"\n--- INDEPENDENCE CHECK ---")
        print(f"Cd at Lin={prev_pos_label}% : {Cd_prev:.4f}")
        print(f"Cd at Lin={pos_label}% : {Cd_mean:.4f}")
        print(f"Relative Difference: {diff_pct:.2f}%")
        
        if diff_pct <= 1.0:
            print("SUCCESS: The difference is <= 1%. Domain independence achieved!")
            print("Stopping parameter sweep early to save time.")
            break 
        else:
            print("WARNING: Difference is > 1%. Boundaries are still affecting the flow. Proceeding to next configuration...")

# ==========================================================
# 5. FINAL OUTPUT SUMMARY & PLOTS
# ==========================================================
print("\n" + "="*50)
print("             PARAMETRIC SWEEP SUMMARY")
print("="*50)
print(f"{'Position (L_in)':<20} | {'Mean Drag (Cd)':<20}")
print("-" * 50)
for label, cd in results_summary.items():
    print(f"{label}%{'':<17} | {cd:.4f}")
print("="*50)

# Plot Domain Independence Curve
if len(results_summary) > 1:
    plt.figure(figsize=(8, 5))
    positions_plot = list(results_summary.keys())
    cds_plot = list(results_summary.values())
    
    plt.plot(positions_plot, cds_plot, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.xlabel('Horizontal Position ($L_{in}$ %)', fontsize=12)
    plt.ylabel('Mean Drag Coefficient ($C_d$)', fontsize=12)
    plt.title('Domain Independence: Drag vs. Cylinder Position', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(positions_plot)
    
    plt.tight_layout()
    plt.savefig(f"Domain_Independence_Summary.png", dpi=300)
    print("\nFinal domain independence summary plot saved as Domain_Independence_Summary.png")