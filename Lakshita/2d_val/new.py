import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from AI4PDEs_utils import create_tensors_2D, get_weights_linear_2D, create_semicircle_body_2D
from AI4PDEs_bounds import boundary_condition_2D_u, boundary_condition_2D_v, boundary_condition_2D_p, boundary_condition_2D_T

# ==========================
# PUBLICATION PLOT STYLING
# ==========================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ==========================
# PAPER REQUIREMENTS
# ==========================
Re = 100.0  
Pr = 0.7    
nu = 1.0 / Re

D_cells = 20                  # Cells per cylinder diameter D
dx = 1.0 / D_cells            # Dimensionless dx
dt = 0.005                    # Time-step

beta = 0.25                   # Blockage ratio D/H = 25%
H_cells = int(D_cells / beta) # ny = 4*D = 80
Xu_cells = int(45 * D_cells)  # Upstream distance = 45D
Xd_cells = int(120 * D_cells) # Downstream distance = 120D

nx = Xu_cells + Xd_cells      # Total length L = 165D (3300 cells)
ny = H_cells                  # Total height H = 4D (80 cells)

ntime = 60000                 # Max timesteps (will exit early upon validation)
iteration = 40                # Poisson solver iterations

# ==========================
# INIT
# ==========================
values_u, values_v, values_p, values_uu, values_vv, values_pp, b_uu, b_vv = create_tensors_2D(nx, ny)

values_T = torch.zeros_like(values_u)
values_TT = torch.zeros_like(values_uu)
b_TT = torch.zeros_like(values_uu)

# Add small noise to trigger instability/vortex shedding faster
values_u[:] = 1e-4 * torch.randn_like(values_u)

# ==========================
# BODY
# ==========================
cor_x = Xu_cells             
cor_y = int(ny / 2)          
radius = int(D_cells / 2)    

sigma = create_semicircle_body_2D(nx, ny, cor_x, cor_y, radius)
mask = (sigma > 0).float()   

w1, w2, w3, wA, w_res, diag = get_weights_linear_2D(dx)
diag_val = float(diag)       

# ==========================
# MODEL (Fractional Step RK2)
# ==========================
class AI4CFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.xadv = nn.Conv2d(1,1,3)
        self.yadv = nn.Conv2d(1,1,3)
        self.diff = nn.Conv2d(1,1,3)
        self.A = nn.Conv2d(1,1,3)

        self.xadv.weight.data = w2.to(device)
        self.yadv.weight.data = w3.to(device)
        self.diff.weight.data = w1.to(device)
        self.A.weight.data = wA.to(device)

        self.xadv.bias.data.zero_()
        self.yadv.bias.data.zero_()
        self.diff.bias.data.zero_()
        self.A.bias.data.zero_()

    def forward(self, u, uu, v, vv, p, pp, T, TT, b_uu, b_vv, b_TT):

        # 1. PREDICTOR STEP
        uu = boundary_condition_2D_u(u, uu, 1.5)
        vv = boundary_condition_2D_v(v, vv, 0.0)
        TT = boundary_condition_2D_T(T, TT)

        u_pred = u + 0.5 * dt * (u * self.xadv(uu) + v * self.yadv(uu) + nu * self.diff(uu))
        v_pred = v + 0.5 * dt * (u * self.xadv(vv) + v * self.yadv(vv) + nu * self.diff(vv))
        T_pred = T + 0.5 * dt * (u * self.xadv(TT) + v * self.yadv(TT) + (1.0 / (Re * Pr)) * self.diff(TT))

        u_pred = u_pred / (1 + 0.5 * dt * sigma)
        v_pred = v_pred / (1 + 0.5 * dt * sigma)
        T_pred = T_pred * (1 - mask) + mask

        b_uu = boundary_condition_2D_u(u_pred, b_uu, 1.5)
        b_vv = boundary_condition_2D_v(v_pred, b_vv, 0.0)
        b_TT = boundary_condition_2D_T(T_pred, b_TT)

        # 2. CORRECTOR STEP 
        u_star = u + dt * (u_pred * self.xadv(b_uu) + v_pred * self.yadv(b_uu) + nu * self.diff(b_uu))
        v_star = v + dt * (u_pred * self.xadv(b_vv) + v_pred * self.yadv(b_vv) + nu * self.diff(b_vv))
        T_new  = T + dt * (u_pred * self.xadv(b_TT) + v_pred * self.yadv(b_TT) + (1.0 / (Re * Pr)) * self.diff(b_TT))

        u_star = u_star / (1 + dt * sigma)
        v_star = v_star / (1 + dt * sigma)
        T_new  = T_new * (1 - mask) + mask

        uu_star = boundary_condition_2D_u(u_star, uu, 1.5)
        vv_star = boundary_condition_2D_v(v_star, vv, 0.0)

        # 3. PRESSURE POISSON SOLVER
        div = self.xadv(uu_star) + self.yadv(vv_star)
        b_rhs = div / dt
        
        for _ in range(iteration):
            pp = boundary_condition_2D_p(p, pp)
            residual = self.A(pp) - b_rhs
            p = p - (residual / diag_val)

        pp = boundary_condition_2D_p(p, pp)

        # 4. VELOCITY CORRECTION
        u_new = u_star + self.xadv(pp) * dt
        v_new = v_star + self.yadv(pp) * dt

        u_new = u_new / (1 + dt * sigma)
        v_new = v_new / (1 + dt * sigma)

        return u_new, v_new, p, T_new, uu_star, vv_star, pp, TT, b_uu, b_vv, b_TT, residual

# ==========================
# EXACT PAPER PLOTTING UTILITY
# ==========================
def save_validation_plots(u_tensor, v_tensor, T_tensor, phase_name):
    u_np = u_tensor[0,0].cpu().numpy()
    T_np = T_tensor[0,0].cpu().numpy()

    # Dimensionless grid spacing for numerical integration
    dy_val = 1.0 / D_cells
    
    # Crop axes to strictly match the paper: X in [43.5, 47.5]
    plot_start_x = cor_x - int(1.5 * D_cells)
    plot_end_x = cor_x + int(2.5 * D_cells)

    U_plot = u_np[:, plot_start_x:plot_end_x]
    T_plot = T_np[:, plot_start_x:plot_end_x]
    mask_plot = mask[0,0,:,plot_start_x:plot_end_x].cpu().numpy()

    x_dim = np.linspace(43.5, 47.5, U_plot.shape[1])
    y_dim = np.linspace(0, 4, ny)
    X, Y = np.meshgrid(x_dim, y_dim)

    # Calculate exact Stream Function (psi = integral of U dy) using Trapezoidal Rule
    psi = np.zeros_like(U_plot)
    for j in range(1, U_plot.shape[0]):
        psi[j, :] = psi[j-1, :] + 0.5 * (U_plot[j, :] + U_plot[j-1, :]) * dy_val

    # Dictionary to map phase names to the specific letter in the paper
    phase_map = {"Tp": "(a)", "Tp_4": "(b)", "2Tp_4": "(c)", "3Tp_4": "(d)"}
    letter = phase_map.get(phase_name, "")
    
    title_text = f"$t=T_p/4$" if phase_name == "Tp_4" else f"$t={phase_name.replace('_', '/')}$"
    if phase_name == "Tp": title_text = "$t=T_p$"

    # ----------------------------------------------------
    # COMBINED SUBPLOTS
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    
    # Top: Isotherms (Matches Fig 5)
    axes[0].contour(X, Y, T_plot, levels=np.linspace(0, 1, 16), colors='red', linewidths=1.0)
    axes[0].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
    axes[0].set_xlim([44, 47.5])
    axes[0].set_ylim([0, 4])
    axes[0].set_xticks([44, 45, 46, 47])
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].text(44.1, 3.8, letter, fontsize=18, fontweight='bold', va='top')
    axes[0].text(47.4, 3.8, title_text, fontsize=18, va='top', ha='right', style='italic')
    axes[0].set_title("Isotherms", fontsize=14)

    # Bottom: Streamlines (Matches Fig 3)
    levels_stream = np.linspace(np.min(psi), np.max(psi), 45) # Dense levels to capture vortices
    axes[1].contour(X, Y, psi, levels=levels_stream, colors='black', linewidths=0.9)
    axes[1].contourf(X, Y, mask_plot, levels=[0.5, 1.5], colors=['black'])
    axes[1].set_xlim([44, 47.5])
    axes[1].set_ylim([0, 4])
    axes[1].set_xticks([44, 45, 46, 47])
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].text(44.1, 3.8, letter, fontsize=18, fontweight='bold', va='top')
    axes[1].text(47.4, 3.8, title_text, fontsize=18, va='top', ha='right', style='italic')
    axes[1].set_title("Streamlines", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"results/Combined_Validation_{phase_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--> Saved exact paper validation plot for phase: {phase_name.replace('_', '/')}")


# ==========================
# RUN
# ==========================
model = AI4CFD().to(device)
os.makedirs("results", exist_ok=True)

print("Starting validation run for Re=100...")

# Phase detection variables
probe_x = cor_x + int(1.5 * D_cells) 
probe_y = cor_y
phase_state = 0
t_start = 0
T_period_steps = 0
target_steps = []
saved_phases = 0
prev_v_probe = 0.0

with torch.no_grad():
    for t in range(ntime):

        values_u, values_v, values_p, values_T, values_uu, values_vv, values_pp, values_TT, b_uu, b_vv, b_TT, residual = model(
            values_u, values_uu, values_v, values_vv, values_p, values_pp, values_T, values_TT, b_uu, b_vv, b_TT
        )

        if t % 500 == 0:
            print(f"Step {t} | Max Pressure Residual = {residual.abs().max().item():.6e}")

        # ==========================
        # DYNAMIC PHASE DETECTION
        # ==========================
        v_probe = values_v[0,0,probe_y,probe_x].item()

        if t > 20000:
            # State 0: Wait for V-velocity to cross zero upward (Marks start of a cycle)
            if phase_state == 0:
                if prev_v_probe < 0 and v_probe >= 0:
                    t_start = t
                    phase_state = 1
                    
            # State 1: Wait for next upward zero crossing to measure exact period
            elif phase_state == 1:
                if prev_v_probe < 0 and v_probe >= 0:
                    T_period_steps = t - t_start
                    print(f"\n[PHASE LOCK] Vortex Shedding Period Detected: {T_period_steps} steps.")
                    
                    target_steps = [
                        t,                             # t = Tp
                        t + T_period_steps // 4,       # t = Tp/4
                        t + 2 * T_period_steps // 4,   # t = 2Tp/4
                        t + 3 * T_period_steps // 4    # t = 3Tp/4
                    ]
                    phase_state = 2
                    
            # State 2: Save the exact validation frames when the targeted steps are hit
            elif phase_state == 2:
                if t in target_steps:
                    phase_index = target_steps.index(t)
                    phase_names = ["Tp", "Tp_4", "2Tp_4", "3Tp_4"]
                    save_validation_plots(values_u, values_v, values_T, phase_names[phase_index])
                    saved_phases += 1
                    
                    if saved_phases == 4:
                        print("\nValidation Complete! All 4 phases successfully extracted matching the paper.")
                        break 

        prev_v_probe = v_probe