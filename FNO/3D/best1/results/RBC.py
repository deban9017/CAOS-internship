#==================================
# BLOCK 1 - Configuration and Setup for 3D
#==================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.fft import rfftn, irfftn, fftfreq
import time
import gc
import os

# --- CONFIGURATION DICTIONARY for 3D ---
CONFIG = {
    'model': {
        'modes1': 32,
        'modes2': 32,
        'modes3': 32,
        'width': 32,  # Reduced width for easier training
        'n_vars': 5,
        'n_layers': 4,
    },
    'data': {
        'train_times': [1, 3],
        'target_time': 5,
        'simulation_dt': 20.0,
        'grid_shape': (256, 256, 256)
    },
    'physics': {
        'kappa': 1e-6,
    },
    'training': {
        'epochs': 4000,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_pct_start': 0.1,
    },
    # --- Using simple, fixed weights that worked in 2D ---
    'loss_weights': {
        'physics': 1.0,
        'data': 10.0,
        'bc': 50.0,
        'smooth': 0.01
    }
}

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#==================================
# BLOCK 2 - Core FNO Components for 3D
#==================================

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3 = in_channels, out_channels, modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
    def compl_mul3d(self, input, weights): return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        return irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))

class FNOBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='gelu'):
        super(FNOBlock3d, self).__init__()
        self.conv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.w = nn.Conv3d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
    def forward(self, x): return self.activation(self.bn(self.conv(x) + self.w(x)))

class MultiVariableFNO3d(nn.Module):
    def __init__(self, config):
        super(MultiVariableFNO3d, self).__init__()
        self.modes1, self.modes2, self.modes3 = config['model']['modes1'], config['model']['modes2'], config['model']['modes3']
        self.width, self.n_vars, self.n_layers = config['model']['width'], config['model']['n_vars'], config['model']['n_layers']
        num_input_timesteps = len(config['data']['train_times'])
        in_channels = self.n_vars * num_input_timesteps
        self.fc0 = nn.Sequential(nn.Conv3d(in_channels, self.width, 1), nn.GELU(), nn.Conv3d(self.width, self.width, 1))
        self.fno_blocks = nn.ModuleList([FNOBlock3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.n_layers)])
        self.fc1 = nn.Sequential(nn.Conv3d(self.width, 128, 1), nn.GELU(), nn.Conv3d(128, 64, 1), nn.GELU(), nn.Conv3d(64, self.n_vars, 1))
    def forward(self, x):
        x = self.fc0(x)
        for fno in self.fno_blocks: x = fno(x)
        return self.fc1(x)

#==================================
# BLOCK 3 - Data Loading and Preprocessing for 3D
#==================================

def extract_fields_at_time_3d(ds, time_idx):
    data = ds.isel(time=time_idx)
    return {
        'u': data['u'].values, 'v': data['v'].values, 'w': data['w'].values,
        'b': data['b'].values, 'p_dyn': data['p_dyn'].values
    }

def interpolate_staggered_to_centers_3d(field, axis):
    if axis == 'z': return 0.5 * (field[:-1, :, :] + field[1:, :, :])
    raise ValueError("Axis must be 'z' for this dataset's staggered grid.")

def normalize_field(field):
    mean, std = np.mean(field), np.std(field) + 1e-8
    return (field - mean) / std, mean, std

def prepare_data_and_tensors_3d(ds, train_times, target_time, config):
    train_data = [extract_fields_at_time_3d(ds, t) for t in train_times]
    target_data = extract_fields_at_time_3d(ds, target_time)
    
    processed_train_data = []
    norm_params = {}

    for i, data in enumerate(train_data):
        u_centered, v_centered = data['u'], data['v']
        w_centered = interpolate_staggered_to_centers_3d(data['w'], 'z')
        b_field, p_field = data['b'], data['p_dyn']
        
        b_profile = np.mean(b_field, axis=(1, 2), keepdims=True)
        b_fluctuations = b_field - b_profile
        p_profile = np.mean(p_field, axis=(1, 2), keepdims=True)
        p_fluctuations = p_field - p_profile
        
        if i == 0:
            u_norm, u_mean, u_std = normalize_field(u_centered)
            v_norm, v_mean, v_std = normalize_field(v_centered)
            w_norm, w_mean, w_std = normalize_field(w_centered)
            b_norm, b_fluc_mean, b_fluc_std = normalize_field(b_fluctuations)
            p_norm, p_fluc_mean, p_fluc_std = normalize_field(p_fluctuations)
            norm_params.update({
                'u':(u_mean, u_std), 'v':(v_mean, v_std), 'w':(w_mean, w_std), 
                'b':(b_fluc_mean, b_fluc_std), 'p':(p_fluc_mean, p_fluc_std),
                'b_profile': b_profile, 'p_profile': p_profile
            })
        else:
            u_norm = (u_centered - norm_params['u'][0]) / norm_params['u'][1]
            v_norm = (v_centered - norm_params['v'][0]) / norm_params['v'][1]
            w_norm = (w_centered - norm_params['w'][0]) / norm_params['w'][1]
            b_norm = (b_fluctuations - norm_params['b'][0]) / norm_params['b'][1]
            p_norm = (p_fluctuations - norm_params['p'][0]) / norm_params['p'][1]

        processed_train_data.append({'u': u_norm, 'v': v_norm, 'w': w_norm, 'b': b_norm, 'p': p_norm})

    target_u_norm = (target_data['u'] - norm_params['u'][0]) / norm_params['u'][1]
    target_v_norm = (target_data['v'] - norm_params['v'][0]) / norm_params['v'][1]
    target_w_norm = (interpolate_staggered_to_centers_3d(target_data['w'], 'z') - norm_params['w'][0]) / norm_params['w'][1]
    target_b_norm = (target_data['b'] - norm_params['b_profile'] - norm_params['b'][0]) / norm_params['b'][1]
    target_p_norm = (target_data['p_dyn'] - norm_params['p_profile'] - norm_params['p'][0]) / norm_params['p'][1]
    
    input_channels = [d[k] for d in processed_train_data for k in ['u', 'v', 'w', 'b', 'p']]
    input_tensor = torch.FloatTensor(np.stack(input_channels, axis=0)).unsqueeze(0).to(device)
    target_tensor = torch.FloatTensor(np.stack([target_u_norm, target_v_norm, target_w_norm, target_b_norm, target_p_norm], axis=0)).unsqueeze(0).to(device)
    
    return input_tensor, target_tensor, processed_train_data, norm_params

#==================================
# BLOCK 4 - Physics-Informed Loss Functions for 3D
#==================================

def compute_derivatives_multi_3d(fields, dx, dy, dz):
    if fields.dim() == 4: fields = fields.unsqueeze(0)
    df_dx = (F.pad(fields, (1, 1, 0, 0, 0, 0), mode='circular')[..., 2:] - F.pad(fields, (1, 1, 0, 0, 0, 0), mode='circular')[..., :-2]) / (2 * dx)
    df_dy = (F.pad(fields, (0, 0, 1, 1, 0, 0), mode='circular')[..., 2:, :] - F.pad(fields, (0, 0, 1, 1, 0, 0), mode='circular')[..., :-2, :]) / (2 * dy)
    df_dz = torch.zeros_like(fields)
    df_dz[..., 1:-1, :, :] = (fields[..., 2:, :, :] - fields[..., :-2, :, :]) / (2 * dz)
    df_dz[..., 0, :, :] = (fields[..., 1, :, :] - fields[..., 0, :, :]) / dz
    df_dz[..., -1, :, :] = (fields[..., -1, :, :] - fields[..., -2, :, :]) / dz
    return df_dx, df_dy, df_dz

def physics_loss_multi_3d(pred_norm, prev_states_norm, dx, dy, dz, dt, kappa, norm_params):
    u_pred = pred_norm[:, 0] * norm_params['u'][1] + norm_params['u'][0]
    v_pred = pred_norm[:, 1] * norm_params['v'][1] + norm_params['v'][0]
    w_pred = pred_norm[:, 2] * norm_params['w'][1] + norm_params['w'][0]
    b_profile = torch.from_numpy(norm_params['b_profile']).float().to(device)
    b_pred_fluc = pred_norm[:, 3] * norm_params['b'][1] + norm_params['b'][0]
    b_pred = b_pred_fluc + b_profile
    p_profile = torch.from_numpy(norm_params['p_profile']).float().to(device)
    p_pred_fluc = pred_norm[:, 4] * norm_params['p'][1] + norm_params['p'][0]
    p_pred = p_pred_fluc + p_profile
    
    last_state = prev_states_norm[-1]
    u_last = torch.from_numpy(last_state['u']).float().to(device) * norm_params['u'][1] + norm_params['u'][0]
    v_last = torch.from_numpy(last_state['v']).float().to(device) * norm_params['v'][1] + norm_params['v'][0]
    w_last = torch.from_numpy(last_state['w']).float().to(device) * norm_params['w'][1] + norm_params['w'][0]
    b_last_fluc = torch.from_numpy(last_state['b']).float().to(device) * norm_params['b'][1] + norm_params['b'][0]
    b_last = b_last_fluc + b_profile
    
    du_dt, dv_dt, dw_dt, db_dt = (u_pred - u_last)/dt, (v_pred - v_last)/dt, (w_pred - w_last)/dt, (b_pred - b_last)/dt
    
    all_fields = torch.stack([u_pred, v_pred, w_pred, b_pred, p_pred], dim=1)
    df_dx, df_dy, df_dz = compute_derivatives_multi_3d(all_fields, dx, dy, dz)
    du_dx, dv_dx, dw_dx, db_dx, dp_dx = [df_dx[:,i] for i in range(5)]
    du_dy, dv_dy, dw_dy, db_dy, dp_dy = [df_dy[:,i] for i in range(5)]
    du_dz, dv_dz, dw_dz, db_dz, dp_dz = [df_dz[:,i] for i in range(5)]
    
    d2u_dx2,_,_ = compute_derivatives_multi_3d(du_dx.unsqueeze(1), dx,dy,dz); _,d2u_dy2,_ = compute_derivatives_multi_3d(du_dy.unsqueeze(1), dx,dy,dz); _,_,d2u_dz2 = compute_derivatives_multi_3d(du_dz.unsqueeze(1), dx,dy,dz)
    d2v_dx2,_,_ = compute_derivatives_multi_3d(dv_dx.unsqueeze(1), dx,dy,dz); _,d2v_dy2,_ = compute_derivatives_multi_3d(dv_dy.unsqueeze(1), dx,dy,dz); _,_,d2v_dz2 = compute_derivatives_multi_3d(dv_dz.unsqueeze(1), dx,dy,dz)
    d2w_dx2,_,_ = compute_derivatives_multi_3d(dw_dx.unsqueeze(1), dx,dy,dz); _,d2w_dy2,_ = compute_derivatives_multi_3d(dw_dy.unsqueeze(1), dx,dy,dz); _,_,d2w_dz2 = compute_derivatives_multi_3d(dw_dz.unsqueeze(1), dx,dy,dz)
    d2b_dx2,_,_ = compute_derivatives_multi_3d(db_dx.unsqueeze(1), dx,dy,dz); _,d2b_dy2,_ = compute_derivatives_multi_3d(db_dy.unsqueeze(1), dx,dy,dz); _,_,d2b_dz2 = compute_derivatives_multi_3d(db_dz.unsqueeze(1), dx,dy,dz)

    lap_u = d2u_dx2.squeeze(1) + d2u_dy2.squeeze(1) + d2u_dz2.squeeze(1)
    lap_v = d2v_dx2.squeeze(1) + d2v_dy2.squeeze(1) + d2v_dz2.squeeze(1)
    lap_w = d2w_dx2.squeeze(1) + d2w_dy2.squeeze(1) + d2w_dz2.squeeze(1)
    lap_b = d2b_dx2.squeeze(1) + d2b_dy2.squeeze(1) + d2b_dz2.squeeze(1)

    adv_u = u_pred*du_dx + v_pred*du_dy + w_pred*du_dz
    adv_v = u_pred*dv_dx + v_pred*dv_dy + w_pred*dv_dz
    adv_w = u_pred*dw_dx + v_pred*dw_dy + w_pred*dw_dz
    adv_b = u_pred*db_dx + v_pred*db_dy + w_pred*db_dz
    
    u_res = du_dt + adv_u + dp_dx - kappa * lap_u
    v_res = dv_dt + adv_v + dp_dy - kappa * lap_v
    w_res = dw_dt + adv_w + dp_dz - kappa * lap_w + b_pred
    b_res = db_dt + adv_b - kappa * lap_b
    cont_res = du_dx + dv_dy + dw_dz
    
    return torch.mean(u_res**2) + torch.mean(v_res**2) + torch.mean(w_res**2) + torch.mean(b_res**2) + torch.mean(cont_res**2)

def boundary_loss_multi_3d(pred_norm, norm_params):
    u_pred = pred_norm[:, 0] * norm_params['u'][1] + norm_params['u'][0]
    v_pred = pred_norm[:, 1] * norm_params['v'][1] + norm_params['v'][0]
    w_pred = pred_norm[:, 2] * norm_params['w'][1] + norm_params['w'][0]
    b_profile = torch.from_numpy(norm_params['b_profile']).float().to(device)
    b_pred_fluc = pred_norm[:, 3] * norm_params['b'][1] + norm_params['b'][0]
    b_pred = b_pred_fluc + b_profile
    
    u_bc = torch.mean(u_pred[:, :, 0, :]**2) + torch.mean(u_pred[:, :, -1, :]**2)
    v_bc = torch.mean(v_pred[:, :, 0, :]**2) + torch.mean(v_pred[:, :, -1, :]**2)
    w_bc = torch.mean(w_pred[:, :, 0, :]**2) + torch.mean(w_pred[:, :, -1, :]**2)
    b_bc = torch.mean((b_pred[:, :, -1, :] - 0.5)**2) + torch.mean((b_pred[:, :, 0, :] - 0.0)**2)
    
    return u_bc + v_bc + w_bc + b_bc

def smoothness_loss_multi_3d(pred):
    tv_loss = 0
    for i in range(pred.shape[1]):
        field = pred[:, i]
        tv_loss += torch.mean(torch.abs(field[:, :, :, 1:] - field[:, :, :, :-1])) # d/dx
        tv_loss += torch.mean(torch.abs(field[:, :, 1:, :] - field[:, :, :-1, :])) # d/dy
        tv_loss += torch.mean(torch.abs(field[:, 1:, :, :] - field[:, :-1, :, :])) # d/dz
    return tv_loss

#==================================
# BLOCK 5 - Main Execution and Visualization
#==================================

if __name__ == '__main__':
    try:
        ds_3d = xr.open_dataset('./RBC_Output.nc', decode_timedelta=False)
        print("Successfully loaded 3D dataset from './RBC_Output.nc'")
    except FileNotFoundError:
        print("FATAL ERROR: 3D data file not found at './RBC_Output.nc'. Please check the path.")
        exit()

    dx = ds_3d.coords[ds_3d['u'].dims[3]].values[1] - ds_3d.coords[ds_3d['u'].dims[3]].values[0]
    dy = ds_3d.coords[ds_3d['v'].dims[2]].values[1] - ds_3d.coords[ds_3d['v'].dims[2]].values[0]
    dz = ds_3d.coords[ds_3d['w'].dims[1]].values[1] - ds_3d.coords[ds_3d['w'].dims[1]].values[0]

    dt = (CONFIG['data']['target_time'] - CONFIG['data']['train_times'][-1]) * CONFIG['data']['simulation_dt']
    print(f"Dynamic 'dt' for physics loss calculated as: {dt}s")

    input_tensor, target_tensor, processed_train_data, norm_params = prepare_data_and_tensors_3d(ds_3d, CONFIG['data']['train_times'], CONFIG['data']['target_time'], CONFIG)

    os.makedirs('results', exist_ok=True)
    log_file_path = 'results/training_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write("--- TRAINING CONFIGURATION ---\n")
        log_file.write(str(CONFIG) + "\n\n")

    model = MultiVariableFNO3d(config=CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in 3D FNO: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['training']['learning_rate'], epochs=CONFIG['training']['epochs'], steps_per_epoch=1, pct_start=CONFIG['training']['scheduler_pct_start'])
    loss_weights = CONFIG['loss_weights']
    
    history = {'total_loss': [], 'physics_loss': [], 'data_loss': [], 'bc_loss': [], 'smooth_loss': []}
    
    print("\nStarting full physics-informed training with fixed weights...")
    start_time = time.time()
    with open(log_file_path, 'a') as log_file:
        for epoch in range(CONFIG['training']['epochs']):
            optimizer.zero_grad()
            pred_norm = model(input_tensor)
            
            l_physics = physics_loss_multi_3d(pred_norm, processed_train_data, dx, dy, dz, dt, CONFIG['physics']['kappa'], norm_params)
            l_data = F.mse_loss(pred_norm, target_tensor)
            l_bc = boundary_loss_multi_3d(pred_norm, norm_params)
            l_smooth = smoothness_loss_multi_3d(pred_norm)
            
            total_loss = (loss_weights['physics'] * l_physics + 
                          loss_weights['data'] * l_data + 
                          loss_weights['bc'] * l_bc + 
                          loss_weights['smooth'] * l_smooth)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            log_line = (f"Epoch {epoch}/{CONFIG['training']['epochs']} | Total: {total_loss.item():.4e} | "
                        f"Physics: {l_physics.item():.4e} | Data: {l_data.item():.4e} | "
                        f"BC: {l_bc.item():.4e} | Smooth: {l_smooth.item():.4e}")
            
            if epoch % 50 == 0:
                print(log_line)
            log_file.write(log_line + '\n')

            for k, v in [('total_loss', total_loss), ('physics_loss', l_physics), ('data_loss', l_data), ('bc_loss', l_bc), ('smooth_loss', l_smooth)]:
                history[k].append(v.item())

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

    print("Saving trained model to 'results/3d_fno_model.pth'...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'norm_params': norm_params,
        'final_loss_weights': loss_weights
    }, 'results/3d_fno_model.pth')
    print("Model saved successfully.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    loss_names = ['total_loss', 'physics_loss', 'data_loss', 'bc_loss', 'smooth_loss']
    for i, name in enumerate(loss_names):
        axes[i].semilogy(history[name])
        axes[i].set_title(name.replace('_', ' ').title())
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss.png', dpi=300)
    plt.show()

    print("\nVisualizing a central 2D slice of the 3D prediction...")
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).squeeze(0)

    u_pred = pred_norm[0].cpu().numpy() * norm_params['u'][1] + norm_params['u'][0]
    v_pred = pred_norm[1].cpu().numpy() * norm_params['v'][1] + norm_params['v'][0]
    w_pred = pred_norm[2].cpu().numpy() * norm_params['w'][1] + norm_params['w'][0]
    b_pred = (pred_norm[3].cpu().numpy() * norm_params['b'][1] + norm_params['b'][0]) + norm_params['b_profile']
    p_pred = (pred_norm[4].cpu().numpy() * norm_params['p'][1] + norm_params['p'][0]) + norm_params['p_profile']
    
    predictions = {'u': u_pred, 'v': v_pred, 'w': w_pred, 'b': b_pred, 'p': p_pred}

    target_data_step = extract_fields_at_time_3d(ds_3d, CONFIG['data']['target_time'])
    targets = {
        'u': target_data_step['u'], 'v': target_data_step['v'],
        'w': interpolate_staggered_to_centers_3d(target_data_step['w'], 'z'),
        'b': target_data_step['b'], 'p': target_data_step['p_dyn']
    }

    fig, axes = plt.subplots(len(predictions), 3, figsize=(18, 22))
    fig.suptitle(f'3D FNO Prediction vs. Target (Central Y-Slice)', fontsize=16)
    slice_idx = CONFIG['data']['grid_shape'][1] // 2

    for i, (var, name) in enumerate(zip(predictions.keys(), ['u', 'v', 'w', 'b', 'p'])):
        pred_slice, target_slice = predictions[var][:, slice_idx, :], targets[var][:, slice_idx, :]
        vmin, vmax = min(pred_slice.min(), target_slice.min()), max(pred_slice.min(), target_slice.max())
        
        axes[i, 0].imshow(pred_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower'); axes[i, 0].set_title(f'{name} - Prediction')
        fig.colorbar(axes[i, 0].images[0], ax=axes[i, 0])
        axes[i, 1].imshow(target_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower'); axes[i, 1].set_title(f'{name} - Target')
        fig.colorbar(axes[i, 1].images[0], ax=axes[i, 1])
        diff = pred_slice - target_slice
        im3 = axes[i, 2].imshow(diff, cmap='seismic', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(), origin='lower'); axes[i, 2].set_title(f'{name} - Error')
        fig.colorbar(im3, ax=axes[i, 2])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('results/prediction_slices.png', dpi=300)
    plt.show()
