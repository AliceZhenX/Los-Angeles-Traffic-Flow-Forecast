import os
import sys
import pickle
import logging
import copy
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
project_root = os.path.dirname(current_dir)

from network import Ultimate_Graphormer
from layers import AdvancedBetaPhysicsLoss

# Utils 
def load_adj(pkl_path):
    with open(pkl_path, 'rb') as f:
        try: _, _, adj_mx = pickle.load(f)
        except: _, _, adj_mx = pickle.load(f, encoding='latin1')
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return [d_mat_inv.dot(adj_mx).todense(), d_mat_inv.dot(adj_mx.T).todense()]

def load_data(data_dir, batch_size):
    data = {}
    for cat in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, f'{cat}.npz'))
        data[f'x_{cat}'] = torch.FloatTensor(cat_data['x'])
        data[f'y_{cat}'] = torch.FloatTensor(cat_data['y'])
    train_dl = DataLoader(TensorDataset(data['x_train'], data['y_train']), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(TensorDataset(data['x_val'], data['y_val']), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_dl = DataLoader(TensorDataset(data['x_test'], data['y_test']), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_dl, val_dl, test_dl

def masked_mae_loss(preds, labels, null_val=0.0):
    mask = (labels > null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return torch.mean(torch.abs(preds - labels) * mask)

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None: self.module.to(device=device)
    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device: model_v = model_v.to(device=self.device)
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Configuration
class Config:
    base_dir = project_root
    data_dir = os.path.join(base_dir, "clean_data/")
    adj_path = os.path.join(base_dir, "processed_data/adj.pkl")
    gpus = "4,5"
    batch_size = 128
    accumulation_steps = 2
    epochs = 200 
    patience = 50
    num_nodes = 207
    seq_len = 12
    output_dim = 1
    d_model = 96
    nhead = 4
    dropout = 0.1
    max_lr = 0.0005
    weight_decay = 1e-3
    ema_decay = 0.999
    use_ema = True
    
    # Baseline Defaults
    use_revin = True
    use_st_attn = True
    encoder_type = 'pagerank'
    decoder_type = 'patch'
    use_gating = True
    output_type = 'point'
    
    max_speed = 100.0

def main(exp_id):
    cfg = Config()
    
    if exp_id == '0_baseline':
        cfg.model_name = "Exp0_Baseline"
        
    elif exp_id == '1_no_revin':
        cfg.model_name = "Exp1_No_RevIN"
        cfg.use_revin = False
        
    elif exp_id == '2_dcrnn':
        cfg.model_name = "Exp2_DCRNN"
        cfg.encoder_type = 'dcrnn'
        
    elif exp_id == '3_no_st':
        cfg.model_name = "Exp3_No_ST"
        cfg.use_st_attn = False
        
    elif exp_id == '4_causal':
        cfg.model_name = "Exp4_Standard_Causal"
        cfg.decoder_type = 'causal'
        
    elif exp_id == '5_transformer':
        cfg.model_name = "Exp5_Transformer_Dec"
        cfg.decoder_type = 'transformer'
        
    elif exp_id == '6_no_gate':
        cfg.model_name = "Exp6_No_Gating"
        cfg.use_gating = False
        
    elif exp_id == '7_beta':
        cfg.model_name = "Exp7_Beta_Head"
        cfg.output_type = 'probabilistic'
        cfg.batch_size = 128 # Beta needs larger batch
        
    else:
        raise ValueError(f"Unknown Exp ID: {exp_id}")

    # Setup
    cfg.log_dir = os.path.join(cfg.base_dir, "experiments", "Ablation_Full", cfg.model_name)
    logger = get_logger(cfg.log_dir)
    logger.info(f" Running {cfg.model_name}")
    logger.info(f"Settings: RevIN={cfg.use_revin}, ST={cfg.use_st_attn}, Enc={cfg.encoder_type}, Dec={cfg.decoder_type}, Gate={cfg.use_gating}, Out={cfg.output_type}")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adj = load_adj(cfg.adj_path)
    train_dl, val_dl, test_dl = load_data(cfg.data_dir, cfg.batch_size)
    
    # Model
    model = Ultimate_Graphormer(cfg, adj_mx=adj)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr, epochs=cfg.epochs, steps_per_epoch=len(train_dl)//cfg.accumulation_steps)
    scaler = GradScaler()
    ema = ModelEMA(model.module if hasattr(model, 'module') else model, decay=cfg.ema_decay, device=device) if cfg.use_ema else None
    
    # Loss for Beta
    if cfg.output_type == 'probabilistic':
        criterion = AdvancedBetaPhysicsLoss(adj_mx=adj, max_speed=cfg.max_speed).to(device)
    
    best_val_mae = float('inf')
    patience_cnt = 0
    
    # Training 
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = []
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            y_target = y[..., 0:1]
            
            with autocast():
                if cfg.output_type == 'probabilistic':
                    alpha, beta = model(x)
                    loss = criterion(alpha, beta, y_target)
                else:
                    pred = model(x)
                    loss = masked_mae_loss(pred, y_target)
                loss = loss / cfg.accumulation_steps
            
            scaler.scale(loss).backward()
            if (i+1) % cfg.accumulation_steps == 0:
                if cfg.output_type == 'probabilistic':
                    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema: ema.update(model.module if hasattr(model, 'module') else model)
            train_loss.append(loss.item() * cfg.accumulation_steps)
            
        # Validation 
        val_model = ema.module if ema else (model.module if hasattr(model, 'module') else model)
        val_model.eval()
        val_maes = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                if cfg.output_type == 'probabilistic':
                    alpha, beta = val_model(x)
                    pred = (alpha / (alpha + beta)) * cfg.max_speed
                else:
                    pred = val_model(x)
                val_maes.append(masked_mae_loss(pred, y[..., 0:1]).item())
        val_mae = np.mean(val_maes)
        
        logger.info(f"Epoch {epoch+1} | Loss: {np.mean(train_loss):.4f} | Val MAE: {val_mae:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            torch.save(val_model.state_dict(), os.path.join(cfg.log_dir, 'best.pth'))
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                logger.info("Early Stopping")
                break
                
    # Testing 
    logger.info("Testing...")
    test_model = Ultimate_Graphormer(cfg, adj_mx=adj).to(device)
    test_model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'best.pth')))
    test_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            if cfg.output_type == 'probabilistic':
                alpha, beta = test_model(x)
                pred = (alpha / (alpha + beta)) * cfg.max_speed
            else:
                pred = test_model(x)
            preds.append(pred.cpu())
            trues.append(y[..., 0:1].cpu())
    
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    mae = masked_mae_loss(preds, trues).item()
    mse = ((preds - trues)**2) * (trues > 0).float()
    rmse = torch.sqrt(torch.sum(mse)/torch.sum(trues>0)).item()
    
    logger.info(f"FINAL RESULT [{exp_id}]: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    with open(os.path.join(cfg.base_dir, "FIX","final_ablation_results.csv"), "a") as f:
        f.write(f"{exp_id},{mae:.4f},{rmse:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, 
                        choices=['0_baseline', '1_no_revin', '2_dcrnn', '3_no_st', 
                                 '4_causal', '5_transformer', '6_no_gate', '7_beta'])
    args = parser.parse_args()
    main(args.exp)