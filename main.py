import os
import sys
import argparse
import yaml
import time
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pulp
import matplotlib.pyplot as plt

# Check for DeepMIMO
try:
    import DeepMIMOv3 as DeepMIMO
    DEEPMIMO_AVAILABLE = True
except ImportError:
    DEEPMIMO_AVAILABLE = False

# ==========================================
# UTILITIES
# ==========================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Robust6G")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
        file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)
    return logger

def save_pickle(data, path):
    with open(path, 'wb') as f: pickle.dump(data, f)

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f: return pickle.load(f)
    return None

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ==========================================
# 1. DATA GENERATION (Hybrid: DeepMIMO or Synthetic)
# ==========================================
class ChannelGenerator:
    def __init__(self, cfg):
        self.c = cfg['system']
        self.n_samples = 300 # Default dataset size
        self.deepmimo_path = './DeepMIMO_Dataset'

    def generate(self, logger):
        logger.info("Generating Channel Data...")
        
        # A. Try DeepMIMO
        if DEEPMIMO_AVAILABLE and os.path.exists(self.deepmimo_path):
            try:
                params = DeepMIMO.default_params()
                params['dataset_folder'] = self.deepmimo_path
                params['scenario'] = 'O1_60'
                params['num_paths'] = 3
                params['active_BS'] = np.array(range(1, self.c['n_aps']+1))
                params['user_row_first'] = 1
                params['user_row_last'] = 100
                params['bs_antenna']['shape'] = np.array([1, self.c['n_ant'], 1])
                params['ue_antenna']['shape'] = np.array([1, 1, 1])
                params['OFDM']['bandwidth'] = 0.5
                params['OFDM']['subcarriers'] = 1
                
                dataset = DeepMIMO.generate_data(params)
                bs_list = [bs['user']['channel'][:, 0, :, 0] for bs in dataset]
                H_pool = np.stack(bs_list, axis=0).transpose(1, 0, 2) # (U_pool, M, N_ant)
                
                indices = np.random.choice(H_pool.shape[0], (self.n_samples, self.c['n_ues']), replace=True)
                H_true = np.array([H_pool[idx] for idx in indices])
                logger.info(f"Loaded DeepMIMO data: {H_true.shape}")
                
            except Exception as e:
                logger.warning(f"DeepMIMO failed ({e}). Reverting to Synthetic.")
                H_true = self._synthetic()
        else:
            logger.info("DeepMIMO not found. Using Synthetic Ray-Based Model.")
            H_true = self._synthetic()
            
        # Normalize
        pwr = np.mean(np.abs(H_true)**2)
        H_true = H_true / np.sqrt(pwr)
        return H_true

    def _synthetic(self):
        # Fallback generator
        H = np.zeros((self.n_samples, self.c['n_ues'], self.c['n_aps'], self.c['n_ant']), dtype=np.complex64)
        for i in range(self.n_samples):
            for u in range(self.c['n_ues']):
                for m in range(self.c['n_aps']):
                    dist = np.random.uniform(10, 100)
                    pl = 1.0 / (dist ** 2.5)
                    gain = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                    H[i, u, m, :] = np.sqrt(pl) * gain
        return H

# ==========================================
# 2. ESTIMATOR (LS-CNN)
# ==========================================
def gnll_loss(y_true, y_pred):
    mu, log_var = tf.split(y_pred, 2, axis=-1)
    precision = tf.exp(-log_var)
    mse = tf.square(y_true - mu)
    return tf.reduce_mean(log_var + precision * mse)

def build_estimator(n_ues, feat_dim):
    inp = layers.Input(shape=(n_ues, feat_dim, 2))
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    out = layers.Conv2D(4, (1,1), padding='same')(x) # Mean_R, Mean_I, Var_R, Var_I
    return keras.Model(inp, out)

# ==========================================
# 3. OPTIMIZATION (Robust MILP)
# ==========================================
def solve_milp(h_flat, sigma_flat, cfg):
    opt = cfg['optimization']
    sys = cfg['system']
    
    # Calculate Robust Gains
    h_flat = np.nan_to_num(h_flat)
    sigma_flat = np.nan_to_num(sigma_flat)
    gains = np.sum(np.abs(h_flat)**2, axis=1)
    unc = np.sum(sigma_flat, axis=1)
    alpha = np.maximum(gains - opt['lambda_risk'] * unc, 0)

    prob = pulp.LpProblem("RobustCluster", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(sys['n_ues']), range(sys['n_subcarriers'])), cat='Binary')
    
    # Objective: Maximize Robust Sum Rate Proxy
    obj = 0
    for u in range(sys['n_ues']):
        for k in range(sys['n_subcarriers']):
            obj += alpha[u] * x[u][k]
    prob += obj
    
    # Constraints
    for u in range(sys['n_ues']): # User assigned exactly once
        prob += pulp.lpSum(x[u][k] for k in range(sys['n_subcarriers'])) == 1
    
    for k in range(sys['n_subcarriers']): # Capacity limit
        prob += pulp.lpSum(x[u][k] for u in range(sys['n_ues'])) <= opt['max_users_per_subcarrier']
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=opt['time_limit_s']))
    
    sol = np.zeros((sys['n_ues'], sys['n_subcarriers']))
    for u in range(sys['n_ues']):
        for k in range(sys['n_subcarriers']):
            if pulp.value(x[u][k]) == 1: sol[u, k] = 1
    return sol

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config) as f: cfg = yaml.safe_load(f)
    
    out_dir = cfg['outputs']['root']
    logger = setup_logger(out_dir)
    set_seed(cfg['seed'])
    
    logger.info("=== Stage 1: Data Generation ===")
    gen = ChannelGenerator(cfg)
    H_true = gen.generate(logger)
    
    # Split Train/Test
    n_train = int(0.8 * len(H_true))
    H_train, H_test = H_true[:n_train], H_true[n_train:]
    
    logger.info("=== Stage 2: Train Estimator ===")
    # Add noise for training
    noise_std = np.sqrt(10**(-cfg['pilots']['snr_train_db']/10)/2)
    H_ls_train = H_train + (np.random.randn(*H_train.shape) + 1j*np.random.randn(*H_train.shape)) * noise_std
    
    # Flatten dims for CNN: (N, U, M*Ant)
    feat_dim = cfg['system']['n_aps'] * cfg['system']['n_ant']
    H_ls_flat = H_ls_train.reshape(n_train, cfg['system']['n_ues'], feat_dim)
    H_true_flat = H_train.reshape(n_train, cfg['system']['n_ues'], feat_dim)
    
    X_train = np.stack([H_ls_flat.real, H_ls_flat.imag], axis=-1)
    Y_train = np.stack([H_true_flat.real, H_true_flat.imag], axis=-1)
    
    model = build_estimator(cfg['system']['n_ues'], feat_dim)
    model.compile(optimizer='adam', loss=gnll_loss)
    model.fit(X_train, Y_train, epochs=cfg['training']['estimator_epochs'], batch_size=cfg['training']['batch_size'], verbose=0)
    logger.info("Estimator trained.")

    logger.info("=== Stage 3: SNR Sweep & Optimization ===")
    results = []
    
    for snr in cfg['snr_sweep']['snr_db']:
        logger.info(f"Simulating SNR: {snr} dB")
        
        # Test Data
        ns = np.sqrt(10**(-snr/10)/2)
        H_ls_test = H_test + (np.random.randn(*H_test.shape) + 1j*np.random.randn(*H_test.shape)) * ns
        
        # 1. Estimation
        h_in = H_ls_test.reshape(len(H_ls_test), cfg['system']['n_ues'], feat_dim)
        preds = model.predict(np.stack([h_in.real, h_in.imag], axis=-1), verbose=0)
        mu, logvar = np.split(preds, 2, axis=-1)
        H_hat = mu[...,0] + 1j*mu[...,1]
        Sigma = np.exp(logvar[...,0]) + np.exp(logvar[...,1])
        
        # 2. Optimization (Sample subset for speed)
        se_list = []
        n_opt = min(20, len(H_test)) # Optimize first 20 samples per SNR for demo
        for i in range(n_opt):
            # Solve MILP
            alloc = solve_milp(H_hat[i], Sigma[i], cfg)
            
            # Eval SE (Simplified OMA/NOMA calc)
            # (In full repo, use your se_eval module here)
            # This is a placeholder for the logic:
            se_list.append(np.sum(alloc) * np.log2(1 + 10**(snr/10))) # Dummy metric for demo
            
        results.append({'snr': snr, 'se': np.mean(se_list)})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    logger.info("Results saved to outputs/results.csv")
    
    # Plotting
    plt.plot(df['snr'], df['se'], marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectral Efficiency')
    plt.title('Performance Sweep')
    plt.savefig(os.path.join(out_dir, "se_vs_snr.png"))
    logger.info("Done.")

if __name__ == "__main__":
    main()
