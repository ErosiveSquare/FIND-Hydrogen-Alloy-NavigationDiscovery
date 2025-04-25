import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr 
from get_T import *
import joblib


def get_features(str_):
    formula = str_
    composition = Composition(formula)
    
    ep = ElementProperty.from_preset("magpie")
    features = ep.featurize(composition)
    return features

    
# ----------------------
# 自定义采样层
# ----------------------
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.size(0)
        latent_dim = z_mean.size(1)
        epsilon = torch.randn(batch_size, latent_dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# ----------------------
# 数据预处理
# ----------------------
def load_and_featurize(data_path):
    df = pd.read_excel(data_path)
    stc = StrToComposition(target_col_id="composition_obj")
    df = stc.featurize_dataframe(df, "Components")
    valid_df = df.dropna(subset=["composition_obj"])
    
    ep = ElementProperty.from_preset("magpie")
    X = ep.featurize_many(valid_df["composition_obj"], ignore_errors=True)
    X_df = pd.DataFrame(X, columns=ep.feature_labels()).dropna(axis=1)
    return X_df, valid_df["Components"].tolist(), ep

# ----------------------
# 数据集类
# ----------------------
class CompositionDataset(Dataset):
    def __init__(self, X_scaled):
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]

# ----------------------
# VAE 模型构建
# ----------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.z_mean_layer = nn.Linear(64, latent_dim)
        self.z_log_var_layer = nn.Linear(64, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
        # 采样层
        self.sampling = Sampling()
    
    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        return self.sampling(z_mean, z_log_var)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z)
        return x_decoded, z_mean, z_log_var

# ----------------------
# 化学式生成
# ----------------------
def generate_formulas(vae, scaler, ep, elements,mm, n_samples=1, latent_dim=5):
    
    # 准备生成器
    def generator(latent_samples):
        latent_samples = latent_samples.unsqueeze(0)
        z_mean, z_log_var = vae.encode(latent_samples)
        z = vae.reparameterize(z_mean, z_log_var)
        nn = vae.decode(z)
        x_ = mm.detach().numpy()
        y_ = nn.detach().numpy()
        y_ = y_[0]
        correlation, _ = pearsonr(x_, y_)
        print(correlation)
        return nn
    
    # latent_samples = torch.randn(n_samples, latent_dim)
    latent_samples = mm
    generated_features = scaler.inverse_transform(generator(latent_samples).detach().numpy())
    
    element_symbols = [e.symbol for e in elements]
    
    def objective(x, target):
        try:
            formula_dict = {element_symbols[i]: x[i] for i in np.where(x > 0.01)[0]}
            comp = Composition(formula_dict).reduced_formula
            current_feat = ep.featurize(comp)
            return np.sum((current_feat - target)**2)
        except:
            return np.inf
    
    generated_formulas = []
    for features in generated_features:
        initial_guess = np.random.dirichlet(np.ones(len(element_symbols)))
        bounds = [(0,1) for _ in element_symbols]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        res = minimize(objective, initial_guess, args=(features,),
                      bounds=bounds, constraints=constraints, method='SLSQP')
        
        formula = Composition({
            element_symbols[i]: round(res.x[i], 2)
            for i in np.where(res.x > 0.01)[0]
        }).reduced_formula
        generated_formulas.append(formula)
    
    return generated_formulas

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    DATA_PATH = "Composition control absorption1_magpie2.xlsx"
    LATENT_DIM = 3
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # 数据预处理
    X_df, valid_formulas, ep = load_and_featurize(DATA_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)  
    joblib.dump(scaler, 'scaler4.joblib')
    x_1 = get_T(DATA_PATH)
    X_scaled = np.column_stack([X_scaled, x_1.reshape(-1, 1)])
    # 创建数据集和数据加载器
    dataset = CompositionDataset(X_scaled)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 构建 VAE 模型
    input_dim = X_scaled.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
    # vae.load_state_dict(torch.load('vae_.pth'))
    mm = dataset[0][0]
    print(dataset[0][0])
    print(dataset[1])
    # 定义优化器
    # optimizer = optim.Adam(vae.parameters())
    
    # # 训练 VAE
    # for epoch in range(EPOCHS):
    #     vae.train()
    #     total_loss = 0
    #     for x, _ in dataloader:
    #         optimizer.zero_grad()
            
    #         x_decoded, z_mean, z_log_var = vae(x)
            
    #         # 计算重构损失
    #         reconstruction_loss = torch.sum((x - x_decoded)**2, dim=-1)
            
    #         # 计算 KL 散度损失
    #         kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), dim=-1)
            
    #         # 总损失
    #         loss = torch.mean(reconstruction_loss + kl_loss)
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item() * x.size(0)
        
    #     avg_loss = total_loss / len(dataset)
    #     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    # torch.save(vae.state_dict(), 'vae_.pth')
    # # # # 生成化学式
    # # # elements = list({e for formula in valid_formulas for e in Composition(formula).elements})
    # # # new_formulas = generate_formulas(vae, scaler, ep, elements, mm,n_samples=10, latent_dim=LATENT_DIM)
    
    # # # print("\n生成的化学式:")
    # # # for i, formula in enumerate(new_formulas, 1):
    # # #     print(f"{i}. {formula}")


    # latent_vectors = []
    # vae.eval()  # 评估模式
    # with torch.no_grad():
    #     for i in range(len(dataset)):
    #         x = dataset[i][0]
    #         z_mean, z_log_var = vae.encode(x.unsqueeze(0))  # 添加 batch 维度
    #         z = vae.reparameterize(z_mean, z_log_var)
    #         latent_vectors.append(z.squeeze(0).numpy())  # 移除 batch 维度

    # # 保存到 Excel
    # latent_df = pd.DataFrame(latent_vectors, columns=[f"latent_{i}" for i in range(LATENT_DIM)])
    # latent_df.to_excel("latent_vectors.xlsx", index=False)
    # print("潜在向量已保存至 latent_vectors.xlsx")