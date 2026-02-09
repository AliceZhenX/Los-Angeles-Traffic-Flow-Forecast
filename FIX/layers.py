import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Normalization (Exp 1) 
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x

#2. Graph Convolutions (Exp 2 & 6) 
class StandardDiffusionConv(nn.Module):
    """ 原版 DCRNN 扩散卷积 (用于 Exp 2: Replace PageRank) """
    def __init__(self, input_dim, output_dim, num_nodes, diffusion_steps=2, num_supports=2, **kwargs):
        super().__init__()
        self.k = diffusion_steps
        self.num_supports = num_supports
        self.weights = nn.Parameter(torch.FloatTensor(num_supports * (self.k + 1), input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x, supports):
        out = 0
        for i, support in enumerate(supports):
            x_k = x
            w_idx = i * (self.k + 1)
            out += torch.einsum('bnf,fo->bno', x_k, self.weights[w_idx])
            for k in range(1, self.k + 1):
                x_k = torch.einsum('nm,bmf->bnf', support, x_k)
                out += torch.einsum('bnf,fo->bno', x_k, self.weights[w_idx + k])
        return out + self.bias

class PageRankDiffusionConv(nn.Module):
    """ 
    PageRank 扩散卷积 (Baseline)
    支持 Exp 6: use_gating 开关
    """
    def __init__(self, input_dim, output_dim, num_nodes, diffusion_steps=2, alpha=0.1, num_supports=2, use_gating=True):
        super().__init__()
        self.k = diffusion_steps
        self.alpha = alpha 
        self.num_supports = num_supports
        self.use_gating = use_gating #Exp 6 
        
        self.weights = nn.Parameter(torch.FloatTensor(num_supports * self.k + 1, input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        
        # 门控参数：只有在 (开启门控) 且 (有自适应图) 时才初始化
        if self.use_gating and num_supports > 2:
            self.gate = nn.Parameter(torch.FloatTensor(1))
            nn.init.constant_(self.gate, 0.5) 
        else:
            self.register_parameter('gate', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x, supports):
        assert len(supports) == self.num_supports
        x0 = x 
        x_list = [x0]
        
        for i, support in enumerate(supports):
            x_k = x0
            for _ in range(self.k):
                diffusion = torch.einsum('nm,bmf->bnf', support, x_k)
                x_k = (1 - self.alpha) * diffusion + self.alpha * x0 
                
                # [Exp 6 Logic] 门控应用
                if self.use_gating and self.gate is not None and i == (self.num_supports - 1):
                    g = torch.sigmoid(self.gate)
                    x_k = x_k * g
                    
                x_list.append(x_k)
        
        xs = torch.stack(x_list, dim=0) 
        out = torch.einsum('kbnf,kfo->bno', xs, self.weights)
        return out + self.bias

class DCGRUCell(nn.Module):
    """ 统一封装的 GRU Cell，支持切换 PageRank/DCRNN """
    def __init__(self, input_dim, hidden_dim, num_nodes, conv_type='pagerank', alpha=0.1, num_supports=2, use_gating=True):
        super().__init__()
        if conv_type == 'pagerank':
            ConvLayer = PageRankDiffusionConv
            kwargs = {'alpha': alpha, 'use_gating': use_gating}
        elif conv_type == 'dcrnn':
            ConvLayer = StandardDiffusionConv
            kwargs = {} # DCRNN 不用 alpha 和 gating
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
            
        self.conv_gate = ConvLayer(input_dim + hidden_dim, hidden_dim * 2, num_nodes, diffusion_steps=2, num_supports=num_supports, **kwargs)
        self.conv_cand = ConvLayer(input_dim + hidden_dim, hidden_dim, num_nodes, diffusion_steps=2, num_supports=num_supports, **kwargs)

    def forward(self, inputs, hidden_state, supports):
        combined = torch.cat([inputs, hidden_state], dim=-1)
        gates = torch.sigmoid(self.conv_gate(combined, supports)) 
        r, u = torch.split(gates, gates.size(-1) // 2, dim=-1)
        combined_r = torch.cat([inputs, r * hidden_state], dim=-1)
        c = torch.tanh(self.conv_cand(combined_r, supports))
        return u * hidden_state + (1.0 - u) * c

# 3. Attention & Embeddings (Exp 3) 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class STTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.temp_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x):
        B, T, N, D = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B*N, T, D) 
        xt = self.norm1(xt + self.temp_attn(xt, xt, xt)[0])
        x = xt.reshape(B, N, T, D).permute(0, 2, 1, 3)
        xs = x.reshape(B*T, N, D) 
        xs = self.norm2(xs + self.spatial_attn(xs, xs, xs)[0])
        x = xs.reshape(B, T, N, D)
        x = self.norm3(x + self.ffn(x))
        return x

# 4. Beta Distribution (Exp 7) 
class BetaProbabilisticHead(nn.Module):
    def __init__(self, d_model, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(d_model, output_dim * 2) 
        self.softplus = nn.Softplus()
        # 初始化 Bias 让初始分布平稳
        with torch.no_grad():
            self.fc.weight.data.normal_(0, 0.001)
            dim = self.fc.bias.shape[0] // 2
            self.fc.bias.data[:dim] = 0.77 
            self.fc.bias.data[dim:] = 0.57 

    def forward(self, x):
        out = self.fc(x)
        raw_alpha, raw_beta = torch.split(out, out.size(-1) // 2, dim=-1)
        alpha = self.softplus(raw_alpha) + 1.01 
        beta = self.softplus(raw_beta) + 1.01
        alpha = torch.clamp(alpha, max=500.0)
        beta = torch.clamp(beta, max=500.0)
        return alpha, beta

class AdvancedBetaPhysicsLoss(nn.Module):
    def __init__(self, adj_mx=None, lambda_phy=0.1, lambda_spatial=0.01, lambda_entropy=0.001, max_speed=100.0):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.lambda_spatial = lambda_spatial
        self.lambda_entropy = lambda_entropy
        self.max_speed = max_speed
        
        if adj_mx is not None:
            if isinstance(adj_mx, list): adj = torch.tensor(adj_mx[0]).float()
            else: adj = torch.tensor(adj_mx).float()
            adj[adj > 0] = 1
            degree = torch.sum(adj, dim=1)
            laplacian = torch.diag(degree) - adj
            self.register_buffer('laplacian', laplacian)
        else:
            self.laplacian = None

    def forward(self, alpha, beta, target, null_val=0.0):
        target_norm = target / self.max_speed
        target_norm = torch.clamp(target_norm, min=1e-4, max=1-1e-4)
        mask = (target > null_val).float()
        
        # NLL
        beta_dist = torch.distributions.Beta(alpha, beta)
        log_prob = beta_dist.log_prob(target_norm)
        loss_nll = -torch.mean(log_prob * mask)
        
        # Physics
        pred_mean = alpha / (alpha + beta) 
        diff_2nd = pred_mean[:, 2:] - 2 * pred_mean[:, 1:-1] + pred_mean[:, :-2]
        loss_time = torch.mean(diff_2nd ** 2)
        
        loss_space = 0.0
        if self.laplacian is not None and self.lambda_spatial > 0:
            B, T, N, _ = pred_mean.shape
            xt = pred_mean.reshape(B*T, N)
            lx = torch.matmul(xt, self.laplacian)
            loss_space = torch.mean(xt * lx)

        entropy = beta_dist.entropy()
        loss_ent = -torch.mean(entropy * mask)

        total_loss = loss_nll + self.lambda_phy * loss_time + \
                     self.lambda_spatial * loss_space + self.lambda_entropy * loss_ent
        return total_loss