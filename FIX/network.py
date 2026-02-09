import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RevIN, DCGRUCell, PositionalEncoding, STTransformerBlock, BetaProbabilisticHead


class PeriodicTemporalEmbedding(nn.Module):
    def __init__(self, d_model, day_len=288): 
        super().__init__()
        self.day_len = day_len
        self.day_emb = nn.Embedding(day_len, d_model)
    def forward(self, x_time_norm):
        time_idx = (x_time_norm * self.day_len).long().clamp(0, self.day_len - 1)
        return self.day_emb(time_idx) 

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, self.padding), dilation=(1, dilation))
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0: x = x[..., :-self.padding]
        return x

# Decoder 变体 (Exp 0, 4, 5) 
class PatchCausalConvDecoder(nn.Module):
    """ [Baseline] Patch + Causal Conv """
    def __init__(self, d_model, n_layers=3, kernel_size=3, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(d_model * patch_size, d_model)
        self.layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(CausalConv2d(d_model, d_model, kernel_size=kernel_size, dilation=dilation))
            self.skip_convs.append(nn.Conv2d(d_model, d_model, 1))
            self.bn.append(nn.BatchNorm2d(d_model))
        self.end_conv_1 = nn.Conv2d(d_model, d_model, 1)
        self.end_conv_2 = nn.Conv2d(d_model, d_model * patch_size, 1) 
        
    def forward(self, x):
        B, T, N, D = x.shape
        pad_len = (self.patch_size - (T % self.patch_size)) % self.patch_size
        T_new = T + pad_len
        if pad_len > 0: x = F.pad(x, (0,0,0,0,0,pad_len))
        
        # Patching
        x = x.view(B, T_new // self.patch_size, self.patch_size, N, D)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, T_new // self.patch_size, N, D * self.patch_size)
        x = self.patch_proj(x).permute(0, 3, 2, 1) 
        
        skip_sum = 0
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = self.bn[i](x)
            x = F.gelu(x)
            s = self.skip_convs[i](x)
            try: skip_sum = skip_sum + s
            except: skip_sum = s
            x = x + residual
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) 
        
        # Unpatching
        x = x.permute(0, 3, 2, 1).view(B, T_new // self.patch_size, N, self.patch_size, -1)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, T_new, N, -1)
        if pad_len > 0: x = x[:, :T, :, :]
        return x # Output: Features (B, T, N, D)

class CausalConvDecoder(nn.Module):
    """ [Exp 4] No Patch """
    def __init__(self, d_model, n_layers=3, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(CausalConv2d(d_model, d_model, kernel_size=kernel_size, dilation=dilation))
            self.skip_convs.append(nn.Conv2d(d_model, d_model, 1))
            self.bn.append(nn.BatchNorm2d(d_model))
        self.end_conv_1 = nn.Conv2d(d_model, d_model, 1)
        self.end_conv_2 = nn.Conv2d(d_model, d_model, 1)
    def forward(self, x):
        x = x.permute(0, 3, 2, 1) 
        skip_sum = 0
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = self.bn[i](x)
            x = F.gelu(x)
            s = self.skip_convs[i](x)
            try: skip_sum = skip_sum + s
            except: skip_sum = s
            x = x + residual
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) 
        return x.permute(0, 3, 2, 1)

class TransformerDecoder(nn.Module):
    """ [Exp 5] Transformer """
    def __init__(self, d_model, nhead=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        
    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B*N, T, D)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.layers(x_flat, mask=mask)
        return out.reshape(B, N, T, -1).permute(0, 2, 1, 3)

#  主模型 
class Ultimate_Graphormer(nn.Module):
    def __init__(self, cfg, adj_mx=None):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model
        
        # 配置参数提取
        self.use_revin = getattr(cfg, 'use_revin', True)
        self.use_st_attn = getattr(cfg, 'use_st_attn', True)
        self.encoder_type = getattr(cfg, 'encoder_type', 'pagerank')
        self.use_gating = getattr(cfg, 'use_gating', True)
        self.decoder_type = getattr(cfg, 'decoder_type', 'patch')
        self.output_type = getattr(cfg, 'output_type', 'point')
        
        # 1. RevIN
        if self.use_revin:
            self.revin = RevIN(self.num_nodes)
        else:
            self.revin = nn.Identity()
            
        # 2. Embedding
        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.d_model))
        self.flow_proj = nn.Linear(1, self.d_model)
        self.time_emb = PeriodicTemporalEmbedding(self.d_model)
        self.input_fusion = nn.Sequential(nn.Linear(self.d_model * 3, self.d_model), nn.ReLU(), nn.Dropout(cfg.dropout))

        # 3. ST-Attention
        if self.use_st_attn:
            self.st_attention = STTransformerBlock(self.d_model, nhead=cfg.nhead, dropout=cfg.dropout)
        
        # 4. Graph Construction
        total_supports = 0
        if adj_mx is not None:
            self.register_buffer('supports', torch.stack([torch.tensor(m).float() for m in adj_mx]))
            total_supports += 2
        else:
            self.register_buffer('supports', torch.empty(0))
        
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 40))
        self.nodevec2 = nn.Parameter(torch.randn(40, self.num_nodes))
        nn.init.xavier_normal_(self.nodevec1)
        nn.init.xavier_normal_(self.nodevec2)
        total_supports += 1

        # 5. Encoder
        self.encoder_layer1 = DCGRUCell(self.d_model, self.d_model, self.num_nodes, conv_type=self.encoder_type, alpha=0.1, num_supports=total_supports, use_gating=self.use_gating)
        self.encoder_layer2 = DCGRUCell(self.d_model, self.d_model, self.num_nodes, conv_type=self.encoder_type, alpha=0.1, num_supports=total_supports, use_gating=self.use_gating)
        self.encoder_layer3 = DCGRUCell(self.d_model, self.d_model, self.num_nodes, conv_type=self.encoder_type, alpha=0.1, num_supports=total_supports, use_gating=self.use_gating)
        self.pos_encoder = PositionalEncoding(self.d_model, cfg.dropout)

        # 6. Decoder Selection
        if self.decoder_type == 'patch':
            self.decoder = PatchCausalConvDecoder(self.d_model, n_layers=3, kernel_size=3, patch_size=2)
        elif self.decoder_type == 'causal':
            self.decoder = CausalConvDecoder(self.d_model, n_layers=3, kernel_size=3)
        elif self.decoder_type == 'transformer':
            self.decoder = TransformerDecoder(self.d_model, nhead=cfg.nhead)
        else:
            raise ValueError(f"Unknown decoder: {self.decoder_type}")
        
        # 7. Output Head
        if self.output_type == 'probabilistic':
            self.head = BetaProbabilisticHead(self.d_model, cfg.output_dim)
        else:
            self.head = nn.Linear(self.d_model, cfg.output_dim)
        
    def forward(self, x):
        B, T, N, C = x.shape
        
        # Norm
        if self.use_revin:
            x_flow_norm = self.revin(x[..., 0], 'norm').unsqueeze(-1)
        else:
            x_flow_norm = x[..., 0].unsqueeze(-1)
            
        # Embed
        emb_flow = self.flow_proj(x_flow_norm)
        emb_time = self.time_emb(x[..., 1]) if C > 1 else 0
        emb_node = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        x_emb = self.input_fusion(torch.cat([emb_flow, emb_time, emb_node], dim=-1))
        
        # ST-Attn
        if self.use_st_attn:
            x_emb = self.st_attention(x_emb)
        
        # Graph
        supports = [self.supports[i] for i in range(len(self.supports))] if len(self.supports) > 0 else []
        adp_mx = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        supports.append(adp_mx)
        
        # Encoder
        h1 = torch.zeros(B, N, self.d_model, device=x.device)
        h2 = torch.zeros(B, N, self.d_model, device=x.device)
        h3 = torch.zeros(B, N, self.d_model, device=x.device)
        enc_outputs = []
        for t in range(self.seq_len):
            inp = x_emb[:, t]
            h1 = self.encoder_layer1(inp, h1, supports)
            h2 = self.encoder_layer2(h1, h2, supports)
            h3 = self.encoder_layer3(h2, h3, supports)
            enc_outputs.append(h3)
        
        enc_output = torch.stack(enc_outputs, dim=1) + self.pos_encoder.pe[:T].unsqueeze(0).unsqueeze(2)
        
        # Decoder (Returns features)
        dec_feat = self.decoder(enc_output)
        
        # Head&Denorm
        if self.output_type == 'probabilistic':
            return self.head(dec_feat) # alpha, beta
        else:
            pred = self.head(dec_feat)
            if self.use_revin:
                return self.revin(pred.squeeze(-1), 'denorm').unsqueeze(-1)
            return pred

Ablation_Graphormer = Ultimate_Graphormer