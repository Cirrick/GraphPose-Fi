import torchvision
import torch.nn as nn
import torch
from torchvision.transforms import Resize
from .GraFormer import GraFormer, adj_mx_from_edges


class AntTimeAggregator(nn.Module):
    def __init__(self, C=512, A=3, W=4, out_dim=256, mode='attn2', heads=4, dropout=0.1):
        super().__init__()
        self.A, self.W, self.mode = A, W, mode
        self.proj = nn.Conv2d(C, out_dim, kernel_size=1, bias=False) 
        self.ln  = nn.LayerNorm(out_dim)
        if mode == 'attn2':
            self.time_score = nn.Conv3d(1, 1, kernel_size=(1,1,1), bias=True)  
            self.ant_score  = nn.Conv3d(1, 1, kernel_size=(1,1,1), bias=True)
        elif mode == 'mhsa':
            self.qkv = nn.Linear(out_dim, out_dim*3, bias=False)
            self.proj_out = nn.Linear(out_dim, out_dim, bias=False)
            self.heads = heads
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B,C,J,12]
        B, C, J, L = x.shape
        assert L == self.A*self.W, "last dimension should be A*W"
        x = self.proj(x)                  # [B,D,J,L]
        D = x.size(1)

        if self.mode == 'mean':
            z = x.mean(dim=3).permute(0,2,1)        # [B,J,D]
            return self.ln(z)

        x = x.view(B, D, J, self.A, self.W)

        if self.mode == 'attn2':
            w_t = self.time_score(x.mean(dim=1, keepdim=True))   # [B,1,J,A,W]
            w_t = torch.softmax(w_t, dim=-1)
            x_t = (x * w_t).sum(dim=-1)                          # [B,D,J,A]

            w_a = self.ant_score(x_t.mean(dim=1, keepdim=True).unsqueeze(-1))  # [B,1,J,A,1]
            w_a = torch.softmax(w_a.squeeze(-1), dim=-1)                       # [B,1,J,A]
            x_a = (x_t * w_a).sum(dim=-1)                                      # [B,D,J]

            z = x_a.permute(0,2,1)                                             # [B,J,D]
            return self.ln(z)

        elif self.mode == 'mhsa':
            x = x.view(B, D, J, L).permute(0,2,3,1).contiguous()  # [B,J,L,D]
            H = self.heads
            Dh = D // H
            qkv = self.qkv(x)                                     # [B,J,L,3D]
            q,k,v = qkv.chunk(3, dim=-1)
            q = q.view(B*J, L, H, Dh).transpose(1,2)              # [BJ,H,L,Dh]
            k = k.view(B*J, L, H, Dh).transpose(1,2)
            v = v.view(B*J, L, H, Dh).transpose(1,2)
            attn = (q @ k.transpose(-2,-1)) / (Dh**0.5)           # [BJ,H,L,L]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            y = (attn @ v).transpose(1,2).contiguous().view(B*J, L, D)  # [BJ,L,D]
            y = self.proj_out(y)                                         # [BJ,L,D]
            y = y.mean(dim=1).view(B, J, D)                              # pool 12 token
            return self.ln(y)

class JointMLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):  # [B, J, D]
        B, J, D = z.shape
        z = self.net(z)     # [B, J, 3]
        return z

class GraphPoseFiNet(nn.Module):
    def __init__(self, num_keypoints, num_coor, num_person=1, num_ant = 3, dataset='mmfi-csi', pretrained_weights=True, num_layers=4, agg_mode='attn2'):
        super(GraphPoseFiNet, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_coor = num_coor
        self.num_person = num_person
        self.diff = self.num_keypoints*self.num_person - 17
        self.dataset = dataset
        self.num_ant = num_ant
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resnet_f = torchvision.models.resnet34(pretrained=pretrained_weights)
        self.f_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.f_bn1 = resnet_f.bn1
        self.f_relu = resnet_f.relu
        self.f_layer1 = resnet_f.layer1
        self.f_layer2 = resnet_f.layer2
        self.f_layer3 = resnet_f.layer3
        self.f_layer4 = resnet_f.layer4

        self.agg = AntTimeAggregator(C=512, A=3, W=4, out_dim=256, mode=agg_mode)  
        self.gnn_in_dim = 256
        edges_17 = torch.tensor([[0, 1], [1, 2], [2, 3],
                    [0, 4], [4, 5], [5, 6],
                    [0, 7], [7, 8], [8, 9], [9,10],
                    [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        
        adj = adj_mx_from_edges(num_pts=17, edges=edges_17, sparse=False).to(self.device)
        self.graformer = GraFormer(adj=adj, in_feat=self.gnn_in_dim, hid_dim=128, out_dim=3, num_layers=self.num_layers, n_head=4, dropout=0.1, n_pts=17)

        self.bn2 = nn.BatchNorm2d(512)
        
    def _encode_freq(self, x):
        x = self.f_conv1(x)
        x = self.f_bn1(x)
        x = self.f_relu(x)
        x = self.f_layer1(x)
        x = self.f_layer2(x)
        x = self.f_layer3(x)
        x = self.f_layer4(x)
        return x


    def forward(self,x): 

        x_f = x.permute(0, 4, 1, 2, 3).contiguous()  
        new_h = 136 + 8 * self.diff
        resize = Resize([new_h, 32])

        freq_list = []
        for i in range(self.num_ant):
            xb = x_f[:, :, i, :, :]     
            xb = resize(xb)          
            xb = self._encode_freq(xb)
            freq_list.append(xb)
        x = torch.cat(freq_list, dim=3)  
        x = self.bn2(x)
        z = self.agg(x)         # [B,17,256]
        pred = self.graformer(z)  # [B,17,3]

        return pred, z

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.rand(32, 3, 128, 10, 2)  # Adjusted size to match input
    data = data[..., 0].unsqueeze(-1) 
    data = data.to(device)
    model = GraphPoseFiNet(num_keypoints=17, num_coor=3, num_person=1).to(device)
    model.apply(_weights_init)
    pose, _ = model(data)
    print(pose.shape)  # Expected: (32, 17, 3)
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}')
