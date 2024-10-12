import torch.nn as nn
import torch
import numpy as np

# from network.knn_search.knn_module import KNN
# from network.ops import get_knn_feats, spectral_smooth, trans, compute_smooth_motion_diff

data_fetech = {
    'node_feats':[],
    'K': [],
    'lamba': [],
    'recovered graph': [],
}

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)topk是自带函数
    return idx

def get_graph_feature(x, k=10, idx=None):
    batch_size = x.size(0)
    num_pts = x.size(2)
    x = x.view(batch_size, -1, num_pts) #change
    if idx is None:
        idx_out = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        idx_out = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_pts #change

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_pts, -1)[idx, :]
    feature = feature.view(batch_size, num_pts, k, num_dims) #change

    x = x.view(batch_size, num_pts, 1, num_dims).repeat(1, 1, k, 1) #change
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class local_transformer_agg(nn.Module):
    def __init__(self,in_channel,out_channels=None):
        nn.Module.__init__(self)
        self.att1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attq1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attk1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.ones(1))

    def forward(self, x_row, x_local):
        # 局部attention
        x_local = self.att1(x_local)
        q = self.attq1(x_local)
        k = self.attk1(x_local)
        v = self.attv1(x_local)

        att = torch.mul(q, k)
        att = torch.softmax(att, dim=3)
        qv = torch.mul(att, v)
        out_local = torch.sum(qv, dim=3).unsqueeze(3)
        out = x_row + self.gamma1 * out_local #x_row只有在这里有用

        return out


class transformer(nn.Module):
    def __init__(self, in_channel):
        nn.Module.__init__(self)

        self.attq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_row, x_local):
        q1 = self.attq(x_local).squeeze(3)
        k1 = self.attk(x_local).squeeze(3)
        v1 = self.attv(x_local).squeeze(3)
        scores = torch.bmm(q1.transpose(1, 2), k1)
        att = torch.softmax(scores, dim=2)
        out = torch.bmm(v1, att.transpose(1, 2))
        out = out.unsqueeze(3)
        out = self.conv(out)
        out = x_row + self.gamma * out
        return out

def GCN_Spectral(feats, eig_vec, eig_val, pre_defined_spectrum, learnable_spectrum_matrix):
    """
    :param feats:    b,f,n,1
    :param eig_vec:  b,n,eig_dim
    :param eig_val:  b,eig_dim
    :param eta:      float
    :return:         b,f,n,1
    """
    feats_proj = torch.bmm(feats[...,0],eig_vec) # b,f,eig_dim
    if pre_defined_spectrum == None:
        feats_norm = feats_proj * learnable_spectrum_matrix # b,f,eig_dim
        data_fetech['K'].append(learnable_spectrum_matrix)
        data_fetech['lamba'].append(eig_val)
    else:
        feats_norm = feats_proj / (1+eig_val.unsqueeze(1)) # b,f,eig_dim
    feats_smooth = torch.bmm(feats_norm,eig_vec.permute(0,2,1)) # b,f,n

    # 计算recovered graph
    if pre_defined_spectrum == None:
        spectrum_diag_matrix = torch.diag(learnable_spectrum_matrix.squeeze(0).squeeze(0)).repeat(eig_vec.shape[0], 1, 1)  # 维度为 eig_dim x eig_dim
        temp = torch.bmm(eig_vec, spectrum_diag_matrix)
        recovered_graph = torch.bmm(temp, eig_vec.permute(0,2,1))
        data_fetech['recovered graph'].append(recovered_graph)
    else:
        pass

    return feats_smooth.unsqueeze(3)


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels

        self.use_short_cut=use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out


class AFAMLayer(nn.Module):
    def __init__(self,feats_dim,use_bn=True):
        super().__init__()
        if use_bn:
            self.conv=nn.Sequential(
                nn.InstanceNorm2d(feats_dim),
                nn.BatchNorm2d(feats_dim),
                nn.ReLU(True),
                nn.Conv2d(feats_dim,feats_dim,1,1),
                nn.InstanceNorm2d(feats_dim),
                nn.BatchNorm2d(feats_dim),
                nn.ReLU(True),
                nn.Conv2d(feats_dim,feats_dim,1,1),
            )
        else:
            self.conv=nn.Sequential(
                nn.InstanceNorm2d(feats_dim),
                nn.ReLU(),
                nn.Conv2d(feats_dim,feats_dim,1,1),
                nn.InstanceNorm2d(feats_dim),
                nn.ReLU(),
                nn.Conv2d(feats_dim,feats_dim,1,1),
            )

        self.fc = nn.Conv2d(2*feats_dim,1,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Correlation_feats, Coherence_residual_feats, feats, pre_defined_alpha):
        if pre_defined_alpha == None:
            # Attention
            alpha = self.sigmoid(self.fc(torch.cat((Correlation_feats, Coherence_residual_feats), 1)))
            aggregated_feature = Correlation_feats - alpha * feats
        else:
            aggregated_feature = Correlation_feats - feats

        # 聚合特征提取
        aggregated_feats=self.conv(aggregated_feature)
        return aggregated_feats


class SFLBlock(nn.Module):
    def __init__(self, in_dim, knn_dim):
        super().__init__()
        # 基础配置
        self.k = knn_dim
        self.learnable_spectrum_matrix = nn.Parameter(torch.randn(1, 1, 32) * 1)  # 初始化矩阵，并使用小的随机值进行初始化

        # block 里边的组成
        #1.Motion Feature Extraction Module
        self.cn0=PointCN(in_dim, in_dim)
        self.tf0 = local_transformer_agg(2*in_dim, in_dim)
        self.tf1 = transformer(in_dim)
        self.cn1=PointCN(in_dim)
        #2.Learnable Graph Structure
        # self.lgs=LGSLayer(eta, in_dim, eta_learnable)
        #3.Attention-Based Adaptive Feature Aggregation Module
        self.afam=AFAMLayer(in_dim)
        self.cn2=PointCN(in_dim)

    def forward(self, feats, eig_vec, eig_val, setting):
        # 除了cn0 cn1 ....不加残差以外，其他都加残差
        # 1 第一部分
        # 局部特征抽取
        local_feature_select = get_graph_feature(feats, k=self.k)
        local_feature = self.tf0(feats, local_feature_select)
        feats=self.cn0(local_feature)
        # 全局特征抽取
        global_feature = self.tf1(feats, feats)
        feats=self.cn1(global_feature)
        # print("第一部分结束", feats.shape)
        # 2 第二部分
        Correlation_feats=GCN_Spectral(feats,eig_vec,eig_val, setting['spectrum'], self.learnable_spectrum_matrix)
        Coherence_residual_feats = Correlation_feats - feats
        # print("第二部分结束", Correlation_feats.shape, Coherence_residual_feats.shape)
        # 3 第三部分
        aggregated_feats = self.afam(Correlation_feats, Coherence_residual_feats, feats, setting['alpha'])
        feats=self.cn2(aggregated_feats)
        # print("第三部分结束", feats.shape)

        return feats

class SFLNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #基础配置
        self.knn_dim=cfg['knn_num']

        # 前两个block用预定义的alpha常数和固定图结构
        self.pre_block_setting = {
            'alpha': 1,
            'spectrum': 1
        }
        # 后面的用可学习的alpha和图结构
        self.post_block_setting = {
            'alpha': None,
            'spectrum': None
        }

        # 网络架构3部分组成
        self.geom_feats_embed=nn.Sequential(
            nn.Conv2d(4,128,1),
            PointCN(128)
        )

        self.sflblock_list=nn.ModuleList()
        for k in range(4):
            self.sflblock_list.append(SFLBlock(128, self.knn_dim))

        self.dimensional_reduction_block = nn.Conv2d(128,1,1)

        self.prob_predictor=nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,1,1),
        )

    def forward(self, data):
        # torch.Size([32, 1, 2000, 4]) torch.Size([32, 2000, 32]) torch.Size([32, 32])
        xs = data['xs'].squeeze(1) # b,n,4
        eig_vec = data['eig_vec'] # b,n,eig_dim
        eig_val = data['eig_val'] # b,eig_dim
        xs = xs.permute(0,2,1).unsqueeze(3) # b,4,n,1

        # 字典清空
        data_fetech['node_feats'].clear()
        data_fetech['K'].clear()
        data_fetech['lamba'].clear()
        data_fetech['recovered graph'].clear()

        corr_feats = self.geom_feats_embed(xs)

        for index, net in enumerate(self.sflblock_list):
            if index < 2:
               corr_feats = net(corr_feats, eig_vec, eig_val, self.pre_block_setting)
            else:
               corr_feats = net(corr_feats, eig_vec, eig_val, self.post_block_setting)
               # 节省计算量 不要用特征向量算
               data_fetech['node_feats'].append(self.dimensional_reduction_block(corr_feats))

        logits=self.prob_predictor(corr_feats) # b,1,n,1
        # print("取回的数据:", data_fetech.keys())

        xs = data['xs'].permute(0, 3, 2, 1)
        logits = logits.squeeze(1).squeeze(2)
        e_hat = weighted_8points(xs, logits)
        return {'logits':logits, 'data_fetech':data_fetech, 'e_hat': e_hat}
    
    
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), 'L')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    # x = x.permute(2,0,1) # 理解为第零维度用原始第二维度填充，第一维度用原始第零维度填充，第二维度用原始第一维度填充

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

    
    
    
