import torch
from utils import torch_skew_symmetric
import numpy as np

def batch_episym(x1, x2, F):
    # print("特征点", x1.shape, x2.shape)
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys


class SpatialFrequencyLoss(object):
    def __init__(self):
        self.quadratic_loss_weight = 1.02
        self.topology_loss_weight = 0.05
        self.post_block_number = 2

    def run(self, data_pr):
        loss1 = []
        loss2 = []
        spatial_frequency_loss = 0

        for i in range(self.post_block_number):
            K = data_pr['K'][i].reshape(1, -1)
            lamba = data_pr['lamba'][i]
            recovered_graph = data_pr['recovered graph'][i]
            # print(data_pr['node_feats'][i].shape, data_pr['K'][i].shape, data_pr['lamba'][i].shape, data_pr['recovered graph'][i].shape)

            # 高频惩罚损失
            temp = torch.mm(K, lamba)
            quadratic_loss = torch.mm(temp, K.T)
            # print("loss", quadratic_loss)
            loss1.append(quadratic_loss.squeeze(0))

            # 计算两个结构之间的损失
            # 1. 计算recover_graph
            # print("graph", recovered_graph)

            # 2. 计算traget_graph
            with torch.no_grad():
                B, _, N, _ = data_pr['node_feats'][i].size()
                w = data_pr['node_feats'][i].squeeze(1).squeeze(2)
                w = w.unsqueeze(-1)
                # 用归一化的图会不稳定
                if 1:
                    A = torch.bmm(w, w.transpose(1, 2))
                    D_out = torch.sum(A, dim=-1)
                    D = torch.diag_embed(D_out)
                    traget_graph = D - A

            # print("结构", traget_graph.shape, recovered_graph.shape)

            # 计算差
            difference = traget_graph - recovered_graph
            # 计算模长（L2范数）
            norm = torch.norm(difference, p=2, dim=(1, 2), keepdim=True)
            # 归一化
            normalized_difference = difference / norm
            # 如果需要计算归一化后的和
            topology_loss = normalized_difference.sum()
            loss2.append(torch.tensor([topology_loss]))

        normalization_quadratic_loss = torch.sum(torch.cat(loss1) / torch.norm(torch.cat(loss1), p='fro'))
        normalization_topology_loss = torch.sum(torch.cat(loss2) / torch.norm(torch.cat(loss2), p='fro'))

        spatial_frequency_loss += normalization_quadratic_loss * self.quadratic_loss_weight
        spatial_frequency_loss += normalization_topology_loss * self.topology_loss_weight

        return spatial_frequency_loss, normalization_quadratic_loss * self.quadratic_loss_weight, normalization_topology_loss * self.topology_loss_weight


class MatchLoss(object):
  def __init__(self, config):
    self.loss_essential = config.loss_essential
    self.loss_classif = config.loss_classif
    self.use_fundamental = config.use_fundamental
    self.obj_geod_th = config.obj_geod_th
    self.geo_loss_margin = config.geo_loss_margin
    self.loss_essential_init_iter = config.loss_essential_init_iter

  def run(self, global_step, data, logits, e_hat):
    R_in, t_in, y_in, pts_virt = data['Rs'], data['ts'], data['ys'], data['virtPts']
    # print("pts", pts_virt.shape)

    # Get groundtruth Essential matrix
    e_gt_unnorm = torch.reshape(torch.matmul(
        torch.reshape(torch_skew_symmetric(t_in), (-1, 3, 3)),
        torch.reshape(R_in, (-1, 3, 3))
    ), (-1, 9))

    e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)

    ess_hat = e_hat
    if self.use_fundamental:
        ess_hat = torch.matmul(torch.matmul(data['T2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['T1s'])
        # get essential matrix from fundamental matrix
        ess_hat = torch.matmul(torch.matmul(data['K2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['K1s']).reshape(-1,9)
        ess_hat = ess_hat / torch.norm(ess_hat, dim=1, keepdim=True)

    # Essential/Fundamental matrix loss
    pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:,:,2:]
    
    geod = batch_episym(pts1_virts, pts2_virts, e_hat)
    # print("geod", geod)
    essential_loss = torch.min(geod, self.geo_loss_margin*geod.new_ones(geod.shape))
    essential_loss = essential_loss.mean()
    
    # essential_loss = 0
    # print(essential_loss)

    # Classification loss
    # The groundtruth epi sqr
    gt_geod_d = y_in[:, :, 0]
    is_pos = (gt_geod_d < self.obj_geod_th).type(logits.type())
    is_neg = (gt_geod_d >= self.obj_geod_th).type(logits.type())
    c = is_pos - is_neg
    # print("AAA", c.shape, logits.shape, logits)
    classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
    # balance
    num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
    num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
    # print("classif_losses", classif_losses.shape)
    classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
    # print("classif_loss_p", classif_loss_p.shape)
    classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
    classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)

    #精确度和召回率
    precision = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum((logits > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
    )
    recall = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )

    # print("loss", classif_loss, essential_loss)
    loss = 0
    # Check global_step and add essential loss
    if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
        loss += self.loss_essential * essential_loss 
    if self.loss_classif > 0:
        loss += self.loss_classif * classif_loss

    # (self.loss_essential * essential_loss).item()
    return [loss, (self.loss_essential * essential_loss).item(), (self.loss_classif * classif_loss).item(), precision.item(), recall.item()]
