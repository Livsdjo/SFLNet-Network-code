import torch
import numpy as np
from network.ops import weighted_8points, batch_epipolar_distance

class Loss:
    def __init__(self):
        self.keys=[]

    def __call__(self, data_pr, data_gt, step):
        pass

def compute_precision_recall_torch(labels_pr,labels_gt):
    with torch.no_grad():
        tp=torch.sum((labels_pr & labels_gt).float(),1)     # b
        fp=torch.sum(((~labels_gt) & labels_pr).float(),1)  # b
        fn=torch.sum(((~labels_pr) & labels_gt).float(),1)  # b
        precision=(tp+1)/(tp+fp+1)
        recall=(tp+1)/(tp+fn+1)
        f1=(2*precision*recall)/(precision+recall)
    return precision, recall, f1

def compute_precision_recall_torch_background(labels_pr,labels_gt,background_mask):
    with torch.no_grad():
        foreground_mask = 1.0 - background_mask.float()
        tp=torch.sum((labels_pr & labels_gt).float()*foreground_mask,1)     # b
        fp=torch.sum(((~labels_gt) & labels_pr).float()*foreground_mask,1)  # b
        fn=torch.sum(((~labels_pr) & labels_gt).float()*foreground_mask,1)  # b
        precision=(tp+1)/(tp+fp+1)
        recall=(tp+1)/(tp+fn+1)
        f1=(2*precision*recall)/(precision+recall)
    return precision, recall, f1

class ClassificationLoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.output_keys=['logits']
        self.keys=['loss_cls_logits','f1','precision','recall']

    def __call__(self, data_pr, data_gt, step):
        results={}
        for key in self.output_keys:
            logits=data_pr[key]
            ys=data_gt['ys'].to(logits.device)
            is_pos = ys.float()
            is_neg = (ys==0).float()
            c = is_pos - is_neg
            classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
            # balance
            num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
            num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
            classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
            classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
            classif_loss = classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg
            results[f'loss_cls_{key}'] = classif_loss

            labels_pr=(logits>0).bool()
            labels_gt=ys.bool()
            precision, recall, f1 = compute_precision_recall_torch(labels_pr, labels_gt)
            results['f1']=f1
            results['recall']=recall
            results['precision']=precision

        return results

class NonRigidClassificationLoss(Loss):
    def __init__(self, cfg):
        super().__init__()
        self.keys=['loss_cls_logits','f1','precision','recall']

    def __call__(self, data_pr, data_gt, step):
        results={}
        logits=data_pr['logits']
        ys=data_gt['ys'].to(logits.device).int()
        is_pos = (ys==1).float()
        is_neg = (ys==-1).float()
        c = is_pos - is_neg
        classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
        # balance
        num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
        num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
        classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
        classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
        classif_loss = classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg
        results[f'loss_cls_logits'] = classif_loss

        labels_pr=(logits>0).bool()
        labels_gt=is_pos.bool()
        background_mask = (ys==0).bool()
        precision, recall, f1 = compute_precision_recall_torch_background(labels_pr, labels_gt, background_mask)
        results['f1']=f1
        results['precision']=precision
        results['recall']=recall
        return results

class GeometricLoss(Loss):
    def __init__(self,cfg):
        self.max_geom_loss=cfg['geom_loss_max']
        self.eps=1e-5
        self.apply_step=cfg['geom_loss_step']
        self.geom_loss_ratio=cfg['geom_loss_ratio']
        self.keys=['loss_geom']

    def __call__(self, data_pr, data_gt, step):
        logits=data_pr['logits'] # b,n
        xs=data_gt['xs'].to(logits.device) # b,n,4
        vxs=data_gt['vxs'].to(logits.device) # b,n,4

        try:
            E_pr=weighted_8points(xs.unsqueeze(1),logits) # b,3,3
        except RuntimeError:
            b=xs.shape[0]
            E_pr=np.repeat(np.identity(3)[None,:,:],b,0)
            E_pr=torch.from_numpy(E_pr).cuda()

        dist=batch_epipolar_distance(vxs[:,:,:2],vxs[:,:,2:],E_pr,1e-5)
        dist=torch.clamp_max(dist,max=self.max_geom_loss)
        loss=torch.mean(dist,1)
        if step<self.apply_step:
            loss=loss.detach()
        return {'loss_geom': loss*self.geom_loss_ratio}

class SpatialFrequencyLoss(Loss):
    def __init__(self,cfg):
        self.quadratic_loss_weight = 5
        self.topology_loss_weight = 0.5
        self.post_block_number = 2

    def __call__(self, data_pr):
        loss1 = []

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

            # 结构恢复损失
            # 1. 计算recover_graph
            # print("graph", recovered_graph)

            # 2. 计算traget_graph
            with torch.no_grad():
                B, _, N, _ = data_pr['node_feats'][i].size()
                w = data_pr['node_feats'][i].squeeze(1).squeeze(2)
                # w = torch.relu(torch.tanh(w)).unsqueeze(-1)
                # w = torch.tanh(w).unsqueeze(-1)
                w = w.unsqueeze(-1)
                A = torch.bmm(w, w.transpose(1, 2))
                I = torch.eye(N).unsqueeze(0).to(data_pr['node_feats'][i].device).detach()
                A = A + I
                # print("未归一化", A)
                D_out = torch.sum(A, dim=-1)
                D = (1 / D_out) ** 0.5
                D = torch.diag_embed(D)
                L = torch.bmm(D, A)
                traget_graph = torch.bmm(L, D)
                # print(traget_graph)

            # 计算两个结构之间的损失 目前有问题

        normalization_quadratic_loss = torch.sum(torch.cat(loss1) / torch.norm(torch.cat(loss1), p='fro'))
        return {'loss_quadratic': normalization_quadratic_loss*self.geom_loss_ratio}    
    
    
name2loss={
    'geom': GeometricLoss,
    'cls': ClassificationLoss,
    'nr_cls': NonRigidClassificationLoss,
    'sfl': SpatialFrequencyLoss,
}