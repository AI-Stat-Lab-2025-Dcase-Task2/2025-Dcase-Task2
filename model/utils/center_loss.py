import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        embedding_dim (int): feature dimension.
    """
    def __init__(self, num_classes, embedding_dim=512):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, embedding_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # centers dtype이 x와 다를 경우 자동 변환
        centers = self.centers.to(dtype=x.dtype, device=x.device)

        B = x.size(0)
        # 변환된 centers 변수를 사용하여 distmat 계산
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(B, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, B).t()

        # addmm 연산 전에 distmat의 dtype을 x의 dtype과 일치시킴킴
        distmat = distmat.to(x.dtype)

        # 변환된 centers 변수를 사용
        distmat.addmm_(1, -2, x, centers.t())

        classes = torch.arange(self.num_classes).long().to(labels.device) # classes도 labels과 동일한 device로 이동
        labels = labels.unsqueeze(1).expand(B, self.num_classes)
        mask = labels.eq(classes.expand(B, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / B

        return loss