import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFace(nn.Module):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
    reference2: <Additive Margin Softmax for Face Verification>
    """

    def __init__(self, feat_dim, num_class, s=30.0, m=0.4):
        super(CosFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce="add")

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)

        return loss, cos_theta


class SphereFace2(nn.Module):
    """reference: <SphereFace2: Binary Classification is All You Need
                for Deep Face Recognition>
    margin='C' -> SphereFace2-C
    margin='A' -> SphereFace2-A
    marign='M' -> SphereFAce2-M
    """

    def __init__(
        self, feat_dim, num_class, magn_type="C", alpha=0.7, r=36.0, m=0.4, t=3.0, lw=50.0
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1.0 - alpha) * (num_class - 1.0))
        if magn_type == "C":
            ay = r * (2.0 * 0.5**t - 1.0 - m)
            ai = r * (2.0 * 0.5**t - 1.0 + m)
        elif magn_type == "A":
            theta_y = min(math.pi, math.pi / 2.0 + m)
            ay = r * (2.0 * ((math.cos(theta_y) + 1.0) / 2.0) ** t - 1.0)
            ai = r * (2.0 * 0.5**t - 1.0)
        elif magn_type == "M":
            theta_y = min(math.pi, m * math.pi / 2.0)
            ay = r * (2.0 * ((math.cos(theta_y) + 1.0) / 2.0) ** t - 1.0)
            ai = r * (2.0 * 0.5**t - 1.0)
        else:
            raise NotImplementedError

        temp = (1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)
        b = math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt(temp))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.0)
        with torch.no_grad():
            if self.magn_type == "C":
                g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
                g_cos_theta = g_cos_theta - self.m * (2.0 * one_hot - 1.0)
            elif self.magn_type == "A":
                theta_m = torch.acos(cos_theta.clamp(-1 + 1e-5, 1.0 - 1e-5))
                theta_m.scatter_(1, y.view(-1, 1), self.m, reduce="add")
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2.0 * ((g_cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
            elif self.magn_type == "M":
                m_theta = torch.acos(cos_theta.clamp(-1 + 1e-5, 1.0 - 1e-5))
                m_theta.scatter_(1, y.view(-1, 1), self.m, reduce="multiply")
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2.0 * ((g_cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1.0 - self.alpha) * (1.0 - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)

        return loss, cos_theta


# This Implementation is taken from https://github.com/kakaoenterprise/BroadFace/blob/main/broadface/loss.py
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=72.0, margin=0.64):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        loss = self.criterion(logit, label)

        return loss, cosine


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, losstype="cosface", eps=1e-7, s=None, m=None):
        """
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['sphereface', 'cosface']
        These losses are described in the following papers:
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super(AngularPenaltySMLoss, self).__init__()
        losstype = losstype.lower()
        assert losstype in ["sphereface", "cosface"]

        if losstype == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if losstype == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.losstype = losstype
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)  # 256 100
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x)

        if self.losstype == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)

        if self.losstype == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(
                        torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps
                    )
                )
            )

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), wf
