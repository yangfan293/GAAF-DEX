import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *
from sklearn.cluster import KMeans
import cv2
import mediapipe as mp
import numpy as np
from torchvision import transforms
from PIL import Image
from mediapipe.python.solutions.drawing_utils import DrawingSpec

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Net(nn.Module):

    def __init__(self, aff_classes=6):
        super(Net, self).__init__()

        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---------yf-hand-----------

        self.gap_hand = nn.AdaptiveAvgPool2d(1)

        # --- hyper-parameters --- #
        self.aff_cam_thd = 0.6
        self.part_iou_thd = 0.6
        self.cel_margin = 0.5

        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        # --- learning parameters --- #
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)
        self.func_aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)
        self.hand_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.func_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc_hand = nn.Conv2d(self.vit_feat_dim, 14, 1)  # yf_hand
        self.fc_classification = nn.Linear(384*28*28, 14)

        self.softmax = nn.Softmax(dim=1)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=5,
            min_detection_confidence=0.01)

        # 初始化mean和std
        self.mean = torch.tensor([0.592, 0.558, 0.523]).view(-1, 1, 1).cuda()
        self.std = torch.tensor([0.228, 0.223, 0.229]).view(-1, 1, 1).cuda()

    def forward(self, exo, ego, aff_label, epoch):
        num_exo = exo.shape[1]
        exo0 = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224
        # --- Extract deep descriptors from DINO-vit --- #
        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # F_exo
            _, exo_key, exo_attn = self.vit_model.get_last_key(exo0)
            ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()
            exo_desc = exo_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()

        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        exo_proj = exo_desc[:, 1:] + self.aff_proj(exo_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w)
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
        head_idxs = [0, 1, 3]
        ego_sam = ego_cls_attn[:, head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1)

        # --- Affordance CAM generation --- #
        exo_proj = self.aff_exo_proj(exo_proj)
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x 36 x h x w 特定类别的热力图，通过最后一个卷积层输出的特征与特征的权重相乘得到，获得exo_aff_cam
        aff_exo_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes)
        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w)

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, aff_label[b_]]  # gt_aff_cam 被填充为对应于这些特定类别的 CAM

        # --- Clustering extracted descriptors based on CAM --- #
        ego_desc_flat = ego_desc.flatten(-2, -1)  # b x 384 x hw
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()
        part_score = torch.zeros(b, self.cluster_num).cuda()
        part_proto = torch.zeros(b, c).cuda()
        part_func_score = torch.zeros(b).cuda()
        part_func_proto = torch.zeros(b, c).cuda()
        sim_func_maps = torch.zeros(b, 1, h * w).cuda()
        exo_funcparts_sim_maps = torch.zeros(b, num_exo, h * w).cuda()
        for b_ in range(b):
            exo_aff_desc = []
            exo_funcparts = []

            # ----------circle----------#
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)  # Pc
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10)
                tmp_desc = exo_desc_re_flat[b_, n]  # Fexo
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T  # n x c
                exo_aff_desc.append(tmp_top_desc)  # 每一张的fexo

                rois_coordinates = self.handaround_feature_extact(exo[b_, n])  # yf
                if rois_coordinates:
                    exo_funcparts = []  # 存储当前exo视角下所有ROI的特征
                    for coordinates in rois_coordinates:
                        circle_x, circle_y, radius = coordinates

                        # 上采样特征图至224x224
                        tmp_desc_upsampled = F.interpolate(tmp_desc.reshape(c, 28, 28).unsqueeze(0), size=(448, 448),
                                                           mode='bilinear',
                                                           align_corners=True).squeeze(0)
                        # 使用numpy创建圆形掩码
                        Y, X = np.ogrid[:448, :448]
                        dist_from_center = np.sqrt((X - circle_x) ** 2 + (Y - circle_y) ** 2)
                        mask = (dist_from_center <= radius).astype(np.float32)

                        # 将numpy数组转换为torch.Tensor
                        mask_tensor = torch.from_numpy(mask).to(tmp_desc_upsampled.device)

                        # 应用掩码并提取特征
                        masked_features = tmp_desc_upsampled * mask_tensor
                        total_area = mask_tensor.sum() + 1e-10  # 防止除以零
                        extracted_features = masked_features.sum(dim=(1, 2)) / total_area

                        exo_funcparts.append(extracted_features.unsqueeze(0))  # 添加特征至列表
            # ----------circle----------#
                if exo_funcparts:
                    # 将所有特征堆叠在一起并进行平均池化
                    exo_funcparts_tensor = torch.cat(exo_funcparts, dim=0).mean(dim=0, keepdim=True).cuda()
                else:
                    exo_funcparts_tensor = torch.rand(1, 384).cuda() * 0.01
            exo_aff_desc_sum = torch.cat(exo_aff_desc, dim=0)  #  3张一起的fexo, (n1 + n2 + n3) x c
            if exo_aff_desc_sum.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, max_iter=300)
            kmeans.fit_predict(exo_aff_desc_sum.contiguous().cpu().numpy())
            clu_cens = F.normalize(torch.from_numpy(kmeans.cluster_centers_), dim=1)   # prototypes

            # save the exocentric similarity maps for visualization in training
            for n_ in range(num_exo):
                exo_desc_re_flat0 = exo_desc_re_flat[b_, n_]
                exo_desc_re_flat0 = exo_desc_re_flat0.to(clu_cens.device)
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat0, dim=0))
                exo_desc_re_flat0 = exo_desc_re_flat0.to(exo_funcparts_tensor.device)
                exo_funcparts_sim_maps[b_, n_] = torch.mm(exo_funcparts_tensor, exo_desc_re_flat0)

                # --- 计算相似度 ---
            ego_desc_flat0 = ego_desc_flat[b_].to(clu_cens.device)
            sam_hard0 = ego_sam_flat[b_].to(clu_cens.device)

            sim_map = torch.mm(clu_cens,
                               F.normalize(ego_desc_flat0, dim=0))  # self.cluster_num x hw，egocentric similarity maps

            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], \
                torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)


            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, dim=0, keepdim=True)).float()
            sam_hard = (sam_hard0 > torch.mean(sam_hard0, 0, keepdim=True)).float()  # saliency map

            inter = (sim_map_hard * sam_hard).sum(1)
            union = sim_map_hard.sum(1) + sam_hard.sum() - inter
            p_score = (inter / sim_map_hard.sum(1) + sam_hard.sum() / union) / 2
            sim_maps[b_] = sim_map
            part_score[b_] = p_score
            if p_score.max() < self.part_iou_thd:
                continue
            part_proto[b_] = clu_cens[torch.argmax(p_score)]  # exo_part_proto

            # ----------- find functioanl parts--------------#
            # ego_funcparts_sim_maps = torch.mm(exo_funcparts_tensor, F.normalize(ego_desc_flat[b_], dim=0))

            part_func_proto[b_] = exo_funcparts_tensor.unsqueeze(0)  # exo_func_part_proto

        # sim_maps = sim_maps.reshape(b, self.cluster_num, h, w)
        exo_sim_maps = exo_sim_maps.reshape(b, num_exo, self.cluster_num, h, w)
        ego_proj2 = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj2)  # Pego
        aff_logits_ego = self.gap(ego_pred).view(b, self.aff_classes)  # classificaton score

        ego_func2 = self.func_ego_proj(ego_proj)
        ego_func = self.aff_fc(ego_func2)  # func_location_map

        # --- concentration loss --- #
        gt_ego_cam = torch.zeros(b, h, w).cuda()
        func_ego_cam = torch.zeros(b, h, w).cuda()
        loss_con = torch.zeros(1).cuda()
        loss_con_func = torch.zeros(1).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])

            func_ego_cam[b_] = ego_func[b_, aff_label[b_]]
            loss_con_func += concentration_loss(ego_func[b_])

        gt_ego_cam = normalize_minmax(gt_ego_cam)
        loss_con /= b

        func_ego_cam = normalize_minmax(func_ego_cam)
        loss_con_func /= b

        # # ------------yf_hand_pred1------------------
        # tmp_feats = ego_desc * gt_ego_cam.unsqueeze(1)  # 深度特征和特定类别定位图值的乘积之和。这一步有效地通过定位图值对深度特征进行加权
        # # embedding = tmp_feats.reshape(tmp_feats.shape[0], -1).sum(1) / mask.sum()
        # ego_proj_hand = self.hand_ego_proj(tmp_feats)
        # ego_pred_hand = self.aff_fc_hand(ego_proj_hand)
        # hand_preds = self.gap_hand(ego_pred_hand).view(b, 14)
        # hand_pred = self.softmax(hand_preds)
        # ------------yf_hand_pred1------------------

        # ------------yf_hand_pred2------------------
        tmp_feats = ego_desc * gt_ego_cam.unsqueeze(1)  # 深度特征和特定类别定位图值的乘积之和。这一步有效地通过定位图值对深度特征进行加权
        tmp_feats_flat = tmp_feats.view(tmp_feats.size(0), -1)
        hand_preds = self.fc_classification(tmp_feats_flat)
        hand_pred = self.softmax(hand_preds)
        # ------------yf_hand_pred2------------------

        # --- prototype guidance loss --- #
        loss_proto = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = gt_ego_cam[b_]  #p_ego
                    tmp_feat = ego_desc[b_] * mask
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum()
                    loss_proto += torch.max(
                        1 - F.cosine_similarity(embedding, part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto = loss_proto / (valid_batch + 1e-15)

        # --- prototype_func guidance loss --- #
        loss_proto_func = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_func_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = func_ego_cam[b_]  # pego
                    tmp_feat = ego_desc[b_] * mask
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum() #fego
                    replacement_value = 1e-1  # 选择一个适当的小值
                    if torch.isnan(part_func_proto[b_]).any() or torch.isinf(part_func_proto[b_]).any():
                        print(f"NaN or Inf found in part_func_proto[{b_}]")
                        mask = torch.isnan(part_func_proto[b_]) | torch.isinf(part_func_proto[b_])
                        part_func_proto_new = part_func_proto[b_].clone()  # 克隆原始张量
                        part_func_proto_new[mask] = replacement_value  # 修改克隆后的张量
                        print('part_func_proto[b_]', part_func_proto[b_])
                    loss_proto_func += torch.max(
                        1 - F.cosine_similarity(embedding, part_func_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto_func = loss_proto_func / (valid_batch + 1e-15)

        masks = {'exo_aff': (gt_aff_cam, exo_aff_desc_sum, exo_funcparts), 'ego_sam': ego_sam,
                 'pred': (sim_func_maps, exo_sim_maps, part_score, gt_ego_cam)}
        logits = {'aff_exo': aff_exo_logits, 'aff_ego': aff_logits_ego}
        return masks, logits, loss_proto, loss_con, hand_pred, loss_proto_func, loss_con_func

    @torch.no_grad()
    def func_test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj2 = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj2)

        ego_func2 = self.func_ego_proj(ego_proj)
        ego_func = self.aff_fc(ego_func2)  # func_location_map

        # ------------yf_hand_pred------------------
        # ego_proj_hand = self.hand_ego_proj(ego_proj)
        # ego_pred_hand = self.aff_fc_hand(ego_proj_hand)
        # hand_pred = self.gap_hand(ego_pred_hand).view(b, 14)
        # hand_pred = self.softmax(hand_pred)
        # gt_ego_cam = torch.zeros(b, h, w).cuda()
        gt_ego_cam = torch.zeros(b, h, w).cpu()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
        gt_ego_cam = normalize_minmax(gt_ego_cam)
        tmp_feats = ego_desc * gt_ego_cam.unsqueeze(1)  # 广播 gt_ego_cam 以匹配 ego_desc 的维度
        # ego_proj_hand = self.hand_ego_proj(tmp_feats)
        # ego_pred_hand = self.aff_fc_hand(ego_proj_hand)
        # hand_preds = self.gap_hand(ego_pred_hand).view(b, 14)
        # hand_pred = self.softmax(hand_preds)
        tmp_feats_flat = tmp_feats.view(tmp_feats.size(0), -1)
        hand_preds = self.fc_classification(tmp_feats_flat)
        hand_pred = self.softmax(hand_preds)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        func_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            a = aff_label[b_]
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

            func_ego_cam[b_] = ego_func[b_, aff_label[b_]]

        return gt_ego_cam, func_ego_cam, hand_pred

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj2 = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj2)

        # ------------yf_hand_pred------------------
        ego_proj_hand = self.hand_ego_proj(ego_proj)
        ego_pred_hand = self.aff_fc_hand(ego_proj_hand)
        hand_pred = self.gap_hand(ego_pred_hand).view(b, 14)
        hand_pred = self.softmax(hand_pred)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            a = aff_label[b_]
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        return gt_ego_cam, hand_pred

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (448 - patch_size) // stride + 1
        width = (448 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

    def handaround_feature_extact(self, exo):
        # --- hand predict from exo --- #
        # 反标准化
        transformed_image00 = exo * self.std + self.mean

        # 转换为NumPy数组，并且像素值范围调整回[0, 255]
        image_np = transformed_image00.cpu().numpy().transpose((1, 2, 0)) * 255
        image_np = image_np.astype(np.uint8)
        image_np = image_np[:, :, ::-1]
        image_np = image_np.copy()
        # --- hand predict from exo --- #

        # --- hand predict from exo_roi --- #
        # image_np = np.array(exo_roi)
        # image_np = image_np.astype(np.uint8)
        # # 调整颜色通道从RGB到BGR
        # image_np = image_np[:, :, ::-1]
        # --- hand predict from exo_roi --- #
        # 处理图像并检测手部
        results = self.hands.process(image_np)
        # 绘制手部关键点
        rois_coordinates = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 为每个手指创建一个9维向量（两个指节和指尖的坐标）
                finger_vectors = []
                for i in range(5):  # 对于每个手指
                    vector = np.concatenate([
                        [hand_landmarks.landmark[i * 4 + j].x, hand_landmarks.landmark[i * 4 + j].y,
                         hand_landmarks.landmark[i * 4 + j].z]
                        for j in range(2, 5)  # 使用最后三个点（两个指节和指尖）
                    ]).flatten()
                    finger_vectors.append(vector)

                # 判断四指是否平行
                # 这里简化判断，仅通过相邻手指尖端点的x坐标差判断
                is_parallel = True
                for i in range(1, 4):
                    angle_cosine = np.dot(finger_vectors[i], finger_vectors[i + 1])  # 计算夹角余弦值
                    if angle_cosine < 3:  # 夹角余弦阈值，对应大约18度，可调整
                        is_parallel = False
                        break

                # 如果四指平行，大拇指是功能手指
                if is_parallel:
                    functional_finger = 0  # 大拇指
                else:
                    # 选择角度、距离及弯曲弧度最小的手指为功能手指
                    min_bend = np.inf  # 初始化为负无穷，寻找最大弯曲度
                    functional_finger = None

                    for i in range(1, 5):  # 跳过大拇指，只考虑其他四指
                        # 获取手指关节的坐标
                        joints = [np.array([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y]) for j in
                                  range(i * 4, i * 4 + 4)]

                        # 计算关节之间的向量
                        vectors = [joints[j + 1] - joints[j] for j in range(len(joints) - 1)]

                        # 计算相邻向量之间的角度，用角度的余弦值来近似
                        angle_cos = np.dot(vectors[0], vectors[1]) / (
                                np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))

                        # 由于角度的余弦值与角度成反比，所以使用1 - angle_cos来表示弯曲度
                        bend = 1 - angle_cos

                        # 选择弯曲度最小的手指
                        if bend < min_bend:
                            min_bend = bend
                            functional_finger = i

                # ---------------------extact functional finger feature------------------
                functional_fingertip_index = 4 * functional_finger + 4  # 功能手指顶点的索引
                functional_fingertip = hand_landmarks.landmark[functional_fingertip_index]

                # 将顶点坐标转换为图像上的像素坐标
                fingertip_pixel_x = int(functional_fingertip.x * image_np.shape[1])
                fingertip_pixel_y = int(functional_fingertip.y * image_np.shape[0])

                # ------------zhengfangxing------------#
                # 定义ROI大小，这里需要根据实际情况调整
                # roi_size = 20  # 像素单位
                #
                # # 计算ROI区域的边界
                # start_point = (max(0, fingertip_pixel_x - roi_size), max(0, fingertip_pixel_y - roi_size))
                # end_point = (
                #     min(image_np.shape[1], fingertip_pixel_x + roi_size),
                #     min(image_np.shape[0], fingertip_pixel_y + roi_size))
                #
                # # 将ROI坐标添加到列表中
                # rois_coordinates.append((start_point, end_point))

                # # 绘制矩形框以可视化ROI区域
                # cv2.rectangle(image_np, start_point, end_point, (255, 0, 0), 2)  # 使用蓝色矩形框标出ROI区域
                #
                # cv2.imwrite('/home/yf/code/LOCATE-main-now/save_preds/image_with_roi3.png', image_np)
                # ------------zhengfangxing------------#

                # ------------circle------------#
                radius = 60  # 圆形ROI的半径
                rois_coordinates.append((fingertip_pixel_x, fingertip_pixel_y, radius))

                # ------------circle------------#
        return rois_coordinates