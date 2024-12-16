import os
import sys
import time
import shutil
import logging
import argparse
import re
import cv2
import torch
import torch.nn as nn
import numpy as np
from models.GAAF_Dex import Net as model  # yf
from utils.viz import viz_pred_train, viz_pred_test
from utils.util import set_seed, process_gt, normalize_map, get_optimizer
from utils.evaluation import cal_kl, cal_sim, cal_nss, AverageMeter, compute_cls_acc
from sklearn.metrics import precision_score, recall_score

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/data1/yf/yinshi1/')
parser.add_argument('--save_root', type=str, default='save_models')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=512)
##  dataloader
parser.add_argument('--num_workers', type=int, default=1)
##  train
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--lr', type=float, default=0.0001)  # resume
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', default=False)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=1)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)
lr = args.lr

if args.divide == "Seen":
    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]
else:
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]

if args.divide == "Seen":
    # args.num_classes = 36
    args.num_classes = 6
else:
    args.num_classes = 25

args.exocentric_root = os.path.join(args.data_root, args.divide, "trainset", "exocentric")
args.egocentric_root = os.path.join(args.data_root, args.divide, "trainset", "egocentric")
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

shutil.copy('./models/locate_func_handpre.py', args.save_path)
shutil.copy('./train.py', args.save_path)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

if __name__ == '__main__':
    set_seed(seed=0)

    from data.datatrain import TrainData

    trainset = TrainData(exocentric_root=args.exocentric_root,
                         egocentric_root=args.egocentric_root,
                         resize_size=args.resize_size,
                         crop_size=args.crop_size, divide=args.divide)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    from data.datatest_func import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes)
    model = model.cuda()
    # checkpoint = torch.load(
    #     '/home/yf/code/LOCATE-main-now/save_models/20241212_123201/best_aff_model_5_1.462_0.327_1.238.pth')
    # model.load_state_dict(checkpoint)
    model.train()
    optimizer, scheduler = get_optimizer(model, args)
    criterion = nn.BCELoss()  # yf
    best_kld = 1000
    best_average_success_rate = 0
    print('Train begining!')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        model.train()
        logger.info('LR = ' + str(scheduler.get_last_lr()))
        exo_aff_acc = AverageMeter()
        ego_obj_acc = AverageMeter()
        accs = 0
        # label_hand = torch.zeros(hand_pred.shape).to(args.divide)
        for step, (exocentric_image, egocentric_image, aff_label, hand_lable, exocentric_image_path, egocentric_image_path) in enumerate(TrainLoader):
            aff_label = aff_label.cuda().long()  # b x n x 6
            exo = exocentric_image.cuda()  # b x n x 3 x 224 x 224
            ego = egocentric_image.cuda()

            masks, logits, loss_proto, loss_con, hand_pred, loss_proto_func, loss_con_func = model(exo, ego, aff_label,egocentric_image_path, (epoch, args.warm_epoch))

            # --------yf------------------
            outputs = torch.zeros(16, 14).cuda()
            for j in range(hand_pred.shape[0]):
                for i in range(hand_pred.shape[1]):
                    if hand_pred[j][i].item() >= 0.5:
                        outputs[j][i] = torch.tensor(1.0)
                    else:
                        outputs[j][i] = torch.tensor(0.0)

            acc = 0
            for i in range(hand_lable.shape[0]):
                for j in range(hand_lable.shape[1]):
                    if outputs[i][j] == hand_lable[i][j]:
                        acc += 1
            acc = acc / (28.0 * 16)
            accs += acc
            hand_label_expanded = hand_lable.expand_as(hand_pred)
            hand_loss = criterion(hand_pred, hand_label_expanded.cuda().float())

            exo_aff_logits = logits['aff_exo']  # (16,3,6)
            num_exo = exo.shape[1]  # 3
            exo_aff_loss = torch.zeros(1).cuda()
            angle_loss = torch.zeros(1).cuda()
            for n in range(num_exo):
                a = exo_aff_logits[:, n]  # 16,6
                exo_aff_loss += nn.CrossEntropyLoss().cuda()(exo_aff_logits[:, n], aff_label)

            exo_aff_loss /= num_exo

            loss_dict = {'ego_ce': nn.CrossEntropyLoss().cuda()(logits['aff_ego'], aff_label),
                         'ego_func_ce': nn.CrossEntropyLoss().cuda()(logits['aff_func_ego'], aff_label),
                         'exo_ce': exo_aff_loss,
                         'con_loss': loss_proto,
                         'loss_cen': loss_con * 0.07,
                         'hand_loss': hand_loss,
                         'con_func_loss': loss_proto_func,
                         'loss_cen_func': loss_con_func * 0.07,
                         }

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_batch = exo.size(0)
            exo_acc = 100. * compute_cls_acc(logits['aff_exo'].mean(1), aff_label)
            exo_aff_acc.updata(exo_acc, cur_batch)
            metric_dict = {'exo_aff_acc': exo_aff_acc.avg}

            if (step + 1) % args.show_step == 0:
                log_str = 'epoch: %d/%d + %d/%d | ' % (epoch + 1, args.epochs, step + 1, len(TrainLoader))
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in metric_dict.items()])
                log_str += ' | '
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in loss_dict.items()])
                logger.info(log_str)
                print("accs", accs)

                # Visualization the prediction during training
            if args.viz:
                viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step + 1)

        scheduler.step()
        KLs = []
        SIM = []
        NSS = []
        model.eval()
        GT_path = args.divide + "_gt.t7"
        if not os.path.exists(GT_path):
            process_gt(args)
        GT_masks = torch.load(args.divide + "_gt.t7")

        # ---------------grasp type success----------------#
        all_preds_per_task = {i: [] for i in range(6)}
        all_hand_labels_per_task = {i: [] for i in range(6)}
        
        all_preds_per_task_tool = {}
        all_hand_labels_per_task_tool = {}
        # ---------------grasp type success----------------#
        for step, (image, label, mask_path, hand_label) in enumerate(TestLoader):
            ego_pred, func_ego_cam, hand_pred = model.func_test_forward(image.cuda(), label.long().cuda())
            cluster_sim_maps = []
            ego_pred = np.array(ego_pred.squeeze().data.cpu())
            ego_pred = normalize_map(ego_pred, args.crop_size)
            func_ego_cam0 = np.array(func_ego_cam.squeeze().data.cpu())
            func_ego_cam1 = normalize_map(func_ego_cam0, args.crop_size)

            names = re.split(r'[/.]', mask_path[0])
            key = names[-4] + "_" + names[-3] + "_" + names[-2] + "_heatmap." + names[-1]
            task = names[-3]
            tool = names[-2]

            # -----------------metric of grasp type---------------#
            hand_pred_probs = hand_pred.squeeze().data.cpu().numpy()
            hand_pred_labels = np.argmax(hand_pred_probs, axis=0)

            hand_label_index = hand_label.argmax(dim=1).item()  # Convert hand_label to the gesture index

            # 根据任务标签来分配预测和真实手势标签
            task_id = label.item()
            all_preds_per_task[task_id].append(hand_pred_labels)
            all_hand_labels_per_task[task_id].append(hand_label_index)
          
            names = mask_path[0].split("/")
            tool = names[-2]
            task_tool_key = f"{task_id}_{tool}"
       
            if task_tool_key not in all_preds_per_task_tool:
                all_preds_per_task_tool[task_tool_key] = []
                all_hand_labels_per_task_tool[task_tool_key] = []
         
            all_preds_per_task_tool[task_tool_key].append(hand_pred_labels)
            all_hand_labels_per_task_tool[task_tool_key].append(hand_label_index)
            # -----------------metric of grasp type---------------#

            GT_mask = GT_masks[key]

            GT_mask = GT_mask / 255.0

            GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

            kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

            # Visualization the prediction during evaluation
            if args.viz:
                if (step + 1) % args.show_step == 0:
                    img_name = key.split(".")[0]
                    viz_pred_test(args, image, ego_pred, GT_mask, aff_list, label, img_name, func_ego_cam1, epoch)

        mKLD = sum(KLs) / len(KLs)
        mSIM = sum(SIM) / len(SIM)
        mNSS = sum(NSS) / len(NSS)

        # 计算6种任务一起的平均操作成功率
        all_preds = [pred for preds in all_preds_per_task.values() for pred in preds]
        all_labels = [label for labels in all_hand_labels_per_task.values() for label in labels]
        average_success_rate = precision_score(all_labels, all_preds, average='macro')

        # 计算每个任务-工具组合的成功率
        precision_per_task_tool = {}
        recall_per_task_tool = {}

        for task_tool_key in all_preds_per_task_tool:
            preds = np.array(all_preds_per_task_tool[task_tool_key])
            labels = np.array(all_hand_labels_per_task_tool[task_tool_key])
            precision_per_task_tool[task_tool_key] = precision_score(labels, preds, average='macro')
            recall_per_task_tool[task_tool_key] = recall_score(labels, preds, average='macro')

        # 计算每种任务的平均操作成功率
        precision_per_task = {}
        recall_per_task = {}

        for task_id in range(6):
            if all_hand_labels_per_task[task_id]:  # 确保不为空
                preds = np.array(all_preds_per_task[task_id])
                labels = np.array(all_hand_labels_per_task[task_id])
                precision_per_task[task_id] = precision_score(labels, preds, average='macro')
                recall_per_task[task_id] = recall_score(labels, preds, average='macro')
            else:
                precision_per_task[task_id] = None
                recall_per_task[task_id] = None

        print("Average success rate across all tasks:", average_success_rate)
        print("Success rate per task-tool combination:", precision_per_task_tool)
        print("Average success rate per task:", precision_per_task)
        print('ddd')

        logger.info(
            "epoch=" + str(epoch + 1) + " mKLD = " + str(round(mKLD, 3))
            + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3))
            + " bestKLD = " + str(round(best_kld, 3)))

        if mKLD < best_kld:
            best_kld = mKLD
            model_name = 'best_aff_model_' + str(epoch + 1) + '_' + str(round(best_kld, 3)) \
                         + '_' + str(round(mSIM, 3)) \
                         + '_' + str(round(mNSS, 3)) \
                         + '.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
        if average_success_rate > best_average_success_rate:
            best_average_success_rate = average_success_rate
            model_name = 'best_grasp_model_' + str(epoch + 1) + '_' + str(round(average_success_rate, 3)) + '.pth'