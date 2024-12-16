import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from utils.util import normalize_map, overlay_mask, overlay_mask_yf
import matplotlib.pyplot as plt


# visualize the prediction of the first batch
def viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step):

    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)

    # 处理 Ego 图像
    ego_0 = ego[0].squeeze(0) * std + mean
    ego_0 = ego_0.detach().cpu().numpy() * 255
    ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))

    exo_img = []
    num_exo = exo.shape[1]
    for i in range(num_exo):
        exo_i = exo[0][i].squeeze(0) * std + mean
        exo_i = exo_i.detach().cpu().numpy() * 255
        exo_i = Image.fromarray(exo_i.transpose(1, 2, 0).astype(np.uint8))
        exo_img.append(exo_i)

    exo_cam = masks['exo_aff'][0][0]
    part_proto = masks['exo_aff'][1][0]
    part_func_proto = masks['exo_aff'][2]
    ego_pred = masks['pred'][3]  # ego_pred = gt_ego_cam

    # 处理 Ego 预测图像
    ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    ego_pred = Image.fromarray(ego_pred, mode='L')
    ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)

    # 处理 Exo 功能图像并叠加
    exo_aff_imgs = []
    part_proto_imgs = []
    part_func_proto_imgs = []

    for i in range(num_exo):
        exo_aff = np.array(exo_cam[i].squeeze().data.cpu())
        exo_aff = normalize_map(exo_aff, args.crop_size)
        exo_aff = Image.fromarray(exo_aff)
        exo_aff = overlay_mask(exo_img[i], exo_aff, alpha=0.5)
        exo_aff_imgs.append(exo_aff)

        if part_proto.numel() > 0:
            part_proto_img = np.array(part_proto.squeeze().data.cpu())
            part_proto_img = normalize_map(part_proto_img, args.crop_size)
            part_proto_img = (part_proto_img * 255).astype(np.uint8)
            part_proto_img = Image.fromarray(part_proto_img, mode='L')
            part_proto_img = overlay_mask(exo_img[i], part_proto_img, alpha=0.5)
            part_proto_imgs.append(part_proto_img)
        else:
            part_proto_imgs.append(None)

        if len(part_func_proto) > 0:
            part_func_proto_img = np.array(part_func_proto[0][i].squeeze().data.cpu())
            part_func_proto_img = normalize_map(part_func_proto_img, args.crop_size)
            part_func_proto_img = Image.fromarray(part_func_proto_img, mode='L')
            part_func_proto_img = overlay_mask(exo_img[i], part_func_proto_img, alpha=0.5)
            part_func_proto_imgs.append(part_func_proto_img)
        else:
            part_func_proto_imgs.append(None)

    # 绘制图像
    fig, ax = plt.subplots(3, num_exo + 1, figsize=(16, 12))
    for axi in ax.ravel():
        axi.set_axis_off()

    # 显示原始 Exo 和 Ego 图像
    for k in range(num_exo):
        ax[0, k].imshow(exo_img[k])
        ax[0, k].set_title(f"Original Exo {k}")
    ax[0, num_exo].imshow(ego_0)
    ax[0, num_exo].set_title("Original Ego")

    # 显示叠加的 Exo 功能图像和 Ego 预测图像
    for k in range(num_exo):
        ax[1, k].imshow(exo_aff_imgs[k])
        ax[1, k].set_title(f"Exo Cam {k}")
    ax[1, num_exo].imshow(ego_pred)
    ax[1, num_exo].set_title("Ego Prediction")

    # 显示 part_proto 和 part_func_proto 叠加图像
    for k in range(num_exo):
        if part_proto_imgs[k] is not None:
            ax[2, k].imshow(part_proto_imgs[k])
            ax[2, k].set_title(f"Part Proto {k}")
        if part_func_proto_imgs[k] is not None:
            ax[2, k].imshow(part_func_proto_imgs[k])
            ax[2, k].set_title(f"Part Func Proto {k}")

    os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    fig_name = os.path.join(args.save_path, 'viz_train', f'cam_{epoch}_{step}.jpg')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def viz_pred_test(args, image, ego_pred, GT_mask, AG_mask, aff_list, aff_label, img_name, func_ego_cam, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)

    ag = Image.fromarray(AG_mask)
    ag_result = overlay_mask(img, ag, alpha=0.5)

    locate = Image.fromarray(ego_pred)
    locate_result = overlay_mask(img, locate, alpha=0.5)

    aff_str = aff_list[aff_label.item()]

    func_ego_cam0 = Image.fromarray(func_ego_cam)
    func_ego_cam, center_x, center_y = overlay_mask_yf(img, func_ego_cam0, alpha=0.5)



    fig, ax = plt.subplots(1, 5, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    # draw_pre = ImageDraw.Draw(ego_pred)
    draw_pre_func = ImageDraw.Draw(func_ego_cam)
    # draw_gt = ImageDraw.Draw(img)
    point_size = 5

    # draw_pre_func.ellipse((center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size),
    #                  fill='red')
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(func_ego_cam)
    ax[1].set_title(aff_str)
    ax[2].imshow(gt_result)
    ax[2].set_title('GT')
    ax[3].imshow(ag_result)
    ax[3].set_title('AG')
    ax[4].imshow(locate_result)
    ax[4].set_title('locate')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()

def viz_pred_test_yf(args, image, ego_pred, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    # gt = Image.fromarray(GT_mask)
    # gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]
    ego_pred0 = Image.fromarray(ego_pred)
    ego_pred, center_x, center_y = overlay_mask_yf(img, ego_pred0, alpha=0.5)

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    # ------------------Plot the grab points------------------#
    # draw_pre = ImageDraw.Draw(ego_pred)
    # draw_gt = ImageDraw.Draw(img)
    # point_size = 5
    # draw_pre.ellipse((center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size),
    #              fill='red')
    # ------------------Plot the grab points------------------#
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(ego_pred)
    ax[1].set_title(aff_str)
    # ax[2].imshow(gt_result)
    # ax[2].set_title('GT')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()
    return center_x, center_y

def viz_pred_test_locate(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    # gt = Image.fromarray(GT_mask)
    # gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]

    ego_pred0 = Image.fromarray(ego_pred)
    ego_pred, center_x, center_y = overlay_mask_yf(img, ego_pred0, alpha=0.5)

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    draw_pre = ImageDraw.Draw(ego_pred)
    draw_gt = ImageDraw.Draw(img)
    point_size = 5
    # draw_pre.ellipse((center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size),
    #              fill='red')
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(ego_pred)
    ax[1].set_title(aff_str)
    # ax[2].imshow(gt_result)
    # ax[2].set_title('GT')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()
