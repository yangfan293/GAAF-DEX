import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms
import json
import numpy as np

class TrainData(data.Dataset):
    def __init__(self, exocentric_root, egocentric_root, resize_size=512, crop_size=448, divide="Seen"):

        self.exocentric_root = exocentric_root
        self.egocentric_root = egocentric_root

        self.image_list = []
        self.exo_image_list = []
        self.resize_size = resize_size
        self.crop_size = crop_size
        # self.crop_size_func = (768, 512)
        if divide == "Seen":
            # --------yf-------------
            self.aff_list = ['hold', 'press', 'click', 'clamp', 'grip', 'open']
            self.obj_list = ['screwdriver', 'plug', 'kettle', 'hammer', 'spraybottle', 'stapler', 'flashlight',
                             'bottle', 'cup', 'mouse', 'knife', 'pliers', 'spatula', 'scissors', 'doorhandle',
                             'lightswitch', 'drill', 'valve']
            # --------yf-------------
        else:

            self.aff_list = ['hold', 'press', 'click', 'clamp', 'grip', 'open']
            self.obj_list = ['screwdriver', 'plug', 'kettle', 'hammer', 'spraybottle', 'stapler', 'flashlight',
                             'bottle', 'cup', 'mouse', 'knife', 'pliers', 'spatula', 'scissors', 'doorhandle',
                             'lightswitch', 'drill', 'valve']

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),  # 如果只提供一个整数而不是元组，图像将被等比例缩放，使得较小的边等于该整数，同时保持图像的宽高比不变。
            transforms.RandomCrop(crop_size),  # 进行随机裁剪
            transforms.RandomHorizontalFlip(),  # 以50%的概率水平翻转图像
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
            # transforms.Normalize(mean=(0.592, 0.558, 0.523),
            #                      std=(0.228, 0.223, 0.229))]) # 进行标准化，即减去均值并除以标准差

        self.transform_ori = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            ])

        # --------yf-------------
        with open('yinshi_labels.json', 'r') as file:
            self.label_map = json.load(file)
        # --------yf-------------

        # image list for egocentric images
        files = os.listdir(self.exocentric_root)
        for file in files:
            file_path = os.path.join(self.exocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    self.image_list.append(img_path)

        # multiple affordance labels for exo-centric samples

    def __getitem__(self, item):

        # load egocentric image
        exocentric_image_path = self.image_list[item]
        names = exocentric_image_path.split("/")
        aff_name, object = names[-3], names[-2]
        exocentric_image = self.load_img(exocentric_image_path)
        aff_label = self.aff_list.index(aff_name)

        # -----------yf-----------------
        category_name = f"{aff_name}_{object}"
        hand_label0 = self.label_map[category_name]
        hand_label = torch.tensor(hand_label0)
        # -----------yf-----------------

        ego_path = os.path.join(self.egocentric_root, aff_name, object)
        obj_images = os.listdir(ego_path)
        idx = random.randint(0, len(obj_images) - 1)
        egocentric_image_path = os.path.join(ego_path, obj_images[idx])
        egocentric_image = self.load_img(egocentric_image_path)

        # pick one available affordance, and then choose & load exo-centric images
        num_exo = 3
        exo_dir = os.path.dirname(exocentric_image_path)
        exocentrics = os.listdir(exo_dir)
        exo_img_name = [os.path.basename(exocentric_image_path)]
        exocentric_images = [exocentric_image]

        # 设置加载额外外向中心图像的数量（num_exo）
        if len(exocentrics) > num_exo:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                while exo_img_ in exo_img_name:
                    exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                exocentric_images.append(tmp_exo)

        else:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                exocentric_images.append(tmp_exo)

        exocentric_images = torch.stack(exocentric_images, dim=0)  # n x 3 x 224 x 224

        return exocentric_images, egocentric_image, aff_label, hand_label, exocentric_image_path, egocentric_image_path

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):

        return len(self.image_list)
