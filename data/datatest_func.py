import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
import json
import torch

class TestData(data.Dataset):
    def __init__(self, image_root, crop_size=448, divide="Seen", mask_root=None):
        self.image_root = image_root
        self.image_list = []
        self.crop_size = crop_size
        self.mask_root = mask_root
        if divide == "Seen":
            # --------yf-------------
            self.aff_list = ['hold', 'press', 'click', 'clamp', 'grip', 'open']
            self.obj_list = ['screwdriver', 'plug', 'kettle', 'hammer', 'spray bottle', 'stapler', 'flashlight',
                             'bottle', 'cup', 'mouse', 'knife', 'pliers', 'spatula', 'scissors', 'door handle',
                             'lightswitch', 'drill']
            # --------yf-------------
        else:
            self.aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
            self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                             'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                             'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                             'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                             'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                             'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                             'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                             'tennis_racket', 'toothbrush', 'wine_glass']

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                      std=(0.229, 0.224, 0.225))])
            transforms.Normalize(mean=(0.592, 0.558, 0.523),
                                 std=(0.228, 0.223, 0.229))])

        # --------yf-------------
        with open('yinshi_labels.json', 'r') as file:
            self.label_map = json.load(file)
        # --------yf-------------

        files = os.listdir(self.image_root)
        for file in files:
            file_path = os.path.join(self.image_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    mask_path = os.path.join(self.mask_root, file, obj_file, img[:-3] + "png")
                    self.image_list.append(img_path)      #yf_func_viz
                    # if os.path.exists(mask_path):
                    #     self.image_list.append(img_path)
        print(self.image_list)

        self.aff2obj_dict = dict()
        for aff in self.aff_list:
            aff_path = os.path.join(self.image_root, aff)
            aff_obj_list = os.listdir(aff_path)
            self.aff2obj_dict.update({aff: aff_obj_list})

        self.obj2aff_dict = dict()
        for obj in self.obj_list:
            obj2aff_list = []
            for k, v in self.aff2obj_dict.items():
                if obj in v:
                    obj2aff_list.append(k)
            for i in range(len(obj2aff_list)):
                obj2aff_list[i] = self.aff_list.index(obj2aff_list[i])
            self.obj2aff_dict.update({obj: obj2aff_list})

    def __getitem__(self, item):

        image_path = self.image_list[item]
        names = image_path.split("/")
        aff_name, object = names[-3], names[-2]

        image = self.load_img(image_path)
        label = self.aff_list.index(aff_name)
        names = image_path.split("/")
        mask_path = os.path.join(self.mask_root, names[-3], names[-2], names[-1][:-3] + "png")

        # -----------yf-----------------
        category_name = f"{aff_name}_{object}"
        hand_label0 = self.label_map[category_name]
        hand_label = torch.tensor(hand_label0)
        # -----------yf-----------------

        return image, label, mask_path, hand_label

    def load_img(self, path):
        img = Image.open(path).convert('RGB')

        img = self.transform(img)
        return img

    def __len__(self):

        return len(self.image_list)
