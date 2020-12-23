import timm
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class classifier_sub:
    def __init__(self, model_path1, model_path2, input_size = 128,back_vs_person_padding = False,
                 back_vs_person_normalize = False, normal_vs_falldown_padding = False,
                 normal_vs_falldown_normalize = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model2 = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=2)

        self.model2.load_state_dict(torch.load(model_path2))
        self.model2.to(self.device)
        self.model2.eval()

        self.input_size = input_size



        transform_normal_vs_falldown = []
        transform_normal_vs_falldown.append(transforms.ToTensor())
        if normal_vs_falldown_normalize:
            transform_normal_vs_falldown.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        if normal_vs_falldown_padding:
            transform_normal_vs_falldown.append(
                lambda x: transforms.Pad(((128 - x.shape[2]) // 2, (128 - x.shape[1]) // 2), fill=0,
                                         padding_mode="constant")(x))
        transform_normal_vs_falldown.append(transforms.Resize((input_size, input_size)))
        self.transform_normal_vs_falldown = transforms.Compose(transform_normal_vs_falldown)
        print("normal_vs_falldown: {}".format(self.transform_normal_vs_falldown))

    def predict(self, input_):
        len_input = len(input_)
        PERSON = 1
        BACKGROUND = 2
        patch_normal_vs_falldown = torch.empty(len_input, 3, self.input_size, self.input_size)
        for n, img in enumerate(input_):
            try:
                img_normal_vs_falldown = self.transform_normal_vs_falldown(img.copy())
                patch_normal_vs_falldown[n] = img_normal_vs_falldown
            except:
                patch_normal_vs_falldown[n] = torch.zeros((3, self.input_size, self.input_size))
        patch_normal_vs_falldown = patch_normal_vs_falldown.to(self.device)
        with torch.no_grad():
            try:
                falldown_or_normal_preds = self.model2(patch_normal_vs_falldown)
                _, falldown_or_normal = torch.max(falldown_or_normal_preds, -1)
                return falldown_or_normal
            except:
                return []









