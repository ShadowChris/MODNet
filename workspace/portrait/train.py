import os
import torch
import time
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import morphology
import cv2
from torch.utils.data import DataLoader
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import logging

os.chdir("../..")
print(os.getcwd())
dataset_path = "datasets/mini_matting_human_half"
# dataset_path = "../datasets/matting_human_half"
ckpt_path = "pretrained/modnet_photographic_portrait_matting.ckpt"

# 生成唯一文件名
uuid_str = uuid.uuid4().hex

# 设置日志
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_filename = f"{uuid_str}_training.log"
log_filepath = os.path.join(log_directory, log_filename)
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, filename=log_filepath, filemode="w")
logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
# logger.addHandler(console_handler)


def cv2_imshow(image):
    # image = cv2.resize(image, (720, 480))

    cv2.imshow("img", image)
    #  add below code
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2plt(img):
    plt.figure(figsize=(7, 7))  # To change the size of figure
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


# 生成数据集pair
data_csv = pd.DataFrame(columns=["images", "matte"])
image_dir = os.path.abspath(dataset_path)
image_list = list()
for folder in os.listdir(image_dir):
    #     if folder==".DS_Store":
    #     continue
    for batch in os.listdir(os.path.join(image_dir, folder)):
        for clip in os.listdir(os.path.join(image_dir, folder, batch)):
            if clip == "._matting_00000000":
                continue
            for img in os.listdir(os.path.join(image_dir, folder, batch, clip)):
                # print(img)
                image = os.path.join(image_dir, folder, batch, clip, img)
                image_list.append(image)

data_csv["images"] = image_list
data_csv["matte"] = data_csv["images"]
data_csv["matte"] = data_csv["matte"].str.replace("jpg", "png").str.replace("clip_img", "matting").str.replace("clip_",
                                                                                                               "matting_")


class ModNetDataLoader(Dataset):
    def __init__(self, annotations_file, resize_dim, transform=None):
        self.img_labels = annotations_file
        self.transform = transform
        self.resize_dim = resize_dim

    def __len__(self):
        # return the total number of images
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        mask_path = self.img_labels.iloc[idx, 1]
        # print(img_path)
        # print(mask_path)

        img = np.asarray(Image.open(img_path))

        in_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # cv2_imshow(in_image)

        mask = in_image[:, :, 3]

        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # convert Image to pytorch tensor
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.transform:
            img = self.transform(img)
            trimap = self.get_trimap(mask)
            mask = self.transform(mask)

        img = self._resize(img)
        mask = self._resize(mask)
        trimap = self._resize(trimap, trimap=True)

        img = torch.squeeze(img, 0)
        mask = torch.squeeze(mask, 0)
        trimap = torch.squeeze(trimap, 1)

        return img, trimap, mask

    def get_trimap(self, alpha):
        # alpha \in [0, 1] should be taken into account
        # be careful when dealing with regions of alpha=0 and alpha=1
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        # image dilation implemented by Euclidean distance transform
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
        trimap = fg
        trimap[unknown] = 0.5
        return torch.unsqueeze(torch.from_numpy(trimap), dim=0)  # .astype(np.uint8)

    def _resize(self, img, trimap=False):
        im = img[None, :, :, :]
        ref_size = self.resize_dim

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        if trimap == True:
            im = F.interpolate(im, size=(im_rh, im_rw), mode='nearest')
        else:
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        return im


transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)
                             )
    ]
)
data = ModNetDataLoader(data_csv, 512, transform=transformer)

img, trimap, mask = data[0]
# 打印图像、matte 和 trimap 的形状
print("Image shape:", img.shape)
print("Trimap shape:", trimap.shape)
print("Matte shape:", mask.shape)

# 训练加载器
train_dataloader = DataLoader(data, batch_size=2, shuffle=True)

bs = 2  # batch size
lr = 0.001  # learn rate
epochs = 10  # total epochs
# step_size = 1  # 学习率将在每 n 个 epoch 之后衰减，epochs < 4 用这行
step_size = int(0.25 * epochs)  # epochs >= 4 用这行

modnet = torch.nn.DataParallel(MODNet()).cuda()
# modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True)).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

# ------------Training---------------
# 创建空列表以保存每个loss的值
avg_semantic_losses = []
avg_detail_losses = []
avg_matte_losses = []
avg_total_losses = []

for epoch in range(0, epochs):
    logger.info(f"Epoch {epoch + 1} Training: ")
    start_time = time.time()

    # 初始化损失值的总和和批次计数
    total_semantic_losses = 0
    total_detail_losses = 0
    total_matte_losses = 0
    total_loss = 0
    batch_count = 0

    train_dataloader_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for idx, (image, trimap, gt_matte) in train_dataloader_progress:
        semantic_loss, detail_loss, matte_loss = \
            supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda())

        # 累加损失值
        total_semantic_losses += semantic_loss.item()
        total_detail_losses += detail_loss.item()
        total_matte_losses += matte_loss.item()
        total_loss += semantic_loss + detail_loss + matte_loss
        batch_count += 1

        # 更新进度条信息
        progress_description = f"Batch {idx + 1}: Semantic Loss {semantic_loss:.4f}, Detail Loss {detail_loss:.4f}, Matte Loss {matte_loss:.4f}"
        train_dataloader_progress.set_description(progress_description)

        # 将进度条信息输入到日志文件中
        logger.info(progress_description)

    end_time = time.time()

    # 计算平均损失值
    avg_semantic_loss = total_semantic_losses / batch_count
    avg_detail_loss = total_detail_losses / batch_count
    avg_matte_loss = total_matte_losses / batch_count
    avg_total_loss = total_loss / batch_count

    # 将每个平均损失值添加到相应的列表中
    avg_semantic_losses.append(avg_semantic_loss)
    avg_detail_losses.append(avg_detail_loss)
    avg_matte_losses.append(avg_matte_loss)
    avg_total_losses.append(avg_total_loss)

    epoch_description = f"Epoch {epoch + 1}/{epochs}, Avg Total Loss: {avg_total_loss:.4f}, Avg Semantic Loss: {avg_semantic_loss:.4f}, Avg Detail Loss: {avg_detail_loss:.4f}, Avg Matte Loss: {avg_matte_loss:.4f}, Time: {end_time - start_time:.2f}s"
    logger.info(epoch_description)
    print(epoch_description)

    lr_scheduler.step()

# 在训练结束后绘制loss曲线
plt.figure()
plt.plot(avg_semantic_losses, label='Avg Semantic Loss')
plt.plot(avg_detail_losses, label='Avg Detail Loss')
plt.plot(avg_matte_losses, label='Avg Matte Loss')
plt.plot(avg_total_losses, label='Avg Total Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Convergence')

plt.savefig(f'logs/{uuid_str}_loss_convergence.png')
plt.show()

logger.info("Saving model...")
torch.save(modnet.state_dict(), f"pretrained/my_train/modnet_{uuid_str}.ckpt")
logger.info(f"Model saved to pretrained/my_train/modnet_{uuid_str}.ckpt")
