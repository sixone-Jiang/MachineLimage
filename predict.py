import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image

# 流程化读取图片
def load_img(img_path, img_w, img_h):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert("RGB") # 读取图片的过程中如果遇到非'RGB'就转换格式(3 通道图像)
    img_convert = img.resize((img_w,img_h,), Image.LANCZOS) # 要快就使用 image.NEARSEST 效果好使用image.LANCZOS
    return np.array(img_convert)

# 输入测试图例1 (真值为 horse)
img_path = "horse.png"
img_w = 32
img_h = 32

# 网络输入形状为 （n, w, h, 3）
test_image = np.array([load_img(img_path, img_w, img_h)])

# 读取模型
my_load_model = models.load_model("model_image.h5")

# 预测
y = my_load_model.predict(np.array(test_image))

# 分类标签
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 查看预测结果(test_image真值为 'horse' 在ipynb文件中可以展示)
print(class_names[np.argmax(y[0])])