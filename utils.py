from import_sets import *
"""
方法集合
"""

# 读取.csv文件
def read_csv_file(filename):
    data = []
    with open(filename, encoding='UTF-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    return data

# 读取多个图片并转换为tensor的序列
def load_sequence(filename, data_aug):
    sequenceDir = filename
    sequence = []
    all_imgs = os.listdir(sequenceDir) # os.listdir 读取顺序不是按照数字
    all_imgs.sort(key = lambda x: int(x[:-4]))
    for i in all_imgs:
        imgDir = os.path.join(sequenceDir, i)
        img = Image.open(imgDir)
        img = img.resize((112, 112))
        # img = img.convert('L')
        if data_aug == False:
            img = functional.to_tensor(img)
        sequence.append(img)
    if data_aug == True:
        sequence = data_augmentation(sequence)
    return sequence

# 将序列合并为一个tensor
def sequence_2_tensor(sequence):
    # temporalvolume = torch.FloatTensor(1, 24, 112, 112) # 数据集中每个文件夹下图片最多有24个
    temporalvolume = torch.FloatTensor(3, 24, 112, 112)
    zeros = torch.zeros([112, 112])
    length = len(sequence)
    for temp_channel in range(24):
        for rgb_channel in range(3):
            if temp_channel < length:
                temporalvolume[rgb_channel][temp_channel] = sequence[temp_channel][rgb_channel]
            else:
                temporalvolume[rgb_channel][temp_channel] = zeros
    return temporalvolume

# 数据增强
def data_augmentation(frames):
    random_seed = np.random.randint(100)
    np.random.seed(random_seed)
    for i in range(0, len(frames)):
        result = randomColor(frames[i], random_seed)
        result = transforms.ToTensor()(result)
        frames[i] = result
    return frames

# 随机裁剪
# 要保证一个序列里的图片增强方式一样，所以这里采用的方法是，先随机产生参数，然后根据该参数进行多次变化
class StatefulRandomCrop(object):
    def __init__(self, insize, outsize):
        self.size = outsize
        self.cropParams = self.get_params(insize, self.size)

    @staticmethod
    def get_params(insize, outsize):
        w, h = insize
        th, tw = outsize
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.cropParams
        return functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

# 随机翻转
# 同上
class StatefulRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.rand = random.random()

    def __call__(self, img):
        if self.rand < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

# 颜色抖动
def randomColor(image, random_seed):

    np.random.seed(random_seed)

    random_factor = np.random.randint(80, 121) / 100.  # 随机因子
    image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

    random_factor = np.random.randint(80, 121) / 100.  # 随机因子
    image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度

    random_factor = np.random.randint(80, 121) / 100.  # 随机因1子
    image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度

    random_factor = np.random.randint(0, 201) / 100.  # 随机因子
    image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度

    return image