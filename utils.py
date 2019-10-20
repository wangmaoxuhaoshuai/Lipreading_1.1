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
def load_sequence(filename):
    sequenceDir = filename
    sequence = []
    for i in os.listdir(sequenceDir):
        imgDir = os.path.join(sequenceDir, i)
        img = Image.open(imgDir).resize((112, 112))
        img = img.convert('L')
        img = functional.to_tensor(img)
        sequence.append(img)
    return sequence

# 将序列合并为一个tensor
def sequence_2_tensor(sequence):
    temporalvolume = torch.FloatTensor(1, 24, 112, 112) # 数据集中每个文件夹下图片最多有24个
    zeros = torch.zeros([112, 112])
    length = len(sequence)
    for i in range(24):
        if i < length:
            temporalvolume[0][i] = sequence[i]
        else:
            temporalvolume[0][i] = zeros
    return temporalvolume

# 数据增强
def data_augmentation(frames):
    crop = StatefulRandomCrop((122, 122), (112, 112))
    flip = StatefulRandomHorizontalFlip(0.5)

    croptransform = transforms.Compose([crop, flip])

    for i in range(0, len(frames)):
        result = transforms.Compose([
            transforms.Resize((127, 127)),
            transforms.CenterCrop((122, 122)),
            croptransform,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize([0.4161,],[0.1688,]),
        ])(frames[i])

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