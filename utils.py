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