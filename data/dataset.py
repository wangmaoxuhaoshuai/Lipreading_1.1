from utils import read_csv_file, load_sequence, sequence_2_tensor
from import_sets import *
"""
所有数据文件都从此读取
"""
# 读取配置文件
print("Loading options...")
with open('config/options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

# 标签到汉字的映射
map_index_hanzi = []
with open(options["dataset"]["map_index_hanzi_file_path"], encoding='UTF-8') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        map_index_hanzi.append([row[0], row[-1]])
map_index_hanzi = dict(map_index_hanzi)

class ReadDataset(Dataset):
    """
    读取 训练/测试 数据集
    """
    def __init__(self, datasetname, augment=True):
        self.file_list = self.build_file_list(datasetname)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filelist = self.file_list[idx]
        label = filelist[2]
        label = int(label)
        label = torch.LongTensor([label])
        frames = load_sequence(filelist[1])
        temporalvolume = sequence_2_tensor(frames)
        sample = {'temporalvolume': temporalvolume, 'label': label}
        return sample

    # [i, dir, label[i]]
    def build_file_list(self, filename):
        data = read_csv_file(filename)
        # 标签序号
        label = []
        # 序列文件夹名
        sequenceDir = []
        for i in range(len(data)):
            label.append(data[i][-1])
            sequenceDir.append(data[i][0])

        completeList = []
        print("load train dataset...")
        for i, dirpath in enumerate(sequenceDir):
            if options['dataset']['add_train_dataset_absolute_path'] == True:
                dirpath = r"{}/{}".format(options['dataset']['train_dataset_absolute_path'], dirpath)
            entry = [i, dirpath, label[i]]
            completeList.append(entry)
            print("load train file: {}".format(dirpath))
        print("load train dataset finished...")
        return completeList

class ReadPredictDataset(Dataset):
    """
    读取预测数据集，只有输入，没有标签
    """
    def __init__(self, augment=True):
        self.file_list = self.get_file_list()
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self):
        testDir = []
        print("load test dataset...")
        for samplename in os.listdir(options['dataset']['test_dataset_absolute_path']):
            dirpath = r"{}/{}".format(options['dataset']['test_dataset_absolute_path'], samplename)
            testDir.append([samplename, dirpath])
            print("load test file: {}".format(dirpath))
        print("load test dataset finished...")
        return testDir

    def __getitem__(self, idx):
        filelist = self.file_list[idx]

        samplename = filelist[0]
        frames = load_sequence(filelist[1])
        temporalvolume = sequence_2_tensor(frames)

        sample = {'temporalvolume': temporalvolume, 'samplename': samplename}

        return sample



