"""
预测
"""
from import_sets import *

if __name__ == '__main__':
    for iter, model in enumerate(options["general"]["pretrainedmodelpath"]):
        model = torch.load(options["general"]["pretrainedmodelpath"][iter])

        # Move the model to the GPU.
        if (options["general"]["usecudnn"]):
            model = model.cuda(options["general"]["gpuid"])

        outputList = []
        predict_dataset = ReadPredictDataset()
        predict_dataset_loader = DataLoader(
            predict_dataset,
            batch_size = 1,
            shuffle = False
        )

        with torch.no_grad():
            for i, sample in enumerate(predict_dataset_loader):
                torch.cuda.empty_cache()
                input = Variable(sample['temporalvolume']).cuda()
                samplename = sample['samplename']
                output = model(input)
                print("predict {} samples".format(i))
                outputList.append([samplename, output])

        # index 用来保存文件夹名和对应的预测值index
        index = []
        for i in range(len(outputList)):
            data = outputList[i][1][0].tolist()
            index.append([outputList[i][0][0], data.index(max(data))])

        # 保存格式：文件夹名，预测值
        with open('outputList{}.csv'.format(iter), 'w', newline='', encoding='UTF-8') as f:
            for i in range(len(index)):
                f_csv = csv.writer(f)
                f_csv.writerow([index[i][0], map_index_hanzi[str(index[i][1])]])
        print("test {} samples".format(len(predict_dataset_loader)))

