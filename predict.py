#coding=utf-8
"""
预测
"""
from import_sets import *

if __name__ == '__main__':
    model = densenet169()
    model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))
    model.eval()

    # Move the model to the GPU.
    if (options["general"]["usecudnn"]):
        model = model.cuda(options["general"]["gpuid"])

    outputList = []
    predict_dataset = ReadPredictDataset()
    predict_dataset_loader = DataLoader(
        predict_dataset,
        batch_size = options["input"]["batchsize"],
        shuffle = False
    )

    with torch.no_grad():
        for i, sample in enumerate(predict_dataset_loader):
            torch.cuda.empty_cache()
            input = Variable(sample['temporalvolume']).cuda()
            samplename = sample['samplename']
            output = model(input)
            # output = output[0].tolist()
            # output = output.index(max(output))
            print("predict {} / {} samples".format(i * len(samplename), len(predict_dataset)))
            for n in range(len(samplename)):
                output_n = output[n].tolist()
                output_n = output_n.index(max(output_n))
                outputList.append([samplename[n], output_n])

    # 保存格式：文件夹名，预测值
    with open('outputLists/outputList.csv', 'w', newline='', encoding='UTF-8') as f:
        for i in range(len(outputList)):
            f_csv = csv.writer(f)
            f_csv.writerow([outputList[i][0], map_index_hanzi[str(outputList[i][1])]])
    print("test {} samples".format(len(predict_dataset_loader)))

