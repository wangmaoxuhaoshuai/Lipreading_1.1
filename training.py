"""
训练器
"""
from import_sets import *
class Trainer():
    def __init__(self, dataset_file):
        self.training_dataset = ReadDataset(dataset_file)
        self.training_data_loader = DataLoader(
            self.training_dataset,
            batch_size = options["input"]["batchsize"],
            shuffle = options["input"]["shuffle"],
            drop_last = True)
        self.usecudnn = options["general"]["usecudnn"]
        self.batchsize = options["input"]["batchsize"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]
        self.learningrate = options["training"]["learningrate"]
        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]

    # 学习率衰减
    def learning_rate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    #
    def training(self, model, epochs):
        # set loss
        criterion = model.loss()
        # transfer the model to the GPU.
        if (self.usecudnn):
            criterion = criterion.cuda(self.gpuid)

        for epoch in range(epochs):
            optimizer = optim.SGD(
                model.parameters(),
                lr = self.learning_rate(epoch),
                momentum = self.learningrate,
                weight_decay = self.weightdecay)

            for i_batch, sample_batched in enumerate(self.training_data_loader):
                optimizer.zero_grad()
                # 图片序列
                input = Variable(sample_batched['temporalvolume'])
                # 标签
                labels = Variable(sample_batched['label'])

                if(self.usecudnn):
                    input = input.cuda(self.gpuid)
                    labels = labels.cuda(self.gpuid)

                outputs = model(input)
                loss = criterion(outputs, labels.squeeze(1))
                if i_batch % 10 == 0:
                    print("epoch : {} / {}, iteration : {}, loss : {}".format(
                        epoch, epochs, i_batch, loss))
                loss.backward()
                optimizer.step()

