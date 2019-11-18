"""
测试
"""
from import_sets import *

class Validator():
    def __init__(self, dataset_file):

        self.validation_dataset = ReadDataset(dataset_file)
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size = options["input"]["batchsize"],
            shuffle = options["input"]["shuffle"],
        )
        self.usecudnn = options["general"]["usecudnn"]
        self.batchsize = options["input"]["batchsize"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.gpuid = options["general"]["gpuid"]

    def validating(self, model):
        print("Starting validation...")
        count = 0
        validator_function = model.validator_function()

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.validation_dataloader):
                input = Variable(sample_batched['temporalvolume'])
                labels = sample_batched['label']

                if (self.usecudnn):
                    input = input.cuda(self.gpuid)
                    labels = labels.cuda(self.gpuid)

                outputs = model(input)
                print("test {} / {} samples".format(i_batch * self.batchsize, len(self.validation_dataset)))

                count += validator_function(outputs, labels)

            total_count = len(self.validation_dataset)
            accuracy = count / total_count
            with open(options["log"]["logs_accuracy_path"], "a") as outputfile:
                outputfile.write(" correct count: {}, total count: {}, accuracy: {}, date: {}, ".format(
                    count, total_count, accuracy, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))



