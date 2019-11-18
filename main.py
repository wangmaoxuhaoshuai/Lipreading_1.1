# -*- coding: utf-8 -*-
from import_sets import *
"""
训练 + 测试
"""
if __name__ == '__main__':
    # options["dataset"]["train_dataset_annotation_file_set"] = 'data/splited_data/0/train.csv' # 训练集注释文档
    options["dataset"]["train_dataset_annotation_file_set"] = 'data/lip_train.csv'
    options["dataset"]["train_test_dataset_annotation_file_set"] = "data/splited_data/0/validate_train.csv" # 用于测试的训练集注释文档
    options["dataset"]["test_dataset_annotation_file_set"] = 'data/splited_data/0/validate.csv' # 测试集注释文档
    options["dataset"]["train_dataset_absolute_path"] = "/home/maggi/dataset/lip/crop_lips/lip_train" # 训练集路径
    options["dataset"]["test_dataset_absolute_path"] = "/home/maggi/dataset/lip/crop_lips/lip_test" # 测试集路径
    options["general"]["model_save_directory_path"] = "model_save" # 模型保存路径

    # 加载模型
    model = densenet201()
    if options["general"]["loadpretrainedmodel"]: # 判断是否加载预训练模型参数
        model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))

    # cudnn
    if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    # Move the model to the GPU.
    if (options["general"]["usecudnn"]):
        model = model.cuda(options["general"]["gpuid"])

    if options["training"]["train"]:
        trainer = Trainer(options["dataset"]["train_dataset_annotation_file_set"])
        validator_train = Validator(options["dataset"]["test_dataset_annotation_file_set"])
        validator_test = Validator(options["dataset"]["train_test_dataset_annotation_file_set"])
        for epoch in range(options["training"]["start_epochs"], options["training"]["end_epochs"] + 1):
            # 训练
            model.train()
            trainer.training(model, epoch, options["training"]["end_epochs"])

            if epoch % 10 == 0:
                if options["general"]["savemodel"]: # 保存模型
                    torch.save(model.state_dict(), r"{}/{}{}.pkl".format(options["general"]["model_save_directory_path"],
                                                                         options["model"]["type"], epoch))
                    with open(options["log"]["logs_train_path"], "a") as outputfile: # 保存模型写入日志
                        outputfile.write("\nsaving model '{}/{}{}.pkl, date: {}' ...".format(
                            options["general"]["model_save_directory_path"],
                            options["model"]["type"], epoch,
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                # 测试：测试集
                model.eval()
                with open(options["log"]["logs_accuracy_path"], "a") as outputfile: # 测试结果写入日志
                    outputfile.write("\nmodel: {}/{}{}.pkl, dataset: test_samples,".format(options["general"]["model_save_directory_path"],
                                                                    options["model"]["type"], epoch))
                validator_train.validating(model)

                # 测试：训练集
                with open(options["log"]["logs_accuracy_path"], "a") as outputfile: # 测试结果写入日志
                    outputfile.write("\nmodel: {}/{}{}.pkl, dataset: train_samples".format(options["general"]["model_save_directory_path"],
                                                                    options["model"]["type"], epoch))
                validator_test.validating(model)
























