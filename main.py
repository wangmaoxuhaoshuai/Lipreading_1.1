from import_sets import *
"""
训练 + 测试
"""
def training(model, datasetname):
    if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    # Move the model to the GPU.
    if (options["general"]["usecudnn"]):
        model = model.cuda(options["general"]["gpuid"])

    trainer = Trainer(datasetname)
    if options["training"]["train"]:
        trainer.training(model, options["training"]["epochs"])

def testing(model, datasetname):

    # Move the model to the GPU.
    if (options["general"]["usecudnn"]):
        model = model.cuda(options["general"]["gpuid"])

    validator = Validator(datasetname)
    validator.validating(model)

if __name__ == '__main__':
    options["dataset"]["train_dataset_annotation_file_set"] = [
        'data/splited_data/0/train.csv',
        'data/splited_data/1/train.csv',
        'data/splited_data/2/train.csv',
        'data/splited_data/3/train.csv',
        'data/splited_data/4/train.csv',
    ] # 训练集注释文档
    options["dataset"]["test_dataset_annotation_file_set"] = [
        'data/splited_data/0/validate.csv',
        'data/splited_data/1/validate.csv',
        'data/splited_data/2/validate.csv',
        'data/splited_data/3/validate.csv',
        'data/splited_data/4/validate.csv',
    ] # 测试集注释文档
    options["dataset"]["train_dataset_absolute_path"] = "/home/maggi/dataset_lipreading/lip_train" # 训练集路径
    options["dataset"]["test_dataset_absolute_path"] = "/home/maggi/dataset_lipreading/lip_test" # 测试集路径
    options["general"]["model_save_directory_path"] = "model_save" # 模型保存路径

    for i in range(len(options["dataset"]["train_dataset_annotation_file_set"])):
        if options["general"]["loadpretrainedmodel"]:
            model = torch.load(options["general"]["pretrainedmodelpath"][i])
        else:
            model = LipRead()

        training(model, options["dataset"]["train_dataset_annotation_file_set"][i])
        if options["general"]["savemodel"]:
            torch.save(model, r"{}/{}{}.pkl".format(options["general"]["model_save_directory_path"], options["model"]["type"], i))
            with open(options["log"]["logs_train_path"], "a") as outputfile:
                outputfile.write("\nsaving model '{}/{}{}.pkl, date: {}' ...".format(
                    options["general"]["model_save_directory_path"], options["model"]["type"], i, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        with open(options["log"]["logs_accuracy_path"], "a") as outputfile:
            outputfile.write("\nmodel: {}/{}{}.pkl,".format(options["general"]["model_save_directory_path"], options["model"]["type"], i))
        testing(model, options["dataset"]["test_dataset_annotation_file_set"][i])






















