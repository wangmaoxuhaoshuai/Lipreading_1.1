from import_sets import *
"""
测试
"""
if __name__ == '__main__':
    def learning_rate(learningrate, epoch):
        decay = math.floor((epoch - 1) / 5)
        return learningrate * pow(0.5, decay)

    epochs = 80
    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0:
            print("epoch : {} / {}, learning rate : {}".format(epoch, epochs, learning_rate(0.1, epoch)))

    for epoch in range(70, 151):
        if epoch % 10 == 0:
            print("epoch : {} / {}, learning rate : {}".format(epoch, 151, learning_rate(0.1, epoch)))

