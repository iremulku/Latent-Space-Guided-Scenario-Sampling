from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from F3_DATASET import satellitedata
from F9_UNET_V2_3 import UNetV2
from F4_TRAIN_Probabilities import train_model
#from F4_TRAIN import train_model
from F6_CROSSVAL import CrossVal
from F8_IMAGES4 import get_images4
from F7_TEST2 import test_model
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
from CMX import CMX 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#This is the first version. And correct version!
if __name__ == '__main__':

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    bg = datetime.datetime.now()
    bgh = bg.hour
    bgm = bg.minute

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)

    # ---------------------------------------------------------
    # Probability JSON path
    # ---------------------------------------------------------
    
    prob_json_path = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\DSTL\Latentfactorizaton\2025_12_8_9_54_model0\dstl_scenario_probabilities.json"
    #prob_json_path = r"C:\Users\Public\Server\experiments\2026_3_15_10_21_model0\dstl_scenario_probabilities.json"
    #prob_json_path = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\DSTL\Others_RandomModalityDropout\2026_1_23_10_27_model0\dstl_scenario_probabilities_cmx.json"

    # ---------------------------------------------------------
    # Pretrained model path
    # Bu model, probabilityGenerator içinde de kullanılan
    # önceden eğitilmiş MMVit4 checkpoint’i olmalı.
    # ---------------------------------------------------------
    pretrained_model_path = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\DSTL\Latentfactorizaton\2025_12_8_9_54_model0\FinaliremmodelLoRA.pt"
    #pretrained_model_path = r"C:\Users\Public\Server\experiments\2026_3_15_10_21_model0\FinaliremmodelLoRA.pt"
    #pretrained_model_path = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\DSTL\Others_RandomModalityDropout\2026_1_23_10_27_model0\FinaliremmodelLoRA.pt"
    

    for i in range(0, 5):
        data_folder = os.path.join("../../experiments")
        file_to_open = os.path.join(data_folder, "model{}.txt".format(i))

        with open(file_to_open) as f:
            lines = [line.rstrip() for line in f]

        trainSetSize = int(lines[0])
        fno = int(lines[1])
        fsiz = int(lines[2])
        valRatio = float(lines[3])
        miniBatchSize = int(lines[4])
        n_epochs = int(lines[5])
        learnRate = float(lines[6])
        optimizerType = str(lines[7])
        trainloss = str(lines[8])
        validationloss = str(lines[9])
        accuracy = str(lines[10])
        initialization = str(lines[11])
        step_size = int(lines[12])
        gamma = float(lines[13])
        lim = int(lines[14])
        modeltype = str(lines[15])
        chindex = str(lines[16])
        transfertype = str(lines[17])

        tsind, trind, vlind = CrossVal(trainSetSize, fno, fsiz)

        input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(
            trainSetSize, fno, fsiz, tsind, trind, vlind, chindex
        )

        params = {'batch_size': miniBatchSize, 'shuffle': False}

        transformations = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.95, 1.05)),
        ])

        training_set = satellitedata(input_images[trind], target_masks[trind], transform=None)
        training_generator = DataLoader(training_set, **params)

        validation_set = satellitedata(input_images[vlind], target_masks[vlind])
        validation_generator = DataLoader(validation_set, **params)

        test_set = satellitedata(input_images[tsind], target_masks[tsind])
        test_generator = DataLoader(test_set, **params)

        # ---------------------------------------------------------
        # Model creation
        # ---------------------------------------------------------
        if modeltype == 'UNetV2':
            model = UNetV2(classes=1).to(device)

        elif modeltype == 'MMVit4':
            model = MMVit4().to(device)
            
        elif modeltype == 'MMVit5':
            model = MMVit5().to(device)   
            
        elif modeltype=='CMX':              
            model = CMX(num_classes=1, decoder_embed_dim=256, random_modality_dropout=False).to(device) 


# model.load_state_dict(torch.load(os.path.join(modelPath, modelName)))
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print("Loaded pretrained model from:")
        print(pretrained_model_path)
        print("irem")


        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------
        if optimizerType == 'Adam':
            optim = torch.optim.Adam(model.parameters(), learnRate)
        elif optimizerType == 'SGD':
            optim = torch.optim.SGD(model.parameters(), learnRate)
        else:
            raise ValueError(f"Unsupported optimizerType: {optimizerType}")

        scheduler = StepLR(optim, step_size, gamma)

        d = datetime.datetime.now()
        pathm = os.path.join(
            data_folder,
            "{}_{}_{}_{}_{}_model{}".format(d.year, d.month, d.day, d.hour, d.minute, i)
        )
        os.mkdir(pathm)

        lrFile = open("lrFile.txt", "w")
        trainaccFile = open("trainaccFile.txt", "w")
        valaccFile = open("valaccFile.txt", "w")
        trainepochFile = open("trainepochFile.txt", "w")
        trainFile = open("trainFile.txt", "w")
        valFile = open("valFile.txt", "w")

        train_model(
            n_epochs,
            trainloss,
            validationloss,
            accuracy,
            model,
            scheduler,
            lrFile,
            training_generator,
            optim,
            lim,
            trainFile,
            trainaccFile,
            trainepochFile,
            validation_generator,
            valFile,
            valaccFile,
            pathm,
            i,
            modeltype,
            prob_json_path=prob_json_path
        )
        
        # train_model(
        #     n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile,
        #     training_generator, optim, lim, trainFile, trainaccFile, trainepochFile,
        #     validation_generator, valFile, valaccFile, pathm, i, modeltype
        # )

        trainFile.close()
        valFile.close()
        trainaccFile.close()
        valaccFile.close()
        trainepochFile.close()
        lrFile.close()

        testaccFile = open("testaccFile.txt", "w")
        testFile = open("testFile.txt", "w")
        test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB)
        testFile.close()
        testaccFile.close()

        x = []
        with open("trainFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                x.append(float(l))

        y = []
        with open("valFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                y.append(float(l))

        tt = []
        with open("testFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                tt.append(float(l))

        z = []
        with open("lrFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                z.append(l)

        xx = []
        with open("trainaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                xx.append(float(l))

        yy = []
        with open("valaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                yy.append(float(l))

        ta = []
        with open("testaccFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                ta.append(float(l))

        e1 = []
        with open("trainepochFile.txt") as f:
            lines = f.readlines()
            for l in lines:
                e1.append(float(l))

        def logfile():
            a = datetime.datetime.now()
            myfile = os.path.join(pathm, "{}_{}_{}_{}_{}.txt".format(a.year, a.month, a.day, a.hour, a.minute))
            LogFile = open(myfile, "w")
            LogFile.write("Date:" + str(datetime.date.today()) + "\n")
            LogFile.write("Ending Time:" + str(a.hour) + ":" + str(a.minute) + "\n")
            LogFile.write("Starting Time:" + str(bgh) + ":" + str(bgm) + "\n")
            LogFile.write("Data set size:" + str(trainSetSize) + "\n")
            LogFile.write("Fold number:" + str(fno) + "\n")
            LogFile.write("Fold size:" + str(fsiz) + "\n")
            LogFile.write("Number of validation images:" + str(len(vlind)) + "\n")
            LogFile.write("Number of training images:" + str(len(trind)) + "\n")
            LogFile.write("Mini batch size:" + str(miniBatchSize) + "\n")
            LogFile.write("Test accuracy:" + str(ta) + "\n")
            LogFile.write("Learning rate:" + str(learnRate) + "\n")
            LogFile.write("Model version:" + str(modeltype) + "\n")
            LogFile.write("Optimizer type:" + str(optimizerType) + "\n")
            LogFile.write("Total number of epochs:" + str(n_epochs) + "\n")
            LogFile.write("Training loss function:" + str(trainloss) + "\n")
            LogFile.write("Validation loss function:" + str(validationloss) + "\n")
            LogFile.write("Accuracy function:" + str(accuracy) + "\n")
            LogFile.write("Channel index:" + str(chindex) + "\n")
            LogFile.write("Original transfer flag from txt:" + str(transfertype) + "\n")
            LogFile.write("Probability json path:" + str(prob_json_path) + "\n")
            LogFile.write("Pretrained model path:" + str(pretrained_model_path) + "\n")
            LogFile.write("Model Summary:" + "\n" + str(model) + "\n")
            for j in range(len(z)):
                LogFile.write(str(z[j]))
            LogFile.close()

        logfile()

        plt.plot(x, "k-", label="Train Loss")
        plt.plot(y, "r--", label="Validation Loss")
        plt.title('Learning Curves')
        plt.legend(loc="upper left")
        l_curve = 'learning_curves.png'
        plt.savefig(os.path.join(pathm, l_curve))
        plt.show()

        plt.plot(xx, "k-", label="Train Accuracy")
        plt.plot(yy, "r--", label="Validation Accuracy")
        plt.title('Accuracy Curves')
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        a_curve = 'accuracy_curves.png'
        plt.savefig(os.path.join(pathm, a_curve), bbox_inches="tight")
        plt.show()

        print("Memory allocated before model {}".format(i), torch.cuda.memory_allocated())
        del model
        torch.cuda.empty_cache()
        print("Memory allocated after model {}".format(i), torch.cuda.memory_allocated())