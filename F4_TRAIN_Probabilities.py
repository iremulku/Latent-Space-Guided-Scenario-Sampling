from __future__ import print_function
import os
import json
import torch
import torch.nn as nn
import numpy as np
from CMX import CMX 
from F5_JACCARD2 import Jaccard2
from F9_UNET_V2_3 import UNetV2
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5

# 3-modal (RGB, NIR, SWIR) için mask array
MASK_ARRAY = torch.tensor([
    [1, 0, 0],  # RGB only
    [0, 1, 0],  # NIR only
    [0, 0, 1],  # SWIR only
    [1, 1, 0],  # RGB + NIR
    [1, 0, 1],  # RGB + SWIR
    [0, 1, 1],  # NIR + SWIR
    [1, 1, 1],  # RGB + NIR + SWIR
], dtype=torch.float32)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)


def load_scenario_probabilities(prob_json_path):
    if prob_json_path is None:
        return None

    if not os.path.exists(prob_json_path):
        raise FileNotFoundError(f"Probability json not found: {prob_json_path}")

    with open(prob_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "final_probs" not in data:
        raise KeyError("JSON file must contain 'final_probs'.")

    probs = torch.tensor(data["final_probs"], dtype=torch.float32)

    if probs.numel() != MASK_ARRAY.shape[0]:
        raise ValueError(
            f"Expected {MASK_ARRAY.shape[0]} probabilities, got {probs.numel()}."
        )

    probs = probs / probs.sum()
    return probs


def sample_modality_mask(batch_size, device, scenario_probs=None):
    mask_array = MASK_ARRAY.to(device)

    if scenario_probs is None:
        # Uniform sampling
        idx = torch.randint(0, mask_array.shape[0], (batch_size,), device=device)
    else:
        # Probability-based sampling
        idx = torch.multinomial(
            scenario_probs.to(device),
            num_samples=batch_size,
            replacement=True
        )

    mod_mask = mask_array[idx]
    return mod_mask, idx


def build_model_for_validation(modeltype):
    if modeltype == 'UNetV2':
        model = UNetV2(classes=1).to(device)
    elif modeltype == 'MMVit4':
        model = MMVit4().to(device)
    elif modeltype == 'MMVit5':
        model = MMVit5().to(device)  
    elif modeltype=='CMX':              
        model = CMX(num_classes=1, decoder_embed_dim=256, random_modality_dropout=False).to(device) 
    else:
        raise ValueError(f"Unsupported modeltype in validation: {modeltype}")

    return model


def train_model(
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
    prob_json_path=None
):
    training_losses = []

    scenario_probs = load_scenario_probabilities(prob_json_path)

    if scenario_probs is not None:
        print("Using learned scenario probabilities:")
        print(scenario_probs.tolist())
        lrFile.write("Using learned scenario probabilities:\n")
        lrFile.write(str(scenario_probs.tolist()) + "\n")
    else:
        print("Using uniform random scenario sampling.")
        lrFile.write("Using uniform random scenario sampling.\n")

    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0

        scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        lrFile.write('Epoch: ' + str(epoch) + ' LR: ' + str(scheduler.get_lr()) + "\n")
        lrFile.write(str(scheduler.state_dict()) + "\n")

        for trainim, trainmas in training_generator:
            optim.zero_grad()

            images = trainim.to(device)
            masks = trainmas.to(device)
            B = images.size(0)

            # Probability-based ya da uniform scenario sampling
            mod_mask, sampled_idx = sample_modality_mask(
                batch_size=B,
                device=images.device,
                scenario_probs=scenario_probs
            )

            if modeltype in ['MMVit4', 'MMVit5', 'CMX']:
                outputs = model(images, mask=mod_mask)
            else:
                outputs = model(images)

            if trainloss == 'BCEWithLogitsLoss':
                loss = nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
            else:
                raise ValueError(f"Unsupported training loss: {trainloss}")

            output.backward()
            optim.step()

            batch_losses.append(output.item())
            batchLoad = len(masks) * lim * lim
            totalBatches += batchLoad

            if accuracy == 'Jaccard':
                masks_eval = masks[:, 0, ...]
                outputs_eval = outputs[:, 0, ...]
                thisJac = Jaccard2(
                    torch.reshape(masks_eval, (batchLoad, 1)),
                    torch.reshape(outputs_eval, (batchLoad, 1))
                ) * batchLoad
                jI = jI + thisJac.data[0]

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        trainFile.write(str(training_losses[epoch]) + "\n")
        trainaccFile.write(str((jI / totalBatches).item()) + "\n")
        trainepochFile.write(str(epoch) + "\n")

        print("Training Jaccard:", (jI / totalBatches).item(), " (epoch:", epoch, ")")
        lrFile.write("Training loss:" + str(training_losses[epoch]) + "\n")
        lrFile.write("Training accuracy:" + str((jI / totalBatches).item()) + "\n")

        torch.save(model.state_dict(), os.path.join(pathm, "iremmodel{}.pt".format(i)))
        validate(
            validationloss,
            accuracy,
            validation_generator,
            valFile,
            valaccFile,
            lim,
            lrFile,
            pathm,
            i,
            modeltype
        )

    torch.save(model.state_dict(), os.path.join(pathm, "FinaliremmodelLoRA.pt"))


def validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype):
    jI = 0
    totalBatches = 0
    validation_losses = []

    model = build_model_for_validation(modeltype)
    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i)), map_location=device))
    model.eval()

    with torch.no_grad():
        val_losses = []
        for valim, valmas in validation_generator:
            images = valim.to(device)
            masks = valmas.to(device)

            B = images.size(0)

            if modeltype in ['MMVit4', 'MMVit5', 'CMX']:
                mod_mask = torch.ones(B, 3, device=images.device)
                outputs = model(images, mask=mod_mask)
            else:
                outputs = model(images)

            if validationloss == 'BCEWithLogitsLoss':
                loss = nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
            else:
                raise ValueError(f"Unsupported validation loss: {validationloss}")

            val_losses.append(output.item())
            batchLoad = len(masks) * lim * lim
            totalBatches += batchLoad

            if accuracy == 'Jaccard':
                masks_eval = masks[:, 0, ...]
                outputs_eval = outputs[:, 0, ...]
                thisJac = Jaccard2(
                    torch.reshape(masks_eval, (batchLoad, 1)),
                    torch.reshape(outputs_eval, (batchLoad, 1))
                ) * batchLoad
                jI = jI + thisJac.data[0]

    dn = jI / totalBatches
    dni = dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)

    valFile.write(str(validation_losses[0]) + "\n")
    valaccFile.write(str(dni) + "\n")
    print("Validation Jaccard:", dni)
    lrFile.write("Validation loss:" + str(validation_losses[0]) + "\n")
    lrFile.write("Validation accuracy:" + str(dni) + "\n")