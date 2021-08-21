import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import wandb
from tqdm import tqdm

PROJECT_NAME = "GameStop-Stock-Price-Prediction"


def train(epochs, optimizer, data_input, model, data_target, criterion, name):
    wandb.init(project=PROJECT_NAME, name=name)
    epochs_iter = tqdm(range(epochs))
    for _ in epochs_iter:

        def closure():
            optimizer.zero_grad()
            preds = model(data_input.float())
            loss = criterion(preds, data_target)
            wandb.log({"Loss": loss.item()})
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            future = 100
            preds = model(data_input.float(), future=future)
            loss = criterion(preds[:, :-future], data_target)
            wandb.log({"Val Loss": loss.item()})
            preds = preds.cpu().detach().numpy()
        plt.figure(figsize=(12, 6))
        n = data_input.shape[1]

        def draw(y_i, color):
            plt.plot(np.arange(n), data_target.cpu().view(-1), color)
            plt.plot(np.arange(n, n + future), y_i[n:], color + ":")

        draw(preds[0], "r")
        plt.savefig("./preds/img.png")
        plt.close()
        wandb.log({"Img": wandb.Image(cv2.imread("./preds/img.png"))})
        epochs_iter.set_description(f"{preds[:5]}")
    wandb.finish()
    torch.save(model, f"./models/{name}.pt")
    torch.save(model, f"./models/{name}.pth")
    return model
