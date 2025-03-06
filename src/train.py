from torchmetrics.classification import MulticlassF1Score
from typing import Union
import torch.nn.functional as F
import torch
from tqdm import tqdm
import time
import wandb

from config.configs import Config
from src.model.model import Model
from tools.sample import generate_text


def train(config: Config, model: Union[Model, torch.nn.Module], dataloaders:dict) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.Optimizer.learning_rate,  weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    f1_metric = MulticlassF1Score(num_classes=config.DataLoader.N_letters).to(model.device)

    since = time.time()
    best_acc = 0.0

    for ep in range(config.Training.epochs):
        print(f'Epoch {ep}/{config.Training.epochs - 1}')
        print('-' * 10)
        f1_metric.reset()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_num_correct = 0
            f1_metric.reset()

            for x, y in tqdm(dataloaders[phase]):
                x, y = x.to(model.device), y.to(model.device)
                hidden = model.model.init_hidden(batch_size=x.size(0), device=model.device)
                pred, hidden = model.forward(x, hidden)
                f1_metric.update(pred.view(-1, pred.shape[-1]), y.view(-1))
                loss = criterion(pred.view(-1, pred.shape[-1]), y.view(-1))

                wandb.log({"Loss": loss.item()})

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                running_loss += loss.item()
                _, predicted_classes = torch.max(pred, dim=-1)
                running_num_correct += torch.sum(predicted_classes.view(-1) == y.view(-1)).item()                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_num_correct / len(dataloaders[phase].dataset)
            epoch_f1 = f1_metric.compute().item()

            if phase == 'train':
                wandb.log(
                    {
                        "train Epoch": ep,
                        "train Loss": epoch_loss,
                        "train Accuracy": epoch_acc,
                        "train F1 Score": epoch_f1,
                        "train number of correct": running_num_correct,
                    }
                )
            if phase == 'val':
                wandb.log(
                    {
                        "val Epoch": ep,
                        "val Loss": epoch_loss,
                        "val Accuracy": epoch_acc,
                        "val F1 Score": epoch_f1,
                        "val number of correct": running_num_correct,
                    }
                )

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Time: {time.time() - since}')

            if phase == 'val':
                start_char = torch.randint(0, config.DataLoader.N_letters, (1,)).item()
                generated_text = generate_text(config, 
                                               model, 
                                               start_char)
                print(f"Generated Text Sample (Epoch {ep}):\n{generated_text}\n")
                wandb.log({f"{phase} Generated Text": generated_text})

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    model.save()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load()
    model.export_to_onnx('model/')

    return model
