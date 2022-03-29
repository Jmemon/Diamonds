import sys
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD

from DiamondDataset2 import DiamondDataset2
from Model import PricePredictor


# from: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
def train_one_epoch(training_loader: DataLoader,
                    model: nn.Module,
                    loss_fn: nn.Module,
                    optimizer: SGD,
                    epoch_index: int,
                    tb_writer: SummaryWriter):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Get current batch and labels
        inputs, labels = data

        # Zero out the gradients for this batch
        optimizer.zero_grad()

        # Get predictions
        pred = model(inputs)

        # Get loss and gradients
        loss = loss_fn(pred, labels)
        loss.backward()

        # Adjust weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 20
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
            running_loss = 0.

        return last_loss


if __name__ == '__main__':
    diamonds = DiamondDataset2()

    train_len = int(0.7 * len(diamonds))
    train, test = random_split(diamonds, (train_len, len(diamonds) - train_len))

    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=128, shuffle=True)

    model = PricePredictor()

    loss = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=3e-4, momentum=0.9)

    epochs = 2
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/price_predictor_{timestamp}')

    best_loss = sys.maxsize

    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}')

        # Turn gradient tracking on
        model.train(True)
        avg_loss = train_one_epoch(training_loader=train_dl,
                                   model=model,
                                   loss_fn=loss,
                                   optimizer=optimizer,
                                   epoch_index=epoch,
                                   tb_writer=writer)

        # Don't need gradient tracking for reporting
        model.train(False)

        print(f'LOSS train {avg_loss}')

        # Log the running loss averaged per batch
        writer.add_scalars('Training Loss', {'Training': avg_loss}, epoch + 1)
        writer.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = f'models/model_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
