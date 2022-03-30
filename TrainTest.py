import sys
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torchvision.transforms as T

from DiamondDataset2 import DiamondDataset2
from Model import PricePredictor


TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 128


def show_sample(img: torch.Tensor, weight: float, price: float):
    transform = T.ToPILImage()
    img = transform(img)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.text(50, 75, f'Weight: {str(weight)}')
    plt.text(275, 75, f'Price: {str(price)}')
    plt.show()


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
        inputs, labels = data  # inputs is (img, weight), labels is the labels without weight

        # Zero out the gradients for this batch
        optimizer.zero_grad()

        # Get predictions
        pred = model(*inputs)

        # Get loss and gradients
        loss = loss_fn(pred, torch.reshape(labels[8], pred.size()))
        loss.backward()

        # Adjust weights
        optimizer.step()

        # Gather data and report
        last_loss = loss.item()
        print(f'  Batch {i + 1}:')
        print(f'    Bch Loss: {last_loss}')
        print(f'    Avg Loss: {last_loss / TRAIN_BATCH_SIZE}')
        print(f'    Act / Exp  [0]: {pred[0].item()} / {labels[8][0].item()}')
        print(f'    Act / Exp [10]: {pred[10].item()} / {labels[8][10].item()}')
        print(f'    Act / Exp [20]: {pred[20].item()} / {labels[8][20].item()}')
        print(f'    Act / Exp [30]: {pred[30].item()} / {labels[8][30].item()}')

        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
        tb_writer.add_scalar('Avg Loss/Train', last_loss / TRAIN_BATCH_SIZE, tb_x)

        running_loss += last_loss

    return running_loss / len(train_dl)


if __name__ == '__main__':
    diamonds = DiamondDataset2()

    train_len = int(0.7 * len(diamonds))
    train, test = random_split(diamonds, (train_len, len(diamonds) - train_len))

    print('Preprocessing Complete.')

    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=128, shuffle=True)

    model = PricePredictor()

    loss = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)

    epochs = 2
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/price_predictor_{timestamp}')

    best_loss = sys.maxsize

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}:')

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
