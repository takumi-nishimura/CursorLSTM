import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class CursorDataset(Dataset):
    def __init__(self, data, seq_len=50, predict_len=50):
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.data = data

        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.predict_len + 1

    def __getitem__(self, idx):
        return (
            self.data[:, 0:2][idx : idx + self.seq_len],
            self.data[:,][
                idx + self.seq_len : idx + self.seq_len + self.predict_len
            ],
        )


class CursorLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, predict_len=50
    ):
        super(CursorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.predict_len = predict_len
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -self.predict_len :])
        out = torch.sigmoid(out)
        return out


def evaluate_model(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total = 0
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(model.device)
            y = y.to(model.device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        loss = total_loss / total
    return loss


INPUT_SIZE = 2
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3
SEQ_LEN = 100
PREDICT_LEN = 30
NUM_LAYERS = 3
NUM_EPOCHS = 300

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(f"runs/cursorlstm")
    data = np.loadtxt("data/record_data_0610.csv", delimiter=",")
    _data = np.loadtxt("data/record_data_0608.csv", delimiter=",")
    data = np.concatenate([data, _data])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    train_data, test_data = train_test_split(
        data, test_size=0.2, shuffle=False
    )
    train_dataset = CursorDataset(train_data, SEQ_LEN, PREDICT_LEN)
    test_dataset = CursorDataset(test_data, SEQ_LEN, PREDICT_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CursorLSTM(
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, PREDICT_LEN
    )
    model.to(model.device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    best_loss = float("inf")
    best_model = None
    patience = 10
    counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(model.device)
            y = y.to(model.device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}"
                )

        test_loss = evaluate_model(model, test_dataloader, criterion)
        writer.add_scalar("loss", test_loss, epoch + 1)

        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            best_model = model.state_dict()
            scheduler.step(test_loss)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    print("Finished Training")
    torch.save(best_model, f"model/cursorlstm_ubuntu_{date}.pth")
