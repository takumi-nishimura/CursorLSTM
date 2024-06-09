import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


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
        return out


def evaluate_model(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(model.device)
            y = y.to(model.device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            predictions = torch.sigmoid(outputs)
            predicted = (predictions > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()

        accuracy = correct / total
        loss = total_loss / len(dataloader)
    return accuracy, loss


INPUT_SIZE = 2
HIDDEN_SIZE = 128
OUTPUT_SIZE = 3
SEQ_LEN = 500
PREDICT_LEN = 50
NUM_LAYERS = 2
NUM_EPOCHS = 300

if __name__ == "__main__":
    data = np.loadtxt("data/record_data_0608.csv", delimiter=",")
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    print("Finished Training")
    torch.save(model.state_dict(), "model/cursorlstm_ubuntu.pth")

    accuracy, test_loss = evaluate_model(model, test_dataloader, criterion)
    print(f"Test Accuracy: {accuracy}, Test Loss: {test_loss}")
