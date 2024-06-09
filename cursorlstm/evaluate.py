import numpy as np
import torch
from learn import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_LAYERS,
    OUTPUT_SIZE,
    PREDICT_LEN,
    SEQ_LEN,
    TRAIN_RATIO,
    CursorDataset,
    CursorLSTM,
    evaluate_model,
)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, random_split

data = np.loadtxt("data/record_data_0608.csv", delimiter=",")
dataset = CursorDataset(data, SEQ_LEN, PREDICT_LEN)
train_size = int(TRAIN_RATIO * len(dataset))
_, test_dataset = random_split(
    dataset, [train_size, len(dataset) - train_size]
)
test_dataloader = DataLoader(test_dataset, 16, shuffle=False)

model = CursorLSTM(
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, PREDICT_LEN
)
model.load_state_dict(torch.load("model/cursorlstm.pth"))
model.to(model.device)
criterion = BCEWithLogitsLoss()

accuracy, loss = evaluate_model(model, test_dataloader, criterion)
print(f"Accuracy: {accuracy}, Loss: {loss}")
