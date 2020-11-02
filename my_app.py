import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from typing import List, Any
from rnn import Model
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import cv2
TOTAL_FRAMES = 20400
LIMIT_FRAMES = 1000

@hydra.main(config_name="config")
def my_app(cfg : DictConfig) -> None:
    batch_size = 400
    train_data = get_mp4_data(to_absolute_path(cfg.dataset.train_mp4))
    train_labels = get_txt_data(to_absolute_path(cfg.dataset.train_txt))
    train_data = train_data[:batch_size]
    train_labels = train_labels[:batch_size]
    train(train_data, train_labels, batch_size)

def train(x, labels, batch_size=400):
    output_size = 1
    n_epochs = 1000
    sequence = 1 # 20 frames per second
    input_size = 640*480
    lr = 0.01
    batch_size = 400
    x = x.reshape((batch_size, input_size))
    input_seq = torch.Tensor(get_input_seq(x, batch_size, sequence))
    target_seq = torch.Tensor(labels)
    model = Model(input_size=input_size, output_size=output_size, hidden_dim=50, n_layers=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        # 8000 outputs, 400 batch size * sequence
        output, hidden = model(input_seq)
        output = output.reshape((400))
        loss = criterion(output, target_seq.view(-1).float())
        loss.backward()
        optimizer.step()
        accuracy = (output == target_seq).sum()
        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
     
def get_input_seq(x, batch_size, seq_len):
    input_seq = np.zeros((seq_len, batch_size, 640*480), dtype=np.float32)

    for u in range(seq_len):
        for i in range(batch_size):
            input_seq[u, i] = x[u, i]
    return input_seq

    
def get_txt_data(path: str) -> List[Any]:
    data = []
    with open(path) as f:
        line = float(f.readline().strip("\n"))
        i = 0
        while line:
            data.append(line)
            line = float(f.readline().strip("\n"))
            i += 1
            if i == LIMIT_FRAMES:
                break
            print("Loading label " + str((i/LIMIT_FRAMES)*100) + "%", end="\r")
    return np.array(data)
        
def get_mp4_data(path: str) -> List[Any]:
    data = []
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ret, frame = cap.read()
        if frame is None: 
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data.append(gray)
        i += 1
        if i == LIMIT_FRAMES:
            break
        print("Loading video frames " + str(int((i/LIMIT_FRAMES)*100)) + "%", end="\r")
    cap.release()
    return np.array(data)

    
if __name__ == "__main__":
    my_app()
