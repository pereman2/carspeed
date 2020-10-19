import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from typing import List, Any
from rnn import Model
import torch
import numpy as np
import cv2
TOTAL_FRAMES = 20400
LIMIT_FRAMES = 1000

@hydra.main(config_name="config")
def my_app(cfg : DictConfig) -> None:
    batch_size = 400
    train_data = get_mp4_data(to_absolute_path(cfg.dataset.train_mp4))[:batch_size]
    train_labels = get_txt_data(to_absolute_path(cfg.dataset.train_txt))[:batch_size]
    train(train_data, train_labels, batch_size)

def train(x, labels, batch_size=400):
    output_size = 1
    n_epochs = 100
    sequence = 20 # 20 frames per second
    input_size = 640*480
    lr = 0.01
    input_seq = torch.Tensor(get_input_seq(x, batch_size, sequence))
    target_seq = torch.Tensor(labels)
    model = Model(input_size=input_size, output_size=output_size, hidden_dim=12, n_layers=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        # 800 outputs, 400 batch size * sequence
        output, hidden = model(input_seq)
     
def get_input_seq(x, batch_size, seq_len):
    input_seq = np.zeros((batch_size, seq_len, 640*480), dtype=np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            input_seq[i, u, x[i, u]] = 1
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
