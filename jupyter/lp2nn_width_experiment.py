
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import schedulefree

import sys

sys.path.append("../auto_LiRPA/")
import auto_LiRPA
from auto_LiRPA.operators.gurobi_maxpool_lp import compute_maxpool_bias

import time


def normalize_bounds(l, u):
    """
    Takes bounds tensors and normalizes them to [0, 1], s.t. smallest lower bound is mapped to 0
    and largest upper bound is mapped to 1

    args:
        l (batch x channels x w x h) - concrete lower bounds
        u (batch x channels x w x h) - concrete upper bounds

    returns:
        l_norm (batch x channels x w x h) - normalized concrete lower bounds
        u_norm (batch x channels x w x h) - normalized concrete upper bounds
    """
    lmin = l.flatten(-2).min(dim=-1)[0]
    umax = u.flatten(-2).max(dim=-1)[0]
    lmin = lmin.unsqueeze(1)
    umax = umax.unsqueeze(1)

    l_norm = (l.flatten(-2) - lmin) / (umax - lmin)
    u_norm = (u.flatten(-2) - lmin) / (umax - lmin)
    l_norm = l_norm.view(l.shape)
    u_norm = u_norm.view(u.shape)

    return l_norm, u_norm


def sort_by_lower_bound(X):
    """
    Sorts tensor of shape (n_neurons, 3, w, h) by concrete lower bounds (the first channel dim).
    """
    _, ind_tensor = X.flatten(-2)[:,0].sort(dim=-1)
    ind_tensor = ind_tensor.unsqueeze(1).expand(-1, X.size(1), -1)

    return torch.gather(X.flatten(-2), dim=2, index=ind_tensor).view(X.shape)


def create_dataset(n_neurons, h, w):
    # since the normalized version suffices, just stick to that
    x1 = torch.rand(n_neurons, 1, h, w)
    x2 = torch.rand(n_neurons, 1, h, w)

    l = torch.where(x1 <= x2, x1, x2)
    u = torch.where(x1  > x2, x1, x2)
    l, u = normalize_bounds(l, u)

    alpha = torch.rand(n_neurons, 1, h, w)

    biases = compute_maxpool_bias(l, u, alpha)

    return l, u, alpha, biases


def create_tensor_dataset(n_neurons_train, n_neurons_val, h, w, sort_by_lb=True):
    l, u, alpha, bias = create_dataset(n_neurons_train, h, w)
    X = torch.cat((l, u, alpha), dim=1)

    if sort_by_lb:
        X = sort_by_lower_bound(X)

    dataset_train = TensorDataset(X, bias)

    l, u, alpha, bias = create_dataset(n_neurons_val, h, w)
    X = torch.cat((l, u, alpha), dim=1)

    if sort_by_lb:
        X = sort_by_lower_bound(X)
        
    dataset_val = TensorDataset(X, bias)

    return dataset_train, dataset_val


class MyNet(nn.Module):
    def __init__(self, n_neurons, h, w, n_features=50):
        super(MyNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*h*w, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, n_neurons)
        self.fc6 = nn.Linear(n_neurons, n_features)
        self.fc7 = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x_features = self.fc6(x)
        x = F.relu(x_features)
        x = self.fc7(x)
        return x, x_features


def train_loop(net, train_dataloader, val_dataloader, patience=10, num_epochs=100, timeout=60, lossfun='mse', opt='adam', l1_weight=0):
    if lossfun == 'mse':
        criterion = nn.MSELoss()
    elif lossfun == 'mae':
        criterion = nn.L1Loss()
    else:
        raise ValueError('Unknown loss function!')
    
    if opt == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif opt == 'schedulefree':
        optimizer = schedulefree.AdamWScheduleFree(net.parameters(), lr=0.0025)
    else:
        raise ValueError('Uknown optimizer!')


    train_losses = []
    train_maes = []
    val_losses = []
    val_maes = []
    val_maxs = []
    best_val_loss = float('inf')
    early_stopping_cnt = 0
    t_start = time.time()
    for epoch in range(num_epochs):
        t_cur = time.time()
        if t_cur - t_start > timeout:
            print(f"Timeout reached ({t_cur - t_start} sec)")
            break 
        
        net.train()

        if opt == 'schedulefree':
            optimizer.train()

        train_loss = 0.
        train_mae = 0.
        for batch_X, batch_y in train_dataloader:
            y_hat, y_features = net(batch_X)
            loss = criterion(y_hat, batch_y)

            l1_loss = 0
            for param in net.parameters():
                l1_loss += param.abs().sum()

            loss += l1_weight * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += torch.abs(y_hat - batch_y).mean().item()

        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataloader)
        train_losses.append(train_loss)
        train_maes.append(train_mae)


        net.eval()

        if opt == 'schedulefree':
            optimizer.eval()
            
        val_loss = 0
        val_mae = 0
        val_max = torch.tensor(0)
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                y_hat, y_features = net(batch_X)
                loss = criterion(y_hat, batch_y)
                val_loss += loss.item()
                val_mae += torch.abs(y_hat - batch_y).mean().item()
                val_max = torch.maximum(val_max, torch.max(torch.abs(y_hat - batch_y)))

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_maxs.append(val_max.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}] - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, train_mae: {train_mae:.6f}, val_mae: {val_mae:.6f}, val_max: {val_max.item():.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_cnt = 0
            best_net_state = net.state_dict()
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt >= patience:
                print(f"Stopping early (patience of {patience} reached)")
                break


    print("Training completed")
    return train_losses, val_losses, train_maes, val_maes, val_maxs, best_net_state


if __name__ == "__main__":
    CREATE_DATASET = False
    train_size = 1_000_000
    val_size = 100_000
    batch_size = 32
    h = 2
    w = 2
    widths = [30, 50, 100, 150, 200, 250]
    timeout = 7200
    patience = 20
    num_epochs = sys.maxsize

    print(f"--- running experiment for (h, w) = {(h, w)} and timeout of {timeout} sec")

    if CREATE_DATASET:
        print(f"-- creating dataset with {train_size} training and {val_size} validation samples") 
        ds_train, ds_val = create_tensor_dataset(train_size, val_size, h, w)
        torch.save(ds_train, './datasets/maxpool2x2_train_clean.pth')
        torch.save(ds_val, './datasets/maxpool2x2_val_clean.pth')
    else:
        print("-- loading dataset")
        ds_train = torch.load('./datasets/maxpool2x2_train_clean.pth')
        ds_val   = torch.load('./datasets/maxpool2x2_val_clean.pth')


    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(ds_val, batch_size=batch_size)

    for n_neurons in widths:
        netname = f"net6x{n_neurons}_width"
        print(f"\n######## {netname} ########\n")
        net = MyNet(n_neurons, h, w)
        train_losses, val_losses, train_maes, val_maes, val_maxs, best_state = train_loop(net, train_dataloader, val_dataloader,
                                                                                          timeout=timeout, patience=patience, num_epochs=num_epochs, 
                                                                                          l1_weight=0.)
        torch.save(best_state, f'./width/{netname}_best_state.pth')
        torch.save(train_losses, f'./width/{netname}_train_losses.pth')
        torch.save(val_losses, f'./width/{netname}_val_losses.pth')
        torch.save(train_maes, f'./width/{netname}_train_maes.pth')
        torch.save(val_maes, f'./width/{netname}_val_maes.pth')
        torch.save(val_maxs, f'./width/{netname}_val_maxs.pth')
