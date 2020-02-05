import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from sklearn import model_selection, preprocessing


class Model(nn.Module):
    def __init__(self):
        """
        Sigmoid model results:
         - 2 neurons 88.56%
         - 3 neurons 94.57%
         - 4 neurons 96.04%
         - 7 neurons 96.92%
        """
        super().__init__()
        input_size = 992
        hidden_size = 7
        output_size = 1

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size)
        )

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)

    def forward(self, X):
        return self.net(X).flatten()


def train(epoch, net, trainloader, criterion, optimizer, acc_fn):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.flatten()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        correct += acc_fn(outputs, targets)

    print(f"E{epoch} Train Loss: {train_loss/(batch_idx+1):.3f} | "
          f"Acc: {100*correct/total:.3f}% ({correct}/{total})")

def test(epoch, net, testloader, criterion, acc_fn):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.flatten()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            correct += acc_fn(outputs, targets)

        print(f"E{epoch} Test  Loss: {test_loss/(batch_idx+1):.3f} | "
              f"Acc: {100*correct/total:.3f}% ({correct}/{total})")

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc


def load_mat_data(fname):
    import scipy.io
    ddict = scipy.io.loadmat(fname)
    X = np.asarray(ddict["X"], dtype=np.float64)
    Y = np.asarray(ddict["Y"], dtype=np.float64).ravel()

    return X, Y

def l2_accuracy(outputs, targets):
    return outputs.sign().eq(targets.sign()).sum().item()

def ce_accuracy(outputs, targets):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    return outputs.eq(targets).sum().item()


## Generic Hyperparameters
subsample = None
loss = "l2"

## Load Data
data_location = "./run_all.mat"
X, Y = load_mat_data(data_location)
subsample = subsample or X.shape[0]
Y = Y.ravel()[:subsample]
X = X[:subsample]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=37)
del X, Y

## Preprocessing
scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True)
scaler.fit(X_train, Y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
if loss == "l2":
    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1

## Create Tensor Datasets
bsize = 128
train_dset = data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
train_loader = data.DataLoader(train_dset, shuffle=True, batch_size=bsize)
test_dset = data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
test_loader = data.DataLoader(test_dset, shuffle=True, batch_size=bsize)

## Create model and optimizer
net = Model()
if loss == "l2":
    criterion = nn.MSELoss()
    acc_fn = l2_accuracy
else:
    criterion = nn.BCEWithLogitsLoss()
    acc_fn = ce_accuracy

optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=0)
start_epoch = 0
best_acc = 0

for epoch in range(start_epoch, start_epoch + 100):
    train(epoch, net, train_loader, criterion, optimizer, acc_fn)
    test(epoch, net, test_loader, criterion, acc_fn)

print(f"Best test accuracy: {best_acc:.3f}")

# Extract model weights and save them to "weights.mat"
l0_weights = net.net[0].weight.detach().numpy()
l1_weights = net.net[2].weight.detach().numpy()
import scipy.io
ddict = scipy.io.savemat("nn_weights.mat", {"W0": l0_weights, "W1": l1_weights})
