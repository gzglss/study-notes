import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as dataset

mb_size = 64
Z_dim = 100
X_dim = 0
# y_dim = 0
h_dim = 128
c = 0
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    dataset.MNIST('data', train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=mb_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset.MNIST('data', train=False, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=mb_size, shuffle=False)

for data in train_loader:
    x, y = data[0], data[1]
    print(x.shape)
    X_dim = x.shape[-1] * (x.shape[-2])
    break

print("x_dim:{}".format(X_dim))


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size, device="cuda") * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim, device="cuda"), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim, device="cuda"), requires_grad=True)


def G(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1)).to(device)
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1)).to(device)
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim, device="cuda"), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1, device="cuda"), requires_grad=True)


def D(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1)).to(device)
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1)).to(device)
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params

""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_optimizer = optim.Adam(G_params, lr=1e-4)
D_optimizer = optim.Adam(D_params, lr=1e-4)

ones_label = Variable(torch.ones(mb_size, 1, device="cuda"))
zeros_label = Variable(torch.zeros(mb_size, 1, device="cuda"))

for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).to(device)
    # X, _ = mnist.train.next_batch(mb_size)
    _, (X, y) = next(enumerate(train_loader))
    X = X.view(-1, X_dim)
    X = Variable(X).to(device)

    # Dicriminator forward-loss-backward-update
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    # $log(1-D(x))$
    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    # $log(D(G(z))$
    D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_optimizer.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).to(device)
    G_sample = G(z)
    D_fake = D(G_sample)

    if it % 2 == 0:
        # $log(1-D(G(z))$
        G_loss = nn.binary_cross_entropy(D_fake, ones_label)

        G_loss.backward()
        G_optimizer.step()

        # Housekeeping - reset gradient
        reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()))

        samples = G(z).data.cpu().numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
