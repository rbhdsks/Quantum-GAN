import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class QuantumBlock(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumBlock, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.dev = qml.device('default.qubit', wires=n_qubits)

    def forward(self, x):
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        x = x.view(-1, self.n_qubits)  # Flatten input to match qubits
        return circuit(x, self.weights)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_quantum=True):
        super().__init__()
        self.use_quantum = use_quantum
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

        if self.use_quantum:
            self.quantum_block = QuantumBlock(n_qubits=4, n_layers=6)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        if self.use_quantum:
            x = self.quantum_block(x)
        return torch.sigmoid(x)

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3, use_quantum=True)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
