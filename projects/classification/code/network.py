"""Rede MLP para classificação binária (PyTorch).

Seguindo recomedações e orientações do exercício, a rede devolve *logits* (números reais, sem sigmoid).
A sigmoid fica dentro da perda ``BCEWithLogitsLoss``, que é mais estável numericamente.

Dropout e BatchNorm vêm desligados por padrão: a configuração base é a rede
"sem nenhuma regularização", usada como ponto de comparação na etapa 4.

Estrutura da rede:
Camadas 64 e 32, ReLU, sem dropout, sem batch norm e inicialização com He para ReLU. Biases começam em zero.

    python docs/projects/classification/code/network.py
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Perceptron multicamadas: entrada -> camadas ocultas -> 1 logit."""

    def __init__(
        self,
        n_inputs: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation

        layers: list[nn.Module] = []
        previous = n_inputs
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))  # 1 unidade de saída, sem ativação

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self) -> None:
        """He (Kaiming) para ReLU; Xavier para tanh. Biases começam em zero.
           Escolha da inicialização em He porque a ativação é ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devolve logits com formato (batch,)."""
        return self.net(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Número de parâmetros treináveis (pesos + biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MLP(n_inputs=21)
    print(model)
    print(f"Parâmetros treináveis: {count_parameters(model):,}")
    logits = model(torch.randn(8, 21))
    print("Formato da saída:", tuple(logits.shape))