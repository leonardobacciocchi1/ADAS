"""
Rete neurale CNN per DQN/DDQN su CarRacing-v3.

Input:  4 frame grayscale 84x84 impilati
Output: Q-value per ciascuna delle 5 azioni discrete
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Approssimatore della funzione Q tramite rete convoluzionale.

    Supporta due architetture selezionabili con n_conv:
      n_conv=2 → baseline NIPS 2013  (2 strati conv, FC 256)
      n_conv=3 → Nature DQN 2015     (3 strati conv, FC 512)

    Il terzo strato conv aggiunge un livello di feature gerarchiche superiori:
    l'agente riesce a "vedere" meglio la curvatura della pista e la propria
    posizione, riducendo la varianza del reward in curva.
    """

    def __init__(self, n_canali: int, n_azioni: int, n_conv: int = 3):
        """
        n_canali : numero di frame impilati usati come input  (tipicamente 4)
        n_azioni : dimensione dello spazio azioni             (5 per CarRacing)
        n_conv   : numero di strati convoluzionali (2 = baseline, 3 = Nature DQN)
        """
        super().__init__()
        assert n_conv in (2, 3), "n_conv deve essere 2 o 3"

        if n_conv == 2:
            # ── Architettura baseline (NIPS 2013, 2 conv) ─────────────────────
            self.rete = nn.Sequential(
                nn.Conv2d(n_canali, 16, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32,       kernel_size=4, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
                nn.Linear(256, n_azioni),
            )
        else:
            # ── Architettura Nature DQN (Mnih et al. 2015, 3 conv) ────────────
            self.rete = nn.Sequential(
                nn.Conv2d(n_canali, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64,       kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64,       kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, n_azioni),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Propaga lo stato attraverso la rete e restituisce i Q-value
        # L'agente sceglierà l'azione con Q-value massimo, politica greedy
        return self.rete(x)

