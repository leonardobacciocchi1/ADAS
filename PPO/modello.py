"""
Rete Actor-Critic per PPO discreto su CarRacing-v3.

Stessa backbone CNN del DQN Nature (3 strati conv), ma con due teste di output:
  - Attore:  5 logits, distribuzione Categorical sulle 5 azioni discrete
  - Critico: 1 scalare, stima del valore di stato V(s)

La backbone è condivisa: le feature estratte dalla CNN vengono usate
sia per decidere l'azione (attore) che per valutare lo stato (critico).
"""

import torch
import torch.nn as nn


class AttoreCritico(nn.Module):
    """
    Rete Actor-Critic con backbone CNN condivisa.

    A differenza del DQN che usa due reti separate (online e target),
    PPO usa una singola rete con due teste finali distinte:
      - Attore  → "cosa fare"   (distribuzione di probabilità sulle azioni)
      - Critico → "quanto è buono lo stato" (valore scalare V(s))
    """

    def __init__(self, n_canali: int, n_azioni: int):
    
        #n_canali : numero di frame impilati (4)
        #n_azioni : numero di azioni discrete (5 per CarRacing)
    
        super().__init__()

        # Backbone CNN condivisa (identica al DQN Nature 3-conv) 
        
        self.backbone = nn.Sequential(
            nn.Conv2d(n_canali, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,       64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,       64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )

        #  Testa Attore 
        # Produce 5 logits trasformati in probabilità dalla distribuzione Categorical
        # L'azione viene campionata da questa distribuzione (non è argmax come nel DQN)
        self.attore  = nn.Linear(512, n_azioni)

        #  Testa Critico 
        # Produce 1 scalare: la stima di V(s), cioè quanto è buono lo stato attuale
        # Usato per calcolare il vantaggio: vantaggio = reward reale - V(s) stimato
        self.critico = nn.Linear(512, 1)

        self._init_pesi()

    def _init_pesi(self):
        # Inizializzazione ortogonale: standard per PPO, stabilizza il training iniziale
        for m in self.backbone:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        
        # gain piccolo: policy inizialmente uniforme
        nn.init.orthogonal_(self.attore.weight,  gain=0.01)  
        nn.init.zeros_(self.attore.bias)
        nn.init.orthogonal_(self.critico.weight, gain=1.0)
        nn.init.zeros_(self.critico.bias)

    def forward(self, x: torch.Tensor):
        # Normalizza i pixel da [0,255] a [0,1] prima di passarli alla CNN
        x    = x.float() / 255.0
        feat = self.backbone(x)             # feature condivise (batch, 512)
        return self.attore(feat), self.critico(feat).squeeze(-1)
        # restituisce: logits (batch, 5)  e  V(s) (batch,1)
