"""
Rollout Buffer per PPO con azioni discrete.

A differenza del ReplayBuffer del DQN (300k esperienze riusate continuamente),
il RolloutBuffer è temporaneo: accumula 2048 step consecutivi con la policy
attuale, li usa per 10 aggiornamenti, poi viene svuotato e reiniziato.

Deve essere svuotato perché salva le log_probm, le probabilità con cui la
policy ATTUALE ha scelto le azioni. Dopo gli aggiornamenti la policy cambia,
quelle probabilità diventano obsolete e il clip PPO non funzionerebbe più.

Salva per ogni step:
  - stato, azione, reward, done         (come il ReplayBuffer)
  - value    : V(s) stimato dal critico (serve per calcolare il vantaggio)
  - log_prob : log P(azione) dell'attore (serve per il clip PPO)
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        n_steps    : int,    # quanti step raccogliere prima di aggiornare (2048)
        forma_stato: tuple,  # shape dell'osservazione (4, 84, 84)
        gamma      : float,  # discount factor per i ritorni
        gae_lambda : float,  # peso GAE (0=solo TD error, 1=solo Monte Carlo)
        device     : str,    # 'cuda' o 'cpu'
    ):
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device

        # Pre-alloca tutti gli array: più efficiente che appendere ad ogni step
        self.stati     = torch.zeros((n_steps, *forma_stato), dtype=torch.uint8)
        self.azioni    = torch.zeros(n_steps, dtype=torch.long)
        self.rewards   = torch.zeros(n_steps)
        self.values    = torch.zeros(n_steps)     # V(s) stimato dal critico
        self.log_probs = torch.zeros(n_steps)     # log P(azione) — necessario per clip PPO
        self.dones     = torch.zeros(n_steps)
        self.vantaggi  = torch.zeros(n_steps)     # calcolati dopo il rollout con GAE
        self.ritorni   = torch.zeros(n_steps)     # vantaggi + values = target per il critico
        self.pos       = 0                         # indice corrente nel buffer

    def aggiungi(self, stato, azione: int, reward: float, value, log_prob: float, done: bool):
        #Aggiunge una transizione al buffer. Chiamare ad ogni step del rollout.
        self.stati[self.pos]     = torch.from_numpy(np.asarray(stato))
        self.azioni[self.pos]    = int(azione)
        self.rewards[self.pos]   = float(reward)
        self.values[self.pos]    = value.item() if isinstance(value, torch.Tensor) else float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.dones[self.pos]     = float(done)
        self.pos += 1

    def calcola_vantaggi(self, last_value, last_done: bool):
        """
        Calcola i vantaggi con GAE (Generalized Advantage Estimation).

        Si calcola all'indietro (dal passo 2048 al passo 1) perché ogni
        vantaggio dipende da quelli futuri.
        """
        lv  = last_value.item() if isinstance(last_value, torch.Tensor) else float(last_value)
        gae = 0.0

        # Calcolo all'indietro: dal passo finale verso il primo
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_val          = lv
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].item()
                next_val          = self.values[t + 1].item()

            # TD error: quanto ha guadagnato in più (o meno) rispetto alle aspettative
            delta = (self.rewards[t].item()
                     + self.gamma * next_val * next_non_terminal
                     - self.values[t].item())

            # GAE: accumula gli errori futuri con peso decrescente
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.vantaggi[t] = gae

        # Ritorni = vantaggi + valori stimati -> target per la loss del critico
        self.ritorni  = self.vantaggi + self.values

        # Normalizza i vantaggi (media 0, std 1) -> stabilizza il training
        self.vantaggi = (self.vantaggi - self.vantaggi.mean()) / (self.vantaggi.std() + 1e-8)

    def get_minibatches(self, batch_size: int):
        
        # Mescola i 2048 step e li divide in minibatch da batch_size=64.
        # Chiamato da agente.aggiorna() per i 10 epoch di aggiornamento.
        
        assert self.pos == self.n_steps, "Buffer non pieno"
        # ordine casuale per rompere correlazioni
        idx = torch.randperm(self.n_steps)     

        # Manda i tensori sul device (GPU) solo ora, non durante la raccolta
        stati     = self.stati.to(self.device)
        azioni    = self.azioni.to(self.device)
        log_probs = self.log_probs.to(self.device)
        vantaggi  = self.vantaggi.to(self.device)
        ritorni   = self.ritorni.to(self.device)

        for start in range(0, self.n_steps, batch_size):
            i = idx[start : start + batch_size]
            yield stati[i], azioni[i], log_probs[i], vantaggi[i], ritorni[i]

    def reset(self):
        #"Svuota il buffer dopo gli aggiornamenti. Le esperienze vengono scartate
        self.pos = 0
