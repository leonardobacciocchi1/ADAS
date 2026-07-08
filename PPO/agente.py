"""
Agente PPO con azioni discrete per CarRacing-v3.

A differenza del DQN che stima Q-value e deriva la policy implicitamente,
PPO ottimizza direttamente la policy tramite policy gradient con clipped loss.
PPO è policy based.

Usa distribuzione Categorical per le azioni discrete:
  - Output dell'attore = 5 logits con distribuzione di probabilità sulle 5 azioni
  - L'azione viene campionata dalla distribuzione (non argmax come DQN)
  - Entropia massima = ln(5) ≈ 1.6 , no divergenze
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mio_ppo.modello import AttoreCritico


class AgentePPO:
    def __init__(
        self,
        forma_stato    : tuple,
        n_azioni       : int,
        lr             : float = 3e-4,    # learning rate Adam
        gamma          : float = 0.99,    # discount factor
        gae_lambda     : float = 0.95,    # peso del GAE (0=solo TD, 1=solo Monte Carlo)
        clip_epsilon   : float = 0.2,     # limite variazione policy: ratio in [0.8, 1.2]
        n_epochs       : int   = 10,      # quante volte ripassare i 2048 step del buffer
        coeff_valore   : float = 0.5,     # peso della loss del critico nella loss totale
        coeff_entropia : float = 0.01,    # peso del bonus entropia (incoraggia esplorazione)
        max_grad_norm  : float = 0.5,     # gradient clipping (più conservativo del DQN)
    ):
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.clip_epsilon   = clip_epsilon
        self.n_epochs       = n_epochs
        self.coeff_valore   = coeff_valore
        self.coeff_entropia = coeff_entropia
        self.max_grad_norm  = max_grad_norm
        self.step_totali    = 0

        # Una sola rete (attore + critico) — a differenza del DQN che ne usa due
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.rete      = AttoreCritico(forma_stato[0], n_azioni).to(self.device)
        self.optimizer = torch.optim.Adam(self.rete.parameters(), lr=lr, eps=1e-5)

    #  Selezione azione 

    @torch.no_grad()
    def scegli_azione(self, stato: np.ndarray, deterministico: bool = False):
        """
        Sceglie un'azione tramite la distribuzione Categorical dell'attore.

        In training (deterministico=False): campiona dalla distribuzione, esplorazione naturale senza bisogno di una politica greedy
        In valutazione (deterministico=True): prende l'azione con logit massimo comportamento greedy puro

        Restituisce:
          azione   : int   — indice dell'azione scelta (0-4)
          log_prob : float — log probabilità dell'azione (serve per il clip PPO)
          value    : tensor scalare — V(s) stimato dal critico (serve per GAE)
        """
        s = torch.from_numpy(np.asarray(stato)).unsqueeze(0).to(self.device)
        logits, value = self.rete(s)              # attore -> logits, critico -> V(s)
        dist = Categorical(logits=logits)          # distribuzione di probabilità sulle 5 azioni

        azione   = logits.argmax(dim=-1) if deterministico else dist.sample()
        log_prob = dist.log_prob(azione)           # log P(azione) — salvato nel buffer

        self.step_totali += 1
        return azione.item(), log_prob.item(), value.squeeze(0)

    @torch.no_grad()
    def get_value(self, stato: np.ndarray) -> torch.Tensor:
        """Stima V(s) senza scegliere un'azione — usato per il bootstrap GAE alla fine del rollout."""
        s = torch.from_numpy(np.asarray(stato)).unsqueeze(0).to(self.device)
        _, value = self.rete(s)
        return value.squeeze(0)

    # Aggiornamento PPO 

    def aggiorna(self, buffer) -> dict:
        """
        Aggiorna la rete usando i 2048 step del buffer.

        Ripete n_epochs=10 volte sull'intero buffer, dividendolo in
        minibatch da 64. Dopo questi aggiornamenti il buffer viene scartato
        (on-policy: le esperienze sono valide solo per la policy che le ha generate).

        Loss totale = loss_policy + coeff_valore * loss_value - coeff_entropia * entropia
        """
        losses_policy = []
        losses_value  = []
        entropie      = []

        for _ in range(self.n_epochs):     # 10 passate sugli stessi dati
            for stati, azioni, log_probs_old, vantaggi, ritorni in buffer.get_minibatches(64):

                # Ricalcola logits e valori con la policy ATTUALE (aggiornata)
                logits, values = self.rete(stati)
                dist = Categorical(logits=logits)

                log_probs_new = dist.log_prob(azioni)   # log P_nuova(azione)
                entropia      = dist.entropy().mean()    # entropia della distribuzione attuale

                #  Clipped Surrogate Loss 
                # quanto è cambiata la policy
                ratio   = (log_probs_new - log_probs_old).exp()
                # obiettivo non clippato
                loss_p1 = ratio * vantaggi  
                # obiettivo clippato [0.8, 1.2]                            
                loss_p2 = ratio.clamp(1 - self.clip_epsilon,
                                      1 + self.clip_epsilon) * vantaggi 
                # Prende il minimo: se il ratio è fuori dal range, usa il valore clippato
                # impedisce aggiornamenti troppo grandi che farebbero collassare la policy
                loss_policy = -torch.min(loss_p1, loss_p2).mean()

                # Loss del critico 
                # MSE tra V(s) stimato e i ritorni reali calcolati dal buffer
                loss_value  = F.mse_loss(values, ritorni)

                # Loss totale
                # L'entropia viene sottratta (massimizzata) per incoraggiare l'esplorazione
                loss = (loss_policy
                        + self.coeff_valore   * loss_value
                        - self.coeff_entropia * entropia)

                # Backpropagation 
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rete.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses_policy.append(loss_policy.item())
                losses_value.append(loss_value.item())
                entropie.append(entropia.item())

        # Restituisce le medie delle loss per il logging in train_ppo.py
        return {
            "loss_policy" : float(np.mean(losses_policy)),
            "loss_value"  : float(np.mean(losses_value)),
            "entropia"    : float(np.mean(entropie)),
        }

    # Salvataggio / Caricamento 
    def salva(self, cartella: str, nome: str) -> str:
        # Salva i pesi della rete e lo stato dell'ottimizzatore.
        os.makedirs(cartella, exist_ok=True)
        path = os.path.join(cartella, f"{nome}_{self.step_totali}.pt")
        torch.save({
            "rete"        : self.rete.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            "step_totali" : self.step_totali,
        }, path)
        print(f"  Modello salvato: {path}")
        return path

    def carica(self, percorso: str, modalita: str = "eval"):
        #Carica un modello salvato in due modalità, train e eval, cruciale per i tempi

        ck = torch.load(percorso, map_location=self.device, weights_only=False)
        self.rete.load_state_dict(ck["rete"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if modalita == "train":
            self.step_totali = ck["step_totali"]   # riprende da dove era rimasto
        if modalita == "eval":
            self.rete.eval()                      
