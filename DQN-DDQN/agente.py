"""
Agente DQN / DDQN

L'agente collega la CNN e il Replay Buffer
Ha due responsabilità principali:
  1. Scegliere le azioni con politica greedy sul Q-value più alto 
  2. Aggiornarsi imparando dalle esperienze memorizzate nel buffer

DQN vs DDQN — differenza nel calcolo del Q-value target:

  DQN:
    target = r + γ * max( rete_target(s') )
    la rete target sceglie E valuta l'azione che potrebbe portare all'overestimation bias

  DDQN (Double DQN):
    azione_migliore = argmax( rete_online(s') )   scelta con rete online
    target = r + γ * rete_target(s')[azione_migliore]  valutazione con target
    disaccoppia scelta e valutazione con due differenti reti e con bias ridotto (training più stabile)

PER (Prioritized Experience Replay):
  Invece di campionare uniformemente dal buffer, campiona di più le esperienze
  con TD error alto — quelle da cui la rete ha ancora più da imparare.
  I pesi IS (Importance Sampling) correggono il bias introdotto dal campionamento
  non uniforme moltiplicando la loss elemento per elemento.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from mio_dqn.modello import CNN
from mio_dqn.buffer  import ReplayBuffer, PERBuffer


class Agente:
    """
    Agente DQN o DDQN per CarRacing-v3, con supporto opzionale a PER.

    Mantiene due reti CNN identiche nella struttura (definite in modello.py):
    - rete_online  : si aggiorna ad ogni step tramite backpropagation
    - rete_target  : copia congelata, sincronizzata ogni sync_ogni step.
                     Fornisce Q-value target stabili per il calcolo della loss. valutando con Bellman
    """

    def __init__(
        self,
        forma_stato  : tuple,
        n_azioni     : int,
        double_q     : bool  = False,   # False = DQN, True = DDQN
        n_conv       : int   = 3,       # 2 = baseline NIPS 2013, 3 = Nature DQN 2015
        per          : bool  = False,   # False = buffer uniforme, True = PER
        gamma        : float = 0.95,    # discount factor: peso delle ricompense future
        lr           : float = 0.0002,  # learning rate dell'ottimizzatore "Adam"
        epsilon      : float = 1.0,     # esplorazione iniziale (tutto casuale)
        epsilon_decay: float = 0.9999925, # moltiplicatore per step (decresce lentamente)
        epsilon_min  : float = 0.05,    # esplorazione minima (5% casuale residuo)
        buffer_cap   : int   = 300_000, # capacità massima del replay buffer
        batch_size   : int   = 32,      # numero di esperienze campionate per aggiornamento
        learn_ogni   : int   = 4,       # aggiorna la rete ogni N step
        sync_ogni    : int   = 5_000,   # copia online del target ogni N step
        # Parametri PER 
        per_alpha    : float = 0.6,     # quanto pesare le priorità (0=uniforme, 1=piena)
        per_beta_start: float= 0.4,     # peso IS iniziale (cresce fino a 1.0)
        per_beta_passi: int  = 740_000, # step per annealare beta a 1.0
    ):
        self.n_azioni      = n_azioni
        self.double_q      = double_q
        self.per           = per
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.batch_size    = batch_size
        self.learn_ogni    = learn_ogni
        self.sync_ogni     = sync_ogni
        self.step_totali   = 0
        self.n_updates     = 0

        # Usa GPU se disponibile
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #  Due reti CNN identiche nella struttura 
        # Stessa architettura (definita in modello.py), pesi inizialmente identici,
        # ma ruoli opposti: online apprende, target fornisce bersagli stabili.
        self.rete_online = CNN(forma_stato[0], n_azioni, n_conv=n_conv).float().to(self.device)
        self.rete_target = CNN(forma_stato[0], n_azioni, n_conv=n_conv).float().to(self.device)
        # pesi identici all'inizio
        self.rete_target.load_state_dict(self.rete_online.state_dict())  
        self.rete_target.eval()  

        #Ottimizzatore (Adam) 
        # Adatta il learning rate per ogni peso in modo indipendente,
        # convergendo più velocemente e stabilmente rispetto a SGD standard.
        self.optimizer = torch.optim.Adam(self.rete_online.parameters(), lr=lr)

        #  Replay Buffer 
        # Buffer uniforme: campionamento casuale tra tutte le esperienze.
        # PER: campiona di più le esperienze con TD error alto (più informative).
        if per:
            self.buffer = PERBuffer(
                capacita    = buffer_cap,
                forma_stato = forma_stato,
                device      = self.device,
                alpha       = per_alpha,
                beta_start  = per_beta_start,
                beta_passi  = per_beta_passi,
            )
        else:
            self.buffer = ReplayBuffer(
                capacita    = buffer_cap,
                forma_stato = forma_stato,
                device      = self.device,
            )

    # Scelta Azione 

    def scegli_azione(self, stato: np.ndarray) -> int:
        """
        Politica greedy:
          - con probabilità epsilon avremo un'azione casuale (esplorazione)
          - altrimenti azione con Q-value massimo (sfruttamento)

        Epsilon decresce da 1.0 a 0.05: all'inizio esplora tutto,
        alla fine sfrutta quasi sempre la policy appresa.
        """
        if np.random.rand() < self.epsilon:
            # esplorazione casuale
            azione = np.random.randint(self.n_azioni)   
        else:
            s = torch.tensor(stato, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                # stima Q-value per ogni azione
                q_values = self.rete_online(s)          
             # scegli l'azione con Q-value massimo
            azione = q_values.argmax(dim=1).item()      

        # Epsilon decresce moltiplicativamente ad ogni step
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.step_totali += 1
        return azione

    # Training con buffer

    def memorizza(self, stato, azione, reward, stato_succ, terminato):
        #Salva una transizione nel replay buffer.
        self.buffer.aggiungi(stato, azione, reward, stato_succ, terminato)

    def aggiorna(self) -> float | None:
        #Aggiorna la rete online se il buffer ha abbastanza esperienze
        # sono passati learn_ogni step dall'ultimo aggiornamento.
        # Restituisce la loss, oppure None se non ha fatto nulla.
        
        if len(self.buffer) < self.batch_size:
            # buffer non ancora abbastanza pieno
            return None   
        if self.step_totali % self.learn_ogni != 0:
            # non è ancora il momento di aggiornare
            return None   

        #  Campionamento dal buffer 
        # PER restituisce anche i pesi IS e gli indici per aggiornare le priorità
        if self.per:
            stati, azioni, rewards, stati_succ, terminato, pesi, indici = \
                self.buffer.campiona(self.batch_size, step_correnti=self.step_totali)
        else:
            stati, azioni, rewards, stati_succ, terminato = \
                self.buffer.campiona(self.batch_size)

        #  Q-value stimato dalla rete online 
        # q_online ha shape (batch, n_azioni); selezioniamo solo il Q-value
        # dell'azione effettivamente eseguita in ogni esperienza del batch.
        q_online  = self.rete_online(stati)
        q_stimato = q_online[torch.arange(self.batch_size), azioni]

        #  Q-value target (bersaglio dell'equazione di Bellman) 
        # no_grad: la rete target non deve accumulare gradienti
        with torch.no_grad():
            if self.double_q:
                # DDQN: rete online SCEGLIE l'azione, rete target la VALUTA
                # disaccoppiamento che riduce l'overestimation bias del DQN
                azione_migliore = self.rete_online(stati_succ).argmax(dim=1)
                q_next = self.rete_target(stati_succ)[torch.arange(self.batch_size), azione_migliore]
            else:
                # DQN: rete target sceglie E valuta overestimation bias
                q_next = self.rete_target(stati_succ).max(dim=1)[0]

        # Equazione di Bellman: se terminato=1 (episodio finito)
        q_target = rewards + self.gamma * q_next * (1.0 - terminato)

        #  Loss (SmoothL1)
        # Più robusta di MSE: lineare per errori grandi, quadratica per piccoli.
        # Riduce l'impatto di esperienze con reward anomali sul gradiente.
        if self.per:
            # Con PER: loss pesata per i pesi IS che correggono il bias di campionamento
            loss_elem = F.smooth_l1_loss(q_stimato, q_target, reduction='none')
            loss      = (pesi * loss_elem).mean()
            # Aggiorna le priorità nel buffer con i nuovi TD errors
            td_errors = (q_stimato - q_target).detach().abs().cpu().numpy()
            self.buffer.aggiorna_priorita(indici, td_errors)
        else:
            loss = F.smooth_l1_loss(q_stimato, q_target)

        #  Backpropagation 
        # azzera i gradienti accumulati dal passo precedente
        self.optimizer.zero_grad()  
        # calcola i gradienti rispetto a tutti i pesi 
        loss.backward()               

        # Gradient clipping: limita la norma del gradiente a max_norm=10.
        # Evita il gradient explosion che nelle prime versioni faceva collassare il training.
        # modifica solo la dimensione massima.
        torch.nn.utils.clip_grad_norm_(self.rete_online.parameters(), max_norm=10)

        # aggiorna i pesi nella direzione che riduce la loss
        self.optimizer.step()         
        self.n_updates += 1

        #  Sincronizzazione rete target 
        # Copia i pesi da online a target ogni sync_ogni step.
        # DQN: sync ogni 10_000 step (target più stabile)
        # DDQN: sync ogni 5_000 step (già stabile per costruzione, può permetterselo)
        if self.step_totali % self.sync_ogni == 0:
            self.rete_target.load_state_dict(self.rete_online.state_dict())

        return loss.item()

    #  Salvataggio / Caricamento 

    def salva(self, cartella: str, nome: str):
       #Salva i pesi delle reti, ottimizzatore e stato dell'agente.
        os.makedirs(cartella, exist_ok=True)
        percorso = os.path.join(cartella, f"{nome}_{self.step_totali}.pt")
        torch.save({
            "rete_online" : self.rete_online.state_dict(),
            "rete_target" : self.rete_target.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            "step_totali" : self.step_totali,
            "epsilon"     : self.epsilon,
            "double_q"    : self.double_q,
        }, percorso)
        print(f"  Modello salvato: {percorso}")
        return percorso

    """
        Carica un modello salvato. FOndamentale per gestire le 16 ore di training

        modalita : 'eval'  → epsilon=0, rete in eval mode (per valutazione)
                   'train' → ripristina epsilon e step (per continuare il training)
    """
    def carica(self, percorso: str, modalita: str = "eval"):
        
        checkpoint = torch.load(percorso, map_location=self.device, weights_only=False)
        self.rete_online.load_state_dict(checkpoint["rete_online"])
        self.rete_target.load_state_dict(checkpoint["rete_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if modalita == "eval":
            self.epsilon = 0.0          # nessuna esplorazione in valutazione
            self.rete_online.eval()
            self.rete_target.eval()
        elif modalita == "train":
            self.step_totali = checkpoint["step_totali"]
            self.epsilon     = checkpoint["epsilon"]    # riprende da dove era rimasto
            self.rete_online.train()
