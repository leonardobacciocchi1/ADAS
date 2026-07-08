"""
Replay Buffer e Prioritized Experience Replay (PER) per DQN/DDQN.

ReplayBuffer
    Buffer circolare classico con campionamento uniforme.
    DeepMind (2013) per rompere le correlazioni temporaliinvece di imparare 
    da esperienze consecutive (fortemente correlate), si campionano 32 esperienze 
    casuali da momenti diversi del training.

PERBuffer
    Prioritized Experience Replay (Schaul et al., 2015).
    Invece di campionare uniformemente, campiona più spesso le esperienze
    con TD error alto — quelle da cui la rete ha ancora più da imparare.

    Implementazione con SumTree:
    - SumTree è un albero binario che porta il campionamento da O(n) a O(log n)
    - Ogni foglia memorizza la priorità di un'esperienza
    - I nodi interni contengono le somme parziali delle priorità
    - La radice contiene la somma totale di tutte le priorità
    - Per campionare: si sceglie un valore casuale v in [0, totale]
      e si scende l'albero finché si trova la foglia corrispondente

    Importance Sampling (IS):
    - Campionare con priorità introduce un bias: le esperienze frequenti
      vengono aggiornate più spesso, la rete rischia di sovradattarsi a quelle
    - Per correggere, si moltiplicano i gradienti per pesi IS:
        w_i = (N * P(i))^(-beta)
      Esperienza con priorità alta -> peso IS basso -> contribuisce poco alla loss
      Esperienza con priorità bassa -> peso IS alto -> contribuisce di più
    - beta sale da 0.4 a 1.0 durante il training (annealing):
      all'inizio accettiamo il bias per esplorare di più,
      alla fine lo correggiamo completamente
"""

import numpy as np
import torch


#  Replay Buffer classico 

class ReplayBuffer:
    """
    Buffer circolare che memorizza le transizioni (s, a, r, s', done).

    Quando il buffer è pieno, le esperienze più vecchie vengono
    sovrascritte dalle nuove (logica FIFO).

    Usato da DQN e DDQN nel training finale.
    """

    def __init__(self, capacita: int, forma_stato: tuple, device: str = "cpu"):
        """
        capacita   : numero massimo di transizioni memorizzabili (300.000)
        forma_stato: shape dell'osservazione, es. (4, 84, 84)
        device     : 'cpu' o 'cuda' 
        """
        self.capacita   = capacita
        self.device     = device
        self.posizione  = 0      # indice circolare: dove scrivere la prossima esperienza
        self.dimensione = 0      # quante esperienze ci sono attualmente nel buffer

        # Salviamo gli stati come uint8 (valori 0-255) invece di float32:
        # occupa 4x meno memoria — la conversione a float avviene solo al campionamento
        self.stati      = np.zeros((capacita, *forma_stato), dtype=np.uint8)
        self.stati_succ = np.zeros((capacita, *forma_stato), dtype=np.uint8)
        self.azioni     = np.zeros(capacita, dtype=np.int64)
        self.reward     = np.zeros(capacita, dtype=np.float32)
        self.terminato  = np.zeros(capacita, dtype=bool)

    def aggiungi(self, stato, azione, reward, stato_succ, terminato):
        """Aggiunge una transizione (s, a, r, s', done) al buffer."""
        self.stati[self.posizione]      = stato
        self.stati_succ[self.posizione] = stato_succ
        self.azioni[self.posizione]     = azione
        self.reward[self.posizione]     = reward
        self.terminato[self.posizione]  = terminato

        # Avanza circolarmente: quando arriva a capacita, riparte da 0
        self.posizione  = (self.posizione + 1) % self.capacita
        self.dimensione = min(self.dimensione + 1, self.capacita)

    def campiona(self, batch_size: int):
        
        #Campiona batch_size esperienze casuali uniformemente dal buffer.
        #Converte da uint8 a float32 solo ora, per risparmiare memoria.
        
        indici     = np.random.randint(0, self.dimensione, size=batch_size)
        stati      = torch.tensor(self.stati[indici],      dtype=torch.float32).to(self.device)
        stati_succ = torch.tensor(self.stati_succ[indici], dtype=torch.float32).to(self.device)
        azioni     = torch.tensor(self.azioni[indici],     dtype=torch.int64).to(self.device)
        reward     = torch.tensor(self.reward[indici],     dtype=torch.float32).to(self.device)
        terminato  = torch.tensor(self.terminato[indici],  dtype=torch.float32).to(self.device)
        return stati, azioni, reward, stati_succ, terminato

    def __len__(self):
        return self.dimensione


# PER Buffer 

class PERBuffer:
    """
    Prioritized Experience Replay Buffer.

    Campiona le esperienze con probabilità proporzionale a p^alpha,
    dove p = |TD error| + eps.

    Usato nelle varianti sperimentali DQN+PER e DDQN+PER .
    """

    def __init__(
        self,
        capacita   : int,
        forma_stato: tuple,
        device     : str   = "cpu",
         # quanto pesare le priorità (0=uniforme, 1=piena)
        alpha      : float = 0.6,   
         # peso IS iniziale (cresce fino a 1.0) 
        beta_start : float = 0.4,    
        # step per annealare beta a 1.0
        beta_passi : int   = 740_000, 
    ):
        self.capacita   = capacita
        self.device     = device
        self.alpha      = alpha       
        self.beta_start = beta_start
        self.beta_passi = beta_passi
        self.posizione  = 0
        self.dimensione = 0

        # Epsilon: garantisce che ogni esperienza abbia priorità > 0
        # evita che esperienze con TD error=0 non vengano mai campionate
        self._eps = 1e-6

        # Le nuove esperienze ricevono la priorità massima vista finora:
        # così vengono campionate almeno una volta prima di essere valutate
        self.max_priorita = 1.0

        #  Dati del buffer 
        self.stati      = np.zeros((capacita, *forma_stato), dtype=np.uint8)
        self.stati_succ = np.zeros((capacita, *forma_stato), dtype=np.uint8)
        self.azioni     = np.zeros(capacita, dtype=np.int64)
        self.reward     = np.zeros(capacita, dtype=np.float32)
        self.terminato  = np.zeros(capacita, dtype=bool)

        #  SumTree 
        # Albero binario con 2*capacita-1 nodi totali:
        # Permette campionamento O(log n) invece di O(n)
        self._albero = np.zeros(2 * capacita - 1, dtype=np.float64)

    # SumTree: operazioni interne 

    def _aggiorna_albero(self, pos_buffer: int, priorita: float):
        
        #Aggiorna la priorità della foglia corrispondente a pos_buffer
        #e risale l'albero aggiornando le somme parziali verso la radice.
        
        # converte indice buffer all'indice foglia
        idx    = pos_buffer + self.capacita - 1   
        # variazione di priorità
        delta  = priorita - self._albero[idx]     
        self._albero[idx] = priorita
        # Propaga il delta verso la radice aggiornando ogni nodo padre
        while idx > 0:
             # sale al nodo padre
            idx = (idx - 1) // 2                 
            self._albero[idx] += delta

    def _trova_foglia(self, valore: float) -> int:
        
        #Dato un valore in [0, totale], discende l'albero e restituisce
        #l'indice nel buffer (0-indexed) della foglia trovata.

        #Funzionamento: se valore <= somma figlio sinistro vai a sinistra,
        #altrimenti sottrai la somma sinistra e vai a destra.
        
        idx = 0   # parte dalla radice
        while True:
            sx = 2 * idx + 1   # indice figlio sinistro
            dx = sx + 1        # indice figlio destro
            if sx >= len(self._albero):
                # Siamo arrivati a una foglia → converti in indice buffer
                return idx - (self.capacita - 1)
            if valore <= self._albero[sx]:
                idx = sx       # vai a sinistra
            else:
                valore -= self._albero[sx]
                idx = dx       # vai a destra (sottraendo la somma sinistra)

    #  Interfaccia pubblica 

    def aggiungi(self, stato, azione, reward, stato_succ, terminato):
        """Aggiunge una transizione con priorità massima (sarà campionata presto)."""
        self.stati[self.posizione]      = stato
        self.stati_succ[self.posizione] = stato_succ
        self.azioni[self.posizione]     = azione
        self.reward[self.posizione]     = reward
        self.terminato[self.posizione]  = terminato

        # Priorità iniziale = max vista finora: garantisce che la nuova esperienza
        # venga campionata almeno una volta prima che il suo TD error sia noto
        self._aggiorna_albero(self.posizione, self.max_priorita)

        self.posizione  = (self.posizione + 1) % self.capacita
        self.dimensione = min(self.dimensione + 1, self.capacita)

    #campionamento 
    def campiona(self, batch_size: int, step_correnti: int):
        """
        Campiona batch_size esperienze con probabilità proporzionale alle priorità.

        Usa campionamento stratificato: divide [0, totale] in batch_size segmenti
        uguali e prende un valore casuale in ciascuno ciò coverage più uniforme.

        Restituisce: 
        stati, azioni, reward, stati_succ, terminato : tensori standard
        pesi   : tensore float32 (batch,) — pesi IS per correggere il bias
        indici : array int64 (batch,)  — indici nel buffer per aggiornare priorità
        """
        # Annealing di beta 
        # beta sale linearmente da beta_start (0.4) a 1.0 durante il training
        # all'inizio accettiamo il bias, alla fine lo correggiamo completamente
        frac = min(step_correnti / self.beta_passi, 1.0)
        beta = self.beta_start + frac * (1.0 - self.beta_start)

        #  Campionamento stratificato 
        # radice = somma totale di tutte le priorità
        totale   = self._albero[0]     
        segmento = totale / batch_size

        indici   = np.zeros(batch_size, dtype=np.int64)
        priorita = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            a = segmento * i
            b = segmento * (i + 1)
            v = np.random.uniform(a, b)            # valore casuale nel segmento i
            idx = self._trova_foglia(v)             # discende il SumTree
            idx = max(0, min(idx, self.dimensione - 1))   # clamp per buffer non pieno
            indici[i]   = idx
            priorita[i] = self._albero[idx + self.capacita - 1]

        #  Importance Sampling weights 
        # Priorità alta -> campionata spesso -> peso IS basso -> contribuisce poco alla loss
        # Priorità bassa -> campionata raramente -> peso IS alto -> contribuisce di più
        proba  = priorita / totale
        # evita divisione per zero
        proba  = np.maximum(proba, 1e-10)          
        pesi   = (self.dimensione * proba) ** (-beta)
        # normalizza: peso massimo = 1
        pesi  /= pesi.max()                       

        #  Conversione
        stati      = torch.tensor(self.stati[indici],      dtype=torch.float32).to(self.device)
        stati_succ = torch.tensor(self.stati_succ[indici], dtype=torch.float32).to(self.device)
        azioni     = torch.tensor(self.azioni[indici],     dtype=torch.int64).to(self.device)
        reward     = torch.tensor(self.reward[indici],     dtype=torch.float32).to(self.device)
        terminato  = torch.tensor(self.terminato[indici],  dtype=torch.float32).to(self.device)
        pesi_t     = torch.tensor(pesi,                    dtype=torch.float32).to(self.device)

        return stati, azioni, reward, stati_succ, terminato, pesi_t, indici

    def aggiorna_priorita(self, indici: np.ndarray, td_errors: np.ndarray):
        """
        Aggiorna le priorità dopo aver calcolato i nuovi TD errors.

        Chiamare dopo ogni aggiornamento della rete, passando gli stessi
        indici restituiti da campiona() e i TD errors calcolati.
        
        """
        for idx, td_err in zip(indici, td_errors):
            p = float((abs(td_err) + self._eps) ** self.alpha)
            self.max_priorita = max(self.max_priorita, p)   # aggiorna il massimo
            self._aggiorna_albero(int(idx), p)

    def __len__(self):
        return self.dimensione
