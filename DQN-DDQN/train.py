"""
Training loop per DQN / DDQN su CarRacing-v3.

Uso:
  python train.py addestra DQN  
  python train.py --ddqn addestra DDQN 

Il training si ferma quando raggiunge MAX_STEP (740k step)
Se RESUME_DA punta a un checkpoint, il training riprende da dove era rimasto.

I risultati (reward per episodio) vengono salvati in:
  .../risultati/rewards_DQN.csv
  .../risultati/rewards_DDQN.csv
"""

import sys
import os
import csv
import time
import numpy as np
import gymnasium as gym
import gymnasium.wrappers as gym_wrap

from mio_dqn.agente import Agente

#Configurazione 

DOUBLE_Q      = "--ddqn" in sys.argv   # True = DDQN, False = DQN
NOME_ALGO     = "DDQN" if DOUBLE_Q else "DQN"

# "" = baseline (2 conv), "_3conv" = Nature DQN, "_3conv_per" = +PER, "_3conv_per_9a" = +9 azioni
VERSIONE      = "_3conv_per"
N_CONV        = 3        # 2 = baseline, 3 = Nature DQN
PER           = True     # True = Prioritized Experience Replay
NOVE_AZIONI   = False    # True = 9 azioni (griglia 3×3 sterzo×velocità)

# Target esteso a 1.1M step per superare i 900 di reward medio
MAX_STEP      = 1_100_000

# Checkpoint da cui riprendere il training (None = parte da zero).
# Aggiorna il path dopo ogni interruzione se vuoi continuare da dove eri rimasto.
RESUME_DQN    = None
RESUME_DDQN   = "mio_dqn/modelli_salvati/DDQN_3conv_per_finale_740037.pt"
RESUME_DA     = RESUME_DDQN if DOUBLE_Q else RESUME_DQN

N_EPISODI     = 50_000   # limite alto: con early termination gli episodi sono corti (~56 step)
SALVA_OGNI    = 100      # salva il modello ogni N episodi
STAMPA_OGNI   = 10       # stampa statistiche ogni N episodi
SKIP_FRAME    = 4        # esegui la stessa azione per N frame consecutivi
STACK_FRAME   = 4        # impila N frame grayscale come input alla CNN

CARTELLA_MODELLI   = "mio_dqn/modelli_salvati"
CARTELLA_RISULTATI = "mio_dqn/risultati"

#  9 Azioni discrete (griglia 3×3: sterzo × velocità) 
# Con 5 azioni (modalità discreta standard) non è possibile sterzare
# E accelerare contemporaneamente, in curva l'agente deve scegliere.
# Con 9 azioni usiamo continuous=True e mappiamo manualmente:

# Questo permette "gas + curva" come azione singola, traiettorie più fluide.

AZIONI_9 = np.array([
    [-1,  0,  0.8],  # 0: sinistra + freno
    [-1,  0,  0  ],  # 1: sinistra
    [-1,  1,  0  ],  # 2: sinistra + gas
    [ 0,  0,  0.8],  # 3: dritto   + freno
    [ 0,  0,  0  ],  # 4: niente
    [ 0,  1,  0  ],  # 5: dritto   + gas
    [ 1,  0,  0.8],  # 6: destra   + freno
    [ 1,  0,  0  ],  # 7: destra
    [ 1,  1,  0  ],  # 8: destra   + gas
], dtype=np.float32)


class DiscretizzaAzioni(gym.ActionWrapper):
    
    #Wrapper che converte 9 azioni discrete in azioni continue [sterzo, gas, freno].
    #Da usare con continuous=True: env.action_space diventa Discrete(9).
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(AZIONI_9))

    def action(self, azione: int) -> np.ndarray:
        return AZIONI_9[azione]


#  Wrapper SkipFrame 

class SkipFrame(gym.Wrapper):

    #Esegue la stessa azione per `skip` frame consecutivi e accumula il reward totale. 
    # Questo velocizza il training: l'agente deve prendere decisioni meno frequentemente, 
    # ma ogni decisione ha un impatto più lungo nel tempo.
    
    def __init__(self, env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, azione):
        reward_totale = 0.0
        for _ in range(self._skip):
            stato, reward, terminated, truncated, info = self.env.step(azione)
            reward_totale += reward
            if terminated:
                break
        return stato, reward_totale, terminated, truncated, info


def crea_ambiente(render: bool = False):
    
    #Crea e prepara l'ambiente CarRacing-v3 con tutti i wrapper.
    #Con NOVE_AZIONI=False (default): continuous=False, 5 azioni standard.
    #Con NOVE_AZIONI=True:  continuous=True + DiscretizzaAzioni -> 9 azioni.

    mode = "human" if render else None

    if NOVE_AZIONI:
        env = gym.make("CarRacing-v3", continuous=True, render_mode=mode)
        env = DiscretizzaAzioni(env)
    else:
        env = gym.make("CarRacing-v3", continuous=False, render_mode=mode)

    # SkipFrame prima degli altri wrapper: opera sui frame originali
    env = SkipFrame(env, skip=SKIP_FRAME)

    # Grayscale: da 3 canali RGB a 1 canale -> 3x meno dati, info visiva simile
    env = gym_wrap.GrayscaleObservation(env)

    # Resize: da 96x96 a 84x84 -> dimensione standard per DQN (DeepMind)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))

    # FrameStack: impila 4 frame consecutivi -> la rete "vede" il movimento
    env = gym_wrap.FrameStackObservation(env, stack_size=STACK_FRAME)

    return env


# Training loop 

def conta_episodi_csv(csv_path: str) -> int:
    #Restituisce il numero di episodi già presenti nel CSV (escludendo header).
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r") as f:
        return max(0, sum(1 for _ in f) - 1)  # -1 per l'header


def train():
    #prendo percorsi
    os.makedirs(CARTELLA_MODELLI,   exist_ok=True)
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

    csv_path = os.path.join(CARTELLA_RISULTATI, f"rewards_{NOME_ALGO}{VERSIONE}.csv")

    # Se parte da zero (no resume), azzera il CSV precedente
    # Se riprende da checkpoint, conta gli episodi già salvati
    if not RESUME_DA:
        episodio_offset = 0
    else:
        episodio_offset = conta_episodi_csv(csv_path)

    print("=" * 60)
    print(f"  Training {NOME_ALGO} su CarRacing-v3")
    print(f"  Target: {MAX_STEP:,} step  |  SkipFrame: {SKIP_FRAME}")
    if RESUME_DA:
        print(f"  Resume: {RESUME_DA}")
    print("=" * 60)

    # Crea ambiente e agente
    env    = crea_ambiente(render=False)
    stato, _ = env.reset()

    agente = Agente(
        forma_stato   = stato.shape,   # (4, 84, 84)
        n_azioni      = env.action_space.n,
        double_q      = DOUBLE_Q,
        n_conv        = N_CONV,
        per           = PER,
        per_beta_passi= MAX_STEP,     # beta raggiunge 1.0 alla fine del training
        sync_ogni     = 10_000 if not DOUBLE_Q else 5_000,
    )

    # Carica checkpoint se specificato
    if RESUME_DA and os.path.exists(RESUME_DA):
        agente.carica(RESUME_DA, modalita="train")
        print(f"  Checkpoint caricato: step={agente.step_totali:,}  eps={agente.epsilon:.3f}")
    elif RESUME_DA:
        print(f"  ATTENZIONE: checkpoint '{RESUME_DA}' non trovato, parto da zero.")

    print(f"  Device: {agente.device}")
    print(f"  Forma stato: {stato.shape}")
    print(f"  Azioni: {env.action_space.n}")
    print(f"  Step già fatti: {agente.step_totali:,} / {MAX_STEP:,}")
    print()

    # Liste per salvare i risultati di questa sessione
    rewards_episodi = []
    losses_episodi  = []
    inizio          = time.time()

    # Crea (o sovrascrive) il CSV se parte da zero, altrimenti append al già presente
    if episodio_offset == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episodio", "reward", "loss"])

    try:
        for episodio in range(1, N_EPISODI + 1):

            # Condizione di stop basata sugli step 
            if agente.step_totali >= MAX_STEP:
                print(f"\n  Raggiunto target {MAX_STEP:,} step — training completato.")
                break

            stato, _ = env.reset()
            reward_ep = 0.0
            losses_ep = []
            done      = False

            # Loop di un episodio 
            while not done:
                # 1. Scegli azione (epsilon-greedy)
                azione = agente.scegli_azione(stato)

                # 2. Esegui azione nell'ambiente
                stato_succ, reward, terminated, truncated, _ = env.step(azione)
                done = terminated or truncated

                # 3. Salva l'esperienza nel buffer
                agente.memorizza(stato, azione, reward, stato_succ, done)

                # 4. Aggiorna la rete (se il buffer è pronto)
                loss = agente.aggiorna()
                if loss is not None:
                    losses_ep.append(loss)

                reward_ep += reward
                stato = stato_succ

            # Fine episodio 
            rewards_episodi.append(reward_ep)
            loss_media = np.mean(losses_ep) if losses_ep else 0.0
            losses_episodi.append(loss_media)

            # Stampa progresso ogni STAMPA_OGNI episodi
            if episodio % STAMPA_OGNI == 0:
                media_10  = np.mean(rewards_episodi[-10:])
                elapsed   = time.time() - inizio
                ep_globale = episodio_offset + episodio
                print(f"  Ep {ep_globale:4d}"
                      f"  reward={reward_ep:7.1f}"
                      f"  media10={media_10:7.1f}"
                      f"  loss={loss_media:.4f}"
                      f"  eps={agente.epsilon:.3f}"
                      f"  step={agente.step_totali:,}/{MAX_STEP:,}"
                      f"  t={elapsed/60:.1f}min")

            # Salva modello ogni SALVA_OGNI episodi
            if episodio % SALVA_OGNI == 0:
                agente.salva(CARTELLA_MODELLI, f"{NOME_ALGO}{VERSIONE}")

    except KeyboardInterrupt:
        print("\n  Interrotto dall'utente — salvo il modello...")

    finally:
        env.close()

        # Salva modello finale
        agente.salva(CARTELLA_MODELLI, f"{NOME_ALGO}{VERSIONE}_finale")

        # Appende i nuovi episodi al CSV (mantiene quelli precedenti)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i, (r, l) in enumerate(zip(rewards_episodi, losses_episodi), episodio_offset + 1):
                writer.writerow([i, round(r, 2), round(l, 6)])

        print(f"\n  Rewards salvati: {csv_path}")
        print(f"  Episodi questa sessione: {len(rewards_episodi)}")
        print(f"  Episodi totali nel CSV: {episodio_offset + len(rewards_episodi)}")
        print(f"  Step totali: {agente.step_totali:,} / {MAX_STEP:,}")
        if rewards_episodi:
            print(f"  Reward finale (media ultimi 10): {np.mean(rewards_episodi[-10:]):.1f}")


if __name__ == "__main__":
    train()
