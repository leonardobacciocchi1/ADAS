"""
Training per PPO discreto su CarRacing-v3.

Stesso ambiente del DQN (continuous=False, 5 azioni, grayscale 84x84),
ma aggiornamento on-policy con clipped surrogate loss invece di Q-learning.

Struttura del loop:
  1. Raccogli 2048 step con la policy attuale (rollout)
  2. Calcola vantaggi GAE sul rollout
  3. Aggiorna la rete 10 volte sul rollout (minibatch da 64)
  4. Svuota il buffer e ricomincia

"""

import os
import csv
import time
import numpy as np
import torch
import gymnasium as gym
import gymnasium.wrappers as gym_wrap

from mio_ppo.agente import AgentePPO
from mio_ppo.buffer import RolloutBuffer
# stesso wrapper SkipFrame usato dal DQN
from train import SkipFrame   

# Configurazione 

MAX_STEP   = 1_000_000   # step totali di training
N_STEPS    = 2048        # step per rollout, aggiorna ogni 2048 step
N_EPOCHS   = 10          # quante volte ripassare il rollout per aggiornamento
BATCH_SIZE = 64          # dimensione minibatch durante l'aggiornamento

GAMMA          = 0.99    # discount factor
GAE_LAMBDA     = 0.95    # peso GAE (bilancia TD error e Monte Carlo)
CLIP_EPSILON   = 0.2     # clip ratio in [0.8, 1.2]: limita la variazione della policy
LR             = 3e-4    # learning rate Adam
COEFF_VALORE   = 0.5     # peso loss critico nella loss totale
COEFF_ENTROPIA = 0.01    # peso bonus entropia (incoraggia esplorazione)
MAX_GRAD_NORM  = 0.5     # gradient clipping

CARTELLA_MODELLI   = "mio_ppo/modelli_salvati"
CARTELLA_RISULTATI = "mio_ppo/risultati"
SALVA_OGNI         = 50    # salva checkpoint ogni N update
STAMPA_OGNI        = 5     # stampa statistiche ogni N update

RESUME_DA = None   # path checkpoint per riprendere il training, None = parte da zero


# Creazione ambiente 

def crea_ambiente():
    """
    Crea CarRacing-v3 con gli stessi wrapper del DQN:
      SkipFrame(4) → GrayscaleObservation → ResizeObservation(84,84) → FrameStack(4)
    Input alla rete: tensore (4, 84, 84) — identico al DQN.
    """
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = SkipFrame(env, skip=4)                              # 1 decisione ogni 4 frame reali
    env = gym_wrap.GrayscaleObservation(env)                  # RGB to grayscale (1 canale)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))     # 96x96 to 84x84
    env = gym_wrap.FrameStackObservation(env, stack_size=4)   # impila 4 frame to (4,84,84)
    return env


# Training 

def train():
    os.makedirs(CARTELLA_MODELLI,   exist_ok=True)
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)
    csv_path = os.path.join(CARTELLA_RISULTATI, "rewards_PPO_disc.csv")

    env      = crea_ambiente()
    stato, _ = env.reset()

    forma_stato = stato.shape          # (4, 84, 84)
    n_azioni    = env.action_space.n   # 5 azioni discrete

    # Crea agente PPO (una sola rete attore+critico, niente rete target)
    agente = AgentePPO(
        forma_stato    = forma_stato,
        n_azioni       = n_azioni,
        lr             = LR,
        gamma          = GAMMA,
        gae_lambda     = GAE_LAMBDA,
        clip_epsilon   = CLIP_EPSILON,
        n_epochs       = N_EPOCHS,
        coeff_valore   = COEFF_VALORE,
        coeff_entropia = COEFF_ENTROPIA,
        max_grad_norm  = MAX_GRAD_NORM,
    )

    # Crea il RolloutBuffer (temporaneo: 2048 step, poi svuotato)
    buffer = RolloutBuffer(
        n_steps    = N_STEPS,
        forma_stato= forma_stato,
        gamma      = GAMMA,
        gae_lambda = GAE_LAMBDA,
        device     = agente.device,
    )

    # Carica checkpoint se specificato (per continuare un training interrotto)
    if RESUME_DA and os.path.exists(RESUME_DA):
        agente.carica(RESUME_DA, modalita="train")
        print(f"  Checkpoint caricato: step={agente.step_totali:,}")

    print("=" * 65)
    print(f"  Training PPO (azioni discrete) su CarRacing-v3")
    print(f"  Target: {MAX_STEP:,} step  |  Rollout: {N_STEPS}  |  Epochs: {N_EPOCHS}")
    print(f"  Device: {agente.device}  |  Azioni: {n_azioni} discrete")
    print("=" * 65)

    # Inizializza CSV con header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episodio", "reward", "step"])

    rewards_episodi = []   # reward di ogni episodio completato
    reward_corrente = 0.0  # reward accumulata nell'episodio in corso
    n_episodi       = 0
    n_update        = 0    # numero di aggiornamenti fatti
    inizio          = time.time()
    done            = False

    try:
        while agente.step_totali < MAX_STEP:

            # Raccolta rollout (2048 step) 
            for _ in range(N_STEPS):
                # Attore sceglie l'azione, critico valuta V(s)
                azione, log_prob, value = agente.scegli_azione(stato)

                stato_succ, reward, terminated, truncated, _ = env.step(azione)
                done = terminated or truncated

                # Salva tutto nel buffer (inclusi log_prob e value — servono per GAE e clip)
                buffer.aggiungi(stato, azione, reward, value, log_prob, done)
                reward_corrente += reward
                stato = stato_succ

                # Se l'episodio finisce durante il rollout, registra il reward e resetta
                if done:
                    rewards_episodi.append(reward_corrente)
                    n_episodi += 1
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [n_episodi, round(reward_corrente, 2), agente.step_totali]
                        )
                    reward_corrente = 0.0
                    stato, _ = env.reset()

            # Calcola vantaggi GAE sull'intero rollout 
            # Bootstrap: stima V(s) dell'ultimo stato per completare il calcolo
            with torch.no_grad():
                last_value = agente.get_value(stato)
            buffer.calcola_vantaggi(last_value, done)

            #  Aggiorna la rete 10 volte sugli stessi 2048 step 
            losses = agente.aggiorna(buffer)

            #  Svuota il buffer — le esperienze non sono più valide 
            buffer.reset()
            n_update += 1

            # Stampa progresso
            if n_update % STAMPA_OGNI == 0:
                media10 = np.mean(rewards_episodi[-10:]) if rewards_episodi else 0.0
                elapsed = time.time() - inizio
                print(f"  Update {n_update:4d}"
                      f"  step={agente.step_totali:>9,}/{MAX_STEP:,}"
                      f"  ep={n_episodi:4d}"
                      f"  media10={media10:7.1f}"
                      f"  lp={losses['loss_policy']:7.4f}"
                      f"  lv={losses['loss_value']:7.4f}"
                      f"  ent={losses['entropia']:.3f}"
                      f"  t={elapsed/60:.1f}min")

            # Salva checkpoint periodico
            if n_update % SALVA_OGNI == 0:
                agente.salva(CARTELLA_MODELLI, "PPO_disc")

    except KeyboardInterrupt:
        print("\n  Interrotto — salvo il modello...")

    finally:
        env.close()
        agente.salva(CARTELLA_MODELLI, "PPO_disc_finale")
        print(f"\n  Step totali: {agente.step_totali:,}")
        print(f"  Episodi:     {n_episodi}")
        if rewards_episodi:
            print(f"  Media ultimi 10: {np.mean(rewards_episodi[-10:]):.1f}")
            print(f"  Media ultimi 50: {np.mean(rewards_episodi[-50:]):.1f}")


if __name__ == "__main__":
    train()
