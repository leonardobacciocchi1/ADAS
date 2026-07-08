"""
Training PPO con Stable-Baselines3 su CarRacing-v3.

A differenza del PPO custom, qui l'algoritmo PPO è gestito
interamente dalla libreria SB3 — configuriamo solo gli iperparametri.

Differenze principali rispetto al PPO custom:
  - 8 ambienti paralleli (SubprocVecEnv) invece di 1
  - RGB invece di grayscale: (84,84,12) invece di (4,84,84) - limite tecnico SB3
  - 2M step totali invece di 1M per ottenre risultati minimamente compatibili

"""

import os
import csv
import numpy as np
import gymnasium as gym
import gymnasium.wrappers as gym_wrap

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from train import SkipFrame   # stesso wrapper SkipFrame usato dal DQN

# Configurazione 

N_ENVS    = 8           # ambienti paralleli su processi separati
MAX_STEP  = 2_000_000   # step totali sommati su tutti gli 8 env

# Iperparametri da SB3 ottimizzati per CarRacing
N_STEPS    = 512     # step per env per rollout -> batch totale = 512 × 8 = 4096
BATCH_SIZE = 128     # dimensione minibatch per l'aggiornamento
N_EPOCHS   = 10      # passate sul rollout per ogni aggiornamento
GAMMA      = 0.99    # discount factor
GAE_LAMBDA = 0.95    # peso GAE
LR         = 1e-4    # learning rate (più basso del custom: 3e-4)
CLIP_RANGE = 0.2     # clip ratio [0.8, 1.2]
ENT_COEF   = 0.0     # nessun bonus entropia (diverso dal custom che usa 0.01)
VF_COEF    = 0.5     # peso loss critico
MAX_GRAD_NORM = 0.5  # gradient clipping

CARTELLA_MODELLI   = "sb3_ppo/modelli_salvati"
CARTELLA_RISULTATI = "sb3_ppo/risultati"
SALVA_OGNI_STEP    = 200_000   # salva checkpoint ogni 200k step

# Funzione ambiente 

def crea_env():
    
    # Crea un singolo ambiente CarRacing-v3.
    
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    env = SkipFrame(env, skip=4)                            # 1 decisione ogni 4 frame
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))   # 96x96 → 84x84 (RGB mantenuto)
    return env


# Callback personalizzata 

class CSVRewardCallback(BaseCallback):
    
    #Callback SB3 che salva le reward per episodio su CSV e i checkpoint periodici.

    #SB3 non espone direttamente le reward degli episodi — le recuperiamo da info["episode"] che VecMonitor 
    # inserisce quando un episodio termina.

    #Salva lo stesso formato CSV degli altri algoritmi (episodio, reward, step) 
    # per poter confrontare le curve di apprendimento.
    
    def __init__(self, csv_path: str, salva_ogni: int, cartella_modelli: str):
        super().__init__(verbose=0)
        self.csv_path         = csv_path
        self.salva_ogni       = salva_ogni
        self.cartella_modelli = cartella_modelli
        self._n_episodi       = 0
        self._prossimo_salva  = salva_ogni

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episodio", "reward", "step"])

    def _on_step(self) -> bool:
        # VecMonitor inserisce info["episode"] quando un episodio termina
        # Può arrivare da uno qualsiasi degli 8 env paralleli
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._n_episodi += 1
                reward = ep["r"]      # reward totale dell'episodio
                step   = self.num_timesteps
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self._n_episodi, round(float(reward), 2), step]
                    )

        # Salva checkpoint ogni "SALVA_OGNI_STEP" step
        if self.num_timesteps >= self._prossimo_salva:
            path = os.path.join(
                self.cartella_modelli,
                f"SB3_PPO_{self.num_timesteps}.zip"
            )
            self.model.save(path)
            print(f"  [checkpoint] salvato: {path}")
            self._prossimo_salva += self.salva_ogni

        return True   # True = continua il training, False = interrompi


# Training 

def train():
    os.makedirs(CARTELLA_MODELLI,   exist_ok=True)
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

    # 8 ambienti paralleli su processi separati 
    # SubprocVecEnv lancia ogni env in un subprocess → sfrutta tutti i core CPU
    # make_vec_env chiama crea_env() 8 volte, una per processo
    vec_env = make_vec_env(
        crea_env,
        n_envs      = N_ENVS,
        vec_env_cls = SubprocVecEnv,
    )

    # VecFrameStack: impila 4 frame RGB 
    # Applicato dopo SubprocVecEnv perché opera sul vettore di env
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # VecMonitor: traccia reward e lunghezza episodi in info["episode"]
    # Necessario per il callback CSVRewardCallback
    vec_env = VecMonitor(vec_env)

    csv_path = os.path.join(CARTELLA_RISULTATI, "rewards_SB3_PPO.csv")
    callback = CSVRewardCallback(
        csv_path         = csv_path,
        salva_ogni       = SALVA_OGNI_STEP,
        cartella_modelli = CARTELLA_MODELLI,
    )

    # Modello PPO di SB3 
    # "CnnPolicy": SB3 usa la sua NatureCNN internamente 
    # L'algoritmo PPO completo (rete, buffer, aggiornamento) è dentro SB3
    model = PPO(
        policy        = "CnnPolicy",   # rete CNN gestita da SB3
        env           = vec_env,
        n_steps       = N_STEPS,       # step per env per rollout
        batch_size    = BATCH_SIZE,    # minibatch per aggiornamento
        n_epochs      = N_EPOCHS,      # passate sul rollout
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        learning_rate = LR,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF,      # 0.0: nessun bonus entropia
        vf_coef       = VF_COEF,
        max_grad_norm = MAX_GRAD_NORM,
        verbose       = 1,
        device        = "cuda",
    )

    print("=" * 65)
    print(f"  Training SB3-PPO su CarRacing-v3")
    print(f"  Ambienti paralleli: {N_ENVS}  |  Target: {MAX_STEP:,} step")
    print(f"  N_steps={N_STEPS}  batch={BATCH_SIZE}  lr={LR}  ent_coef={ENT_COEF}")
    print(f"  Tempo stimato: ~6-10 ore (RTX 4060)")
    print("=" * 65)

    # Lancia il training — SB3 gestisce internamente il loop raccolta/aggiornamento
    model.learn(
        total_timesteps     = MAX_STEP,
        callback            = callback,
        reset_num_timesteps = True,
        progress_bar        = False,
    )

    # Salva il modello finale
    finale = os.path.join(CARTELLA_MODELLI, f"SB3_PPO_finale_{MAX_STEP}.zip")
    model.save(finale)
    print(f"\n  Modello finale salvato: {finale}")
    print(f"  Reward CSV: {csv_path}")
    vec_env.close()


if __name__ == "__main__":
    train()
