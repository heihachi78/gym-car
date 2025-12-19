# Megerősítéses Tanulás: Autóverseny AI

Ez a dokumentáció részletesen bemutatja, hogyan tanítunk meg egy mesterséges intelligenciát autót vezetni a CarRacing-v3 környezetben. A projekt a **Proximal Policy Optimization (PPO)** algoritmust használja **LSTM** (Long Short-Term Memory) neurális hálózattal.

---

## Tartalomjegyzék

1. [Bevezetés a Megerősítéses Tanulásba](#1-bevezetés-a-megerősítéses-tanulásba)
2. [A Környezet (Environment)](#2-a-környezet-environment)
3. [Megfigyelések Feldolgozása](#3-megfigyelések-feldolgozása)
4. [A Neurális Hálózat Architektúrája](#4-a-neurális-hálózat-architektúrája)
5. [A PPO Algoritmus](#5-a-ppo-algoritmus)
6. [Generalized Advantage Estimation (GAE)](#6-generalized-advantage-estimation-gae)
7. [A Tanítási Folyamat](#7-a-tanítási-folyamat)
8. [Kód Struktúra](#8-kód-struktúra)
9. [Hiperparaméterek](#9-hiperparaméterek)
10. [Gyakori Kérdések](#10-gyakori-kérdések)

---

## 1. Bevezetés a Megerősítéses Tanulásba

### Mi az a Megerősítéses Tanulás (Reinforcement Learning)?

A megerősítéses tanulás (RL) a gépi tanulás egy ága, ahol egy **ágens** (agent) megtanul döntéseket hozni egy **környezetben** (environment) azáltal, hogy **jutalmakat** (rewards) kap vagy büntetéseket szenved el.

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Alapvető Ciklus                       │
│                                                             │
│    ┌─────────┐    akció (a_t)     ┌─────────────┐          │
│    │         │ ────────────────>  │             │          │
│    │  Ágens  │                    │  Környezet  │          │
│    │         │ <────────────────  │             │          │
│    └─────────┘  állapot (s_t+1)   └─────────────┘          │
│                 jutalom (r_t)                               │
└─────────────────────────────────────────────────────────────┘
```

### Kulcsfogalmak

| Fogalom | Jelentés | Példa a projektben |
|---------|----------|-------------------|
| **Állapot (State)** | A környezet aktuális helyzete | A feldolgozott kamerakép (40×48 pixel) |
| **Akció (Action)** | Az ágens döntése | Balra, jobbra, gáz, fék, semmi |
| **Jutalom (Reward)** | Visszajelzés a döntésről | +1 ha előrehalad, -1 ha letér |
| **Politika (Policy)** | Döntési stratégia | A neurális hálózat kimenete |
| **Érték (Value)** | Várható jövőbeli jutalom | Mennyire "jó" egy állapot |

### Policy Gradient módszerek

A projektben **policy gradient** módszert használunk. Ez közvetlenül a politikát (π) optimalizálja, szemben a value-based módszerekkel (pl. DQN), amelyek az értékfüggvényt tanulják.

**Miért policy gradient?**
- Természetesen kezel sztochasztikus politikákat
- Jól működik folytonos és diszkrét akciótérrel
- Stabilabb konvergencia bizonyos feladatoknál

---

## 2. A Környezet (Environment)

### CarRacing-v3

A **Gymnasium** (korábban OpenAI Gym) könyvtár CarRacing-v3 környezete egy felülnézeti autóversenyzős szimuláció.

```python
# A környezet létrehozása (utils/env_factory.py)
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",      # Kép kimenet
    lap_complete_percent=0.95,    # 95%-nál vége az epizódnak
    domain_randomize=False,       # Konzisztens pálya
    continuous=False              # Diszkrét akciók
)
```

### Akciótér (Action Space)

A környezet **5 diszkrét akciót** kínál:

| Akció ID | Jelentés | Leírás |
|----------|----------|--------|
| 0 | Semmi | Nincs input |
| 1 | Balra | Kormány balra |
| 2 | Jobbra | Kormány jobbra |
| 3 | Gáz | Gyorsítás |
| 4 | Fék | Lassítás |

### Jutalom Rendszer

A CarRacing-v3 beépített jutalom rendszere:

```
+1000/N  - Minden megtett pálya tile után (N = összes tile)
-0.1     - Minden lépésért (időbüntetés)
-100     - Ha teljesen letér a pályáról
```

**Tipikus epizód jutalom:**
- Rossz ágens: -100 körül (gyorsan letér)
- Közepes ágens: 200-500 (lassú, de a pályán marad)
- Jó ágens: 700-900+ (gyors és pontos)

---

## 3. Megfigyelések Feldolgozása

### Miért kell feldolgozni a képet?

A nyers kép (96×96×3 RGB) feldolgozása segít a hatékonyabb tanulásban:
1. **Eltávolítja a felesleges területeket** → A műszerfal nem releváns
2. **Normalizálja az értékeket** → Stabilabb gradiens flow
3. **Megtartja a színinformációt** → A pálya és a fű színe hasznos

### Feldolgozási Pipeline

```
Nyers kép (96×96×3 RGB)
        │
        ▼
┌───────────────────┐
│  CropObservation  │  Levágja a műszerfalat (alsó 16 sor)
│    (80×96×3)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ NormalizeObserv.  │  Normalizálás (futó átlag és szórás)
│    (80×96×3)      │
└───────────────────┘
```

**Megjegyzés:** A korábbi verziók használtak szürkeárnyalatos konverziót, élesítést és éldetektálást is. Ezek a wrapperek továbbra is elérhetők a `wrappers/` mappában, de a jelenlegi konfiguráció RGB képekkel dolgozik, mert a színinformáció (zöld fű vs. szürke pálya) hasznos a tanuláshoz.

### Wrapper-ek Részletesen

#### 1. CropObservation (`wrappers/crop_observation.py`)

**Cél:** Eltávolítja a műszerfalat a kép aljáról.

```python
class CropObservation(ObservationWrapper):
    def __init__(self, env, height=80, width=96):
        # A kép felső 80 sorát tartjuk meg
        # Ez eltávolítja a műszerfalat az aljáról

    def observation(self, observation):
        return observation[:self.height, :self.width, :]
```

**Vizuális magyarázat:**
```
┌────────────────────┐
│                    │  ← Pálya (ezt tartjuk meg: 80 sor)
│     PÁLYA          │
│                    │
├────────────────────┤
│   MŰSZERFAL        │  ← Ezt levágjuk (16 sor)
└────────────────────┘
```

#### 2. NormalizeObservation (Gymnasium beépített)

**Cél:** Normalizálja a megfigyeléseket 0 átlagra és 1 szórásra

```python
# Belső működés (egyszerűsítve):
normalized = (observation - running_mean) / running_std
```

**Miért fontos?**
- A neurális hálózatok jobban tanulnak normalizált inputtal
- Stabilizálja a gradiens flow-t
- Gyorsabb konvergencia

##### Normalizációs Statisztikák Mentése és Betöltése

A `NormalizeObservation` wrapper **futó statisztikákat** (running mean/std) használ, amelyek a tanítás során frissülnek. Ezeket a statisztikákat el kell menteni a checkpoint-tal együtt, különben az inferenciánál a modell rosszul fog működni!

**A probléma:**
```
Tanítás közben:     obs_rms.mean ≈ 0.15,  obs_rms.var ≈ 0.08
Új környezetben:    obs_rms.mean = 0.0,   obs_rms.var = 1.0  ← ROSSZ!
```

Ha nem mentjük/töltjük be a statisztikákat, a modell teljesen más normalizált értékeket kap, mint amikre tanítva lett.

**Megoldás: Statisztikák mentése checkpoint-ba** (`training/ppo_trainer.py`):

```python
def save_checkpoint(self, path: str, step: int, episode_rewards=None, env=None):
    checkpoint = {
        'step': step,
        'network_state_dict': self.network.state_dict(),
        # ...
    }

    # Normalizációs statisztikák mentése
    if env is not None:
        norm_wrapper = get_normalize_wrapper(env)
        if norm_wrapper is not None:
            checkpoint['obs_rms'] = {
                'mean': norm_wrapper.obs_rms.mean,
                'var': norm_wrapper.obs_rms.var,
                'count': norm_wrapper.obs_rms.count,
            }

    torch.save(checkpoint, path)
```

**Statisztikák betöltése inferenciánál** (`main.py`):

```python
# Checkpoint betöltése
checkpoint = torch.load(checkpoint_path)

# Normalizációs statisztikák visszaállítása
if 'obs_rms' in checkpoint:
    norm_wrapper = get_normalize_wrapper(env)
    if norm_wrapper is not None:
        norm_wrapper.obs_rms.mean = checkpoint['obs_rms']['mean']
        norm_wrapper.obs_rms.var = checkpoint['obs_rms']['var']
        norm_wrapper.obs_rms.count = checkpoint['obs_rms']['count']
        norm_wrapper.update_running_mean = False  # Fagyasztás inferenciánál!
```

**`update_running_mean = False` fontossága:**

Inferenciánál nem akarjuk, hogy a statisztikák frissüljenek:
- A modell a tanításkori statisztikákra lett optimalizálva
- Ha folyamatosan frissülnének, az változtatná a normalizált értékeket
- Ez instabil viselkedést eredményezne

```
┌─────────────────────────────────────────────────────────────────┐
│           Normalizációs Statisztikák Életciklusa                │
│                                                                 │
│   TANÍTÁS                         INFERENCIA                    │
│   ┌──────────────────┐           ┌──────────────────┐          │
│   │ obs_rms frissül  │    ──>    │ obs_rms FAGYASZTVA│          │
│   │ minden step-nél  │  MENTÉS   │ nem változik      │          │
│   └──────────────────┘           └──────────────────┘          │
│          │                              ▲                       │
│          ▼                              │                       │
│   ┌──────────────────┐           ┌──────────────────┐          │
│   │   Checkpoint     │    ──>    │   Checkpoint     │          │
│   │   + obs_rms      │  BETÖLTÉS │   + obs_rms      │          │
│   └──────────────────┘           └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**Helper függvény a wrapper megtalálásához** (`utils/env_factory.py`):

```python
def get_normalize_wrapper(env: gym.Env) -> NormalizeObservation | None:
    """
    Megkeresi a NormalizeObservation wrapper-t a wrapper láncban.
    """
    current = env
    while current is not None:
        if isinstance(current, NormalizeObservation):
            return current
        current = getattr(current, 'env', None)
    return None
```

---

## 4. A Neurális Hálózat Architektúrája

### Actor-Critic Architektúra

Az **Actor-Critic** egy hibrid megközelítés, amely kombinálja:
- **Actor (Színész):** Eldönti, melyik akciót válassza
- **Critic (Kritikus):** Megbecsüli, mennyire jó az aktuális állapot

```
┌─────────────────────────────────────────────────────────────────┐
│                   ActorCriticLSTM Architektúra                  │
│                                                                 │
│   Input: (batch, seq_len, 3, 80, 96) - RGB kép                 │
│              │                                                  │
│              ▼                                                  │
│   ┌─────────────────────────┐                                  │
│   │   CNN Feature Extractor │  Térbeli jellemzők kinyerése     │
│   │   (4 konvolúciós réteg) │  BatchNorm + ReLU                │
│   └───────────┬─────────────┘                                  │
│               │ (batch, seq_len, 512)                          │
│               ▼                                                  │
│   ┌─────────────────────────┐                                  │
│   │        LSTM             │  Időbeli függőségek tanulása     │
│   │   (hidden_size=512)     │                                  │
│   └───────────┬─────────────┘                                  │
│               │ (batch, seq_len, 512)                          │
│       ┌───────┴───────┐                                        │
│       │               │                                        │
│       ▼               ▼                                        │
│   ┌───────────┐   ┌───────────┐                                │
│   │   Actor   │   │  Critic   │  Külön MLP fejek               │
│   │ 512→512→5 │   │ 512→512→1 │  (2 réteg mindkettő)           │
│   └─────┬─────┘   └─────┬─────┘                                │
│         │               │                                       │
│         ▼               ▼                                       │
│     5 akció          1 érték                                    │
│     logits           becslés                                    │
└─────────────────────────────────────────────────────────────────┘
```

### CNN Feature Extractor (`models/cnn_feature_extractor.py`)

A **Convolutional Neural Network (CNN)** a képekből jellemzőket (feature) von ki. A jelenlegi implementáció 4 konvolúciós réteget használ BatchNorm normalizálással.

```python
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape=(3, 80, 96), output_size=512):
        self.conv = nn.Sequential(
            # Conv1: 3 → 64 csatorna (RGB input)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Input: (3, 80, 96) → Output: (64, 40, 48)

            # Conv2: 64 → 128 csatorna
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Input: (64, 40, 48) → Output: (128, 20, 24)

            # Conv3: 128 → 256 csatorna
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Input: (128, 20, 24) → Output: (256, 10, 12)

            # Conv4: 256 → 256 csatorna
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Input: (256, 10, 12) → Output: (256, 5, 6)

            nn.Flatten(),
            # Output: 256 × 5 × 6 = 7680
        )

        self.fc = nn.Sequential(
            nn.Linear(7680, 512),  # Dinamikusan számított
            nn.ReLU()
        )
```

**BatchNorm előnyei:**
- Stabilizálja a tanulást a belső covariate shift csökkentésével
- Lehetővé teszi magasabb learning rate használatát
- Enyhe regularizációs hatás

**Konvolúció működése:**

```
Input kép (3, 80, 96)           Kernel (4×4)
┌───────────────────┐           ┌───────┐
│ ░░░░░░░░░░░░░░░░ │           │ w w w │
│ ░░░░░░░░░░░░░░░░ │     *     │ w w w │  =  Feature map
│ ░░░░░░░░░░░░░░░░ │           │ w w w │
│ ░░░░░░░░░░░░░░░░ │           └───────┘
└───────────────────┘
```

**Paraméterek magyarázata:**
- `kernel_size`: A szűrő mérete (4×4 vagy 3×3)
- `stride`: Lépésköz (2 = minden második pozíció → felbontás felezése)
- `padding`: Szegély kitöltés (1 = méret megőrzés stride=2 esetén)

### LSTM Réteg

Az **LSTM (Long Short-Term Memory)** egy rekurrens neurális hálózat, amely "emlékszik" a korábbi lépésekre.

**Miért kell LSTM autóvezetéshez?**

Az autóvezetés **szekvenciális döntések** sorozata:
- A jelenlegi sebesség a múltbeli gyorsításoktól függ
- A kormányzás a korábbi pozíciótól függ
- A pálya alakja időben változik

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Működése                            │
│                                                             │
│  Időpont:    t-2       t-1        t        t+1             │
│              │         │          │         │               │
│              ▼         ▼          ▼         ▼               │
│           ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐            │
│  Input:   │kép_1│   │kép_2│   │kép_3│   │kép_4│            │
│           └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘            │
│              │         │          │         │               │
│              ▼         ▼          ▼         ▼               │
│           ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐            │
│  LSTM:    │cell │──>│cell │──>│cell │──>│cell │            │
│           └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘            │
│              │         │          │         │               │
│  Hidden:   h_1───────>h_2───────>h_3───────>h_4            │
│           (memória átadása időben)                          │
└─────────────────────────────────────────────────────────────┘
```

**LSTM belső szerkezete:**

```python
# Egyszerűsített LSTM egyenletek:

# Forget gate - mit felejtsünk el
f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)

# Input gate - mit jegyezzünk meg
i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)

# Candidate values - új információ
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

# Cell state update - memória frissítése
c_t = f_t * c_{t-1} + i_t * c̃_t

# Output gate - mit adjunk ki
o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)

# Hidden state - kimenet
h_t = o_t * tanh(c_t)
```

**LSTMState osztály (`models/lstm_state.py`):**

```python
@dataclass
class LSTMState:
    """LSTM rejtett állapot tárolása"""
    h: torch.Tensor  # Hidden state (rövid távú memória)
    c: torch.Tensor  # Cell state (hosszú távú memória)

    # Shape: (num_layers, batch_size, hidden_size)
```

### Actor és Critic Fejek

A jelenlegi implementáció **külön MLP fejeket** használ az actor és critic számára, nem egyszerű lineáris rétegeket. Ez lehetővé teszi, hogy az actor és critic különböző reprezentációkat tanuljanak.

```python
# Actor: Politika kimenet (melyik akciót válasszuk)
# 2 rétegű MLP: 512 → 512 → 5
self.actor = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),  # 512 → 512
    nn.ReLU(),
    nn.Linear(hidden_size, num_actions)   # 512 → 5
)

# Critic: Érték becslés (mennyire jó ez az állapot)
# 2 rétegű MLP: 512 → 512 → 1
self.critic = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),  # 512 → 512
    nn.ReLU(),
    nn.Linear(hidden_size, 1)             # 512 → 1
)
```

**Miért külön MLP fejek?**
- Az actor és critic különböző célokra optimalizálnak
- A közös LSTM reprezentáció felett külön "szakértő" rétegek tanulnak
- Csökkenti az interferenciát a policy és value tanulás között

**Actor kimenet feldolgozása:**

```python
# Logits → Valószínűségek
policy_logits = self.actor(lstm_out)      # [-2.1, 0.5, 1.2, -0.3, 0.8]
probs = softmax(policy_logits)            # [0.02, 0.14, 0.45, 0.06, 0.33]

# Kategorikus eloszlásból mintavételezés
dist = Categorical(probs)
action = dist.sample()                     # pl. 2 (jobbra)
log_prob = dist.log_prob(action)           # log(0.45) = -0.80
```

### Súly Inicializálás

A jó inicializálás kritikus a stabil tanuláshoz:

```python
def _init_weights(self):
    """CNN és LSTM súlyok inicializálása."""
    for module in self.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Orthogonal inicializálás ReLU-hoz
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

def _init_heads(self):
    """Actor és Critic fejek inicializálása."""
    # Actor: Rejtett réteg ReLU gain-nel, kimenet kis gain-nel
    nn.init.orthogonal_(self.actor[0].weight, gain=nn.init.calculate_gain('relu'))
    nn.init.zeros_(self.actor[0].bias)
    nn.init.orthogonal_(self.actor[2].weight, gain=0.01)  # Kis súlyok
    nn.init.zeros_(self.actor[2].bias)

    # Critic: Rejtett réteg ReLU gain-nel, kimenet gain=1
    nn.init.orthogonal_(self.critic[0].weight, gain=nn.init.calculate_gain('relu'))
    nn.init.zeros_(self.critic[0].bias)
    nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
    nn.init.zeros_(self.critic[2].bias)
```

**Miért orthogonal inicializálás?**
- Megőrzi a gradiens nagyságát a rétegeken át
- Elkerüli a "vanishing/exploding gradient" problémát
- Gyorsabb konvergenciát eredményez

**Actor kis gain (0.01):**
- Kezdetben közel egyenletes akció-eloszlást eredményez
- Ösztönzi a felfedezést a tanulás elején

---

## 5. A PPO Algoritmus

### Mi az a PPO (Proximal Policy Optimization)?

A **PPO** egy policy gradient algoritmus, amelyet az OpenAI fejlesztett 2017-ben. Célja a **stabil** és **hatékony** policy tanulás.

### A Policy Gradient Probléma

Az alap policy gradient módszer (REINFORCE) problémája:
- **Nagy variancia:** A jutalmak nagyon változékonyak
- **Instabil frissítések:** Egy rossz lépés tönkreteheti a tanulást
- **Minta-hatékonytalan:** Sok tapasztalat kell

### PPO Megoldása: Clipped Surrogate Objective

A PPO **korlátozza**, mennyit változhat a politika egy frissítés során.

```
┌────────────────────────────────────────────────────────────────┐
│                    PPO Objective Function                      │
│                                                                │
│   L^{CLIP}(θ) = E[ min( r_t(θ) · A_t,                        │
│                        clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]       │
│                                                                │
│   ahol:                                                        │
│   - r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (ratio)      │
│   - A_t = advantage (előny)                                   │
│   - ε = 0.2 (clip range)                                      │
└────────────────────────────────────────────────────────────────┘
```

### Ratio és Clipping Vizualizáció

```
                    Advantage > 0
                    (jó akció)
                         │
        r(θ) < 1-ε      │      r(θ) > 1+ε
        ┌───────────────┼───────────────┐
        │   túl sokat   │   túl sokat   │
        │   csökkent    │   növekedett  │
        │               │               │
   0.8 ─┼───────────────┼───────────────┼─ 1.2
        │    megengedett tartomány      │
        │    (1-ε) ─────┼───── (1+ε)    │
        │               │               │
        │     ratio     │               │
        └───────────────┼───────────────┘
                        1.0
                    (változatlan)
```

**A clipping hatása:**
- Ha `ratio > 1.2`: A frissítés le van vágva (nem nő tovább)
- Ha `ratio < 0.8`: A frissítés le van vágva (nem csökken tovább)
- Ez **megakadályozza a túl nagy politika változásokat**

### PPO Implementáció (`training/ppo_trainer.py`)

```python
def update(self, buffer: RolloutBuffer) -> dict:
    for epoch in range(self.num_epochs):  # 4 epoch
        for batch in buffer.get_batches(batch_size, seq_len):
            obs = batch['observations']
            actions = batch['actions']
            old_log_probs = batch['old_log_probs']
            advantages = batch['advantages']
            returns = batch['returns']
            hidden_state = batch['hidden_state']
            masks = batch['masks']  # Maszk epizód végek kezelésére

            # 1. Forward pass - új politika értékelése
            _, new_log_probs, entropy, values, _ = self.network.get_action_and_value(
                obs, hidden_state, actions
            )

            # 2. Ratio számítás
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 3. Clipped és unclipped objective
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)

            policy_loss_unclipped = -advantages * ratio
            policy_loss_clipped = -advantages * clipped_ratio

            # A ROSSZABBAT választjuk (pesszimista becslés)
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped)

            # Maszkolás: epizód vége utáni lépések kizárása
            policy_loss = (policy_loss * masks).sum() / (masks.sum() + 1e-8)

            # 4. Value loss (MSE) - opcionális clipping-gel
            value_loss = (values - returns) ** 2
            value_loss = (value_loss * masks).sum() / (masks.sum() + 1e-8)

            # 5. Entropy bonus (felfedezés ösztönzése)
            entropy_loss = -(entropy * masks).sum() / (masks.sum() + 1e-8)

            # 6. Teljes veszteség
            loss = policy_loss + 0.5 * value_loss + 0.03 * entropy_loss

            # 7. Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # 8. Gradient clipping
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)

            optimizer.step()
```

**Maszkolás fontossága:**
Az LSTM szekvenciákon belül előfordulhat, hogy egy epizód véget ér. A maszk biztosítja, hogy az epizód vége utáni lépések (amelyek már egy új epizódból származnak) ne befolyásolják a loss számítást.

### Veszteség Komponensek Részletesen

#### 1. Policy Loss (Politika Veszteség)

```python
# Cél: Növelni a jó akciók valószínűségét
policy_loss = -advantages * ratio

# Ha advantage > 0 (jó akció volt):
#   → ratio növelése csökkenti a loss-t
#   → tehát a hálózat növeli az akció valószínűségét

# Ha advantage < 0 (rossz akció volt):
#   → ratio csökkentése csökkenti a loss-t
#   → tehát a hálózat csökkenti az akció valószínűségét
```

#### 2. Value Loss (Érték Veszteség)

```python
# Cél: A critic pontosan becsülje az értéket
value_loss = (values - returns) ** 2

# MSE (Mean Squared Error) a becsült és valós return között
# Minél kisebb, annál pontosabb a critic
```

#### 3. Entropy Loss (Entrópia Veszteség)

```python
# Cél: Megakadályozni a túl korai konvergenciát
entropy = -sum(p * log(p))  # Shannon entrópia

entropy_loss = -entropy  # Negatív, mert maximalizálni akarjuk

# Magas entrópia = egyenletesebb eloszlás = több felfedezés
# Alacsony entrópia = egy akció dominál = kevesebb felfedezés
```

**Entrópia példa:**
```
Egyenletes eloszlás:    [0.2, 0.2, 0.2, 0.2, 0.2]  → Entrópia: 1.61
Koncentrált eloszlás:   [0.9, 0.025, 0.025, 0.025, 0.025] → Entrópia: 0.47
```

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
```

**Miért kell?**
- Megakadályozza a "gradient explosion"-t
- Stabilizálja a tanulást
- A gradiensek L2 normáját 0.5-re korlátozza

---

## 6. Generalized Advantage Estimation (GAE)

### Mi az Advantage?

Az **advantage** megmutatja, mennyivel jobb (vagy rosszabb) egy akció az átlagosnál:

```
A(s, a) = Q(s, a) - V(s)

ahol:
- Q(s, a) = Várható jutalom, ha 'a' akciót választjuk 's' állapotban
- V(s) = Átlagos várható jutalom 's' állapotban
```

**Intuitívan:**
- `A > 0`: Az akció jobb volt az átlagosnál → ösztönözni
- `A < 0`: Az akció rosszabb volt az átlagosnál → büntetni
- `A ≈ 0`: Az akció átlagos volt → nem változtatni sokat

### A GAE Probléma: Bias vs. Variance Tradeoff

Kétféleképpen számíthatjuk az advantage-t:

**1. Monte Carlo (MC) - Alacsony bias, magas variancia:**
```python
# Várjuk meg az epizód végét és használjuk a valós return-t
A_t = R_t - V(s_t)
# R_t = r_t + r_{t+1} + r_{t+2} + ... (teljes jövőbeli jutalom)
```

**2. TD (Temporal Difference) - Magas bias, alacsony variancia:**
```python
# Csak egy lépést nézünk és becslést használunk
A_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

### GAE: A Legjobb Mindkét Világból

A **GAE (Generalized Advantage Estimation)** egy súlyozott átlag a különböző hosszúságú TD becslések között.

```
┌────────────────────────────────────────────────────────────────┐
│                        GAE Formula                             │
│                                                                │
│   A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}              │
│                                                                │
│   ahol:                                                        │
│   - δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD error)            │
│   - γ = 0.99 (discount factor)                                │
│   - λ = 0.95 (GAE lambda)                                     │
└────────────────────────────────────────────────────────────────┘
```

### GAE Implementáció (`training/rollout_buffer.py`)

```python
def compute_returns_and_advantages(self, last_value: float, last_done: bool):
    """
    GAE számítás visszafelé iterálva az időben.
    """
    last_gae = 0

    # Visszafelé iterálunk (t = T-1, T-2, ..., 0)
    for step in reversed(range(self.buffer_size)):
        # Következő állapot értéke
        if step == self.buffer_size - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(last_done)
        else:
            next_value = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step + 1]

        # TD error (δ) számítás
        delta = (
            self.rewards[step]                           # r_t
            + self.gamma * next_value * next_non_terminal  # + γ·V(s_{t+1})
            - self.values[step]                          # - V(s_t)
        )

        # GAE rekurzív formula
        # A_t = δ_t + γλ·A_{t+1}
        last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        self.advantages[step] = last_gae

    # Return = Advantage + Value
    self.returns = self.advantages + self.values
```

### Lambda (λ) Hatása

```
λ = 0:   Csak TD(0) - 1 lépéses becslés
         A_t = δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
         → Alacsony variancia, de biased

λ = 1:   Teljes Monte Carlo
         A_t = R_t - V(s_t)
         → Magas variancia, de unbiased

λ = 0.95: Kompromisszum (ezt használjuk)
          → Közepes variancia és bias
```

**Vizuális magyarázat:**

```
λ=0    λ=0.5   λ=0.95  λ=1
 │       │        │      │
 ▼       ▼        ▼      ▼
┌───┬───┬───┬───┬───┬───┬───┐
│TD0│TD1│TD2│TD3│...│TDn│ MC│
└───┴───┴───┴───┴───┴───┴───┘
 ▲                        ▲
 │                        │
Kevés lépést             Összes lépést
néz előre                néz előre
(gyors, de pontatlan)    (pontos, de zajos)
```

---

## 7. A Tanítási Folyamat

### Vektorizált Környezetek

A jelenlegi implementáció **16 párhuzamos környezetet** használ az adatgyűjtés felgyorsítására. Ez `AsyncVectorEnv`-et használ, ahol minden környezet külön subprocess-ben fut.

```python
# Vektorizált környezet létrehozása (utils/env_factory.py)
def make_vec_env(num_envs=16, render_mode="rgb_array", normalize=True):
    def _make_env():
        return make_env(render_mode=render_mode, normalize=normalize)
    return AsyncVectorEnv([_make_env for _ in range(num_envs)])
```

**Előnyök:**
- 16× több adat ugyanannyi idő alatt
- Változatosabb tapasztalatok (különböző pályák)
- Jobb GPU kihasználtság (nagyobb batch-ek)

### Rollout Gyűjtés

A **rollout** az ágens és a környezet közötti interakció rögzítése. A vektorizált változatban egyszerre 16 környezetből gyűjtünk.

```python
def collect_rollout(env, network, buffer, num_steps, num_envs, device,
                    obs=None, hidden=None, current_episode_rewards=None):
    """2048 × 16 = 32768 lépésnyi tapasztalat gyűjtése"""

    # Állapot perzisztens a rollout-ok között
    if obs is None:
        obs, _ = env.reset()  # (num_envs, H, W, C)
    if hidden is None:
        hidden = network.get_initial_hidden(batch_size=num_envs, device=device)

    for step in range(num_steps):  # 2048 lépés
        # Hidden state mentése AKCIÓ ELŐTT
        hidden_before = LSTMState(h=hidden.h.clone(), c=hidden.c.clone())

        # 1. Akció kiválasztása (vektorizált)
        obs_tensor = torch.from_numpy(obs).float().permute(0, 3, 1, 2).unsqueeze(1).to(device)
        with torch.no_grad():
            action, log_prob, _, value, hidden = network.get_action_and_value(
                obs_tensor, hidden
            )

        # 2. Környezet léptetése (vektorizált)
        actions_np = action.squeeze(1).cpu().numpy()  # (num_envs,)
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions_np)
        dones = np.logical_or(terminateds, truncateds)

        # 3. Átmenet tárolása
        buffer.add(
            obs=obs,               # (num_envs, H, W, C)
            action=actions_np,     # (num_envs,)
            reward=rewards,        # (num_envs,)
            done=dones,            # (num_envs,)
            log_prob=log_probs_np, # (num_envs,)
            value=values_np,       # (num_envs,)
            hidden_state=hidden_before
        )

        obs = next_obs

        # 4. Hidden state reset befejezett epizódoknál
        for i, done in enumerate(dones):
            if done:
                hidden.h[:, i, :] = 0
                hidden.c[:, i, :] = 0

    return episode_rewards, obs, dones, hidden, last_value, current_episode_rewards
```

**Fontos különbségek a nem-vektorizált változattól:**
- Az állapot (obs, hidden) **perzisztens** a rollout-ok között
- Nincs explicit reset - az `AsyncVectorEnv` automatikusan reseteli a befejezett környezeteket
- A hidden state-et csak a befejezett környezeteknél nullázzuk

### RolloutBuffer (`training/rollout_buffer.py`)

A buffer vektorizált környezetekhez van tervezve, így 2D tömbökben tárolja az adatokat: `(buffer_size, num_envs, ...)`.

```python
class RolloutBuffer:
    """Tapasztalatok tárolása és batch-elése vektorizált környezetekhez"""

    def __init__(self, buffer_size=2048, num_envs=16, obs_shape=(80, 96, 3), ...):
        # Tárolók inicializálása: (buffer_size, num_envs, ...)
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape))  # (2048, 16, 80, 96, 3)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, num_envs))
        self.dones = np.zeros((buffer_size, num_envs))
        self.log_probs = np.zeros((buffer_size, num_envs))
        self.values = np.zeros((buffer_size, num_envs))

        # LSTM állapotok: (buffer_size, num_layers, num_envs, hidden_size)
        self.hidden_h = np.zeros((buffer_size, num_layers, num_envs, hidden_size))
        self.hidden_c = np.zeros((buffer_size, num_layers, num_envs, hidden_size))

        # GAE után számítva: (buffer_size, num_envs)
        self.advantages = np.zeros((buffer_size, num_envs))
        self.returns = np.zeros((buffer_size, num_envs))
```

**Adatméret:**
- Megfigyelések: 2048 × 16 × 80 × 96 × 3 = ~754 MB (float32)
- Összes adat egy rollout-ban: ~1 GB

### Batch Generálás Szekvenciákkal

Az LSTM szekvenciákat igényel, nem független mintákat. A vektorizált buffer esetén először laposítjuk az env dimenziót.

```python
def get_batches(self, batch_size=64, seq_len=32):
    """Szekvenciális batch-ek generálása"""

    # Flatten env dimenzió: (buffer_size, num_envs, ...) -> (buffer_size * num_envs, ...)
    flat_obs = self.observations.reshape(-1, *self.obs_shape)  # (32768, 80, 96, 3)
    flat_actions = self.actions.reshape(-1)  # (32768,)
    # ... többi adat hasonlóan

    # Hidden states: (buffer_size, num_layers, num_envs, hidden) -> (buffer_size * num_envs, num_layers, hidden)
    flat_hidden_h = self.hidden_h.transpose(0, 2, 1, 3).reshape(-1, self.num_lstm_layers, self.hidden_size)

    # Szekvencia kezdőpontok
    total_steps = self.buffer_size * self.num_envs  # 2048 * 16 = 32768
    num_sequences = total_steps // seq_len  # 32768 // 32 = 1024 szekvencia
    indices = np.arange(num_sequences) * seq_len  # [0, 32, 64, ...]

    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]  # 64 szekvencia

        for idx in batch_indices:
            # 32 lépés kezdve idx-től
            batch_obs.append(flat_obs[idx:idx+seq_len])

            # Maszk: 0 az epizód vége után
            mask = np.ones(seq_len)
            for i in range(seq_len - 1):
                if flat_dones[idx + i]:
                    mask[i + 1:] = 0
                    break

        # Advantage normalizálás batch-en belül
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
```

**Maszkolás magyarázata:**

```
Szekvencia: [s1, s2, s3, DONE, s5, s6, s7, ...]
Maszk:      [1,  1,  1,  1,    0,  0,  0,  ...]

A DONE után az állapotok egy ÚJ epizódból származnak,
ezért nem szabad őket figyelembe venni a loss számításnál.
```

**Batch méret számítás:**
- Total szekvenciák: 32768 / 32 = 1024
- Batch-ek száma: 1024 / 64 = 16 batch / epoch
- Epoch-ok: 4
- Összesen: 64 gradient update / rollout

### A Teljes Tanítási Ciklus

```python
def main():
    # 1. Inicializálás
    env = make_vec_env(num_envs=16, render_mode="rgb_array")
    network = ActorCriticLSTM(obs_shape=(80, 96, 3), num_actions=5, hidden_size=512)
    trainer = PPOTrainer(network, learning_rate=2.5e-4, entropy_coef=0.03, ...)
    buffer = RolloutBuffer(buffer_size=2048, num_envs=16, obs_shape=(80, 96, 3), ...)

    # Perzisztens állapot rollout-ok között
    current_obs = None
    current_hidden = None
    current_ep_rewards = None

    # 2. Fő tanítási ciklus
    global_step = 0
    steps_per_update = 2048 * 16  # 32768

    while global_step < 15_000_000:  # 15 millió lépés

        # 3. Rollout gyűjtés (2048 × 16 = 32768 lépés)
        episode_rewards, current_obs, last_done, current_hidden, last_value, current_ep_rewards = collect_rollout(
            env, network, buffer, num_steps=2048, num_envs=16, device,
            obs=current_obs, hidden=current_hidden, current_episode_rewards=current_ep_rewards
        )

        # 4. GAE számítás
        buffer.compute_returns_and_advantages(last_value, last_done)

        # 5. Learning rate decay
        progress = global_step / 15_000_000
        trainer.update_learning_rate(progress)  # Lineáris decay

        # 6. PPO frissítés (4 epoch × 16 batch = 64 update)
        stats = trainer.update(buffer)

        # 7. Logolás és checkpoint mentés
        global_step += steps_per_update
        trainer.log_training_stats(stats, episode_rewards, global_step)

        if global_step % 250000 < steps_per_update:
            trainer.save_checkpoint(f"model_{global_step}.pt", step=global_step, env=env)

        # 8. Buffer reset
        buffer.reset()
```

**Learning Rate Decay:**
```python
def update_learning_rate(self, progress: float):
    """Lineáris decay: lr = initial_lr * (1 - progress)"""
    lr = self.initial_lr * (1.0 - progress)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
```

### Tanulási Görbe

```
Jutalom
  ^
900│                                    ●●●●●●
   │                               ●●●●●
800│                           ●●●●
   │                       ●●●●
700│                   ●●●●
   │               ●●●●
600│           ●●●●
   │       ●●●●
500│   ●●●●
   │●●●
400│●
   │
   └────────────────────────────────────────────> Lépések
   0      500k     1M      1.5M    2M
```

---

## 8. Kód Struktúra

### Könyvtárszerkezet

```
gym-car/
├── main.py                 # Játék/inferencia betanított modellel
├── train.py                # Fő tanítási szkript (vektorizált)
├── evaluate.py             # Modell értékelése több epizódon
├── requirements.txt        # Függőségek
├── CLAUDE.md               # Projekt útmutató
├── README.md               # Projekt leírás
├── learn.md                # Ez a dokumentáció
├── review.md               # Kód review
│
├── models/                 # Neurális hálózat komponensek
│   ├── __init__.py
│   ├── actor_critic.py     # ActorCriticLSTM (CNN + LSTM + Actor/Critic fejek)
│   ├── cnn_feature_extractor.py  # 4 rétegű CNN BatchNorm-mal
│   └── lstm_state.py       # LSTMState dataclass (h, c)
│
├── training/               # Tanítási komponensek
│   ├── __init__.py
│   ├── ppo_trainer.py      # PPO algoritmus + checkpoint mentés
│   └── rollout_buffer.py   # Vektorizált buffer + GAE + batch generálás
│
├── agents/                 # Ágens wrapperek
│   ├── __init__.py
│   └── lstm_ppo_agent.py   # Inferencia wrapper (from_checkpoint)
│
├── wrappers/               # Gymnasium wrapperek
│   ├── __init__.py
│   ├── crop_observation.py     # Kép vágás (80×96)
│   ├── sharpen_observation.py  # Élesítés (opcionális)
│   ├── edge_observation.py     # Canny éldetektálás (opcionális)
│   └── render_observation.py   # Debug vizualizáció
│
├── utils/                  # Segédfüggvények
│   ├── __init__.py
│   └── env_factory.py      # make_env(), make_vec_env(), get_normalize_wrapper()
│
├── checkpoints/            # Mentett modellek
│   ├── model_latest.pt     # Legújabb checkpoint (obs_rms-sel)
│   ├── model_XXXXXX.pt     # Periodikus checkpointok
│   └── model_final.pt      # Végleges modell
│
└── runs/                   # TensorBoard logok
    └── <run_name>/events.out.tfevents.*
```

### Fájlok Kapcsolata

```
┌─────────────────────────────────────────────────────────────────┐
│                      TANÍTÁS (train.py)                         │
│                                                                 │
│   ┌─────────────┐     ┌──────────────┐     ┌────────────────┐  │
│   │ env_factory │────>│ Environment  │────>│ RolloutBuffer  │  │
│   │ (wrappers)  │     │              │     │                │  │
│   └─────────────┘     └──────────────┘     └───────┬────────┘  │
│                              │                      │           │
│                              ▼                      ▼           │
│                       ┌─────────────┐        ┌───────────┐     │
│                       │ActorCritic  │<───────│PPOTrainer │     │
│                       │   LSTM      │        │           │     │
│                       └─────────────┘        └───────────┘     │
│                              │                      │           │
│                              └──────────┬───────────┘           │
│                                         │                       │
│                                         ▼                       │
│                                  ┌────────────┐                 │
│                                  │ Checkpoint │                 │
│                                  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCIA (main.py)                        │
│                                                                 │
│   ┌────────────┐     ┌────────────┐     ┌──────────────────┐   │
│   │ Checkpoint │────>│LSTMPPOAgent│────>│ Environment      │   │
│   └────────────┘     │            │     │ (with rendering) │   │
│                      └────────────┘     └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Hiperparaméterek

### Összefoglaló Táblázat

| Paraméter | Érték | Leírás |
|-----------|-------|--------|
| **Tanítás** |
| `total_timesteps` | 15,000,000 | Összes tanítási lépés |
| `num_envs` | 16 | Párhuzamos környezetek száma |
| `num_steps` | 2,048 | Lépések rollout-onként (per env) |
| `num_epochs` | 4 | PPO epoch-ok frissítésenként |
| `batch_size` | 64 | Szekvenciák batch-enként |
| `seq_len` | 32 | Szekvencia hossz (LSTM) |
| **PPO** |
| `learning_rate` | 2.5e-4 | Tanulási ráta (Adam, lineáris decay) |
| `adam_eps` | 1e-5 | Adam optimizer epsilon |
| `gamma` | 0.99 | Diszkont faktor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping tartomány |
| `value_coef` | 0.5 | Value loss súly |
| `entropy_coef` | 0.03 | Entropy bonus súly |
| `max_grad_norm` | 0.5 | Gradiens clipping |
| **Hálózat** |
| `hidden_size` | 512 | LSTM és CNN feature méret |
| `num_lstm_layers` | 1 | LSTM rétegek száma |
| **Megfigyelés** |
| `obs_shape` | (80, 96, 3) | Feldolgozott kép méret (RGB) |
| `crop_height` | 80 | Vágott magasság |

### Paraméter Hangolási Tippek

#### Learning Rate

```
Túl magas (1e-3):  Instabil tanulás, oszcilláció
Túl alacsony (1e-5): Nagyon lassú konvergencia
Optimális (3e-4):  Stabil, gyors tanulás
```

#### Gamma (γ)

```
γ = 0.9:   Rövidtávú fókusz (10 lépés horizontál)
γ = 0.99:  Hosszútávú fókusz (100 lépés horizontál)
γ = 0.999: Nagyon hosszú táv (1000 lépés)

Autóvezetéshez γ = 0.99 jó kompromisszum
```

#### GAE Lambda (λ)

```
λ = 0.9:   Több bias, kevesebb variancia
λ = 0.95:  Kiegyensúlyozott (ajánlott)
λ = 1.0:   Nincs bias, sok variancia (MC)
```

#### Entropy Coefficient

```
entropy_coef = 0.0:   Nincs felfedezés ösztönzés
entropy_coef = 0.01:  Enyhe felfedezés (ajánlott)
entropy_coef = 0.1:   Túl sok véletlenszerűség
```

---

## 10. Gyakori Kérdések

### Q: Miért LSTM és nem sima CNN?

**A:** Az autóvezetés időbeli döntéseket igényel:
- A sebesség nem látszik a képen, de az LSTM megjegyzi a gyorsításokat
- A pálya alakja időben változik (kanyar közeledik)
- A korábbi kormánymozgások befolyásolják a jelenlegi pozíciót

### Q: Miért nem használunk Frame Stacking-et?

**A:** Az LSTM és a frame stacking hasonló célt szolgál (időbeli információ), de:
- LSTM elegánsabb és kevesebb memóriát használ
- Nincs fix "ablak méret" - az LSTM dinamikusan dönt
- Kevesebb CNN paraméter (1 csatorna vs. 4)

### Q: Miért RGB és nem éldetektálás?

**A:** A jelenlegi verzió RGB képeket használ, mert:
- A színinformáció hasznos (zöld fű vs. szürke pálya)
- A nagyobb CNN (4 réteg, BatchNorm) képes kezelni a komplexitást
- A 16 párhuzamos környezet elegendő adatot biztosít

Az éldetektáló wrapperek továbbra is elérhetők a `wrappers/` mappában, és könnyen bekapcsolhatók az `env_factory.py`-ban.

### Q: Mi történik, ha a tanítás elakad?

**A:** Lehetséges okok és megoldások:

| Tünet | Ok | Megoldás |
|-------|-----|----------|
| Reward nem nő | Learning rate túl alacsony | Növeld 1e-3-ra |
| Reward oszcillál | Learning rate túl magas | Csökkentsd 1e-4-re |
| Reward csökken | Entropy collapse | Növeld entropy_coef-et |
| Lassú tanulás | Kis batch | Növeld batch_size-t |

### Q: Hogyan monitorozhatom a tanulást?

**A:** TensorBoard használatával:

```bash
# Terminálban
tensorboard --logdir runs/

# Böngészőben
http://localhost:6006
```

Figyeld ezeket a metrikákat:
- `Reward/mean`: Növekednie kell
- `Loss/policy`: Stabilan alacsony
- `Loss/entropy`: Ne csökkenjen túl gyorsan

### Q: Mennyi ideig tart a tanítás?

**A:** Hardvertől függően (16 párhuzamos környezettel):
- **GPU (RTX 3080):** ~4-6 óra 15M lépésre
- **GPU (RTX 4090):** ~2-3 óra 15M lépésre
- **CPU:** Nem ajánlott (nagyon lassú)

A legtöbb javulás az első 5M lépésben történik. A vektorizált környezetek jelentősen felgyorsítják a tanítást.

### Q: Hogyan használhatom a betanított modellt?

**A:**

```bash
# Legújabb checkpoint betöltése
python main.py

# Specifikus checkpoint
python main.py --checkpoint checkpoints/model_5000000.pt

# Random akciók (összehasonlításhoz)
python main.py --random

# Értékelés több epizódon
python evaluate.py --checkpoint checkpoints/model_latest.pt --episodes 10
```

### Q: A modell jól tanult, de inferenciánál rosszul teljesít. Miért?

**A:** Valószínűleg a **normalizációs statisztikák** nincsenek megfelelően betöltve.

A `NormalizeObservation` wrapper futó átlagot és szórást használ, amelyek a tanítás alatt frissülnek. Ha ezeket nem mentjük/töltjük be a checkpoint-tal:

| Tanítás | Inferencia (rossz) | Inferencia (helyes) |
|---------|-------------------|---------------------|
| mean ≈ 0.15 | mean = 0.0 | mean ≈ 0.15 |
| var ≈ 0.08 | var = 1.0 | var ≈ 0.08 |

**Megoldás:**
A jelenlegi implementáció automatikusan menti az `obs_rms` statisztikákat a checkpoint-ba. A `main.py` és `evaluate.py` szkriptek helyesen betöltik és fagyasztják ezeket.

```python
# main.py - Normalizációs statisztikák betöltése
agent, obs_rms = LSTMPPOAgent.from_checkpoint(checkpoint_path, device=device)

if obs_rms is not None:
    norm_wrapper = get_normalize_wrapper(env)
    norm_wrapper.obs_rms.mean = obs_rms['mean']
    norm_wrapper.obs_rms.var = obs_rms['var']
    norm_wrapper.obs_rms.count = obs_rms['count']
    norm_wrapper.update_running_mean = False  # Fagyasztás!
```

Ha régi checkpoint-ot használsz, ami nem tartalmazza az `obs_rms`-t, újra kell tanítani.

---

## Összefoglalás

Ez a projekt bemutatja a modern megerősítéses tanulás kulcselemeit:

1. **Vektorizált környezetek:** 16 párhuzamos CarRacing-v3 AsyncVectorEnv-vel
2. **Neurális architektúra:** 4 rétegű CNN (BatchNorm) + LSTM az Actor-Critic keretrendszerben
3. **PPO algoritmus:** Stabil policy gradient optimalizálás maszkolással és value clipping-gel
4. **GAE:** Hatékony advantage becslés vektorizált környezetekre
5. **Szekvenciális tanulás:** LSTM-specifikus batch kezelés hidden state megőrzéssel

A kód moduláris felépítése lehetővé teszi:
- Könnyű kísérletezést különböző hiperparaméterekkel
- Új wrapper-ek hozzáadását (éldetektálás, élesítés, stb.)
- A hálózat architektúrájának módosítását
- Más Gymnasium környezetekre való adaptálást

**Kulcs implementációs részletek:**
- Learning rate lineáris decay a tanítás során
- Normalizációs statisztikák mentése és betöltése checkpoint-okkal
- Perzisztens állapot (obs, hidden) rollout-ok között
- Entropy coefficient (0.03) a felfedezés ösztönzésére

---

*Készítette: Claude Code oktatási célból*
