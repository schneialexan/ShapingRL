# Checkpointing Erklärung

## Was sind Checkpoints?

Checkpoints sind gespeicherte Zustände deines trainierten Models zu bestimmten Zeitpunkten während des Trainings. Sie enthalten:
- **Model Weights**: Die aktuellen Gewichte des Neural Networks
- **Optimizer State**: Der Zustand des Optimizers (für kontinuierliches Training)
- **Training Metadata**: Timesteps, Hyperparameter, etc.

## Was sind VecNormalize Stats?

**VecNormalize** ist ein Wrapper, der Observations und Rewards normalisiert, um das Training zu stabilisieren. Die "Stats" sind:
- **Running Mean**: Durchschnittswerte der Observations/Rewards
- **Running Variance**: Varianz der Observations/Rewards
- **Count**: Anzahl der Samples, die für die Statistik verwendet wurden

### Warum sind sie wichtig?

Wenn du von einem Checkpoint weiterlernst, **müssen** die VecNormalize Stats geladen werden, sonst:
- Die Normalisierung startet von neuem
- Das Model bekommt plötzlich andere normalisierte Inputs
- Das Training kann instabil werden oder die Performance einbrechen

### Beispiel:

```python
# Während Training:
obs = [100, 200, 300]  # Raw observation
normalized = (obs - mean) / std  # z.B. [-0.5, 0.0, 0.5]

# Wenn du ohne Stats weiterlernst:
# Neue mean/std werden berechnet → normalized könnte [0.1, 0.2, 0.3] sein
# → Model bekommt andere Inputs → Performance bricht ein!
```

## Checkpoint-Dateien

Stable-Baselines3 speichert Checkpoints im Format:
- **Model**: `checkpoint_XXXXXX_steps.zip`
  - `XXXXXX` = Anzahl der Timesteps (z.B. `1000000`)
  - Enthält: Model weights, optimizer state, metadata
  
- **VecNormalize Stats**: `checkpoint_XXXXXX_steps_vecnormalize.pkl`
  - Separate Datei mit den Normalisierungs-Statistiken
  - Wird automatisch gespeichert wenn `save_vecnormalize=True`

### Beispiel-Dateien:

```
checkpoints/
├── checkpoint_1000000_steps.zip          # Model nach 1M Steps
├── checkpoint_1000000_steps_vecnormalize.pkl  # Stats nach 1M Steps
├── checkpoint_2000000_steps.zip          # Model nach 2M Steps
└── checkpoint_2000000_steps_vecnormalize.pkl  # Stats nach 2M Steps
```

## Wie funktioniert das Laden?

```python
# 1. VecNormalize Stats laden (WICHTIG: vor dem Model!)
vec_env = VecNormalize.load("checkpoint_1000000_steps_vecnormalize.pkl", vec_env)

# 2. Model laden (mit der korrekten Umgebung)
model = PPO.load("checkpoint_1000000_steps.zip", env=vec_env, device="cuda")

# 3. Weiterlernen
model.learn(total_timesteps=500_000, reset_num_timesteps=False)
```

## Warum separate Dateien?

- **Flexibilität**: Du kannst Model und Stats getrennt laden
- **Größe**: Stats sind klein, Model kann groß sein
- **Optional**: Stats sind optional (aber empfohlen)

## Troubleshooting

**Problem**: "VecNormalize stats not found"
- **Lösung**: Nicht kritisch, aber Training könnte instabil sein
- **Grund**: Checkpoint wurde vor dem Speichern der Stats erstellt

**Problem**: "Model loaded but performance dropped"
- **Lösung**: Stelle sicher, dass VecNormalize Stats geladen wurden
- **Grund**: Normalisierung stimmt nicht mehr

