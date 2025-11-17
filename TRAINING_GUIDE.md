# Training Guide - Montezuma's Revenge

## Training System

Das Projekt verwendet ein einheitliches Environment und Training-System für alle Montezuma's Revenge Experimente.

## Environment-Optionen

### Observation Types
- `default`: RGB image (210, 160, 3)
- `grey`: Grayscale image (210, 160, 1)
- `ram`: Dictionary mit RGB image + RAM features
- `ram-only`: Nur RAM features (12,)
- `grey-ram`: Flattened grayscale + RAM features

### Action Space
- `--include-noop`: 7 Aktionen (mit NOOP) - Standard
- `--no-include-noop`: 6 Aktionen (ohne NOOP) - Für Speedrunning empfohlen

## Training starten

### Basis-Training

```bash
# RAM-only, 6 Aktionen (ohne NOOP), 2.5M Steps, Checkpoints alle 500k
python src/training/montezuma.py \
    --observation-type ram-only \
    --no-include-noop \
    --timesteps 2500000 \
    --checkpoint-interval 500000 \
    --checkpoint-path ./checkpoints
```

### Weitere Beispiele

```bash
# Default RGB, 7 Aktionen (mit NOOP)
python src/training/montezuma.py \
    --observation-type default \
    --timesteps 1000000 \
    --checkpoint-interval 100000

# Grayscale + RAM
python src/training/montezuma.py \
    --observation-type grey-ram \
    --no-include-noop \
    --timesteps 2000000

# Dictionary RAM (für MultiInputPolicy)
python src/training/montezuma.py \
    --observation-type ram \
    --timesteps 1000000
```

## Weiterlernen von Checkpoint

```bash
# Von Checkpoint weitertrainieren
python src/training/montezuma.py \
    --continue-from ./checkpoints/checkpoint_1000000_steps.zip \
    --timesteps 500000 \
    --observation-type ram-only \
    --no-include-noop \
    --save-checkpoints ./checkpoints/continued
```

**WICHTIG**: `--observation-type` und `--include-noop` müssen exakt mit dem ursprünglichen Training übereinstimmen!

## Checkpointing

Checkpointing ist **immer aktiviert**. Checkpoints werden automatisch gespeichert:
- Model: `checkpoint_XXXXXX_steps.zip`
- VecNormalize Stats: `checkpoint_XXXXXX_steps_vecnormalize.pkl`

## Alle CLI-Optionen

```bash
python src/training/montezuma.py --help
```

### Wichtige Optionen:
- `--timesteps`: Anzahl Training Steps
- `--observation-type`: default, grey, ram, ram-only, grey-ram
- `--include-noop` / `--no-include-noop`: Action Space Größe
- `--checkpoint-path`: Verzeichnis für Checkpoints
- `--checkpoint-interval`: Checkpoint alle N Steps
- `--continue-from`: Checkpoint zum Weiterlernen
- `--device`: cuda oder cpu
- `--seed`: Random Seed
- `--no-wandb`: Wandb deaktivieren

## Empfohlene Konfigurationen

### Speedrunning (ohne NOOP)
```bash
python src/training/montezuma.py \
    --observation-type ram-only \
    --no-include-noop \
    --timesteps 2500000 \
    --checkpoint-interval 500000
```

### Vergleich: RAM-only vs Image+RAM
```bash
# RAM-only
python src/training/montezuma.py --observation-type ram-only --timesteps 1000000

# Image+RAM
python src/training/montezuma.py --observation-type ram --timesteps 1000000
```

