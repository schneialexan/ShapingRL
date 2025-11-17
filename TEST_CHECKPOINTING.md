# Checkpointing Test Guide

## Übersicht

Das Checkpointing-System ist vollständig in `test_mr_ram.py` integriert. Es ermöglicht:
1. **Periodisches Speichern** während des Trainings
2. **Nahtloses Weiterlernen** von einem Checkpoint
3. **Automatisches Speichern** von VecNormalize Stats

## Verwendung

### 1. Training mit Checkpointing

```bash
# Training starten
python src/training/montezuma.py \
    --timesteps 2150000 \
    --device cuda \
    --seed 42 \
    --observation-type ram-only \
    --no-include-noop \
    --checkpoint-path ./checkpoints/my_training_run \
    --checkpoint-interval 100000
```

### 2. Weiterlernen von Checkpoint

```bash
python src/training/montezuma.py \
    --continue-from checkpoints/my_training_run/checkpoint_1000000_steps.zip \
    --timesteps 1000000 \
    --device cuda \
    --seed 42 \
    --observation-type ram-only \
    --no-include-noop \
    --save-checkpoints checkpoints/continued \
    --checkpoint-interval 100000
```

## Tests ausführen

Um die Funktionalität zu testen:

```bash
# Alle Checkpointing-Tests
python -m pytest tests/test_checkpointing.py -v -s

# Einzelne Tests
python -m pytest tests/test_checkpointing.py::test_checkpointing_basic -v -s
python -m pytest tests/test_checkpointing.py::test_resume_training -v -s
python -m pytest tests/test_checkpointing.py::test_full_workflow -v -s
```

## Wichtige Hinweise

1. **Seed muss gleich bleiben**: Beim Weiterlernen muss der gleiche Seed verwendet werden wie beim ursprünglichen Training
2. **Environment Setup**: Muss exakt gleich sein (flatten, reward shaping, etc.)
3. **VecNormalize Stats**: Werden automatisch gespeichert und geladen
4. **Timestep Counter**: Wird nicht zurückgesetzt (`reset_num_timesteps=False`)

## Dateien

- **Checkpoint**: `checkpoint_XXXXXX_steps.zip` - Enthält das Model
- **VecNormalize Stats**: `checkpoint_XXXXXX_steps_vecnormalize.pkl` - Enthält Normalisierungs-Statistiken
- **Replay Buffer**: Wird optional gespeichert (kann groß sein)

## Steps für 6 Stunden

Basierend auf deinem letzten Training (1.000.000 Steps in 2:47:47 h):
- **Empfohlene Steps**: ~2.150.000 Steps
- **Checkpoint-Interval**: 100.000 Steps (empfohlen)

