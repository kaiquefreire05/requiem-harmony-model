# Requiem — Modelo de Machine Learning (LSTM)

Documentação técnica do modelo de Deep Learning que substitui o motor de harmonia baseado em Cadeias de Markov.

---

## Por que ML/DL em vez de Markov?

O motor Markov original tem uma limitação fundamental: ele só enxerga **2 acordes anteriores** para decidir o próximo. Isso é chamado de "janela de contexto limitada".

```
Markov:  [C] [Am]  →  ?          (enxerga só 2)
LSTM:    [C] [Am] [F] [G]  →  ?  (enxerga 4, configurável)
```

Com uma rede neural LSTM, o modelo aprende **padrões de longo prazo** em progressões harmônicas — como a tensão que se cria ao longo de um compasso e resolve no acorde seguinte.

---

## Arquitetura do Modelo

```
Entrada: [PAD, PAD, C, Am]     <- 4 acordes anteriores (índices inteiros)
         |
[Embedding  (vocab_size -> 32)]   Aprende representação vetorial dos acordes
         |  shape: (batch, 4, 32)
[LSTM  128 unidades            ]  Processa sequência temporal
[return_sequences=True         ]  Passa estado para próxima camada
[Dropout 0.30                  ]  Regularização
         |  shape: (batch, 4, 128)
[LSTM  64 unidades             ]  Extrai padrões de nível mais alto
[return_sequences=False        ]  Retorna apenas o estado final
[Dropout 0.30                  ]
         |  shape: (batch, 64)
[Dense  64 unidades  (ReLU)    ]  Camada de fusão
[Dropout 0.20                  ]
         |  shape: (batch, 64)
[Dense  vocab_size  (Softmax)  ]  Probabilidade por acorde
         |  shape: (batch, vocab_size)
Saída: [0.03, 0.45, 0.02, ...]   <- probabilidade de cada acorde
```

**Total de parâmetros:** ~50.000 (modelo pequeno e eficiente)

---

## Dataset e Data Augmentation

### Dados brutos
- **51 músicas** com suas progressões e tonalidades originais
- Comprimento médio: ~5 acordes por progressão
- Acordes únicos: 27 (após normalização para Dó Maior)

### Por que 51 músicas é pouco?
Redes neurais precisam de **muitas amostras** para generalizar. 51 músicas × 5 acordes = ~255 amostras brutas. Isso seria insuficiente.

### Data Augmentation por transposição
A solução: transpomos cada progressão para **todas as 12 tonalidades**.

```
"Numb" (Fm): [Fm, Db, Ab, Eb]
  -> Normaliza para C: [Am, F, C, G]
  -> Transpõe +1: [A#m, F#, C#, G#]
  -> Transpõe +2: [Bm, G, D, A]
  -> ...
  -> Transpõe +11: [Gm, Eb, Bb, F]
```

Isso é **matematicamente correto**: uma progressão I->IV->I->V em Dó é idêntica em padrão à mesma progressão em Ré. O modelo aprende o **padrão relativo**, não a tonalidade absoluta.

**Resultado: 51 × 12 = 612 progressões -> ~3.000 amostras de treino**

---

## Método de Treino (Sliding Window)

O treino é um problema de **classificação sequencial**: dado um contexto de N acordes anteriores, prever qual acorde vem a seguir.

```
Progressão: [C, Am, F, G, C, Am] com window_size=4

Amostra 1:  X=[PAD, PAD, PAD,  C]  ->  y=Am
Amostra 2:  X=[PAD, PAD,   C, Am]  ->  y=F
Amostra 3:  X=[PAD,   C,  Am,  F]  ->  y=G
Amostra 4:  X=[  C,  Am,   F,  G]  ->  y=C
Amostra 5:  X=[Am,   F,   G,   C]  ->  y=Am
```

O token PAD (índice 0) é ignorado pelo LSTM via `mask_zero=True` no Embedding.

---

## Função de Loss e Otimizador

### `sparse_categorical_crossentropy`
Mede o quão errada está a distribuição de probabilidade prevista vs. o acorde correto.

```
Previsão: [C=0.05, Am=0.60, F=0.20, G=0.15]
Correto:   Am
Loss = -log(0.60) = 0.51   <- quanto menor, melhor
```

### Otimizador Adam
- Adaptativo: ajusta o learning rate individualmente para cada parâmetro
- Ideal para embeddings (gradientes esparsos)
- `lr=1e-3` -> padrão da literatura, reduzido automaticamente via `ReduceLROnPlateau`

---

## Como o LSTM "Aprende" Harmonia

A LSTM usa **células de memória** com portões (gates):

| Portão | Função |
|--------|--------|
| **Forget gate** | "Devo esquecer acordes antigos?" |
| **Input gate**  | "Qual informação nova salvar?" |
| **Output gate** | "O que enviar para a próxima camada?" |

Na prática, a LSTM aprende que:
- Após `I -> IV`, o próximo quase sempre é `I` ou `V`
- `V7` quase sempre resolve para `I`
- `vi` pode ir para `IV` ou `ii`

Esses padrões emergem automaticamente do treinamento, sem regras explícitas.

---

## Inferência: Como o Motor Decide o Próximo Acorde

```
Histórico [C, Am, F]   ->  [ Modelo LSTM ]  ->  P(próximo acorde)
                                    +
Notas tocadas [60,64,67] ->  Scoring Acústico  ->  S(compatibilidade)
                                    |
          Score Final = (1.0 × S_acústico) + (15.0 × P_neural)
                                    |
                        Acorde com maior score
```

A **combinação** garante:
- O estilo aprendido orienta a progressão (P_neural)
- As notas reais do usuário refinam a escolha (S_acústico)

---

## Como Usar

### 1. Instalação

```bash
pip install tensorflow
```

### 2. Treinar o modelo

```bash
cd requiem-harmony-model
python3 train_model.py
```

O treino gera:
```
saved_model/          <- modelo TensorFlow
  vocab.json           <- vocabulário de acordes
training_history.json  <- curvas de loss/accuracy
```

### 3. Usar o motor neural

```python
from neural_harmony_engine import NeuralHarmonyEngine
from tonality_adapter import DetectedNote

engine = NeuralHarmonyEngine(model_dir="saved_model")

notes = [
    DetectedNote(pitch=69, start_time=0.0, end_time=0.5, amplitude=0.8),
    DetectedNote(pitch=72, start_time=0.5, end_time=1.0, amplitude=0.7),
    DetectedNote(pitch=76, start_time=1.0, end_time=2.0, amplitude=0.9),
]

progression = engine.generate(notes, bpm=90, time_signature=(4, 4))
for r in progression:
    print(f"Acorde: {r.chord}  |  Velocity: {r.velocity:.2f}")
```

### 4. Trocar entre Markov e LSTM

```python
# Motor Markov (original — sem treino necessário)
from harmony_engine import generate_progression

# Motor LSTM (precisa de treino prévio)
from neural_harmony_engine import generate_progression_neural

# Assinaturas idênticas -> drop-in replacement
progression = generate_progression_neural(notes, bpm=120)
```

---

## Métricas Esperadas

| Condição | Val Accuracy | Interpretação |
|----------|:---:|---|
| < 40% | Ruim | Não aprendeu — adicione mais JSONs |
| 40–60% | OK | Normal para este tamanho de dataset |
| 60–75% | Bom | Padrões harmônicos aprendidos |
| > 75% | Checar | Possível overfitting — verifique val_loss |

> **Por que não 100%?** Harmonia musical não é determinística. Vários acordes são igualmente válidos em cada contexto — e isso é desejado. Um modelo com 100% de accuracy estaria "decorando" os dados.

---

## Como Melhorar o Modelo

### Adicionar mais músicas (mais impactante)

Crie JSONs em `music_data/` no formato:
```json
{
  "title": "Nome da Música",
  "artist": "Artista",
  "originalTonality": "Am",
  "normalizedProgression": ["Am", "F", "C", "G"]
}
```

Cada JSON novo gera +12 progressões aumentadas. Com 100 músicas você terá 1.200 progressões e accuracy significativamente melhor.

### Ajustar hiperparâmetros (`train_model.py`)

| Parâmetro | Padrão | Efeito de aumentar |
|-----------|:------:|---|
| `MAX_EPOCHS` | 300 | Mais tempo de treino |
| `PATIENCE` | 25 | Aguarda mais antes de parar |
| `BATCH_SIZE` | 32 | Treino mais estável |

### Ajustar arquitetura (`lstm_harmony_model.py`)

| Parâmetro | Padrão | Efeito de aumentar |
|-----------|:------:|---|
| `lstm_units_1` | 128 | Maior capacidade de memória |
| `embedding_dim` | 32 | Representação mais rica dos acordes |
| `dropout_rate` | 0.3 | Reduz overfitting (use 0.4–0.5 com mais dados) |

### Ajustar pesos da inferência (`neural_harmony_engine.py`)

| Parâmetro | Padrão | Efeito de aumentar |
|-----------|:------:|---|
| `NEURAL_WEIGHT` | 15.0 | Mais influência do estilo aprendido |
| `ACOUSTIC_WEIGHT` | 1.0 | Mais influência das notas tocadas |

---

## Arquivos do Projeto

| Arquivo | Função |
|---------|--------|
| `chord_dataset.py` | Carrega JSONs, augmentation ×12, sliding window |
| `lstm_harmony_model.py` | Arquitetura LSTM com TensorFlow/Keras |
| `train_model.py` | Script de treinamento com callbacks |
| `neural_harmony_engine.py` | Motor de inferência (substituto do Markov) |
| `tonality_adapter.py` | Transposição de acordes e notas |
| `audio_analyzer.py` | Detecção de tonalidade e BPM |
| `harmony_engine.py` | Motor Markov original (referência) |
| `requiem_model.py` | API de alto nível |
