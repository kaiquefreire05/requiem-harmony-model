# Requiem — Manual Completo de Execução e Testes

> **Stacks:** React + TypeScript (frontend) · Spring Boot / Tomcat (backend) · Python/Keras (modelo ML)

---

## Índice

1. [Visão geral da arquitetura](#1-visão-geral-da-arquitetura)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Estrutura de diretórios](#3-estrutura-de-diretórios)
4. [Configuração do ambiente](#4-configuração-do-ambiente)
5. [Rodando o frontend](#5-rodando-o-frontend)
6. [Rodando o backend](#6-rodando-o-backend)
7. [Testes do modelo ML](#7-testes-do-modelo-ml-python)
8. [Retreinando o modelo](#8-retreinando-o-modelo)
9. [Adicionando novas músicas](#9-adicionando-novas-músicas)
10. [Exportando o modelo](#10-exportando-o-modelo-para-o-frontend)
11. [Testes do frontend](#11-testes-do-frontend)
12. [Referência rápida](#12-referência-rápida-de-comandos)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Visão geral da arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                        Requiem App                          │
│                                                             │
│  ┌──────────────────┐      ┌──────────────────────────────┐ │
│  │  Frontend (Vite) │ ←──→ │  Backend (Spring Boot/Java)  │ │
│  │  React + TS      │      │  Porta 3001                  │ │
│  │  Porta 5173      │      │  PostgreSQL / H2              │ │
│  └────────┬─────────┘      └──────────────────────────────┘ │
│           │                                                  │
│  ┌────────▼─────────────────────────────────────────────┐   │
│  │         Motor Neural LSTM (browser, sem servidor)    │   │
│  │  public/model/weights.json  ←  Python train_model.py │   │
│  │  public/model/vocab.json    ←  chord_dataset.py      │   │
│  │  HarmonyEngine.ts  (inferência LSTM em TypeScript)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

O motor de harmonia **roda inteiramente no navegador** — sem servidor Python em produção.

---

## 2. Pré-requisitos

| Ferramenta | Versão mínima | Verificar |
|---|---|---|
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ | `npm --version` |
| **Python** | 3.10+ | `python3 --version` |
| **Java** | 17+ | `java --version` |

### Instalar dependências Python

```bash
cd requiem-harmony-model/

# Ambiente virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# OU global (Linux/Ubuntu)
pip3 install -r requirements.txt --break-system-packages
```

---

## 3. Estrutura de diretórios

```
ia/
├── requiem-app-source/requiem-app/    ← Frontend React/Vite
│   ├── .env                           ← ✅ VITE_API_URL=http://localhost:3001/api
│   ├── src/engine/HarmonyEngine.ts    ← Motor LSTM em TypeScript
│   ├── src/lib/api.ts                 ← Cliente HTTP
│   └── public/model/
│       ├── weights.json               ← Pesos do modelo (gerado pelo Python)
│       └── vocab.json                 ← Vocabulário de acordes
│
└── requiem-harmony-model/             ← Modelo ML (Python)
    ├── music_data/                    ← 201 JSONs de progressões musicais
    ├── saved_model/model.keras        ← Modelo treinado
    ├── train_model.py                 ← Treina o modelo LSTM
    ├── export_weights_json.py         ← Exporta pesos para o frontend
    ├── test_ts_parity.py              ← Testes de paridade TS↔Python (7 testes)
    ├── add_more_songs.py              ← Adiciona novas músicas
    ├── chord_dataset.py               ← Pipeline de dados
    └── lstm_harmony_model.py          ← Arquitetura do modelo
```

---

## 4. Configuração do ambiente

### `.env` do frontend

**Arquivo:** `requiem-app-source/requiem-app/.env`

```env
# URL base da API backend — inclui o prefixo /api
VITE_API_URL=http://localhost:3001/api
```

> **Atenção:** Reinicie `npm run dev` após criar/alterar o `.env`.

### Verificar backend

```bash
# 401 {"error":"Token inválido"} → backend OK
curl http://localhost:3001/api/auth/me

# 401 {"error":"Credenciais inválidas"} → rota de login OK
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"x","password":"x"}'
```

---

## 5. Rodando o frontend

```bash
cd requiem-app-source/requiem-app/

npm install        # Só na primeira vez
npm run dev        # Inicia em http://localhost:5173
```

**Sinais de que o modelo neural carregou** (DevTools → Console):

```
[NeuralHarmonyEngine] Carregando pesos LSTM...
[NeuralHarmonyEngine] Pronto. Vocab: 145 acordes.
```

No rodapé da interface: `✦ REQUIEM ENGINE ATIVO` e `● MODELO PRONTO`.

```bash
npm run build      # Build de produção → dist/
npm run preview    # Servir build localmente
```

---

## 6. Rodando o backend

```bash
# JAR
java -jar requiem-backend.jar

# Maven
mvn spring-boot:run

# Verificar
curl http://localhost:3001/api/auth/me
```

> O motor neural não depende do backend — funciona sem ele.

---

## 7. Testes do modelo ML (Python)

### 7.1 Suite completa de paridade TypeScript ↔ Keras

```bash
cd requiem-harmony-model/

python3 test_ts_parity.py
```

| # | Teste | Critério de aprovação |
|---|---|---|
| 1 | Carregamento do `weights.json` | Shapes e buffers corretos, sem erro |
| 2 | Paridade numérica TS vs Keras | Erro máx < 1e-4 |
| 3 | Top-5 previsões musicais | Top-1 idêntico ao Keras |
| 4 | Determinismo | Δ = 0 para mesma entrada |
| 5 | Validade das probabilidades | Soma = 1.0, valores em [0, 1] |
| 6 | Cobertura do HARMONY_GRAPH | Acordes do grafo presentes no vocab |
| 7 | Pipeline de geração completo | 8 acordes coerentes por progressão |

**Saída esperada:**

```
============================================================
  RESULTADO FINAL
============================================================
  ✓ Paridade numérica TS↔Keras: erro máx = 1.77e-07
  ✓ Todos os 7 testes passaram!
============================================================
```

### 7.2 Inferência rápida

```bash
cd requiem-harmony-model/

TF_CPP_MIN_LOG_LEVEL=3 python3 - << 'EOF'
import json, numpy as np
from tensorflow import keras

model = keras.models.load_model("saved_model/model.keras")
vocab = json.load(open("saved_model/vocab.json"))["vocab"]
vocab_map = {c: i for i, c in enumerate(vocab)}

context = [vocab_map[c] for c in ["C", "Am", "F", "G"]]
probs = model.predict([context], verbose=0)[0]
top5 = sorted(enumerate(probs), key=lambda t: t[1], reverse=True)[:5]

print("Após [C, Am, F, G]:")
for idx, p in top5:
    print(f"  {vocab[idx]:12s} {p:.1%}")
EOF
```

### 7.3 Estatísticas do dataset

```bash
cd requiem-harmony-model/

ls music_data/*.json | wc -l    # Conta músicas

TF_CPP_MIN_LOG_LEVEL=3 python3 - << 'EOF'
from chord_dataset import prepare_dataset, WINDOW_SIZE
import pathlib

data = prepare_dataset("music_data", WINDOW_SIZE, augment=True)
n = len(list(pathlib.Path("music_data").glob("*.json")))
print(f"Músicas     : {n}")
print(f"Amostras    : {len(data.X)} (após augmentation ×12)")
print(f"Vocabulário : {len(data.vocab)} acordes únicos")
EOF
```

---

## 8. Retreinando o modelo

```bash
cd requiem-harmony-model/

python3 train_model.py
```

**Pipeline:**
1. Lê todos os JSONs de `music_data/`
2. Augmentation ×12 tonalidades → 201 músicas ≈ 15.000+ amostras
3. Divisão 80% treino / 20% validação
4. Treina até 300 épocas com Early Stopping (patience=25)
5. Salva melhor modelo em `saved_model/model.keras`

**Métricas de referência:**

| Métrica | Mínimo | Excelente |
|---|---|---|
| `val_accuracy` | > 35% | > 60% |
| `val_loss` | < 3.0 | < 2.0 |

> **Após retreinar, sempre exporte:** `python3 export_weights_json.py`

---

## 9. Adicionando novas músicas

### Formato do JSON

```json
{
  "title": "Nome da Música",
  "originalTonality": "C",
  "normalizedProgression": ["C", "Am", "F", "G", "C", "G", "Am", "F"]
}
```

- Mínimo 4 acordes, ideal 8–16
- Nomes exatos: `"Am"` não `"A minor"`, `"G7"` não `"G dominant7"`

### Via script

Edite `add_more_songs.py`, adicione ao array `new_songs` e execute:

```bash
python3 add_more_songs.py
```

### Fluxo completo

```bash
python3 add_more_songs.py        # 1. Adiciona músicas
python3 train_model.py           # 2. Retreina
python3 export_weights_json.py   # 3. Exporta para o frontend
python3 test_ts_parity.py        # 4. Valida paridade
```

---

## 10. Exportando o modelo para o frontend

```bash
cd requiem-harmony-model/

python3 export_weights_json.py
```

**Gera em `requiem-app/public/model/`:**

| Arquivo | Tamanho | Conteúdo |
|---|---|---|
| `weights.json` | ~780 KB | Pesos do modelo em base64 |
| `vocab.json` | ~2 KB | Lista de acordes |

**Verificar:**

```bash
curl http://localhost:5173/model/weights.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('vocab_size:', d['vocab_size'])
print('total_params:', d['total_params'])
print('layers:', list(d['layers'].keys()))
"
```

---

## 11. Testes do frontend

### TypeScript — sem erros de compilação

```bash
cd requiem-app-source/requiem-app/

node_modules/.bin/tsc --noEmit
# Saída esperada: (vazio)
```

### Lint

```bash
npm run lint
```

### Teste manual no navegador

1. Acesse **http://localhost:5173**
2. Console → `[NeuralHarmonyEngine] Pronto. Vocab: 145 acordes.`
3. Network → `weights.json` com status **200**
4. Rodapé → `● MODELO PRONTO`
5. Criar composição → gravar → gerar harmonia → verificar Studio

### Teste de integração da API via curl

```bash
# Registro
curl -X POST http://localhost:3001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Teste","email":"teste@requiem.com","password":"senha123"}'

# Login — salva token
TOKEN=$(curl -s -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"teste@requiem.com","password":"senha123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

echo "Token: $TOKEN"

# Listar sessões
curl http://localhost:3001/api/sessions \
  -H "Authorization: Bearer $TOKEN"

# Criar sessão
curl -X POST http://localhost:3001/api/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title":"Minha Composição"}'
```

---

## 12. Referência rápida de comandos

### Frontend

```bash
cd requiem-app-source/requiem-app/

npm install                        # Instala dependências
npm run dev                        # Dev server → http://localhost:5173
npm run build                      # Build de produção
npm run preview                    # Servir build
npm run lint                       # Lint
node_modules/.bin/tsc --noEmit     # Verificar TypeScript
```

### Modelo ML

```bash
cd requiem-harmony-model/

python3 train_model.py             # Treinar
python3 export_weights_json.py     # Exportar para o frontend
python3 test_ts_parity.py          # Testes de paridade (7 testes)
python3 add_more_songs.py          # Adicionar músicas ao dataset
```

### Backend

```bash
curl http://localhost:3001/api/auth/me           # Health check
curl -o /dev/null -w "%{http_code}\n" \
  http://localhost:3001/api/sessions             # Status HTTP
```

---

## 13. Troubleshooting

### ❌ `[NeuralHarmonyEngine] Falha ao pré-carregar`

```bash
ls requiem-app-source/requiem-app/public/model/
# Se weights.json ausente:
cd requiem-harmony-model/ && python3 export_weights_json.py
```

---

### ❌ Frontend não conecta ao backend

```bash
# Verificar .env
cat requiem-app-source/requiem-app/.env
# Esperado: VITE_API_URL=http://localhost:3001/api

# Reiniciar Vite após alterar .env:
# Ctrl+C → npm run dev

# Verificar backend
curl http://localhost:3001/api/auth/me
# Esperado: {"error":"Token inválido..."} (JSON, não HTML)
```

---

### ❌ `tsc --noEmit` com erros

```bash
node_modules/.bin/tsc --noEmit 2>&1 | head -30
```

---

### ❌ `val_accuracy < 35%` após retreino

```bash
# Ver histórico
python3 - << 'EOF'
import json
h = json.load(open("training_history.json"))
best = min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i])
print(f"Melhor época : {best+1}")
print(f"Val accuracy : {h['val_accuracy'][best]:.1%}")
print(f"Val loss     : {h['val_loss'][best]:.4f}")
EOF

# Adicionar mais músicas e retreinar
python3 add_more_songs.py && python3 train_model.py
```

---

### ❌ `test_ts_parity.py` falha no TEST 2 (erro > 1e-4)

```bash
# Reexportar e testar novamente
python3 export_weights_json.py
python3 test_ts_parity.py
```

---

### ❌ `weights.json` muito grande (> 2 MB)

Em `lstm_harmony_model.py`, reduza a arquitetura:

```python
embedding_dim=16,    # era 32
lstm_units_1=64,     # era 128
lstm_units_2=32,     # era 64
```

Depois retreine e reexporte.

---

*© 2026 Requiem Labs · Harmony Engine v1.0*
