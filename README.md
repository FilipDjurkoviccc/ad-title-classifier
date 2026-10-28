# Ad Title Classifier: klasifikacija oglasa po naslovu (MLP i transformer)

Zadatak: na osnovu naslova oglasa (npr. `Ugaona brusilica Bosch GWS 750`) odrediti jednu od 10 grupa iz tabele `ml_dataset`. Isti podaci i ista podela na trening/validaciju/test (80/10/10) rešeni su na dva načina i upoređeni:

1. **MLP nad TF-IDF vektorima** (`sklearn.neural_network.MLPClassifier`, treniran epohu po epohu),
2. **transformer mreža** (PyTorch, `nn.TransformerEncoder`, trenirana od nule, bez pretreniranih modela).

| model | tačnost na testu | macro F1 | parametara |
|---|---|---|---|
| MLP + TF-IDF (reči + 3-4 grami karaktera) | **99.30%** | 0.9926 | 1.11M |
| transformer (rečnik reči, 4 sloja, d=128) | 97.57% | 0.9748 | 1.07M |

Test skup ima 4151 oglas koji modeli nikad nisu videli. Za ovaj zadatak je bolji jednostavniji MLP; razlog je opisan niže.

## Struktura

| putanja | sadržaj |
|---|---|
| `notebooks/01_podaci.ipynb` | preuzimanje iz MySQL-a, analiza, čišćenje, podela (zajedničko za oba modela) |
| `notebooks/02a_mlp_model.ipynb` | MLP: feature-i i ablacija, kriva učenja, broj neurona, regularizacija, 22 arhitekture, konačan model |
| `notebooks/03a_mlp_evaluacija.ipynb` | MLP: metrike, matrica konfuzije, preprilagođavanje |
| `notebooks/02b_transformer_model.ipynb` | transformer: tokenizacija, arhitektura, normalizacija sa/bez, rečnik, veličina, dropout, konačan model |
| `notebooks/03b_transformer_evaluacija.ipynb` | transformer: metrike, konfuzija, preprilagođavanje, analiza grešaka, poređenje sa MLP-om |
| `src/data.py`, `src/textprep.py` | preuzimanje i priprema podataka, normalizacija teksta (zajedničko) |
| `src/nn.py` | trening MLP-a epohu po epohu sa ranim zaustavljanjem |
| `src/model.py` | rečnik, transformer, petlja za treniranje sa ranim zaustavljanjem |
| `models/mlp_model.joblib`, `models/transformer_model.pt` | već obučeni modeli (oko 35 MB i 4 MB) |
| `docs/` | slika arhitekture MLP-a i PDF sa objašnjenjem transformer mreža (priprema za odbranu) |

Svaki korak u sveskama je objašnjen (šta je urađeno i zašto). Redosled čitanja: 01, zatim 02a/03a (MLP) i 02b/03b (transformer).

## Pokretanje

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # torch je potreban samo za transformer
.venv/bin/jupyter notebook notebooks/
```
