# Molecules Anomaly Detection

## Projekt
W niniejszej pracy badamy, czy modele GAE i VGAE potrafią nauczyć się struktury niemutagennych cząsteczek oraz czy w połączeniu ze stratą rekonstrukcji lub modelem GMM mogą skutecznie wspomagać detekcję mutacji. Dodatkowo analizujemy jakość uzyskanych reprezentacji w zadaniu klasyfikacji. Eksperymenty pokazują, że pretrenowanie modeli na dodatkowym zbiorze niemutagennych cząsteczek - nawet z innej dziedziny - zwykle poprawia wyniki. Ponadto metoda oparta na GMM osiągała lepsze rezultaty niż podejście ze stratą rekonstrukcji. 
## Reprodukcja wyników (jak uruchomić pipeline)
1. Przygotowanie środowiska
  - Używając pip w wirtualnym środowisku:
      - Utwórz wirtualne środowisko: `python -m venv myenv`
      - Aktywuj środowisko: `source myenv/bin/activate` (Linux/macOS) lub `myenv\Scripts\activate` (Windows)
      - Zainstaluj paczki: `pip install -r requirements.txt`
  - Używając Conda:
      - Utwórz nowe środowisko: `conda create --name myenv`
      - Aktywuj środowisko: `conda activate myenv`
      - Zainstaluj paczki: `pip install -r requirements.txt`
2. Uruchom pipeline dla modelu GAE: `python experiments/run_experiments_gae.py`
3. Uruchom pipeline dla modelu VGAE: `python experiments/run_experiments_vgae.py`

## Raport z omówionymi wynikami projektu
W folderze `report` dostępny jest pełen raport z omówionymi wynikami prac.

## Autorzy
- [Jakub Antczak](https://github.com/khiras1)
- [Joanna Wojciechowicz](https://github.com/joannaww)
