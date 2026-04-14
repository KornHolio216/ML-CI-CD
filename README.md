# FastAPI ML API

To repozytorium przygotowałem w ramach laboratoriów z przedmiotu **Nowoczesne Technologie Przetwarzania Danych**, ale potraktowałem je też jako praktyczny projekt pokazujący, jak wystawić model ML przez API, skonteneryzować aplikację i wdrożyć ją w chmurze.

W projekcie zbudowałem aplikację w **FastAPI**, dodałem endpoint do predykcji, przygotowałem środowisko w **Dockerze**, uruchomiłem całość przez **Docker Compose** i dostosowałem aplikację do wdrożenia na **Google Cloud Run**.

Model udostępniany przez API to prosty klasyfikator oparty na `LogisticRegression`, przygotowany jako demonstracja serwowania predykcji w aplikacji webowej.

## Co pokazuje ten projekt

W tym projekcie zaimplementowałem:

- API w **FastAPI** do serwowania modelu ML,
- endpointy `GET /`, `GET /health`, `GET /info` i `POST /predict`,
- walidację danych wejściowych przy użyciu **Pydantic**,
- prosty model klasyfikacyjny oparty na **LogisticRegression**,
- konteneryzację aplikacji w **Dockerze**,
- obsługę zmiennej środowiskowej `APP_ENV`,
- konfigurację pod wdrożenie na **Google Cloud Run**.

## Technologie

- Python
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- Joblib
- Docker
- Docker Compose
- Google Cloud Run

## Struktura projektu

```text
lab03ntpd/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── artifacts/
    └── model_v1.joblib
```

## Uruchomienie lokalne

### 1. Utworzenie i aktywacja środowiska wirtualnego

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalacja zależności

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uruchomienie aplikacji

```bash
uvicorn app:app --reload
```

Po uruchomieniu aplikacja będzie dostępna pod adresami:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Endpointy

### `GET /health`

Zwracam status działania aplikacji oraz aktywne środowisko uruchomienia.

```json
{
  "status": "ok",
  "app_env": "development"
}
```

### `GET /info`

Zwracam podstawowe informacje o modelu oraz środowisku.

```json
{
  "model_type": "LogisticRegression",
  "number_of_features": 2,
  "classes": ["klasa_0", "klasa_1"],
  "model_path": "artifacts/model_v1.joblib",
  "app_env": "development"
}
```

### `POST /predict`

Przyjmuję dane wejściowe w formacie JSON i zwracam predykcję modelu.

Przykładowe dane wejściowe:

```json
{
  "feature_1": 1.0,
  "feature_2": 1.1
}
```

Przykładowa odpowiedź:

```json
{
  "prediction": 0,
  "predicted_class_name": "klasa_0",
  "probability_class_0": 0.8486,
  "probability_class_1": 0.1514,
  "input_data": {
    "feature_1": 1.0,
    "feature_2": 1.1
  },
  "app_env": "development"
}
```

## Szybkie testy

### Windows PowerShell

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/info
curl.exe --% -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"feature_1\": 1.0, \"feature_2\": 1.1}"
```

## Docker

### Budowa obrazu

```bash
docker build -t fastapi-ml-api .
```

### Uruchomienie kontenera

```bash
docker run -d -p 8000:8080 --name fastapi-ml-api-container fastapi-ml-api
```

### Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose down
```

W konfiguracji `docker-compose.yml` uruchamiam:
- `api` – aplikację FastAPI z modelem ML,
- `mongodb` – dodatkowy serwis w tej samej sieci dockerowej.

## Zmienne środowiskowe

Dodałem obsługę zmiennej środowiskowej `APP_ENV`, dzięki czemu aplikacja może działać w różnych środowiskach bez zmiany kodu.

```python
APP_ENV = os.getenv("APP_ENV", "development")
```

Przykład uruchomienia lokalnego:

### Windows PowerShell

```powershell
$env:APP_ENV="development"
uvicorn app:app --reload
```

### Linux / macOS

```bash
APP_ENV=development uvicorn app:app --reload
```

## Deployment na Google Cloud Run

Projekt przygotowałem także pod wdrożenie na **Google Cloud Run**.

### Build i push obrazu

```bash
docker build -t gcr.io/YOUR_PROJECT_ID/my-ml-app:v1 .
docker push gcr.io/YOUR_PROJECT_ID/my-ml-app:v1
```

### Deploy usługi

```bash
gcloud run deploy my-ml-app \
  --image gcr.io/YOUR_PROJECT_ID/my-ml-app:v1 \
  --platform managed \
  --region europe-central2 \
  --allow-unauthenticated
```

### Konfiguracja środowiska w Cloud Run

```bash
gcloud run deploy my-ml-app \
  --image gcr.io/YOUR_PROJECT_ID/my-ml-app:v2 \
  --platform managed \
  --region europe-central2 \
  --allow-unauthenticated \
  --set-env-vars APP_ENV=production \
  --max-instances 2
```

## Konfiguracja portu

Cloud Run przekazuje port aplikacji przez zmienną `PORT`, dlatego dostosowałem `Dockerfile` do takiego uruchomienia:

```dockerfile
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
```