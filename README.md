# Bike Rental Prediction

A machine learning model for forecasting bike rental demand using a Random Forest Regressor. Features a production-ready sklearn pipeline with custom transformers for data imputation, encoding, outlier handling, and categorical mapping.

## Features

- **Random Forest Regressor** for demand prediction with configurable hyperparameters
- **Custom sklearn transformers:**
  - `WeekdayImputer` -- Extracts day name from date when weekday is missing
  - `WeathersitImputer` -- Fills missing weather data with the most frequent category
  - `Mapper` -- Maps categorical values to numerical encodings (season, month, hour, etc.)
  - `OutlierHandler` -- Caps outliers using IQR-based bounds
  - `WeekdayOneHotEncoder` -- One-hot encodes weekday categories
- **Full ML pipeline** from data preprocessing through prediction
- **YAML-driven configuration** for features, mappings, and model parameters
- **Input validation** with Pydantic schemas
- **Model persistence** via joblib serialization
- **Unit tests** for feature transformers

## Tech Stack

- **Language:** Python
- **ML Framework:** scikit-learn (RandomForestRegressor)
- **Data Processing:** pandas, NumPy
- **Configuration:** strictyaml, ruamel.yaml
- **Validation:** Pydantic
- **Serialization:** joblib
- **Testing:** pytest

## Project Structure

```
bike-rental-prediction/
├── Application/
│   ├── bikeshare_model/
│   │   ├── config.yml             # Feature mappings & model parameters
│   │   ├── config/core.py         # Configuration loading
│   │   ├── pipeline.py            # sklearn Pipeline definition
│   │   ├── train_pipeline.py      # Training script
│   │   ├── predict.py             # Prediction interface
│   │   ├── processing/
│   │   │   ├── features.py        # Custom transformers
│   │   │   ├── data_manager.py    # Data loading & model persistence
│   │   │   └── validation.py      # Input validation
│   │   ├── datasets/              # Training data (CSV)
│   │   ├── trained_models/        # Serialized model output
│   │   └── tests/                 # Unit tests
│   └── requirements/
│       └── requirements.txt
```

## Setup / Installation

```bash
git clone https://github.com/bnarasimha21/bike-rental-prediction.git
cd bike-rental-prediction
python -m venv venv
source venv/bin/activate
pip install -r Application/requirements/requirements.txt
```

## Usage

### Train the Model

```bash
python Application/bikeshare_model/train_pipeline.py
```

### Make Predictions

```python
from bikeshare_model.predict import make_prediction

data = {
    'dteday': ['2012-04-10'],
    'season': ['summer'],
    'hr': ['3am'],
    'holiday': ['No'],
    'weekday': ['Tue'],
    'workingday': ['Yes'],
    'weathersit': ['Clear'],
    'temp': [8.92],
    'atemp': [7.001],
    'hum': [71.0],
    'windspeed': [8.9981],
}

result = make_prediction(input_data=data)
print(result)
```

### Run Tests

```bash
pytest Application/bikeshare_model/tests/
```

## License

MIT
