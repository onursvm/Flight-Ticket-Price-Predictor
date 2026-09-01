# ✈️ Flight Ticket Price Prediction

A machine learning project for predicting airline ticket prices using flight-related features such as airline, route, departure/arrival time, flight duration, number of stops, travel class, and date information.

The project combines **data preprocessing, feature engineering, exploratory analysis, model comparison, hyperparameter tuning, and regression evaluation** in a Jupyter Notebook.

---

## 📌 Project Overview

The notebook works with two flight-ticket datasets:

- `business.csv`
- `economy.csv`

The two datasets are combined into a single dataset and a `class` feature is created to distinguish the ticket type:

- `1` → Business
- `0` → Economy

After preprocessing and feature engineering, several regression algorithms are trained and evaluated to find a strong model for ticket-price prediction.

---

## 🎯 Objective

The main objective is to build a regression model capable of estimating the price of a flight from the information available about that flight.

The target variable is:

```text
price
```

The final processed dataset contains **300,261 records**.

The original data covers dates from **11 February 2022 to 31 March 2022**.

---

## 📊 Dataset Features

The source datasets contain the following fields:

| Feature | Description |
|---|---|
| `date` | Flight date |
| `airline` | Airline operating the flight |
| `ch_code` | Airline code |
| `num_code` | Flight number/code |
| `dep_time` | Departure time |
| `from` | Departure city |
| `time_taken` | Flight duration |
| `stop` | Number/type of stops |
| `arr_time` | Arrival time |
| `to` | Destination city |
| `price` | Flight ticket price |
| `class` | Ticket class: Business = 1, Economy = 0 |

The notebook identifies 8 airlines in the data, including Air India, Vistara, SpiceJet, AirAsia, GO FIRST, Indigo, Trujet, and StarAir.

---

## 🔄 Data Processing & Feature Engineering

The notebook follows a multi-stage preprocessing pipeline.

### 1. Load the datasets

```python
df1 = pd.read_csv("business.csv")
df2 = pd.read_csv("economy.csv")
```

### 2. Add ticket-class information

```python
df1["class"] = 1
df2["class"] = 0
```

The datasets are then concatenated into one DataFrame.

### 3. Date transformation

The `date` column is converted to a datetime format and transformed into a numerical `day-of-year` feature.

```python
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df["day-of-year"] = df["date"].dt.dayofyear
```

The original `date` column is subsequently removed.

### 4. Airline encoding

The categorical `airline` feature is converted into dummy variables using one-hot encoding.

### 5. Departure-time conversion

Departure times such as:

```text
18:00
19:00
07:05
```

are converted into minutes after midnight.

For example:

```text
18:00 → 1080
19:00 → 1140
07:05 → 425
```

This creates the:

```text
dep_time_minutes
```

feature.

### 6. Duration and arrival-time transformation

Flight duration and arrival-time information are converted into numerical representations so that they can be used by regression algorithms.

### 7. Route/city encoding

Origin/destination city information is transformed into numerical city features such as:

```text
Delhi_city
Mumbai_city
Bangalore_city
Kolkata_city
Hyderabad_city
Chennai_city
```

### 8. Price transformation

The ticket price is converted from its original string representation into a numerical target used by the regression models.

---

## 🧠 Machine Learning Approach

The project treats ticket-price prediction as a **supervised regression problem**.

The data is divided into training and test sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

A `StandardScaler` is also used to scale the model inputs.

### Evaluation Metrics

The models are evaluated using:

- **MAE — Mean Absolute Error**
- **MSE — Mean Squared Error**
- **R² — R-Squared**
- **RMSE — Root Mean Squared Error**

A higher R² and lower error values indicate better predictive performance.

---

## 🤖 Models Evaluated

The notebook experiments with multiple regression algorithms, including:

- Linear Regression
- Lasso
- Ridge
- ElasticNet
- KNeighborsRegressor
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- ExtraTreesRegressor
- BaggingRegressor
- MLPRegressor
- SVR
- AdaBoostRegressor
- and additional regressors through `LazyRegressor`

This broad comparison helps identify which model family performs best for the prepared flight-ticket data.

---

## 🔎 Hyperparameter Tuning

A `DecisionTreeRegressor` is optimized using `GridSearchCV` with 5-fold cross-validation.

The searched parameters include:

```text
criterion
max_depth
max_features
splitter
```

The best configuration found in the notebook is:

```text
criterion    = squared_error
max_depth    = None
max_features = None
splitter     = best
```

The corresponding cross-validation score reported by the notebook is approximately:

```text
0.98058
```

---

## 🏆 Results

### Final Random Forest Model

The notebook trains a `RandomForestRegressor` using parallel processing (`n_jobs=-1`).

Reported test-set results:

| Metric | Score |
|---|---:|
| MAE | **0.8980** |
| MSE | **5.5296** |
| R² | **0.9893** |

The resulting **R² ≈ 0.9893** indicates that the final Random Forest model explains approximately 98.9% of the variance in the test-set target values.

### LazyRegressor Comparison

A 5,000-sample subset is also used with `LazyRegressor` to compare a large collection of regression algorithms.

Top-performing models reported by the notebook:

| Model | R² | RMSE | Time Taken |
|---|---:|---:|---:|
| **RandomForestRegressor** | **0.9678** | **4.1091** | 3.4393 s |
| HistGradientBoostingRegressor | 0.9664 | 4.2024 | 1.6499 s |
| ExtraTreesRegressor | 0.9650 | 4.2888 | 2.5534 s |
| BaggingRegressor | 0.9634 | 4.3862 | 0.3647 s |
| GradientBoostingRegressor | 0.9522 | 5.0099 | 0.9137 s |

> **Note:** The LazyRegressor comparison uses a 5,000-sample subset, while the separately trained final Random Forest reports its own test-set metrics. Therefore, these scores should not be treated as a direct one-to-one benchmark.

---

## 🏗️ Project Pipeline

```text
business.csv ──────┐
                   ├──► Data Loading
economy.csv ───────┘
                         │
                         ▼
                  Dataset Combination
                         │
                         ▼
                 Data Cleaning
                         │
                         ▼
                Feature Engineering
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Time Features   City Encoding  Airline Encoding
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Train / Test Split
                         │
                         ▼
                   Standard Scaling
                         │
                         ▼
                Regression Models
                         │
                         ▼
                Model Evaluation
                         │
                         ▼
              Hyperparameter Tuning
                         │
                         ▼
                 Random Forest
                         │
                         ▼
                 Price Prediction
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Exploratory data visualization |
| **Scikit-learn** | Machine learning and evaluation |
| **LazyPredict** | Automated regression model comparison |
| **Jupyter Notebook** | Development and experimentation environment |

---

## 📁 Project Structure

A recommended repository structure is:

```text
Flight-Ticket-Prediction/
│
├── Flight Ticket Predict.ipynb
├── business.csv
├── economy.csv
└── README.md

```

The current notebook expects `business.csv` and `economy.csv` to be available in the working directory.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Flight-Ticket-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lazypredict jupyter
```

### 4. Add the datasets

Place:

```text
business.csv
economy.csv
```

in the project root directory.

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Flight Ticket Predict.ipynb
```

and execute the notebook cells.

---

## 📈 Key Takeaways

- The project uses **300K+ flight records** after combining the Business and Economy datasets.
- Time, duration, airline, route, stop, and class information are transformed into machine-learning-ready features.
- Multiple regression algorithms are compared instead of relying on a single model.
- `GridSearchCV` is used for Decision Tree hyperparameter optimization.
- `LazyRegressor` is used to quickly benchmark many regression algorithms.
- The separately trained **Random Forest Regressor achieved an R² of approximately 0.9893** on the reported test set.
- Random Forest also ranked first in the notebook's 5,000-sample LazyRegressor comparison.

---

## ⚠️ Notes

This repository contains an experimental machine-learning notebook rather than a production deployment.

For a production-ready version, the project could be extended with:

- A reproducible preprocessing pipeline using `Pipeline` / `ColumnTransformer`
- Cross-validation and systematic hyperparameter optimization for the final model
- Model serialization with `joblib`
- A REST API using FastAPI or Flask
- A web interface for entering flight details and receiving price predictions
- Feature-importance analysis and model explainability
- Automated data validation and testing
- Docker-based deployment

---


## 👤 Author

**Onur Sevim**

If you found this project useful, feel free to ⭐ the repository and explore the notebook.
