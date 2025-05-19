# ECG_disease_classification
Creating models for identifying and classification of important disorders of the heart from electrocardiogram (ECG)

## Project: "11. analyzing medical data for health monitoring"
This project focuses on developing machine learning models for identifying and classifying heart diseases using ECG time series data. By leveraging advanced deep learning techniques, the models will analyze ECG signals to detect abnormalities and classify different types of heart conditions. 

## Team
Curator: Nikita Karagodin (tg: @Einstein_30)

Maxim Pigalskiy (tg: @EIisLog)

Nikita Rodionov (tg: white_shpengler)

## Directory Structure
```
backend
├── app
│   ├── data
│   │   └── loader.py
│   ├── managers
│   │   └── model_manager.py
│   ├── routers
│   │   └── endpoints.py
│   ├── schemas
│   │   └── schemas.py
│   ├── utils
│   │   └── logger.py
│   ├── trainer.py
│   └── main.py
├── models
├── logs
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd backend
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
docker compose up
```
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/photo.jpg?raw=true)

## Features
- **Data Loading**: Load and preprocess data from various sources.
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/upload.jpg?raw=true)
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/predict.jpg?raw=true)
- **Model Management**: Train, evaluate, and save machine learning models.
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/create_model.jpg?raw=true)
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/set.jpg?raw=true)
![alt text](https://github.com/Nicky-Georgia/ECG_disease_classification/blob/main/pics/fit.jpg?raw=true)
- **API Endpoints**: Define and handle API requests for the application.
- **Data Validation**: Validate and serialize data using schemas.
- **Logging**: Set up logging for monitoring and debugging.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.