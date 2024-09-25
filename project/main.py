# Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import torch.nn as nn
import torch.optim as optim
from processing.data_preprocessing import load_data, preprocess_data
from models.model import FraudDetectionModel
from models.train import train_in_series
from utils.utils import to_tensor

# Carregar e processar os dados
data = load_data('data/creditcard.csv')
X_train_res, X_test, y_train_res, y_test = preprocess_data(data)

# Converter os dados de treino e teste para tensores
X_train_tensor = to_tensor(X_train_res)
y_train_tensor = to_tensor(y_train_res).view(-1, 1)
X_test_tensor = to_tensor(X_test)
y_test_tensor = to_tensor(y_test).view(-1, 1)

# Inicializar o modelo, a função de perda e o otimizador
model = FraudDetectionModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento do modelo em séries
train_in_series(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, optimizer)
# train_in_series(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, optimizer, series_count=5, epochs=50, batch_size=32)
