import os
import torch
from .model import FraudDetectionModel
from .evaluate import evaluate_model

def get_model_save_path(base_dir='trainings'):
    # Cria o diretório base se não existir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Encontrar o próximo número de set_x
    set_num = len([d for d in os.listdir(base_dir) if d.startswith('set_')])
    new_set_folder = os.path.join(base_dir, f'set_{set_num + 1}')

    # Cria a nova pasta set_x
    os.makedirs(new_set_folder, exist_ok=True)

    return new_set_folder

def train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=50, batch_size=64):
    for epoch in range(epochs):
        model.train()
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

def train_in_series(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, optimizer, series_count=5, epochs=50):
    # Obtém o caminho para salvar o modelo
    save_path = get_model_save_path()

    for series in range(series_count):
        print(f'Treinamento n. {series + 1}, Série n. {series + 1}/{series_count}')

        # Mapeia o último arquivo salvo para continuar a partir dele
        if series > 0:
            last_model_file = max([f for f in os.listdir(save_path) if f.startswith('model_series_')], key=os.path.getctime)
            model.load_state_dict(torch.load(os.path.join(save_path, last_model_file)))
        
        # Treinamento do modelo
        train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=epochs)

        # Avaliação do modelo
        evaluate_model(model, X_test_tensor, y_test_tensor)

        # Salvar o modelo após a série
        torch.save(model.state_dict(), os.path.join(save_path, f'model_series_{series + 1}.pth'))
