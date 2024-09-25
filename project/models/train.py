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

    # Encontrar o próximo número de serie_y
    series_num = len([d for d in os.listdir(new_set_folder) if d.startswith('serie_')])
    new_series_folder = os.path.join(new_set_folder, f'serie_{series_num + 1}')

    # Cria a nova pasta serie_y
    os.makedirs(new_series_folder, exist_ok=True)

    return new_series_folder

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

def train_in_series(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, optimizer, series_count=5, epochs=50, batch_size=64):
    # Obtém o caminho para salvar o modelo
    save_path = get_model_save_path()

    for series in range(series_count):
        print(f'Treinamento n. {series + 1}')  # Exibe o número do treinamento

        # Se não for a primeira série, carregue o modelo salvo
        if series > 0:
            model.load_state_dict(torch.load(os.path.join(save_path, f'model_series_{series}.pth')))

        print(f'Série n. {series + 1}')  # Exibe o número da série

        # Treinamento do modelo
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
            
            # Exibir detalhes do epoch e loss
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # Avaliação do modelo
        evaluate_model(model, X_test_tensor, y_test_tensor)

        # Salvar o modelo após a série
        torch.save(model.state_dict(), os.path.join(save_path, f'model_series_{series + 1}.pth'))
