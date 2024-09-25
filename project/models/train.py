import os
import torch
from .model import FraudDetectionModel
from .evaluate import evaluate_model

def get_model_save_path(base_dir='trainings'):
    # Cria o diretório base se não existir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Encontrar o último número de set_x
    set_dirs = [d for d in os.listdir(base_dir) if d.startswith('set_')]
    if set_dirs:
        set_num = max([int(d.split('_')[1]) for d in set_dirs])
    else:
        set_num = 0  # Inicia com set_0, e incrementa na primeira vez

    # Incrementa o número do set
    set_num += 1
    new_set_folder = os.path.join(base_dir, f'set_{set_num}')
    if not os.path.exists(new_set_folder):
        os.makedirs(new_set_folder)

    return new_set_folder, set_num

def get_next_series_num(set_folder):
    # Encontrar o último número de série_y
    model_files = [f for f in os.listdir(set_folder) if f.startswith('model_series_')]
    if model_files:
        series_num = max([int(f.split('_')[2].split('.')[0]) for f in model_files])
    else:
        series_num = 0  # Inicia em 0 se não houver arquivos

    return series_num + 1  # Sempre inicia na próxima série

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
    # Obtém o caminho e o número do próximo set_x
    save_path, set_num = get_model_save_path()

    # Obtém o próximo número de série_y
    next_series_num = get_next_series_num(save_path)

    for series in range(series_count):
        current_series = next_series_num + series

        # Mapeia o último arquivo salvo para continuar a partir dele
        if current_series > 1:
            last_model_file = max([f for f in os.listdir(save_path) if f.startswith('model_series_')], key=lambda f: os.path.getctime(os.path.join(save_path, f)))
            model.load_state_dict(torch.load(os.path.join(save_path, last_model_file)))

        print(f'Treinamento n. {set_num}, Série n. {current_series}')

        # Treinamento do modelo
        train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=epochs)

        # Avaliação do modelo
        evaluate_model(model, X_test_tensor, y_test_tensor)

        # Salvar o modelo após a série
        torch.save(model.state_dict(), os.path.join(save_path, f'model_series_{current_series}.pth'))
