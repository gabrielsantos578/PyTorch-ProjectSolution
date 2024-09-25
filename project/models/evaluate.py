import torch
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = outputs.round()
        
        accuracy = accuracy_score(y_test_tensor, predictions)
        print(f'Acurácia do modelo: {accuracy:.4f}')
        
        print("Relatório de Classificação:")
        print(classification_report(y_test_tensor.numpy(), predictions.numpy()))  # Converte para numpy para evitar erros
