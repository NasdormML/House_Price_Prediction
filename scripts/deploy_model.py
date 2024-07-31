import os
import shutil

def deploy_model(source_model_path, deployment_directory):
    """
    :param source_model_path: Путь к файлу модели
    :param deployment_directory: Директория для развертывания модели
    """
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Модель не найдена по пути {source_model_path}")
    
    os.makedirs(deployment_directory, exist_ok=True)
    destination_model_path = os.path.join(deployment_directory, os.path.basename(source_model_path))
    
    shutil.copy2(source_model_path, destination_model_path)
    print(f"Модель развернута в {destination_model_path}")

if __name__ == "__main__":
    model_path = './models/trained_model.pkl'  # Путь к вашей сохраненной модели
    deployment_dir = 'deployment'  # Директория для развертывания модели

    deploy_model(model_path, deployment_dir)
