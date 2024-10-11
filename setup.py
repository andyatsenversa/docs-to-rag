import os
import subprocess
import sys
import venv
import yaml

def create_virtual_environment():
    venv_dir = 'rag_env'
    # check if venv alread exists
    if os.path.exists('rag_env'):
        print("Virtual environment already exists. Skipping creation.")
        return 'rag_env'

    print(f"Creating virtual environment in {venv_dir}...")
    venv.create(venv_dir, with_pip=True)
    return venv_dir

def install_dependencies(venv_dir):
    pip = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip')
    requirements = [
        'numpy',
        'pandas',
        'scikit-learn',
        'sentence-transformers',
        'faiss-cpu',
        'python-docx',
        'PyPDF2',
        'pyyaml',
        'python-dotenv',
        'psutil',
        'sqlalchemy',
        'nltk',
        'tqdm',
        'docx'
    ]
    print("Installing dependencies...")
    subprocess.check_call([pip, 'install'] + requirements)
    print("/nDownloading nltk files...")
    import nltk
    nltk.download('punkt', download_dir = venv_dir+'/nltk_data')
    nltk.download('punkt_tab', download_dir = venv_dir+'/nltk_data')
    nltk.download('wordnet', download_dir = venv_dir+'/nltk_data')
    nltk.download('omw-1.4', download_dir = venv_dir+'/nltk_data')

def create_config_file():
    if not os.path.exists('config.yaml'):
        config = {
            'folder_path': '''..original_documents''',
            'chunk_size': 256,
            'overlap': 50,
            'model_name': '''all-MiniLM-L6-v2''',
            # 'model_name': '''distilbert-base-nli-stsb-mean-tokens''',
            'output_index': '''rag_index.faiss''',
            'db_path': '''rag_database.sqlite''',
            'model_cache_dir': '''model_dir'''

        }
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        print("Created config.yaml with default values. Please update it with your settings.")
    else:
        print("config.yaml already exists. Please update it if necessary.")

def check_files():
    required_files = ['rag_preparation.py', 'chunking.py', 'test_rag_preparation.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Warning: {file} is missing. Please ensure all required files are present.")

## WONT RUN ON SEN MACHINE _ HUGGINFACE BLOCKED
def download_model(model_name, model_cache_dir):
    print(f"Downloading model {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, cache_folder=model_cache_dir)
    print("Model downloaded successfully.")

def main():
    venv_dir = create_virtual_environment()
    install_dependencies(venv_dir)
    create_config_file()
    check_files()
    
    # Download model
    config = yaml.safe_load(open('config.yaml'))
    download_model(config['model_name'], config['model_cache_dir'])

    print("\nSetup complete!")
    print("To activate the virtual environment:")
    if os.name != 'nt':
        print(f"source {venv_dir}/bin/activate.bat")
    else:
        print(f"{venv_dir}\\Scripts\\activate")
    
    print("\nThen run the tool with:")
    print("python rag_preparation.py config.yaml")

if __name__ == "__main__":
    main()