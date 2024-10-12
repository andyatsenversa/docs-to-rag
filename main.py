import argparse
import yaml
from src.preparation import rag_preparation
from src.retrieval import retrieval
from src.generation import generation
from src.optimization import optimization
from src.deployment import deployment

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(config):
    print("Preparing data...")
    rag_preparation.process_documents(config['folder_path'], config)

def run_retrieval(config, query):
    print("Running retrieval...")
    retriever = retrieval.RetrievalComponent(config)
    return retriever.get_relevant_chunks(query)

def generate_response(config, query, context):
    print("Generating response...")
    generator = generation.GenerationComponent(config.get('generation_model', 'gpt2'))
    return generator.generate_response(query, context)

def optimize_model(config):
    print("Optimizing model...")
    optimizer = optimization.OptimizationComponent(config)
    optimizer.fine_tune()

def start_server(config):
    print("Starting server...")
    deployment.run_server(host=config.get('host', '0.0.0.0'), port=config.get('port', 5000))

def main():
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--prepare', action='store_true', help='Prepare data')
    parser.add_argument('--optimize', action='store_true', help='Optimize model')
    parser.add_argument('--query', help='Run a query through the RAG system')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    
    args = parser.parse_args()
    config = load_config(args.config)

    if args.prepare:
        prepare_data(config)
    
    if args.optimize:
        optimize_model(config)
    
    if args.query:
        chunks = run_retrieval(config, args.query)
        context = " ".join(chunks)
        response = generate_response(config, args.query, context)
        print(f"Query: {args.query}")
        print(f"Response: {response}")
    
    if args.serve:
        start_server(config)

if __name__ == "__main__":
    main()