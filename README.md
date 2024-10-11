# RAG Preparation Tool

This tool prepares documents for Retrieval-Augmented Generation (RAG) by processing various document types, chunking the text, creating embeddings, and storing the results in a vector database.

## Features

- Supports multiple document types (DOCX, PDF, TXT)
- Sentence-based chunking for more coherent text segments
- Multi-processing for faster document processing
- FAISS index creation for efficient similarity search
- SQLite database for storing documents, chunks, and embeddings
- Configurable settings via YAML file
- Command-line interface for easy use
- Comprehensive unit tests

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rag-preparation-tool.git
   cd rag-preparation-tool
   ```

2. Run the setup script:
   ```
   python setup.py
   ```

   This script will create a virtual environment, install dependencies, and check for required files/settings.

## Configuration

Create a `config.yaml` file in the project directory with the following structure:

```yaml
folder_path: "path/to/your/documents"
chunk_size: 256
overlap: 50
model_name: "all-MiniLM-L6-v2"
output_index: "rag_index.faiss"
db_path: "rag_database.sqlite"
```

Adjust the values according to your needs.

## Usage

Run the tool from the command line:

```
python rag_preparation.py path/to/your/config.yaml
```

## Running Tests

To run the unit tests:

```
python -m unittest test_rag_preparation.py
```

## File Structure

- `rag_preparation.py`: Main script
- `chunking.py`: Contains the sentence-based chunking algorithm
- `test_rag_preparation.py`: Unit tests
- `config.yaml`: Configuration file
- `setup.py`: Setup script for environment preparation

## How It Works

1. **Document Processing**: The tool reads documents from the specified folder, extracting text from DOCX, PDF, and TXT files.

2. **Text Chunking**: The extracted text is cleaned and chunked using a sentence-based algorithm, which respects sentence boundaries for more coherent segments.

3. **Embedding Creation**: The tool uses a Sentence Transformer model to create embeddings for each text chunk.

4. **FAISS Index**: A FAISS index is created from the embeddings, allowing for efficient similarity search.

5. **Database Storage**: Documents, chunks, and embeddings are stored in a SQLite database for easy retrieval and management.

## Extending the Tool

- To support additional document types, add new extraction functions in `rag_preparation.py` and update the `process_document` function.
- To use a different embedding model, change the `model_name` in the configuration file.
- To modify the chunking algorithm, update the `sentence_based_chunking` function in `chunking.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.