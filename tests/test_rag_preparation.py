import unittest
import tempfile
import os
from rag_preparation import (
    load_config,
    clean_text,
    process_document,
    create_embeddings,
    create_faiss_index,
    save_to_database,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt
)
from chunking import sentence_based_chunking
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

class TestRAGPreparation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'folder_path': self.temp_dir,
            'chunk_size': 100,
            'overlap': 20,
            'model_name': 'all-MiniLM-L6-v2',
            'output_index': 'test_index.faiss',
            'db_path': 'test_db.sqlite'
        }
        with open(os.path.join(self.temp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)

    def test_load_config(self):
        loaded_config = load_config(os.path.join(self.temp_dir, 'config.yaml'))
        self.assertEqual(loaded_config, self.config)

    def test_clean_text(self):
        dirty_text = "This is a\n messy\r text   with   spaces"
        clean = clean_text(dirty_text)
        self.assertEqual(clean, "This is a messy text with spaces")

    def test_sentence_based_chunking(self):
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = sentence_based_chunking(text, chunk_size=20, overlap=5)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("This is the first sentence."))
        self.assertTrue(chunks[1].endswith("This is the third sentence."))

    def test_process_document(self):
        with open(os.path.join(self.temp_dir, 'test.txt'), 'w') as f:
            f.write("This is a test document. It has multiple sentences. We will process it.")
        result = process_document('test.txt', self.temp_dir, self.config)
        self.assertIn('test.txt', result)
        self.assertTrue(len(result['test.txt']) > 0)

    def test_create_embeddings(self):
        documents = {'test.txt': ['This is a test chunk.', 'This is another test chunk.']}
        embeddings = create_embeddings(documents, self.config['model_name'])
        self.assertIn('test.txt', embeddings)
        self.assertEqual(len(embeddings['test.txt']), 2)
        self.assertIsInstance(embeddings['test.txt'][0], np.ndarray)

    def test_create_faiss_index(self):
        embeddings = {'test.txt': [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]}
        index = create_faiss_index(embeddings)
        self.assertEqual(index.ntotal, 2)

    def test_save_to_database(self):
        documents = {'test.txt': ['This is a test chunk.', 'This is another test chunk.']}
        embeddings = {'test.txt': [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]}
        save_to_database(documents, embeddings, self.config['db_path'])
        self.assertTrue(os.path.exists(self.config['db_path']))

    def test_extract_text_from_docx(self):
        # You'll need to create a sample .docx file for this test
        pass

    def test_extract_text_from_pdf(self):
        # You'll need to create a sample .pdf file for this test
        pass

    def test_extract_text_from_txt(self):
        with open(os.path.join(self.temp_dir, 'test.txt'), 'w') as f:
            f.write("This is a test document.")
        text = extract_text_from_txt(os.path.join(self.temp_dir, 'test.txt'))
        self.assertEqual(text, "This is a test document.")

if __name__ == '__main__':
    unittest.main()