#!/usr/bin/env python3
"""
NeuSpell BERT Fine-tuning Example
Demonstrates how to fine-tune a BERT model for spelling correction using NeuSpell
"""

import os
import sys
import time
from typing import List, Tuple

# Add neuspell to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuspell import BertChecker
from neuspell.seq_modeling.helpers import get_tokens, load_data, train_validation_split
from neuspell.commons import DEFAULT_TRAINTEST_DATA_PATH

def create_sample_data(output_dir: str = "sample_data", num_samples: int = 1000):
    """
    Create sample training data for demonstration purposes.
    In practice, you would have your own training data files.
    
    Args:
        output_dir: Directory to save sample data files
        num_samples: Number of training samples to generate
    """
    print(f"Creating sample training data with {num_samples} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample clean sentences (correct spelling)
    clean_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Computer vision enables machines to interpret visual information.",
        "Data science combines statistics and programming to extract insights.",
        "Python is a popular programming language for data analysis.",
        "Spell checking is an important feature in text processing.",
        "Transformers have revolutionized the field of natural language processing.",
        "BERT is a powerful language model for various NLP tasks.",
        "Fine-tuning allows us to adapt pre-trained models to specific tasks.",
        "Training data quality is crucial for model performance.",
        "Evaluation metrics help us understand model effectiveness.",
        "Batch processing improves computational efficiency.",
        "Neural networks learn patterns from training data.",
        "Attention mechanisms help models focus on relevant information.",
        "Embeddings capture semantic relationships between words.",
        "Gradient descent optimizes model parameters during training.",
        "Validation sets help prevent overfitting during training.",
        "Transfer learning leverages pre-trained models for new tasks."
    ]
    
    # Function to introduce spelling errors
    def introduce_errors(text: str) -> str:
        """Introduce common spelling errors"""
        words = text.split()
        corrupted_words = []
        
        for word in words:
            # Skip punctuation and short words
            if len(word) <= 2 or not word.isalpha():
                corrupted_words.append(word)
                continue
            
            # Introduce errors with some probability
            import random
            if random.random() < 0.3:  # 30% chance of error
                error_type = random.choice(['swap', 'delete', 'insert', 'substitute'])
                
                if error_type == 'swap' and len(word) > 3:
                    # Swap adjacent characters
                    pos = random.randint(1, len(word) - 2)
                    word_list = list(word)
                    word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                    word = ''.join(word_list)
                elif error_type == 'delete' and len(word) > 3:
                    # Delete a character
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos + 1:]
                elif error_type == 'insert':
                    # Insert a random character
                    pos = random.randint(1, len(word) - 1)
                    char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    word = word[:pos] + char + word[pos:]
                elif error_type == 'substitute':
                    # Substitute a character
                    pos = random.randint(1, len(word) - 2)
                    char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    word = word[:pos] + char + word[pos + 1:]
            
            corrupted_words.append(word)
        
        return ' '.join(corrupted_words)
    
    # Generate training data
    clean_data = []
    corrupt_data = []
    
    for i in range(num_samples):
        # Cycle through clean sentences
        clean_sentence = clean_sentences[i % len(clean_sentences)]
        corrupt_sentence = introduce_errors(clean_sentence)
        
        clean_data.append(clean_sentence)
        corrupt_data.append(corrupt_sentence)
    
    # Save to files
    clean_file = os.path.join(output_dir, "clean.txt")
    corrupt_file = os.path.join(output_dir, "corrupt.txt")
    
    with open(clean_file, 'w', encoding='utf-8') as f:
        for sentence in clean_data:
            f.write(sentence + '\n')
    
    with open(corrupt_file, 'w', encoding='utf-8') as f:
        for sentence in corrupt_data:
            f.write(sentence + '\n')
    
    print(f"Sample data created:")
    print(f"  Clean sentences: {clean_file}")
    print(f"  Corrupt sentences: {corrupt_file}")
    print(f"  Total samples: {len(clean_data)}")
    
    # Show some examples
    print("\nSample training pairs:")
    for i in range(min(5, len(clean_data))):
        print(f"  Clean:   {clean_data[i]}")
        print(f"  Corrupt: {corrupt_data[i]}")
        print()
    
    return clean_file, corrupt_file

def finetune_bert_basic(clean_file: str, corrupt_file: str, data_dir: str = ""):
    """
    Basic fine-tuning using the built-in finetune method.
    
    Args:
        clean_file: Path to file with correct sentences
        corrupt_file: Path to file with corrupted sentences
        data_dir: Directory containing the data files
    """
    print("=== Basic BERT Fine-tuning ===")
    
    # Initialize BERT checker
    print("Initializing BERT checker...")
    checker = BertChecker()
    checker.from_pretrained()
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    start_time = time.time()
    
    checker.finetune(
        clean_file=clean_file,
        corrupt_file=corrupt_file,
        data_dir=data_dir,
        validation_split=0.2,
        n_epochs=2  # Use more epochs for better results
    )
    
    end_time = time.time()
    print(f"Fine-tuning completed in {end_time - start_time:.2f} seconds")
    
    return checker

def finetune_bert_custom(clean_file: str, corrupt_file: str, data_dir: str = ""):
    """
    Custom fine-tuning with more control over the process.
    
    Args:
        clean_file: Path to file with correct sentences
        corrupt_file: Path to file with corrupted sentences
        data_dir: Directory containing the data files
    """
    print("=== Custom BERT Fine-tuning ===")
    
    # Step 1: Load and prepare data
    print("Loading training data...")
    train_data = load_data(data_dir, clean_file, corrupt_file)
    train_data, valid_data = train_validation_split(train_data, 0.8, seed=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    
    # Step 2: Create vocabulary
    print("Creating vocabulary...")
    vocab = get_tokens(
        [item[0] for item in train_data],  # Clean sentences
        keep_simple=True,
        min_max_freq=(1, float("inf")),
        topk=10000
    )
    
    print(f"Vocabulary size: {len(vocab['token2idx'])}")
    
    # Step 3: Initialize model with custom configuration
    print("Initializing BERT model...")
    checker = BertChecker(device="cuda" if sys.platform != "darwin" else "cpu")
    
    # Use from_huggingface for more control
    checker.from_huggingface(
        bert_pretrained_name_or_path="bert-base-cased",
        vocab=vocab
    )
    
    # Step 4: Fine-tune
    print("Starting custom fine-tuning...")
    start_time = time.time()
    
    checker.finetune(
        clean_file=clean_file,
        corrupt_file=corrupt_file,
        data_dir=data_dir,
        validation_split=0.2,
        n_epochs=3
    )
    
    end_time = time.time()
    print(f"Custom fine-tuning completed in {end_time - start_time:.2f} seconds")
    
    return checker

def evaluate_model(checker: BertChecker, clean_file: str, corrupt_file: str, data_dir: str = ""):
    """
    Evaluate the fine-tuned model.
    
    Args:
        checker: Fine-tuned BERT checker
        clean_file: Path to clean test sentences
        corrupt_file: Path to corrupt test sentences
        data_dir: Directory containing the data files
    """
    print("=== Model Evaluation ===")
    
    # Use the built-in evaluation method
    print("Running evaluation...")
    checker.evaluate(
        clean_file=clean_file,
        corrupt_file=corrupt_file,
        data_dir=data_dir
    )

def test_corrections(checker: BertChecker):
    """
    Test the fine-tuned model on sample sentences.
    
    Args:
        checker: Fine-tuned BERT checker
    """
    print("=== Testing Corrections ===")
    
    # Test sentences with errors
    test_sentences = [
        "Ths is a sampel sentece with erors.",
        "Machien lerning is fascinatng and powerfull.",
        "Natral langauge procesing helps computrs understand text.",
        "Transfomers have revolutionzed the feild of NLP.",
        "BERT is a powerfull langauge modell for many tasks."
    ]
    
    print("Testing corrections on sample sentences:")
    for i, sentence in enumerate(test_sentences, 1):
        corrected = checker.correct(sentence)
        print(f"{i}. Original:  {sentence}")
        print(f"   Corrected: {corrected}")
        print()

def use_different_bert_models():
    """
    Example of using different BERT model variants.
    """
    print("=== Using Different BERT Models ===")
    
    # Available BERT variants
    bert_models = [
        "bert-base-cased",
        "bert-base-uncased",
        "distilbert-base-cased",
        "distilbert-base-uncased",
    ]
    
    print("Available BERT models for fine-tuning:")
    for i, model in enumerate(bert_models, 1):
        print(f"{i}. {model}")
    
    # Example: Using DistilBERT
    print("\nExample: Using DistilBERT")
    try:
        # Create sample data
        clean_file, corrupt_file = create_sample_data("distilbert_data", 100)
        data_dir = "distilbert_data"
        
        # Load data for vocab creation
        train_data = load_data(data_dir, os.path.basename(clean_file), os.path.basename(corrupt_file))
        vocab = get_tokens([item[0] for item in train_data], keep_simple=True, topk=5000)
        
        # Initialize with DistilBERT
        checker = BertChecker(device="cpu")
        checker.from_huggingface(
            bert_pretrained_name_or_path="distilbert-base-cased",
            vocab=vocab
        )
        
        print("DistilBERT model initialized successfully!")
        
        # Test correction
        test_text = "This is a test sentece with som erors."
        corrected = checker.correct(test_text)
        print(f"Test: {test_text}")
        print(f"Corrected: {corrected}")
        
    except Exception as e:
        print(f"Error with DistilBERT: {e}")

def main():
    """
    Main function demonstrating BERT fine-tuning for spell correction.
    """
    print("NeuSpell BERT Fine-tuning Example")
    print("=" * 40)
    
    try:
        # Step 1: Create sample training data
        print("Step 1: Creating sample training data...")
        clean_file, corrupt_file = create_sample_data("training_data", 500)
        data_dir = "training_data"
        
        # Step 2: Basic fine-tuning
        print("\nStep 2: Basic fine-tuning...")
        checker = finetune_bert_basic(
            clean_file=os.path.basename(clean_file),
            corrupt_file=os.path.basename(corrupt_file),
            data_dir=data_dir
        )
        
        # Step 3: Test the fine-tuned model
        print("\nStep 3: Testing fine-tuned model...")
        test_corrections(checker)
        
        # Step 4: Evaluate the model
        print("\nStep 4: Evaluating model...")
        # Create separate test data for evaluation
        test_clean_file, test_corrupt_file = create_sample_data("test_data", 100)
        test_data_dir = "test_data"
        
        evaluate_model(
            checker,
            clean_file=os.path.basename(test_clean_file),
            corrupt_file=os.path.basename(test_corrupt_file),
            data_dir=test_data_dir
        )
        
        # Step 5: Example with different BERT models
        print("\nStep 5: Different BERT models...")
        use_different_bert_models()
        
        print("\n" + "=" * 40)
        print("Fine-tuning example completed successfully!")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 