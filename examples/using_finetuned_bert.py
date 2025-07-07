#!/usr/bin/env python3
"""
Using a Fine-tuned BERT Model
Demonstrates how to load and use a previously fine-tuned BERT model for spelling correction
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuspell import BertChecker

def load_finetuned_model(model_path: str, vocab_path: str = None):
    """
    Load a fine-tuned BERT model from a saved checkpoint.
    
    Args:
        model_path: Path to the directory containing the fine-tuned model
        vocab_path: Optional path to vocabulary file (will auto-detect if not provided)
    """
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Initialize checker
    checker = BertChecker()
    
    # Load the fine-tuned model
    # The model path should contain pytorch_model.bin and vocab.pkl
    checker.from_pretrained(
        ckpt_path=model_path,
        vocab_path=vocab_path
    )
    
    print("✓ Fine-tuned model loaded successfully")
    return checker

def load_huggingface_finetuned_model(model_path: str, bert_model_name: str = "bert-base-cased"):
    """
    Load a fine-tuned model that was trained with from_huggingface().
    
    Args:
        model_path: Path to the saved model directory
        bert_model_name: Name of the base BERT model that was fine-tuned
    """
    print(f"Loading HuggingFace fine-tuned model from: {model_path}")
    
    # Initialize checker
    checker = BertChecker()
    
    # Load with the original BERT model name and the checkpoint path
    checker.from_pretrained(
        bert_pretrained_name_or_path=bert_model_name,
        ckpt_path=model_path
    )
    
    print("✓ HuggingFace fine-tuned model loaded successfully")
    return checker

def batch_correction(checker: BertChecker, sentences: list):
    """
    Apply spelling correction to a batch of sentences.
    
    Args:
        checker: Fine-tuned BERT checker
        sentences: List of sentences to correct
    """
    print("Performing batch correction...")
    
    # Method 1: Using correct_strings for batch processing
    corrected_sentences = checker.correct_strings(sentences)
    
    # Display results
    print("\nBatch Correction Results:")
    print("-" * 40)
    for i, (original, corrected) in enumerate(zip(sentences, corrected_sentences), 1):
        print(f"{i}. Original:  {original}")
        print(f"   Corrected: {corrected}")
        print()
    
    return corrected_sentences

def interactive_correction(checker: BertChecker):
    """
    Interactive spelling correction mode.
    
    Args:
        checker: Fine-tuned BERT checker
    """
    print("\nInteractive Spelling Correction")
    print("=" * 35)
    print("Enter sentences to correct (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
                
            # Correct the sentence
            corrected = checker.correct(user_input)
            print(f"Corrected: {corrected}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def evaluate_on_test_data(checker: BertChecker, test_data_dir: str):
    """
    Evaluate the fine-tuned model on test data.
    
    Args:
        checker: Fine-tuned BERT checker
        test_data_dir: Directory containing test data files
    """
    print(f"\nEvaluating model on test data from: {test_data_dir}")
    
    # Check if test data exists
    clean_file = os.path.join(test_data_dir, "clean.txt")
    corrupt_file = os.path.join(test_data_dir, "corrupt.txt")
    
    if not (os.path.exists(clean_file) and os.path.exists(corrupt_file)):
        print("❌ Test data not found. Expected files:")
        print(f"   {clean_file}")
        print(f"   {corrupt_file}")
        return
    
    try:
        # Run evaluation
        checker.evaluate(
            clean_file="clean.txt",
            corrupt_file="corrupt.txt",
            data_dir=test_data_dir
        )
        print("✓ Evaluation completed")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def compare_with_pretrained(finetuned_checker: BertChecker, test_sentences: list):
    """
    Compare fine-tuned model with the original pre-trained model.
    
    Args:
        finetuned_checker: Fine-tuned BERT checker
        test_sentences: List of test sentences
    """
    print("\nComparing Fine-tuned vs Pre-trained Model")
    print("=" * 45)
    
    # Load pre-trained model for comparison
    pretrained_checker = BertChecker()
    pretrained_checker.from_pretrained()
    
    print("Correction Comparison:")
    print("-" * 25)
    
    for i, sentence in enumerate(test_sentences, 1):
        finetuned_result = finetuned_checker.correct(sentence)
        pretrained_result = pretrained_checker.correct(sentence)
        
        print(f"{i}. Original:    {sentence}")
        print(f"   Fine-tuned:  {finetuned_result}")
        print(f"   Pre-trained: {pretrained_result}")
        
        if finetuned_result != pretrained_result:
            print("   ⚠️  Results differ!")
        else:
            print("   ✓ Results match")
        print()

def main():
    """Main function demonstrating how to use fine-tuned BERT models"""
    print("Using Fine-tuned BERT Models with NeuSpell")
    print("=" * 45)
    
    # Example model paths (adjust these to your actual paths)
    model_paths = [
        "quickstart_data/new_models/bert-base-cased",
        "training_data/new_models/bert-base-cased",
        "training_data/new_models/distilbert-base-cased"
    ]
    
    # Find available models
    available_models = []
    for path in model_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "pytorch_model.bin")):
            available_models.append(path)
    
    if not available_models:
        print("❌ No fine-tuned models found.")
        print("Please run one of the following first:")
        print("   python bert_finetuning_quickstart.py")
        print("   python example_bert_finetuning.py")
        return
    
    # Use the first available model
    model_path = available_models[0]
    print(f"\nUsing model from: {model_path}")
    
    try:
        # Load the fine-tuned model
        print("\n1. Loading fine-tuned model...")
        checker = load_finetuned_model(model_path)
        
        # Test sentences with spelling errors
        test_sentences = [
            "Ths is a test sentece with multipel erors.",
            "Machien lerning algoritms are fascinatng.",
            "Natral langauge procesing is a chalenging feild.",
            "Deep lerning modells reqire large amunts of data.",
            "Fine-tuning alows us to adapt pre-traind modells."
        ]
        
        # 2. Batch correction
        print("\n2. Batch correction...")
        corrected_sentences = batch_correction(checker, test_sentences)
        
        # 3. Compare with pre-trained model
        print("\n3. Comparing with pre-trained model...")
        compare_with_pretrained(checker, test_sentences[:3])  # Use first 3 for comparison
        
        # 4. Evaluate on test data if available
        test_data_dirs = ["test_data", "quickstart_data"]
        for test_dir in test_data_dirs:
            if os.path.exists(test_dir):
                evaluate_on_test_data(checker, test_dir)
                break
        
        # 5. Interactive mode (optional)
        print("\n5. Interactive mode (optional)...")
        response = input("Would you like to try interactive correction? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_correction(checker)
        
        print("\n✓ Fine-tuned model usage example completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the model path is correct")
        print("2. Check if pytorch_model.bin and vocab.pkl exist in the model directory")
        print("3. Ensure the model was trained with compatible neuspell version")

if __name__ == "__main__":
    main() 