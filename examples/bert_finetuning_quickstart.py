#!/usr/bin/env python3
"""
Quick Start: BERT Fine-tuning with NeuSpell
A minimal example showing how to fine-tune BERT for spelling correction
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuspell import BertChecker

def create_sample_data():
    """Create minimal sample data for demonstration"""
    # Clean sentences (correct spelling)
    clean_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a powerful tool for data analysis.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can achieve impressive results.",
        "Python is widely used for artificial intelligence projects.",
        "Spell checking is an important feature in text editors.",
        "BERT models are pre-trained on large text corpora.",
        "Fine-tuning adapts models to specific tasks.",
        "Training data quality affects model performance.",
        "Evaluation metrics help assess model effectiveness."
    ]
    
    # Corrupt sentences (with spelling errors)
    corrupt_sentences = [
        "The quik brown fox jumps over the lasy dog.",
        "Machien lerning is a powerfull tool for data analisis.",
        "Natral langauge procesing helps computrs understand text.",
        "Deep lerning modells can achive impresive results.",
        "Python is widly used for artifical inteligence projects.",
        "Spel cheking is an importnt feture in text editors.",
        "BERT modells are pre-traind on large text corpora.",
        "Fine-tuning adapts modells to specifc tasks.",
        "Traing data qualiy afects model performnce.",
        "Evaluaton metircs help ases model efectivenes."
    ]
    
    # Create data directory
    os.makedirs("quickstart_data", exist_ok=True)
    
    # Save to files
    with open("quickstart_data/clean.txt", "w") as f:
        for sentence in clean_sentences:
            f.write(sentence + "\n")
    
    with open("quickstart_data/corrupt.txt", "w") as f:
        for sentence in corrupt_sentences:
            f.write(sentence + "\n")
    
    print("Sample data created in 'quickstart_data' directory")
    return "quickstart_data"

def main():
    """Main function demonstrating BERT fine-tuning quickstart"""
    print("NeuSpell BERT Fine-tuning Quickstart")
    print("=" * 40)
    
    try:
        # Step 1: Create sample data
        print("\n1. Creating sample training data...")
        data_dir = create_sample_data()
        
        # Step 2: Initialize BERT checker
        print("\n2. Initializing BERT model...")
        checker = BertChecker()
        checker.from_pretrained()
        print("✓ BERT model loaded successfully")
        
        # Step 3: Fine-tune the model
        print("\n3. Fine-tuning model...")
        print("   This may take a few minutes...")
        
        checker.finetune(
            clean_file="clean.txt",
            corrupt_file="corrupt.txt",
            data_dir=data_dir,
            validation_split=0.2,
            n_epochs=2
        )
        
        print("✓ Fine-tuning completed")
        
        # Step 4: Test the fine-tuned model
        print("\n4. Testing the fine-tuned model...")
        
        test_sentences = [
            "Ths is a test sentece with erors.",
            "Machien lerning is fascinatng.",
            "Spel cheking is importnt."
        ]
        
        for i, sentence in enumerate(test_sentences, 1):
            corrected = checker.correct(sentence)
            print(f"   {i}. Original:  {sentence}")
            print(f"      Corrected: {corrected}")
        
        print("\n✓ Fine-tuning example completed successfully!")
        print("\nYour fine-tuned model is saved in:")
        print(f"   {data_dir}/new_models/bert-base-cased/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed: pip install torch transformers")
        print("2. Make sure neuspell is installed: pip install -e .")
        print("3. Check if you have internet connection for model download")

if __name__ == "__main__":
    main() 