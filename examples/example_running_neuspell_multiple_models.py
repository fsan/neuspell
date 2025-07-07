#!/usr/bin/env python3
"""
NeuSpell Different Models Demo
Demonstrates usage of various NeuSpell models for spell correction
"""

from neuspell import available_checkers, BertChecker, SclstmChecker

def check_available_checkers():
    """Display all available spell checkers"""
    print("Available NeuSpell Checkers:")
    print("=" * 40)
    checkers = available_checkers()
    for i, checker in enumerate(checkers, 1):
        print(f"{i}. {checker}")
    print()

def test_bert_checker():
    """Test BERT-based spell checker"""
    print("Testing BERT Checker:")
    print("-" * 20)
    
    # Initialize BERT checker
    bert_checker = BertChecker()
    bert_checker.from_pretrained()
    
    # Test sentences with spelling errors
    test_sentences = [
        "This is a sampel sentence with erors.",
        "I have a frend who loves readng books.",
        "The weahter is beutiful today.",
        "Programming is challening but rewrding."
    ]
    
    for sentence in test_sentences:
        corrected = bert_checker.correct(sentence)
        print(f"Original:  {sentence}")
        print(f"Corrected: {corrected}")
        print()

def test_sclstm_checker():
    """Test SC-LSTM spell checker"""
    print("Testing SC-LSTM Checker:")
    print("-" * 23)
    
    # Initialize SC-LSTM checker
    sclstm_checker = SclstmChecker()
    sclstm_checker.from_pretrained()
    
    # Test sentences
    test_sentences = [
        "Ther are many diferent aproaches to solv this problm.",
        "Comunication is vry importnt in teamwork.",
        "Educaton is the key to succes in lif."
    ]
    
    for sentence in test_sentences:
        corrected = sclstm_checker.correct(sentence)
        print(f"Original:  {sentence}")
        print(f"Corrected: {corrected}")
        print()

def compare_models():
    """Compare different models on the same text"""
    print("Model Comparison:")
    print("=" * 20)
    
    # Initialize all checkers
    bert_checker = BertChecker()
    bert_checker.from_pretrained()
    
    sclstm_checker = SclstmChecker()
    sclstm_checker.from_pretrained()
    
    # Test sentence with multiple errors
    test_text = "The studnt was writting a leter to his teachr about the assigment."
    
    print(f"Original text: {test_text}")
    print()
    
    # Test with each model
    bert_result = bert_checker.correct(test_text)
    sclstm_result = sclstm_checker.correct(test_text)
    
    print(f"BERT correction:    {bert_result}")
    print(f"SC-LSTM correction: {sclstm_result}")
    print()

def batch_correction_example():
    """Example of batch correction"""
    print("Batch Correction Example:")
    print("=" * 25)
    
    # Initialize checker
    checker = BertChecker()
    checker.from_pretrained()
    
    # Multiple sentences for batch processing
    sentences = [
        "Artficial inteligence is revolutionzing the world.",
        "Machine lerning helps us make predictons.",
        "Deep lerning uses neural netwrks.",
        "Natural langauge procesing is a fascinatng field.",
        "Computer visoin enables machins to see."
    ]
    
    print("Processing batch of sentences...")
    
    # Correct each sentence
    corrected_sentences = []
    for i, sentence in enumerate(sentences, 1):
        corrected = checker.correct(sentence)
        corrected_sentences.append(corrected)
        print(f"{i}. Original:  {sentence}")
        print(f"   Corrected: {corrected}")
        print()
    
    return corrected_sentences

def custom_model_loading():
    """Example of loading specific model variants"""
    print("Custom Model Loading:")
    print("=" * 20)
    
    try:
        # You can specify different model variants
        # These are examples - actual availability depends on your installation
        
        # Load BERT with specific configuration
        bert_checker = BertChecker()
        bert_checker.from_pretrained()
        
        print("BERT model loaded successfully")
        
        test_text = "This is a test sentece with som erors."
        corrected = bert_checker.correct(test_text)
        print(f"Test: {test_text}")
        print(f"Corrected: {corrected}")
        
    except Exception as e:
        print(f"Error loading custom model: {e}")

def main():
    """Main function to run all demonstrations"""
    print("NeuSpell Models Demonstration")
    print("=" * 35)
    print()
    
    try:
        # Show available checkers
        check_available_checkers()
        
        # Test individual models
        test_bert_checker()
        test_sclstm_checker()
        compare_models()
        batch_correction_example()
        custom_model_loading()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure to install neuspell: pip install neuspell")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure models are properly downloaded and initialized")

if __name__ == "__main__":
    main()
