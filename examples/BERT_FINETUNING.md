# BERT Fine-tuning for NeuSpell

This example demonstrates how to fine-tune a BERT model for spelling correction using the NeuSpell library.

## Overview

The `example_bert_finetuning.py` script shows several approaches to fine-tuning BERT models for spelling correction:

1. **Basic Fine-tuning** - Using the built-in `finetune()` method
2. **Custom Fine-tuning** - More control over the training process
3. **Different BERT Models** - Using various BERT variants (BERT, DistilBERT, etc.)
4. **Data Preparation** - Creating and formatting training data
5. **Model Evaluation** - Testing the fine-tuned model

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install torch transformers
pip install -e .  # Install neuspell from the project root
```

## Data Format

The training data should consist of two files:

1. **Clean file** (`clean.txt`) - Contains correctly spelled sentences
2. **Corrupt file** (`corrupt.txt`) - Contains corresponding sentences with spelling errors

Each line in both files should contain one sentence, and the lines should correspond:

**clean.txt:**
```
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating and powerful.
Natural language processing helps computers understand text.
```

**corrupt.txt:**
```
The quik brown fox jumps over the lasy dog.
Machien lerning is fascinatng and powerfull.
Natral langauge procesing helps computrs understand text.
```

## Usage

### Basic Usage

Run the complete example:

```bash
cd examples
python example_bert_finetuning.py
```

This will:
1. Create sample training data
2. Fine-tune a BERT model
3. Test the model on sample sentences
4. Evaluate the model performance

### Custom Training Data

To use your own training data:

```python
from neuspell import BertChecker

# Initialize and load pre-trained model
checker = BertChecker()
checker.from_pretrained()

# Fine-tune on your data
checker.finetune(
    clean_file="your_clean_sentences.txt",
    corrupt_file="your_corrupt_sentences.txt",
    data_dir="path/to/your/data/",
    validation_split=0.2,
    n_epochs=3
)

# Test the model
corrected = checker.correct("Your sentece with erors.")
print(corrected)
```

### Advanced Configuration

For more control over the fine-tuning process:

```python
from neuspell import BertChecker
from neuspell.seq_modeling.helpers import get_tokens, load_data, train_validation_split

# Load and prepare data
train_data = load_data("data_dir", "clean.txt", "corrupt.txt")
train_data, valid_data = train_validation_split(train_data, 0.8, seed=42)

# Create vocabulary
vocab = get_tokens(
    [item[0] for item in train_data],  # Clean sentences
    keep_simple=True,
    min_max_freq=(1, float("inf")),
    topk=10000
)

# Initialize with custom BERT model
checker = BertChecker(device="cuda")
checker.from_huggingface(
    bert_pretrained_name_or_path="distilbert-base-cased",
    vocab=vocab
)

# Fine-tune
checker.finetune(
    clean_file="clean.txt",
    corrupt_file="corrupt.txt",
    data_dir="data_dir",
    validation_split=0.2,
    n_epochs=5
)
```

## Supported BERT Models

The script supports various BERT model variants:

- `bert-base-cased`
- `bert-base-uncased`
- `distilbert-base-cased`
- `distilbert-base-uncased`
- `xlm-roberta-base`
- `bert-base-multilingual-cased`

## Parameters

### Fine-tuning Parameters

- `clean_file`: Path to file with correct sentences
- `corrupt_file`: Path to file with corrupted sentences
- `data_dir`: Directory containing the data files
- `validation_split`: Proportion of data to use for validation (default: 0.2)
- `n_epochs`: Number of training epochs (default: 2)

### Model Parameters

- `device`: Device to run on ("cuda", "cpu", or "auto")
- `bert_pretrained_name_or_path`: HuggingFace model name or path
- `vocab`: Custom vocabulary dictionary

## Training Process

The fine-tuning process includes:

1. **Data Loading**: Load and split training data
2. **Tokenization**: Convert text to BERT tokens
3. **Model Initialization**: Load pre-trained BERT model
4. **Training Loop**: 
   - Forward pass through BERT
   - Calculate loss
   - Backward pass and optimization
   - Validation evaluation
5. **Model Saving**: Save best model checkpoint

## Evaluation Metrics

The evaluation provides:

- **Accuracy**: Percentage of correctly predicted tokens
- **Word Correction Rate**: Percentage of misspelled words corrected
- **Confusion Matrix**: Breakdown of correction vs. non-correction

## Tips for Better Results

1. **Data Quality**: Ensure your training data has diverse spelling errors
2. **Data Size**: Use at least 1000+ training examples for good results
3. **Epochs**: Start with 2-3 epochs, increase if needed
4. **Validation**: Always use a validation set to monitor overfitting
5. **GPU**: Use GPU for faster training if available

## Example Output

```
=== Basic BERT Fine-tuning ===
Initializing BERT checker...
Starting fine-tuning...
len of train and test data:  400 100
Training model params
In epoch: 1
train_data size: 400
...
Fine-tuning completed in 245.67 seconds

=== Testing Corrections ===
1. Original:  Ths is a sampel sentece with erors.
   Corrected: This is a sample sentence with errors.

2. Original:  Machien lerning is fascinatng and powerfull.
   Corrected: Machine learning is fascinating and powerful.
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size or use CPU
```python
checker = BertChecker(device="cpu")
```

**Import Errors**: Make sure neuspell is installed correctly
```bash
pip install -e .
```

**Model Download Issues**: Ensure internet connection for downloading pre-trained models

## File Structure

After running the script, you'll have:

```
examples/
├── example_bert_finetuning.py
├── README_BERT_FINETUNING.md
├── training_data/
│   ├── clean.txt
│   ├── corrupt.txt
│   └── new_models/
│       └── bert-base-cased/
│           ├── pytorch_model.bin
│           └── vocab.pkl
└── test_data/
    ├── clean.txt
    └── corrupt.txt
```

## Next Steps

1. **Collect More Data**: Gather domain-specific spelling error data
2. **Experiment with Models**: Try different BERT variants
3. **Hyperparameter Tuning**: Adjust learning rate, batch size, etc.
4. **Evaluation**: Test on real-world data
5. **Deployment**: Integrate the fine-tuned model into your application

For more information, see the [NeuSpell documentation](https://github.com/neuspell/neuspell). 