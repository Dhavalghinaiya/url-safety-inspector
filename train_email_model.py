import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import transformers

# --- DIAGNOSTIC STEP ---
# This line will print the exact version of the library you are using.
print(f"Using transformers version: {transformers.__version__}")


# --- 1. DATA PREPARATION ---
print("Step 1: Preparing data...")
# Load your dataset
try:
    df = pd.read_csv("se_phishing_test_set.csv")
except FileNotFoundError:
    print("Error: 'se_phishing_test_set.csv' not found.")
    print("Please download the dataset and place it in the same directory.")
    exit()

# Clean up column names to remove hidden characters/spaces
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)

# Map labels to integers (0 for Benign, 1 for Malicious)
df['label'] = df['label'].map({'Benign': 0, 'Malicious': 1})
# Rename columns to what the model expects
df.rename(columns={'email_text': 'text'}, inplace=True)

# Drop rows where 'text' or 'label' might be empty/NaN
df.dropna(subset=['text', 'label'], inplace=True)


# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].astype(str), df['label'], test_size=0.2, random_state=42
)
print("Data prepared successfully.")

# --- 2. MODEL AND TOKENIZER LOADING ---
print("\nStep 2: Loading pre-trained BERT model...")
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("Model and tokenizer loaded.")

# --- 3. TOKENIZATION ---
print("\nStep 3: Tokenizing datasets...")
# Convert pandas Series to lists
train_texts_list = train_texts.tolist()
val_texts_list = val_texts.tolist()

# Tokenize the texts
train_encodings = tokenizer(train_texts_list, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts_list, truncation=True, padding=True, max_length=512)
print("Tokenization complete.")

# --- 4. CREATE PYTORCH DATASET ---
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert labels to lists before creating the dataset
train_dataset = PhishingDataset(train_encodings, train_labels.tolist())
val_dataset = PhishingDataset(val_encodings, val_labels.tolist())
print("PyTorch datasets created.")

# --- 5. MODEL TRAINING ---
print("\nStep 5: Starting model training...")

# CORRECTED: Removed the problematic 'evaluation_strategy' argument
# to ensure compatibility with your environment.
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for checkpoints
    num_train_epochs=1,              # A single epoch is often enough for fine-tuning
    per_device_train_batch_size=8,   # Batch size for training
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
)

# We also remove the 'eval_dataset' since we are no longer evaluating during training.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
print("Training complete.")

# --- 6. SAVE THE MODEL ---
print("\nStep 6: Saving the fine-tuned model...")
output_model_dir = './email_classifier_model'
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model saved to '{output_model_dir}'. You can now run the Flask app.")

