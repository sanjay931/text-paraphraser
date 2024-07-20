import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from sklearn.model_selection import train_test_split


class ParaphraseDataset(Dataset):
    """Dataset class for paraphrase generation."""

    def __init__(self, inputs, outputs, tokenizer, max_length=512):
        """
        Initialize the dataset with inputs, outputs, tokenizer, and max_length.

        Args:
            inputs (list): List of input texts.
            outputs (list): List of output paraphrases.
            tokenizer (T5Tokenizer): Tokenizer for encoding the texts.
            max_length (int): Maximum token length for encoding.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Encoded input and output texts.
        """
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        output_encodings = self.tokenizer(
            output_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = output_encodings.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encodings.input_ids.squeeze(),
            'attention_mask': input_encodings.attention_mask.squeeze(),
            'labels': labels
        }


class ParaphraseModel:
    """Model class for paraphrase generation using T5."""

    def __init__(self, model_name='t5-base', max_length=256, batch_size=8, lr=3e-5, epochs=1):
        """
        Initialize the model with specified parameters.

        Args:
            model_name (str): Pretrained model name.
            max_length (int): Maximum token length for encoding.
            batch_size (int): Batch size for training and validation.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

    def train(self, inputs, outputs, val_inputs, val_outputs):
        """
        Train the model.

        Args:
            inputs (list): List of training input texts.
            outputs (list): List of training output paraphrases.
            val_inputs (list): List of validation input texts.
            val_outputs (list): List of validation output paraphrases.
        """
        train_dataset = ParaphraseDataset(inputs, outputs, self.tokenizer, self.max_length)
        val_dataset = ParaphraseDataset(val_inputs, val_outputs, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    def generate(self, text, num_return_sequences=5):
        """
        Generate paraphrases for a given text.

        Args:
            text (str): Input text.
            num_return_sequences (int): Number of paraphrases to generate.

        Returns:
            list: List of generated paraphrases.
        """
        input_encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = input_encodings.input_ids.to(self.model.device)
        attention_mask = input_encodings.attention_mask.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            early_stopping=True
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def save_model(self, save_directory):
        """
        Save the model and tokenizer to the specified directory.

        Args:
            save_directory (str): Directory to save the model and tokenizer.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    def load_model(self, load_directory):
        """
        Load the model and tokenizer from the specified directory.

        Args:
            load_directory (str): Directory to load the model and tokenizer from.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        self.tokenizer = T5Tokenizer.from_pretrained(load_directory)
        print(f"Model loaded from {load_directory}")


def load_data(csv_path):
    """
    Load data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: Two lists containing inputs and outputs.
    """
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.dropna(inplace=True)
        df['paraphrases'] = df['paraphrases'].apply(lambda x: eval(x))
        inputs = df['text'].tolist()
        outputs = df['paraphrases'].tolist()
        flat_outputs = [item for sublist in outputs for item in sublist]
        repeated_inputs = [inputs[i // 5] for i in range(len(flat_outputs))]
        return repeated_inputs, flat_outputs
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

"""Start of training"""

csv_path = './data/chatgpt_paraphrases.csv'
inputs, outputs = load_data(csv_path)
print(inputs[:5])
print(outputs[:5])
train_inputs = inputs[:1000]
train_outputs = outputs[:1000]
val_inputs = inputs[:50]
val_outputs = outputs[:50]

model = ParaphraseModel()
model.train(train_inputs, train_outputs, val_inputs, val_outputs)
model.save_model('./trained_model')
