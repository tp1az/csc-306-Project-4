import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments



class TabularQADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: T5Tokenizer,
                 max_input_length: int = 512, max_output_length: int = 128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        table_text = row["table_data"]
        question = row["question"]
        answer = row["answer"]

        # Create a prompt that incorporates the table data and the question.
        # You can adjust the format to match your specific task.
        input_text = f"Data: {table_text} Question: {question}"
        target_text = answer

        # Tokenize the input and target texts.
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_output_length,
            return_tensors="pt"
        )

        input_encodings = {key: tensor.squeeze() for key, tensor in input_encodings.items()}
        target_encodings = {key: tensor.squeeze() for key, tensor in target_encodings.items()}

        labels = target_encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels,
        }

# Choose a pre-trained model (here, t5-small is used as an example)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Create the dataset.
df = pd.read_parquet("../data/066_IBM_HR/sample.parquet")
train_dataset = TabularQADataset(df, tokenizer)

# Set up training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=10,
    evaluation_strategy="no",  # Change to 'epoch' if you have an evaluation dataset.
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model.
trainer.train()

# Save the fine-tuned model and tokenizer.
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")