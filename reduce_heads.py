from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from datasets import load_dataset
import torch.nn as nn
import torch

# Step 1: Custom block with reduced heads in the last layer
class CustomFinalBlock(GPT2Block):
    def __init__(self, config, reduced_n_head=3):
        super().__init__(config)

        head_dim = config.n_embd // config.n_head
        reduced_emb_dim = reduced_n_head * head_dim

        # Projection to smaller input for this block
        self.input_proj = nn.Linear(config.n_embd, reduced_emb_dim)

        # Replace attention module with fewer heads
        self.attn = GPT2Attention(config)
        self.attn.num_heads = reduced_n_head
        self.attn.split_size = reduced_emb_dim

        # Project back to original emb dim after attention
        self.output_proj = nn.Linear(reduced_emb_dim, config.n_embd)

    def forward(self, hidden_states, **kwargs):
        # Project down
        reduced_in = self.input_proj(hidden_states)

        # Replace the hidden states for attn forward
        attn_out = self.attn(reduced_in)[0]

        # Project back to original
        projected = self.output_proj(attn_out)
        return projected + hidden_states  # residual connection

# Step 2: Define model
class ReducedHeadGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.h[-1] = CustomFinalBlock(config)

# Step 3: Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Load and tokenize dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True).remove_columns("text").select(range(1000))

# Step 5: Load config and model
config = GPT2Config()
model = ReducedHeadGPT2(config)

# Step 6: Set training args
training_args = TrainingArguments(
    output_dir="./reduced_head_gpt",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    report_to="none"
)

# Step 7: Train
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()


base_model = GPT2LMHeadModel(config)

trainer = Trainer(model=base_model, args=training_args, train_dataset=tokenized)
trainer.train()