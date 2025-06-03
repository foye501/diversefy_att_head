from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from datasets import load_dataset
import torch.nn as nn
import torch
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2Block, GPT2Attention

class ReducedHeadAttention(GPT2Attention):
    def __init__(self, config, reduced_n_head):
        # Keep head_dim the same
        self.reduced_n_head = reduced_n_head
        self.original_n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.embed_dim = self.head_dim * reduced_n_head

        # Create a new config copy to fool the base class
        reduced_config = GPT2Config.from_pretrained('gpt2')
        reduced_config.n_head = reduced_n_head
        reduced_config.n_embd = self.embed_dim

        super().__init__(reduced_config)

    def forward(self, hidden_states, **kwargs):
        return super().forward(hidden_states, **kwargs)

class CustomFinalBlock(GPT2Block):
    def __init__(self, config, reduced_n_head=3):
        super().__init__(config)
        self.reduced_attn_in_proj = nn.Linear(config.n_embd, reduced_n_head * (config.n_embd // config.n_head))
        self.attn = ReducedHeadAttention(config, reduced_n_head)
        self.reduced_attn_out_proj = nn.Linear(reduced_n_head * (config.n_embd // config.n_head), config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, **kwargs):
        residual = hidden_states

        # Project input down
        reduced_input = self.reduced_attn_in_proj(hidden_states)

        # Run attention
        attn_outputs = self.attn(reduced_input, layer_past=layer_past, attention_mask=attention_mask)
        attn_output = attn_outputs[0]  # shape: [batch, seq_len, reduced_dim]

        # Project back up to original dim
        attn_output = self.reduced_attn_out_proj(attn_output)

        hidden_states = residual + attn_output  # residual connection

        # --- MLP block ---
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class ReducedHeadGPT2(GPT2LMHeadModel):
    def __init__(self, config, reduced_n_head=3):
        super().__init__(config)
        self.transformer.h[-1] = CustomFinalBlock(config, reduced_n_head=reduced_n_head)


# Step 3: Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Load and tokenize dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")["train"]

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True).remove_columns("text").select(range(1000))

# Step 5: Load config and model
config = GPT2Config(n_layer=12, n_head=12, n_embd=768)
model = ReducedHeadGPT2(config, reduced_n_head=3)


# Step 6: Set training args
training_args = TrainingArguments(
    output_dir="./reduced_head_gpt",
    per_device_train_batch_size=32,
    num_train_epochs=4,
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