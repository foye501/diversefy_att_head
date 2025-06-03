from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from datasets import load_dataset
import torch.nn as nn
import torch
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2Block, GPT2Attention

class ReducedHeadAttention(GPT2Attention):
    def __init__(self, config, n_head_custom):
        super().__init__(config)
        # Override number of heads and embed dim
        self.num_heads = n_head_custom
        self.split_size = self.num_heads * (config.n_embd // config.n_head)

        # Adjust projections
        self.c_attn = nn.Linear(self.split_size, 3 * self.split_size)
        self.c_proj = nn.Linear(self.split_size, self.split_size)

class CustomBlock(GPT2Block):
    def __init__(self, config, n_head_custom):
        super().__init__(config)
        self.reduced_attn_in_proj = nn.Linear(config.n_embd, n_head_custom * (config.n_embd // config.n_head))
        self.attn = ReducedHeadAttention(config, n_head_custom)
        self.reduced_attn_out_proj = nn.Linear(n_head_custom * (config.n_embd // config.n_head), config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, **kwargs):
        residual = hidden_states

        # Pre-attention projection
        reduced_input = self.reduced_attn_in_proj(hidden_states)
        attn_outputs = self.attn(reduced_input, layer_past=layer_past, attention_mask=attention_mask)
        attn_output = attn_outputs[0]
        attn_output = self.reduced_attn_out_proj(attn_output)

        hidden_states = residual + attn_output
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + feed_forward_hidden_states
        return (hidden_states,) + attn_outputs[1:]

class CustomGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

        layer_custom_heads = {
            1: 3,
            7: 3,
            8: 5,
            9: 3,
            10: 5,
            11: 3
        }

        for i, n_head in layer_custom_heads.items():
            self.transformer.h[i] = CustomBlock(config, n_head_custom=n_head)


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

config = GPT2Config()
model = CustomGPT2(config)

print(model)  # Print the structure to verify changes

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
