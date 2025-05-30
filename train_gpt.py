from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,GPT2Config

# Load model and tokenizer
config = GPT2Config(
    n_embd=256, n_layer=4, n_head=4, # for smaller/faster training
)
model = GPT2LMHeadModel(config)
# model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load a small dataset (e.g., wikitext, or your own)
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

# Tokenize
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True).remove_columns("text")

# Training args
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    max_grad_norm=1.0,
    no_cuda=False,
    num_train_epochs=2,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="no",
)


# --- Diversity regularization and baseline training ---
import copy
import torch
import torch.nn.functional as F

lambda_coeff = 0.01  # adjust for strength of diversity loss

def cosine_divergence_loss(q_heads):
    n = q_heads.shape[0]
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = q_heads[i].flatten()
            vj = q_heads[j].flatten()
            loss += F.cosine_similarity(vi, vj, dim=0)
    return loss / (n * (n - 1) / 2)

class DiverseGPT2(GPT2LMHeadModel):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Enable attention outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            **kwargs
        )

        loss = outputs.loss
        logits = outputs.logits
        all_attentions = outputs.attentions  # list of tensors, one per layer

        # Get last layer attention: shape (batch, num_heads, seq_len, seq_len)
        last_attn = all_attentions[-1]

        # Average over batch and query sequence dim
        # Result: shape (num_heads, seq_len)
        head_outputs = last_attn.mean(dim=0).mean(dim=1)  # [num_heads, seq_len]

        # Optional: normalize or flatten if needed
        n = head_outputs.shape[0]
        div_loss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                div_loss += F.cosine_similarity(head_outputs[i], head_outputs[j], dim=0)

        div_loss = div_loss / (n * (n - 1) / 2)

        total_loss = loss + lambda_coeff * div_loss
        return {"loss": total_loss, "logits": logits}


# Train baseline model
model = model.to("cuda:0")

trainer_baseline = Trainer(
    model=model,

    args=args,
    train_dataset=tokenized,
)

print("Training baseline model...")
trainer_baseline.train()

# Train diversity-regularized model
diverse_model = DiverseGPT2.from_pretrained("gpt2")
trainer_diverse = Trainer(
    model=diverse_model,
    args=args,
    train_dataset=tokenized,
)

print("Training diversity-regularized model...")
trainer_diverse.train()