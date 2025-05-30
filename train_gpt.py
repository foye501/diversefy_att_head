from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load a small dataset (e.g., wikitext, or your own)
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
tokenized = dataset.map(tokenize, batched=True).remove_columns("text")

# Training args
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
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
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        loss = outputs.loss
        logits = outputs.logits

        # Extract Q projection weights from final layer
        q_weight = self.transformer.h[-1].attn.c_attn.weight[:self.config.n_embd]
        q_heads = q_weight.view(self.config.n_head, -1, self.config.n_embd)
        div_loss = cosine_divergence_loss(q_heads)

        total_loss = loss + lambda_coeff * div_loss
        return {"loss": total_loss, "logits": logits}

# Train baseline model
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