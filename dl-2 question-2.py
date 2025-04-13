#question-2
# =======================================
# STEP 1: Install Required Libraries
# =======================================
!pip install transformers datasets --quiet

# =======================================
# STEP 2: Import Libraries
# =======================================
import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)

# Disable external logging (like wandb)
os.environ["WANDB_DISABLED"] = "true"

# =======================================
# STEP 3: Load Tokenizer and Base GPT-2
# =======================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # set pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# =======================================
# STEP 4: Create Sample Training Data
# =======================================
lyrics_file = "lyrics.txt"
if not os.path.exists(lyrics_file):
    sample_lyrics = [
        "You're the one that I want\n",
        "Hello from the other side\n",
        "Cause baby you're a firework\n",
        "Let it go, let it go\n",
        "We will, we will rock you\n"
    ]
    with open(lyrics_file, "w", encoding="utf-8") as f:
        f.writelines(sample_lyrics)

# =======================================
# STEP 5: Load & Tokenize Lyrics Data
# =======================================
dataset = load_dataset("text", data_files={"train": lyrics_file})

def tokenize_text(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_data = dataset.map(tokenize_text, batched=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =======================================
# STEP 6: Set Up Training Arguments
# =======================================
training_args = TrainingArguments(
    output_dir="./gpt2-lyrics-output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_steps=5,
    save_total_limit=1,
    prediction_loss_only=True
)

# =======================================
# STEP 7: Train the GPT-2 Model
# =======================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    data_collator=collator
)

print("🎶 Training GPT-2 on sample lyrics...")
trainer.train()
print("✅ Training complete.")

# Save the fine-tuned model and tokenizer for later reuse
model.save_pretrained("gpt2-lyrics-model")
tokenizer.save_pretrained("gpt2-lyrics-model")

# =======================================
# STEP 8: Define a Function to Generate Lyrics
# =======================================
def generate_lyrics(prompt, max_new_tokens=60):
    # Encode the user-provided prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate new tokens based on the prompt
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# =======================================
# STEP 9: Get User Song Prompt and Generate Lyrics
# =======================================
user_prompt = input("🎤 Enter your song prompt: ")
lyrics = generate_lyrics(user_prompt)
print("\n🎵 Generated Lyrics:")
print(lyrics)
