
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils .data import DataLoader
import pandas as pd
import random
import os


# Load dataset
data = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(data['train'])
df.dropna(subset=['quote', 'author'], inplace=True)
df['text'] = df['quote'] + " - " + df['author'] + " (" + df['tags'].apply(lambda x: ', '.join(x)) + ")"

# Prepare triplet samples
samples = []
quotes = df['text'].tolist()
for i in range(len(df)):
    anchor = df.iloc[i]['text']
    positive = df.iloc[(i + 1) % len(df)]['text']
    negative = random.choice(quotes)
    samples.append(InputExample(texts=[anchor, positive, negative]))

# Build model
word_embedding_model = models.Transformer("sentence-transformer/all-MiniLM-L6-v2")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Train model
train_dataloader = DataLoader(samples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save model
os.makedirs("models/quote_model", exist_ok=True)
model.save("models/quote_model")
