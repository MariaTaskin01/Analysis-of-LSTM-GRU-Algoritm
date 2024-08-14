import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
import csv

# Load and preprocess the dataset
df = pd.read_csv('Selected Data_FINAL.csv')
print(df.head())

# Drop rows with missing gloss_tokens
df = df.dropna(subset=['gloss_tokens'])
print(f'Shape of dataset after dropping invalid rows: {df.shape}')

# Tokenize the gloss_tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['gloss_tokens'])

# Convert gloss_tokens to sequences
sequences = tokenizer.texts_to_sequences(df['gloss_tokens'])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print(f'Vocabulary Size: {vocab_size}')

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Prepare features (X) and labels (y)
X = np.array(data)
y = np.array(data)  # using sequences as labels

# Reshape y to be (num_samples, sequence_length, 1)
y = y.reshape((y.shape[0], y.shape[1], 1))

# Convert y_train and y_test to one-hot encoded format
y = to_categorical(y, num_classes=vocab_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

# Define the GRU model
embedding_dim = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(GRU(128, return_sequences=True))
model.add(GRU(128, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Extract Training Accuracy, Validation Accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Generate predictions
y_pred = model.predict(X_test)

# Convert predictions and ground truth to sequences
y_test_sequences = np.argmax(y_test, axis=-1)
y_pred_sequences = np.argmax(y_pred, axis=-1)

# Convert sequences to texts
y_test_texts = tokenizer.sequences_to_texts(y_test_sequences)
y_pred_texts = tokenizer.sequences_to_texts(y_pred_sequences)

# Debug prints
print("Sample reference texts:")
print(y_test_texts[:5])
print("Sample predicted texts:")
print(y_pred_texts[:5])

# Calculate ROUGE and manual BLEU-4 scores
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_scores = []
rouge_scores = []

smoothie = SmoothingFunction().method4

for ref, hyp in zip(y_test_texts, y_pred_texts):
    if not ref.strip():  # Skip empty references
        continue

    # Calculate manual BLEU-4 score
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    
    # Calculate precision for each n-gram
    precisions = []
    for n in range(1, 5):
        ref_ngrams = list(zip(*[ref_tokens[i:] for i in range(n)]))
        hyp_ngrams = list(zip(*[hyp_tokens[i:] for i in range(n)]))
        ref_count = len(ref_ngrams)
        hyp_count = len(hyp_ngrams)
        match_count = sum(1 for gram in hyp_ngrams if gram in ref_ngrams)
        precision = match_count / hyp_count if hyp_count > 0 else 0
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if all(p == 0 for p in precisions):
        geo_mean = 0
    else:
        geo_mean = np.exp(np.mean(np.log([p if p > 0 else 1e-10 for p in precisions])))
    
    # Calculate brevity penalty
    r = len(ref_tokens)
    c = len(hyp_tokens)
    brevity_penalty = np.exp(1 - r / c) if c < r else 1
    
    # Calculate BLEU-4 score
    bleu_score = brevity_penalty * geo_mean
    bleu_scores.append(bleu_score)
    
    # Calculate ROUGE scores
    scores = rouge.score(ref, hyp)
    rouge_scores.append(scores)

# Calculate average ROUGE and BLEU-4 scores
avg_bleu_score = np.mean(bleu_scores)
avg_rouge_score = {key: np.mean([score[key].fmeasure for score in rouge_scores]) for key in rouge_scores[0]}

# Append ROUGE and BLEU-4 scores to the CSV file
with open('gru_training_history_New.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['BLEU-4', avg_bleu_score])
    for key, value in avg_rouge_score.items():
        writer.writerow([key, value])

# Add Training Accuracy, Validation Accuracy, and Test Accuracy to the CSV file
history_df = pd.DataFrame(history.history)
history_df['Training Accuracy'] = history_df['accuracy']
history_df['Validation Accuracy'] = history_df['val_accuracy']
history_df['Test Accuracy'] = test_acc

history_df.to_csv('gru_training_history_with_accuracies.csv', index=False)

print(f'Average BLEU-4 Score: {avg_bleu_score}')
for key, value in avg_rouge_score.items():
    print(f'Average {key} Score: {value}')
print(f'Test Accuracy: {test_acc}')
