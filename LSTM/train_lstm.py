import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import CSVLogger
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter
from math import exp, log
from tensorflow.keras.utils import to_categorical

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

# Load the dataset
file_path = 'Selected Data_FINAL.csv'
data = pd.read_csv(file_path)

# Preprocess gloss_tokens to convert string representations of lists to actual lists
import ast
data['gloss_tokens'] = data['gloss_tokens'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Remove rows with empty gloss_tokens
data = data[data['gloss_tokens'].map(len) > 0]

# Display the shape after preprocessing
print(f"Shape of dataset after dropping invalid rows: {data.shape}")

# Prepare the input and output sequences
X = data['gloss_tokens'].values
y = data['text'].values

# Initialize the tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the data
tokenizer.fit_on_texts(X)
tokenizer.fit_on_texts(y)

# Convert text to sequences
X_seq = tokenizer.texts_to_sequences(X)
y_seq = tokenizer.texts_to_sequences(y)

# Define maximum sequence length
max_sequence_length = max(max(len(seq) for seq in X_seq), max(len(seq) for seq in y_seq))

# Pad sequences
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length, padding='post')
y_padded = pad_sequences(y_seq, maxlen=max_sequence_length, padding='post')

# Convert y_train and y_test to one-hot encoded format
y_padded = np.expand_dims(y_padded, -1)
y_padded = to_categorical(y_padded, num_classes=len(tokenizer.word_index) + 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Print shapes of the datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dense(units=len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a CSV logger callback to save the training history
csv_logger = CSVLogger('training_history.csv')

# Train the model for 50 epochs
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[csv_logger])

# Generate predictions
y_pred = model.predict(X_test)

# Convert predictions and ground truths back to text
y_pred_texts = []
y_true_texts = []

for seq in y_pred:
    y_pred_texts.append(' '.join([tokenizer.index_word.get(idx, '') for idx in np.argmax(seq, axis=1) if idx != 0]))

for seq in np.argmax(y_test, axis=-1):  # Convert y_test back from one-hot to indices
    y_true_texts.append(' '.join([tokenizer.index_word.get(idx, '') for idx in seq if idx != 0]))

# Function to calculate n-gram precision
def n_gram_precision(reference, candidate, n):
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    overlapping_ngrams = cand_ngrams & ref_ngrams
    return sum(overlapping_ngrams.values()), max(1, len(candidate) - n + 1)

# Calculate BLEU score as per the formula
def calculate_bleu(reference, candidate):
    precisions = []
    for n in range(1, 4):  # Using BLEU-3, calculate precision for n=1,2,3
        correct, total = n_gram_precision(reference, candidate, n)
        precisions.append(correct / total)
    
    log_precisions = [log(p) if p > 0 else float('-inf') for p in precisions]
    avg_log_precision = sum(log_precisions) / len(precisions)
    c = len(candidate)
    r = len(reference)
    
    if c == 0:  # Handle empty candidate
        brevity_penalty = 0
    else:
        brevity_penalty = min(1.0 - (r / c), 0.0)
    
    bleu_score = exp(brevity_penalty + avg_log_precision)
    return bleu_score

# Calculate BLEU-3 and ROUGE scores
bleu_scores = []
rouge_scores = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for pred, true in zip(y_pred_texts, y_true_texts):
    reference = true.split()
    candidate = pred.split()
    
    bleu_score = calculate_bleu(reference, candidate)
    bleu_scores.append(bleu_score)
    
    rouge_score = scorer.score(true, pred)
    rouge_scores.append(rouge_score)

# Calculate average BLEU-3 and ROUGE scores
avg_bleu = np.mean(bleu_scores)
avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

# Get training and validation accuracy from history
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Save all results to the CSV file
history_df = pd.read_csv('training_history.csv')
history_df['avg_bleu'] = avg_bleu
history_df['avg_rouge1'] = avg_rouge1
history_df['avg_rouge2'] = avg_rouge2
history_df['avg_rougeL'] = avg_rougeL
history_df['train_accuracy'] = train_acc
history_df['validation_accuracy'] = val_acc
history_df['test_accuracy'] = test_acc

history_df.to_csv('training_history_with_scores_New.csv', index=False)
print(history_df)

# Print training, validation, and test accuracy
print(f"Training Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")
print(f"Test Accuracy: {test_acc}")


