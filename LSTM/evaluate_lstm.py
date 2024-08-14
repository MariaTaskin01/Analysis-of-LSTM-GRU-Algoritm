import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import ast

# Load the training history CSV file
history_df = pd.read_csv('training_history.csv')

# Print the final training and validation accuracies
final_training_accuracy = history_df['accuracy'].iloc[-1]
final_validation_accuracy = history_df['val_accuracy'].iloc[-1]

print(f'Final Training Accuracy: {final_training_accuracy}')
print(f'Final Validation Accuracy: {final_validation_accuracy}')

# If you also want to print the accuracy for each epoch, you can do so as follows
print("\nTraining and Validation Accuracies for each epoch:")
print(history_df[['epoch', 'accuracy', 'val_accuracy']])

# Load the dataset
file_path = 'Selected Data_FINAL.csv'
data = pd.read_csv(file_path)

# Preprocess gloss_tokens to convert string representations of lists to actual lists
data['gloss_tokens'] = data['gloss_tokens'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Remove rows with empty gloss_tokens
data = data[data['gloss_tokens'].map(len) > 0]

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dense(units=len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the model weights if you have saved them previously
# model.load_weights('model_weights.h5')

# Evaluate the model on the test data to get the test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy}')

# Save all results to a new CSV file
output_df = history_df.copy()
output_df['final_training_accuracy'] = final_training_accuracy
output_df['final_validation_accuracy'] = final_validation_accuracy
output_df['test_accuracy'] = test_accuracy

output_df.to_csv('final_training_output_with_accuracies.csv', index=False)
print("Final output saved to 'final_training_output_with_accuracies.csv'")
