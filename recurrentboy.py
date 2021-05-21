import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os
import time
text = open("/Users/ryan/Desktop/speeches/adams/adams_speeches_000.txt",'rb').read().decode(encoding='utf-8')

#
# print(text[:1000])
# print(f"{len(vocab)} unique characters.")
#
# Preprocessing

vocab = sorted(set(text))
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab),mask_token=None)
char_ids = ids_from_chars(tf.strings.unicode_split(text,'UTF-8'))

# print(char_ids)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# chars = chars_from_ids(char_ids)
# print(chars)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text,'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 200
examples_per_epoch = len(text)//(seq_length+1)
print(examples_per_epoch)
sequences = ids_dataset.batch(seq_length+1,drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
# batch and buffer
batch_size = 64
buffer_size = 10000
#
dataset = (dataset.shuffle(buffer_size)
.batch(batch_size,drop_remainder=True)
.prefetch(tf.data.experimental.AUTOTUNE))


# WHY WOULD ANYONE DO THIS TO THEMSELVES. THIS IS A CTRL C + CTRL V JOB.
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
# This is TensorFlow's RNN model subclass from their tutorial
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(vocab_size = len(
    ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units = rnn_units)
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
model.summary()
# Testing the first example in the batch.
def model_test():
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
  sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
  print(sampled_indices)
  print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
  print()
  print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
model_test()
# Training stuff
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)
print(tf.exp(mean_loss).numpy())
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = '/Users/ryan/Desktop/models/training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

Epochs = 400

print(dataset)

history = model.fit(dataset,epochs=Epochs,callbacks=[checkpoint_callback])
model_test()

#One time step generation. This is also a ctrl c+v job, essentially this model just loops over and over for a set number of times.
#This model is from tensorflow
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
one_step_m = OneStep(model, chars_from_ids, ids_from_chars)
# Here, we're telling it to start generating based on an initial string.
# We use time to measure how long it runs, and use a for loop to make it generate x characters.
# States is defined to make sure the model starts with no GRU/embedding state.
start = time.time()
states = None
next_char = tf.constant(['My fellow Americans,'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_m.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

tf.saved_model.save(model, 'recurrentboi')