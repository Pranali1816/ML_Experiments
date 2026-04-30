import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1. Sample dataset
# -----------------------------
documents = [
    "machine learning is a field of artificial intelligence",
    "deep learning uses neural networks for complex tasks",
    "natural language processing deals with text data"
]

summaries = [
    "machine learning",
    "deep learning",
    "nlp text"
]

# -----------------------------
# 2. Tokenization
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents + summaries)

vocab_size = len(tokenizer.word_index) + 1

# Convert to sequences
doc_seq = tokenizer.texts_to_sequences(documents)
sum_seq = tokenizer.texts_to_sequences(summaries)

# Padding
max_doc_len = max(len(seq) for seq in doc_seq)
max_sum_len = max(len(seq) for seq in sum_seq)

doc_seq = pad_sequences(doc_seq, maxlen=max_doc_len, padding='post')
sum_seq = pad_sequences(sum_seq, maxlen=max_sum_len, padding='post')

# -----------------------------
# 3. Prepare decoder input/output
# -----------------------------
decoder_input = sum_seq[:, :-1]
decoder_output = sum_seq[:, 1:]

# -----------------------------
# 4. Build Encoder-Decoder Model
# -----------------------------
embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_doc_len,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_sum_len-1,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# -----------------------------
# 5. Compile & Train
# -----------------------------
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(
    [doc_seq, decoder_input],
    np.expand_dims(decoder_output, -1),
    epochs=200,
    verbose=1
)

# -----------------------------
# 6. Simple Prediction Function
# -----------------------------
reverse_word_index = {v:k for k,v in tokenizer.word_index.items()}

def decode_sequence(input_seq):
    states = encoder_lstm(enc_emb).output
    target_seq = np.zeros((1,1))
    
    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_lstm(dec_emb, initial_state=states)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        if sampled_token_index == 0:
            break
        
        word = reverse_word_index.get(sampled_token_index, '')
        decoded_sentence += ' ' + word
        
        if len(decoded_sentence.split()) > max_sum_len:
            stop_condition = True

    return decoded_sentence

# -----------------------------
# 7. Test summarization
# -----------------------------
for i in range(len(documents)):
    print("\nDocument:", documents[i])
    print("Summary (expected):", summaries[i])