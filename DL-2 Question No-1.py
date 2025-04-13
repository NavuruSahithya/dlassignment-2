#Question No-1
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input, LSTM, GRU, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------
# Custom Config
# ------------------------
embed_sz = 32
units = 64
depth = 1
cell_kind = 'LSTM'  # 'GRU' or 'RNN'

# ------------------------
# Sample Character Data
# ------------------------
eng = ['dil', 'pyar', 'namaste']
hin = ['दिल', 'प्यार', 'नमस्ते']

def tokenize(char_list):
    chars = sorted(set(''.join(char_list)))
    c2i = {c: i+1 for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c

src_vocab, rev_src = tokenize(eng)
tgt_vocab, rev_tgt = tokenize(hin)

src_vocab_len = len(src_vocab) + 1
tgt_vocab_len = len(tgt_vocab) + 1

# ------------------------
# Encode and pad
# ------------------------
def encode_sequence(word_list, mapper):
    return [[mapper[ch] for ch in word] for word in word_list]

src_encoded = encode_sequence(eng, src_vocab)
tgt_encoded = encode_sequence(hin, tgt_vocab)

sos_token = tgt_vocab_len
eos_token = tgt_vocab_len + 1

dec_input = pad_sequences([[sos_token] + seq for seq in tgt_encoded], padding='post')
dec_output = pad_sequences([seq + [eos_token] for seq in tgt_encoded], padding='post')
dec_output = np.expand_dims(dec_output, -1)
src_input = pad_sequences(src_encoded, padding='post')

# ------------------------
# RNN Selector
# ------------------------
def make_rnn(units, name, return_sequences=False, return_state=True):
    if cell_kind == 'GRU':
        return GRU(units, name=name, return_sequences=return_sequences, return_state=return_state)
    elif cell_kind == 'RNN':
        return SimpleRNN(units, name=name, return_sequences=return_sequences, return_state=return_state)
    else:
        return LSTM(units, name=name, return_sequences=return_sequences, return_state=return_state)

# ------------------------
# Model Assembly
# ------------------------
enc_input_layer = Input(shape=(None,), name="src_input")
enc_embed = Embedding(input_dim=src_vocab_len, output_dim=embed_sz, mask_zero=True, name="src_embed")(enc_input_layer)

# Encoder Stack
enc_out = enc_embed
states = []
for layer_num in range(depth):
    rnn = make_rnn(units, name=f"enc_rnn_{layer_num}")
    if cell_kind == 'LSTM':
        enc_out, state_h, state_c = rnn(enc_out)
        states = [state_h, state_c]
    else:
        enc_out, state_h = rnn(enc_out)
        states = [state_h]

dec_input_layer = Input(shape=(None,), name="tgt_input")
dec_embed = Embedding(input_dim=tgt_vocab_len + 2, output_dim=embed_sz, mask_zero=True, name="tgt_embed")(dec_input_layer)

# Decoder Stack
dec_out = dec_embed
for layer_num in range(depth):
    rnn = make_rnn(units, name=f"dec_rnn_{layer_num}", return_sequences=True)
    if cell_kind == 'LSTM':
        dec_out, _, _ = rnn(dec_out, initial_state=states)
    else:
        dec_out, _ = rnn(dec_out, initial_state=states)

final_dense = Dense(tgt_vocab_len + 2, activation='softmax', name="out_layer")(dec_out)

model = Model(inputs=[enc_input_layer, dec_input_layer], outputs=final_dense)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------
# Train
# ------------------------
model.fit([src_input, dec_input], dec_output, batch_size=2, epochs=50)