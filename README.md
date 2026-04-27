## TRANSOFRMERS

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# 🔹 SETUP
name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_attentions=True)

# Choose your mode: "extract" / "heads" / "heatmap" / "layers" / "modify" / "analyze"
mode = "analyze" 

# Sample Inputs
text1 = "Virat Kohli lives in Delhi"
text2 = "Apple is a tech company in California"

def get_attention(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return outputs.attentions, tokens

# --- TASKS ---

# 71. Load & Extract Attention Weights
if mode == "extract":
    attentions, tokens = get_attention(text1)
    # Extracting Layer 0, Head 0 weights
    weights = attentions[0][0][0].detach().numpy()
    print(f"Tokens: {tokens}")
    print(f"Attention Matrix Shape: {weights.shape}")
    print(f"Sample Weights (First Row):\n{weights[0]}")

# 72. Visualize Attention Heads
elif mode == "heads":
    attentions, tokens = get_attention(text1)
    layer = 0
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for i in range(4): # Visualizing first 4 heads
        attn = attentions[layer][0][i].detach().numpy()
        axs[i].imshow(attn, cmap='viridis')
        axs[i].set_title(f"Head {i}")
        axs[i].axis('off')
    plt.show()

# 73. Plot Attention Heatmaps
elif mode == "heatmap":
    attentions, tokens = get_attention(text1)
    attn = attentions[0][0][0].detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='hot')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title("Attention Heatmap (Layer 0, Head 0)")
    plt.colorbar()
    plt.show()

# 74. Compare Attention Across Layers
elif mode == "layers":
    attentions, tokens = get_attention(text1)
    layers_to_show = [0, 5, 11] # Early, Middle, Final
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, l in enumerate(layers_to_show):
        attn = attentions[l][0][0].detach().numpy()
        axs[i].imshow(attn)
        axs[i].set_title(f"Layer {l}")
        axs[i].set_xticks(range(len(tokens)))
        axs[i].set_xticklabels(tokens, rotation=90)
    plt.show()

# 75. Modify Input & Observe Changes
elif mode == "modify":
    for t in [text1, text2]:
        attentions, tokens = get_attention(t)
        attn = attentions[0][0][0].detach().numpy()
        plt.figure(figsize=(6, 4))
        plt.imshow(attn)
        plt.title(f"Input: {t[:20]}...")
        plt.show()

# 76. Analyze Multi-Head Attention Outputs
elif mode == "analyze":
    attentions, tokens = get_attention(text1)
    layer = 0
    print(f"Analyzing Layer {layer} for token '{tokens[1]}':")
    for h in range(4):
        # Get attention values for the second token (index 1) in head 'h'
        row = attentions[layer][0][h][1].detach().numpy()
        top_token_idx = row.argmax()
        print(f"Head {h} focuses most on: {tokens[top_token_idx]} ({row[top_token_idx]:.4f})")


#####################################################
LSTM
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input

# --- Configuration & Mode Selection ---
# Set mode to any number from 41 to 50
mode = 43 
seq_len = 25 

# --- Helper: Data Generation ---
def get_data(s_len):
    t = np.linspace(0, 100, 1000)
    data = np.sin(t)
    xs, ys = [], []
    for i in range(len(data) - s_len):
        xs.append(data[i:i+s_len])
        ys.append(data[i+s_len])
    X = np.expand_dims(np.array(xs), -1)
    y = np.array(ys)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]

# --- Helper: Model Factory ---
def build_model(m_type, s_len, units=64):
    layer = LSTM(units, return_sequences=True) if m_type == "LSTM" else GRU(units, return_sequences=True)
    return Sequential([Input(shape=(s_len, 1)), layer, Dense(1)])

# --- Main Execution Block ---
X_train, y_train, X_test, y_test = get_data(seq_len)

if mode == 41: # Build LSTM
    model = build_model("LSTM", seq_len)
    model.summary()

elif mode == 42: # Build GRU
    model = build_model("GRU", seq_len)
    model.summary()

elif mode == 43 or mode == 46: # Compare Performance / Training Time
    for m_type in ["LSTM", "GRU"]:
        m = build_model(m_type, seq_len)
        m.compile(optimizer='adam', loss='mse')
        start = time.time()
        m.fit(X_train, y_train, epochs=5, verbose=0)
        print(f"{m_type} Training Time: {time.time() - start:.2f}s")

elif mode == 44: # Different Sequence Lengths
    for s in [10, 50]:
        xt, yt, _, _ = get_data(s)
        m = build_model("LSTM", s)
        m.compile(optimizer='adam', loss='mse')
        h = m.fit(xt, yt, epochs=5, verbose=0)
        print(f"Seq Len {s} - Final Loss: {h.history['loss'][-1]:.6f}")

elif mode == 45 or mode == 50: # Visualize Hidden States / Temporal Deps
    model = build_model("LSTM", seq_len)
    # Extracting internal activations
    feat_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
    hidden_states = feat_extractor.predict(X_test[:1])
    plt.imshow(hidden_states[0].T, aspect='auto', cmap='viridis')
    plt.title("Task 45/50: Hidden State Visualizer")
    plt.show()

elif mode == 47 or mode == 49: # Forecast & Plot
    model = build_model("GRU", seq_len)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, verbose=0)
    preds = model.predict(X_test)
    plt.plot(y_test[:100], label="Actual")
    plt.plot(preds[:100], label="GRU Forecast")
    plt.legend()
    plt.title("Task 47/49: Forecast Visualization")
    plt.show()

elif mode == 48: # Hyperparameter Tuning (Example: Hidden Units)
    for u in [32, 128]:
        m = build_model("LSTM", seq_len, units=u)
        m.compile(optimizer='adam', loss='mse')
        loss = m.evaluate(X_test, y_test, verbose=0)
        print(f"Units: {u}, Test MSE: {loss:.6f}")

###########################################################
SEQ2SEQ
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model

# --- 1. THE SHORTEST DATASET (Task 59: Math Translation) ---
# We treat "12+4" as the source sequence and "16" as the target.
def gen_data(num_samples=1000):
    questions, answers = [], []
    for _ in range(num_samples):
        a, b = np.random.randint(0, 50, 2)
        q, ans = f"{a}+{b}", f"={a+b}"
        questions.append(q.ljust(5)) # Padding to fixed length
        answers.append(ans.ljust(5))
    
    chars = sorted(list(set("0123456789+ =")))
    char_to_int = {c: i for i, c in enumerate(chars)}
    
    encode = lambda s: [char_to_int[c] for c in s]
    X = np.array([encode(q) for q in questions])
    Y = np.array([encode(a) for a in answers])
    return X, Y, len(chars), char_to_int

X, Y, vocab_size, char_map = gen_data()

# --- 2. MODES 51-60 ---
mode = 51  # Set to 51, 52, 54, 56, etc.

# --- Task 51 & 57: Encoder-Decoder (No Attention) ---
def build_seq2seq(use_attention=False):
    # Encoder
    enc_inputs = Input(shape=(5,))
    enc_emb = Embedding(vocab_size, 8)(enc_inputs)
    enc_outputs, state_h, state_c = LSTM(32, return_sequences=True, return_state=True)(enc_emb)
    
    # Decoder
    dec_inputs = Input(shape=(5,))
    dec_emb = Embedding(vocab_size, 8)(dec_inputs)
    dec_lstm = LSTM(32, return_sequences=True)(dec_emb, initial_state=[state_h, state_c])
    
    if use_attention: # Task 52: Adding Attention
        # Simple Luong-style Dot-product Attention
        query = dec_lstm
        value = enc_outputs
        attn_layer = Attention()
        context_vector, attn_weights = attn_layer([query, value], return_attention_scores=True)
        decoder_combined_context = Concatenate()([dec_lstm, context_vector])
        outputs = Dense(vocab_size, activation='softmax')(decoder_combined_context)
        return Model([enc_inputs, dec_inputs], outputs), attn_weights
    
    outputs = Dense(vocab_size, activation='softmax')(dec_lstm)
    return Model([enc_inputs, dec_inputs], outputs), None

# --- EXECUTION ---
if mode == 51 or mode == 57:
    model, _ = build_seq2seq(use_attention=False)
    model.summary()

elif mode == 52:
    model, _ = build_seq2seq(use_attention=True)
    print("Seq2Seq with Attention Layer Built.")

elif mode == 54 or mode == 58: # Visualize Attention Weights
    model, attn_weights_layer = build_seq2seq(use_attention=True)
    # Extract weights for the first sample
    debug_model = Model(inputs=model.input, outputs=attn_weights_layer)
    weights = debug_model.predict([X[:1], Y[:1]])
    plt.imshow(weights[0], cmap='hot')
    plt.title("Task 54: Attention Alignment Map")
    plt.xlabel("Encoder Steps"); plt.ylabel("Decoder Steps")
    plt.show()

elif mode == 56: # Evaluate using BLEU (Conceptual)
    from nltk.translate.bleu_score import sentence_bleu
    reference = [["1", "5", " ", " ", " "]]
    candidate = ["1", "5", " ", " ", " "]
    score = sentence_bleu(reference, candidate)
    print(f"Task 56: Example BLEU Score: {score:.4f}")

elif mode == 53 or mode == 60: # Compare Baseline vs Attention
    for attn in [False, True]:
        m, _ = build_seq2seq(use_attention=attn)
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        start = time.time()
        m.fit([X, Y], Y, epochs=1, batch_size=32, verbose=0)
        print(f"Attention={attn} | Train Time: {time.time()-start:.2f}s")

####################################################################################################
initilaizers, activations, 
from sklearn.datasets import load_diabetes
from tensorflow.keras import models, layers
import tensforflow as tf
import pandas as pd
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
def create_model(act, optim, loo, init='glorot_uniform'):
    model = models.Sequential([
        # Added 'kernel_initializer' to help you with tasks 11-20
        layers.Dense(1, activation=act, kernel_initializer=init, input_shape=(X_train.shape[1],))
    ])
    model.compile(optimizer=optim, loss=loo, metrics=['mae']) # Using MAE for regression
    return model
def train_n_eval(epchs, bs, act, optim, loo, init='glorot_uniform'):
    m = create_model(act, optim, loo, init)
    hist = m.fit(X_train, y_train, epochs=epchs, batch_size=bs, 
                 validation_data=(X_test, y_test), verbose=0)
    loss, acc = model.evaluate(X_test, y_test)
    return m, hist, loss, acc

'''_, hist_sgd = train_n_eval(50, 32, 'relu', 'sgd', 'mse')
_, hist_adam = train_n_eval(50, 32, 'relu', 'adam', 'mse')
Use init='glorot_normal' for Xavier.
Use init='he_normal' for He.
_, hist_zero = train_n_eval(50, 32, 'relu', 'adam', 'mse', init='zeros')'''

#########################################################################################################
IMAGE FILTERS 
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/content/cpic.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_edge = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])

kernel_blur = np.ones((5,5)) / 25

kernel_sharp = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])

edges = cv2.filter2D(gray, -1, kernel_edge)
blur = cv2.filter2D(img, -1, kernel_blur)
sharp = cv2.filter2D(img, -1, kernel_sharp)

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Edge")
plt.imshow(edges, cmap='gray')

plt.subplot(2,2,2)
plt.title("Blur")
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,3)
plt.title("Sharpen")
plt.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,4)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for i in range(1,5):
    plt.subplot(2,2,i).axis('off')

plt.show()


############## MAUAL CONVULATION
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/content/cpic.jpg", 0)

kernel = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])

# Flip kernel (important)
kernel = np.flipud(np.fliplr(kernel))

img_h, img_w = img.shape
k_h, k_w = kernel.shape

output = np.zeros((img_h - k_h + 1, img_w - k_w + 1))

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        patch = img[i:i+k_h, j:j+k_w]
        output[i, j] = np.sum(patch * kernel)

plt.imshow(output, cmap='gray')
plt.title("Manual Convolution Output")
plt.axis('off')
plt.show()

##############feature map
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# ---------- Load image ----------
img = image.load_img("/content/cpic.jpg", target_size=(128,128))
img_arr = np.expand_dims(image.img_to_array(img)/255.0, axis=0)

# ---------- Build CNN ----------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,128,3)),
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', name='conv1'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', name='conv2'),
])

# ---------- Build model ----------
_ = model.predict(img_arr)

# ---------- Get feature maps ----------
feature_model = Model(
    inputs=model.inputs,
    outputs=model.get_layer("conv1").output
)

feature_maps = feature_model.predict(img_arr)

# ---------- Print matrix ----------
print("Feature map shape:", feature_maps.shape)

for i in range(3):  # print first 3 maps
    print(f"\nFeature Map {i+1} (5x5 portion):")
    print(feature_maps[0, :5, :5, i])

# ---------- Plot ----------
plt.figure(figsize=(10,3))

for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(feature_maps[0,:,:,i], cmap='gray')
    plt.axis('off')

plt.suptitle("Feature Maps (Conv1)")
plt.show()

################################# FILTERS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------- Build CNN ----------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,128,3)),
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', name='conv1')
])

# ---------- Build model ----------
dummy = np.random.rand(1,128,128,3)
_ = model.predict(dummy)

# ---------- Extract filters ----------
conv_layer = model.get_layer("conv1")
filters, bias = conv_layer.get_weights()

print("Filter shape:", filters.shape)

# ---------- Normalize filters ----------
filters = (filters - filters.min()) / (filters.max() - filters.min())

# ---------- Visualize ----------
plt.figure(figsize=(10,4))
for i in range(3):  # print first 3 maps
    print(f"\nFeature Map {i+1} (5x5 portion):")
    print(filters[0, :5, :5, i])
for i in range(filters.shape[-1]):
    plt.subplot(2,4,i+1)
    plt.imshow(filters[:,:,:,i])
    plt.title(f"F{i+1}")
    plt.axis('off')

plt.suptitle("Learned Filters (Conv Layer)")
plt.show()


############################################################# MNIST
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ---------- Load MNIST ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ---------- Preprocess ----------
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# ---------- CNN with Dropout ----------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),   # ⭐ important
    layers.Dense(10, activation='softmax')
])

# ---------- Compile ----------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------- Train ----------
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# ---------- Plot ----------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')

plt.legend()
plt.title("Overfitting Analysis")
plt.show()

##########################################################################################################################

gGRU-LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------- Create dummy time series ----------
data = np.sin(np.linspace(0, 50, 500))

# ---------- Prepare dataset ----------
X, y = [], []
for i in range(20, len(data)):
    X.append(data[i-20:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

X = X[..., np.newaxis]   # (samples, timesteps, features)

# ---------- Split ----------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------- LSTM Model ----------
lstm_model = tf.keras.Sequential([
    layers.LSTM(50, input_shape=(20,1)),
    layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

# ---------- Train ----------
lstm_model.fit(X_train, y_train, epochs=5)

# ---------- Predict ----------
lstm_pred = lstm_model.predict(X_test)

---------------- GRU----------------------------------
# ---------- GRU Model ----------
gru_model = tf.keras.Sequential([
    layers.GRU(50, input_shape=(20,1)),
    layers.Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse')

# ---------- Train ----------
gru_model.fit(X_train, y_train, epochs=5)

# ---------- Predict ----------
gru_pred = gru_model.predict(X_test)

---------------------------------------------------------
# ---------- Evaluate ----------
lstm_loss = lstm_model.evaluate(X_test, y_test)
gru_loss = gru_model.evaluate(X_test, y_test)

print("LSTM Loss:", lstm_loss)
print("GRU Loss:", gru_loss)

# ---------- Plot ----------
plt.plot(y_test, label="Actual")
plt.plot(lstm_pred, label="LSTM")
plt.plot(gru_pred, label="GRU")

plt.legend()
plt.title("LSTM vs GRU")
plt.show()

------------------------------------------------SEQUNECE
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ---------- Create dummy time series ----------
data = np.sin(np.linspace(0, 50, 500))

# ---------- Function to prepare data ----------
def create_dataset(seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

# ---------- Try different sequence lengths ----------
seq_lengths = [5, 10, 20]

for seq_len in seq_lengths:
    print(f"\n===== Sequence Length: {seq_len} =====")
    
    X, y = create_dataset(seq_len)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # ---------- LSTM Model ----------
    model = tf.keras.Sequential([
        layers.LSTM(50, input_shape=(seq_len,1)),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # ---------- Train ----------
    model.fit(X_train, y_train, epochs=3, verbose=0)
    
    # ---------- Evaluate ----------
    loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", loss)

#########################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ================= SETTINGS =================
MODE = "adam"        # sgd, momentum, rmsprop, adam, nadam
INIT = "he"          # zeros, random, xavier, he
GD_TYPE = "mini"     # batch, mini, stochastic
TASK = "classification"  # classification / regression
# ===========================================

# ---------- DATA ----------
if TASK == "classification":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train/255.0, X_test/255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
else:
    X = np.random.rand(1000,1)
    y = 3*X + np.random.randn(1000,1)*0.1
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

# ---------- INITIALIZATION ----------
def get_init():
    if INIT == "zeros": return tf.keras.initializers.Zeros()
    if INIT == "random": return tf.keras.initializers.RandomNormal()
    if INIT == "xavier": return tf.keras.initializers.GlorotUniform()
    if INIT == "he": return tf.keras.initializers.HeNormal()

init = get_init()

# ---------- MODEL ----------
if TASK == "classification":
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', kernel_initializer=init, input_shape=(28,28,1)),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_initializer=init),
        layers.Dense(10, activation='softmax')
    ])
else:
    model = models.Sequential([
        layers.Dense(32, activation='relu', kernel_initializer=init, input_shape=(1,)),
        layers.Dense(1)
    ])

# ---------- OPTIMIZER ----------
def get_optimizer():
    if MODE == "sgd":
        return tf.keras.optimizers.SGD()
    if MODE == "momentum":
        return tf.keras.optimizers.SGD(momentum=0.9)
    if MODE == "rmsprop":
        return tf.keras.optimizers.RMSprop()
    if MODE == "adam":
        return tf.keras.optimizers.Adam()
    if MODE == "nadam":
        return tf.keras.optimizers.Nadam()

opt = get_optimizer()

# ---------- LOSS ----------
if TASK == "classification":
    loss = "sparse_categorical_crossentropy"
else:
    loss = "mse"

model.compile(optimizer=opt, loss=loss, metrics=["accuracy"] if TASK=="classification" else [])

# ---------- GD TYPE ----------
batch_size = {
    "batch": len(X_train),
    "mini": 32,
    "stochastic": 1
}[GD_TYPE]

# ---------- TRAIN ----------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1
)

# ---------- EVALUATE ----------
print("\nTest Results:")
model.evaluate(X_test, y_test)

# ---------- VISUALIZE LOSS ----------
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title(f"{MODE} | {INIT} | {GD_TYPE}")
plt.show()

# ---------- WEIGHT DISTRIBUTION ----------
weights = model.layers[0].get_weights()[0].flatten()
plt.hist(weights, bins=50)
plt.title("Weight Distribution")
plt.show()