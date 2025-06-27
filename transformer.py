import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

# Assume you've already cleaned your dataset: train_file
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

train_file = pd.read_csv(TRAIN_FILE)
train_file['Age'].fillna(train_file['Age'].median(), inplace=True)

train_file['Embarked'].fillna(train_file['Embarked'].mode()[0], inplace= True)


train_file.drop(columns= ['Cabin', 'Ticket', 'Name'], inplace= True)

train = pd.get_dummies(train_file, columns=['Sex', 'Embarked'], drop_first=True)
df = train_file.copy()

# Separate target
y = df['Survived'].values
X = df.drop(columns=['Survived', 'PassengerId'])

# Categorical and continuous columns
cat_cols = ['Sex', 'Embarked', 'Pclass']
cont_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Encode categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Scale continuous values
scaler = StandardScaler()
X[cont_cols] = scaler.fit_transform(X[cont_cols])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras import layers, Model

# Helper: transformer block
def transformer_block(x, heads=2, ff_dim=32, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=heads, key_dim=x.shape[-1])(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization()(x + attn_output)

    ffn = layers.Dense(ff_dim, activation="relu")(out1)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    return layers.LayerNormalization()(out1 + ffn)

# Inputs
inputs = []
cat_embeds = []

# Embedding categorical features
for col in cat_cols:
    vocab_size = int(X[col].max()) + 1
    inp = layers.Input(shape=(1,), name=f"{col}_in")
    embed = layers.Embedding(input_dim=vocab_size, output_dim=8)(inp)
    embed = layers.Reshape((1, 8))(embed)
    inputs.append(inp)
    cat_embeds.append(embed)

# Concatenate categorical embeddings (sequence of tokens)
x_cat = layers.Concatenate(axis=1)(cat_embeds)

# Pass through transformer block
x = transformer_block(x_cat)
x = layers.Flatten()(x)

# Add continuous inputs
cont_input = layers.Input(shape=(len(cont_cols),), name="cont_in")
inputs.append(cont_input)

x = layers.Concatenate()([x, cont_input])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, x)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])
model.summary()

# Prepare inputs as dict
def prepare_inputs(X):
    return {f"{col}_in": X[col].values for col in cat_cols} | {"cont_in": X[cont_cols].values}

train_inputs = prepare_inputs(X_train)
val_inputs = prepare_inputs(X_val)

# Train
history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

results = model.evaluate(val_inputs, y_val)
print(f"\n✅ Validation Accuracy: {results[1]:.4f} — AUC: {results[2]:.4f}")

y_pred_probs = model.predict(val_inputs).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)
print("F1 Score:", f1_score(y_val, y_pred))