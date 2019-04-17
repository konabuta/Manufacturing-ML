
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import plot_model
import tensorflow as tf

np.random.seed(1234)  
PYTHONHASHSEED = 0

from azureml.core import Run
run = Run.get_context()

parser = argparse.ArgumentParser(description='Keras DogCat example:')
parser.add_argument('--dataset', '-d', dest='data_folder',help='The datastore')
args = parser.parse_args()

train_df = pd.read_csv(args.data_folder+"/data/train.csv", sep=",", header=0)
train_df['RUL'] = train_df['RUL'].astype(float)
test_df = pd.read_csv(args.data_folder+"/data/test.csv", sep=",", header=0)
train_df['RUL'] = train_df['RUL'].astype(float)

sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):
    #指定された列の値を取得
    data_array = id_df[seq_cols].values
    #num_elements : 特定idのデータ数 (for id = 1, it is 192)
    num_elements = data_array.shape[0]
    # for id = 1, zip from both range(0, 142) & range(50, 192)
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        #print(start,stop)
        yield data_array[start:stop, :]
        
        
#  特徴量となる列の抽出 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# 学習データのsequences作成
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# generate labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)


epochs=10
batch_size=200
validation_split=0.05

# Hyper-Parameter
run.log("エポック数",epochs)
run.log("バッチサイズ",batch_size)
run.log("検証データ分割",validation_split)


class RunCallback(tf.keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
        
    def on_epoch_end(self, batch, logs={}):
        self.run.log(name="training_loss", value=float(logs.get('loss')))
        self.run.log(name="validation_loss", value=float(logs.get('val_loss')))
        self.run.log(name="training_acc", value=float(logs.get('acc')))
        self.run.log(name="validation_acc", value=float(logs.get('val_acc')))

callbacks = list()
callbacks.append(RunCallback(run))

# モデルネットワークの定義
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]
print("nb_features:",seq_array.shape[2])
print("nb_out:",label_array.shape[1])

model = Sequential()

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x = seq_array, y = label_array, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1,
          callbacks = callbacks)



# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
run.log("損失",scores[0])
run.log("モデル精度", scores[1])

os.makedirs('./outputs/model', exist_ok=True)
model.save_weights('./outputs/mnist_mlp_weights.h5')
