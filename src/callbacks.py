"""
Inputs to all these functions are Tensorflow tensors with INPUT_SHAPE=(H,H,2) I think.
Loss functions listed here are not to be used. We will use Cross Entropy only.
But these metrics matter anda we can paass them all in one training itself.
model.compile(blah, blah, metrics = ['accuracy', JS, DC, SP])
Like so.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def create_mask(pred_mask):
    pred_mask = tf.cast(tf.argmax(pred_mask, -1), tf.dtypes.float32)
    return pred_mask


def JS(y_true, y_pred, smooth=100):
    # Jaccard Similarity
    # Also known as IoU (Intersection over Union)
    
    y_pred = create_mask(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    J = K.mean((intersection + smooth) / (union - intersection + smooth), axis=0)
    return J

def DC(y_true, y_pred, smooth=1):
    # Dice Coefficient

    y_pred = create_mask(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def SE(y_true, y_pred):
    y_pred = create_mask(y_pred)

    FN = K.sum((y_true) * (1 - y_pred), axis=[1, 2])
    TP = K.sum((y_true) * (y_pred), axis=[1, 2])
    S = K.mean(TP /(TP + FN + K.epsilon()), axis=0)
    return S

def SP(y_true, y_pred):
    y_pred = create_mask(y_pred)

    FP = K.sum((1 - y_true) * y_pred, axis=[1, 2])
    TN = K.sum((1 - y_true) * (1 - y_pred), axis=[1, 2])
    S = K.mean(TN /(TN + FP + K.epsilon()), axis=0)
    return S

def F1(y_true, y_pred):
    y_pred = create_mask(y_pred)
    FP = K.sum((1 - y_true) * y_pred, axis=[1, 2])
    TN = K.sum((1 - y_true) * (1 - y_pred), axis=[1, 2])
    FN = K.sum((y_true) * (1 - y_pred), axis=[1, 2])
    TP = K.sum((y_true) * (y_pred), axis=[1, 2])
    recall = TP / (TP + FN + K.epsilon())
    precision = TP / (TP + FP + K.epsilon())
    F = K.mean(2.0*recall*precision/(precision + recall + K.epsilon()), axis=0)
    return F


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics_map = None):
        self.batch_loss = []
        self.batch_acc = []
        self.batch_SP = []
        self.batch_DC = []
        self.batch_JS = []
        self.batch_val_acc = []
        self.batch_val_SP = []
        self.batch_val_DC = []
        self.batch_val_JS = []
        self.metrics_map = metrics_map
    def on_train_batch_end(self, batch, logs=None):
        self.batch_loss.append(logs['loss'])
        self.batch_acc.append(logs['accuracy'])
        self.batch_SP.append(logs['SP'])
        self.batch_DC.append(logs['DC'])
        self.batch_JS.append(logs['JS'])
        # print(logs)
        # self.batch_val_acc.append(logs['val_accuracy'])
        # self.batch_val_SP.append(logs['val_SP'])
        # self.batch_val_DC.append(logs['val_DC'])
        # self.batch_val_JS.append(logs['val_JS'])
        self.model.reset_metrics()
        
def plot_history(history, sp=False, dc=False, js=True):
    
    plt.figure(figsize = (12,8))
    plt.ylabel("Metric values")
    plt.xlabel("Epochs")
    plt.ylim([0.3,1.0])

    print(len(history))
    plt.plot(history['accuracy'], label='Training Accuracy')
    if sp: plt.plot(history['SP'], label='Training Specificity')
    if dc: plt.plot(history['DC'], label='Training Dice Coefficient')
    if js: plt.plot(history['JS'], label='Training Jaccard Similarity (IoU)')
    plt.plot(history['val_accuracy'], label='Valiation Accuracy')
    if sp: plt.plot(history['val_SP'], label='Validation Specificity')
    if dc: plt.plot(history['val_DC'], label='Validation Dice Coefficient')
    if js: plt.plot(history['val_JS'], label='Validation Jaccard Similarity')
    plt.legend(loc='upper right',  prop={'size': 10})
    
    #plt.savefig('/content/drive/My Drive/data/111.png')

def plot_batch_stats(callback_object, sp=False, dc=False, js=True, prop = {'size' : 10}):
    plt.figure()
    plt.ylabel("Metric values")
    plt.xlabel("Training steps")
    plt.ylim([0.5,1.0])
    
    
    print(len(callback_object.batch_acc))
    plt.plot(callback_object.batch_acc, label='Training Accuracy')
    if sp: plt.plot(callback_object.batch_SP, label='Training Specificity')
    if dc: plt.plot(callback_object.batch_DC, label='Training Dice Coefficient')
    if js: plt.plot(callback_object.batch_JS, label='Training Jaccard Similarity (IoU)')
    plt.plot(callback_object.batch_val_acc, label='Valiation Accuracy')
    if sp: plt.plot(callback_object.batch_val_SP, label='Validation Specificity')
    if dc: plt.plot(callback_object.batch_val_DC, label='Validation Dice Coefficient')
    if js: plt.plot(callback_object.batch_val_JS, label='Validation Jaccard Similarity')
    plt.legend(loc='lower right',  prop=prop)
    plt.show()

# batch_stats = CustomCallback();

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy', SP, DC, JS])

# train_size = 2000
# val_size = 1000
# num_epochs = 15

# history = model.fit(
#     train_data_gen,
#     steps_per_epoch=(train_size//batch_size),
#     epochs=num_epochs,
#     validation_data=val_data_gen,
#     validation_steps=(val_size//batch_size),
#     callbacks = [batch_stats]
# )