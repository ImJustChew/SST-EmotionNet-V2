import os
import sys
import matplotlib.pyplot as plt
import ast

# take first argument and read the file
file_name = sys.argv[1]
# read tf history file

with open(os.path.join(os.path.curdir,'result',file_name)) as f:
    data = ast.literal_eval(f.read())

    plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(data['loss'], label='Training Loss')
plt.plot(data['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(data['accuracy'], label='Training Accuracy')
plt.plot(data['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()