import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', '/home/navallo/Documents/data/flowers')

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
print([label])
sess = tf.InteractiveSession()
init_op = tf.initialize_all_variables()

node1 = label
print(node1)
#node1.eval()