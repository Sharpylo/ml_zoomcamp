import tensorflow as tf


IMG_SIZE = (96, 96)
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
PATH_DATA = 'data_plants/'
NAME = 'best_model.h5'
FOLDER_PATH = 'images\img_test'

classes = {
    0: 'aloevera', 1: 'banana', 2: 'bilimbi', 3: 'cantaloupe', 4: 'cassava', 5: 'coconut',
    6: 'corn', 7: 'cucumber', 8: 'curcuma', 9: 'eggplant', 10: 'galangal', 11: 'ginger',
    12: 'guava', 13: 'kale', 14: 'longbeans', 15: 'mango', 16: 'melon', 17: 'orange',
    18: 'paddy', 19: 'papaya', 20: 'peperchili', 21: 'pineapple', 22: 'pomelo', 23: 'shallot',
    24: 'soybeans', 25: 'spinach', 26: 'sweetpotatoes', 27: 'tobacco', 28: 'waterapple', 29: 'watermelon'
}