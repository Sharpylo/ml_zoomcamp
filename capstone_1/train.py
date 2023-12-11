import time

from logic.constants import IMG_SIZE, BATCH_SIZE, AUTO, SEED, LIST_SEED, PATH_DATA, classes
from logic.train_logic import evaluate_best_model, load_and_preprocess_data, img_preprocessing, augmentation, create_dataset_loader, create_custom_model, train_model

start_time = time.time()


train_data = load_and_preprocess_data(PATH_DATA + 'train.csv')
valid_data = load_and_preprocess_data(PATH_DATA + 'val.csv')
test_data = load_and_preprocess_data(PATH_DATA + 'test.csv')

print("train images: ", train_data.shape[0])
print("validation images: ", valid_data.shape[0])
print("test images: ", test_data.shape[0])

# Common preprocessing functions
common_preprocessing_functions = [img_preprocessing, augmentation]

# Creating datasets
train_dataset = create_dataset_loader(train_data, BATCH_SIZE, common_preprocessing_functions)
valid_dataset = create_dataset_loader(valid_data, BATCH_SIZE, [img_preprocessing])
test_dataset = create_dataset_loader(test_data, BATCH_SIZE, [img_preprocessing])

num_classes = len(classes)
# Create the model using the function
custom_model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
# Display the model summary
custom_model.summary()

history_1 = train_model(custom_model, train_dataset, valid_dataset, name='best_model_1.h5', epochs=100)

best_model_path_2 = 'best_model_1.h5'
evaluate_best_model(best_model_path_2, test_data, test_dataset)

end_time = time.time()
execution_time = end_time = start_time
print(f"Execution time: {round(execution_time, 2)} seconds")