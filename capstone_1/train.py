import time

from logic.constants import IMG_SIZE, BATCH_SIZE, PATH_DATA, NAME, classes
from logic.train_logic import evaluate_best_model, load_and_preprocess_data, img_preprocessing, augmentation, create_dataset_loader, create_custom_model, train_model

def load_and_prepare_datasets():
    # Load and preprocess data
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

    return train_dataset, valid_dataset, test_dataset, test_data

def create_and_train_model(train_dataset, valid_dataset):
    num_classes = len(classes)

    # Create and train the model
    custom_model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
    custom_model.summary()

    history = train_model(custom_model, train_dataset, valid_dataset, name=NAME, epochs=100)

    return custom_model, history

def load_and_evaluate_recreated_model(model, model_path, test_data, test_dataset):
    # If the model is provided, use it; otherwise, load the model from the path
    if model is None:
        # Recreate the model architecture
        recreated_model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(classes))

        # Load the saved weights
        recreated_model.load_weights(model_path)
    else:
        recreated_model = model

    # Evaluate the recreated model on the test set
    evaluate_best_model(recreated_model, test_data, test_dataset)

def main():
    start_time = time.time()

    train_dataset, valid_dataset, test_dataset, test_data = load_and_prepare_datasets()

    # Create and train the original model
    original_model, history = create_and_train_model(train_dataset, valid_dataset)

    # Save the weights of the original model
    original_model.save_weights(NAME)

    # Load and evaluate the recreated model (saved without further training)
    load_and_evaluate_recreated_model(None, NAME, test_data, test_dataset)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {round(execution_time, 2)} seconds")


if __name__ == "__main__":
    main()