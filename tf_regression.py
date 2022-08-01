"""
This script performs DNN model building, training, and evaluation.
"""
import os
from globalVar import *
from processBusData import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# Tensorflow-------------------------------------------------------------------------------------------------------------------
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        # print(f'{epoch}.', end='')


def build_and_compile_model():
    # Build model
    model = keras.Sequential([
        # normalizer,
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    # Compile model. Optimizers control learning rate
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['mae', 'mse', 'RootMeanSquaredError'])
    # metrics = ['mae', 'mse', 'accuracy', 'RootMeanSquaredError', tf.keras.metrics.CategoricalAccuracy()]
    return model


# Prep-------------------------------------------------------------------------------------------------------------------
def get_normalizer(data):
    # Normalizer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(data))

    return normalizer


def normalize_data(data):
    data_stats = data.describe()
    data_stats = data_stats.transpose()

    cols_change = []
    for col_name in data.columns:
        if len(data[col_name].unique()) != 2:
            cols_change.append(col_name)

    for col_change in cols_change:
        data[col_change] = (data[col_change] - data_stats['mean'][col_change]) / data_stats['std'][col_change]

    return data


def check_gpu(disable_gpu=False):
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def prepare_data(df, label_name, feature_names, num_of_cate):
    categorize_col(df, 'label_in', num_of_cate)
    feature = df.copy()
    label = feature.pop(label_name)

    for col in feature.columns:
        if col not in feature_names:
            feature.pop(col)

    feature = normalize_data(feature)
    # label = normalize_data(label.to_frame())

    return feature, label


def categorize_col(df, col_name, num_of_cate=4):  # Categorize the data column
    separator = 1 / num_of_cate
    separators = [df.eval(col_name).quantile(separator)]
    for i in range(num_of_cate-2):
        separator += 1 / num_of_cate
        separators.append(df.eval(col_name).quantile(separator))

    for i in range(len(df.index)):
        for j in range(len(separators)):
            separator = separators[j]
            if df.at[i, 'label_in'] < separator:
                df.at[i, 'label_in'] = j
                break
            if j == len(separators) - 1:
                if df.at[i, 'label_in'] >= separator:
                    df.at[i, 'label_in'] = j + 1

    return df


# Result log-------------------------------------------------------------------------------------------------------------------
def plot_train_history(history, general_info):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plt.figure()
    plt.title(f'Mean Absolute Error - Epoch Graph', fontsize='small', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f'Mean Square Error - Epoch Graph', fontsize='small', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    plt.show()


def write_training_log(label_name, feature_names, epochs, batch_size, number_of_categories, evaluation, accuracy):
    try:
        log = read_csv('log/log.csv')
    except:
        log = [['label', 'feature1', 'feature2', 'epochs', 'batch_size', 'number_of_categories', 'MeanAbsoluteError', 'MeanSquareError', 'RootMeanSquaredError', 'accuracy']]

    new_line = [label_name, feature_names[0], feature_names[1], epochs, batch_size, number_of_categories, evaluation[1], evaluation[2], evaluation[3], accuracy]
    log.append(new_line)

    write_csv(log, 'log/log.csv')


def check_identical_param(epochs, feature_combination, category):  # Skip the initial settings that have already been run
    try:
        log = pd.read_csv('log/log.csv')
    except:
        return False

    for i in range(len(log.index)):
        if log.at[i, 'epochs'] == epochs and log.at[i, 'feature1'] == feature_combination[0] and log.at[i, 'feature2'] == feature_combination[1] and log.at[i, 'number_of_categories'] == category:
            return True

    return False


# Main functions-----------------------------------------------------------------------------------------------------------------------

def get_eval_accuracy(model, eval_label, eval_feature, general_info):  # Evaluate the model, write training logs and draw graphs.
    evaluation = model.evaluate(eval_feature, eval_label, verbose=2)
    prediction = model.predict(eval_feature).flatten()

    for i in range(len(prediction)):
        prediction[i] = int(prediction[i] + 0.5)

    # Draw the graph that shows how true value and predictions
    plt.title(f'True - Pred Graph', fontsize='small', fontweight='bold')
    plt.scatter(eval_label, prediction, alpha=0.003)
    plt.xlabel('True Values [Volume]')
    plt.ylabel('Predictions [Volume]')
    plt.axis('equal')  # Equal length for x and y
    plt.axis('square')  # Square shape graph
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    fig1 = plt.gcf()
    fig1.savefig(f'./graph/{general_info}_TPG.jpg', bbox_inches='tight', dpi=450)
    plt.clf()

    # Draw the graph that shows error distribution
    errors = prediction - eval_label
    plt.title(f'Prediction Error Distribution Graph', fontsize='small', fontweight='bold')
    plt.hist(errors, bins=25)
    plt.xlabel("Prediction Error [Volume]")
    plt.ylabel("Count")
    fig2 = plt.gcf()
    fig2.savefig(f'./graph/{general_info}_PEDG.jpg', bbox_inches='tight', dpi=450)
    plt.clf()

    # How many stops are correctly predicted in volume category
    eval_accuracy = errors.value_counts()[0] / len(errors)
    return evaluation, eval_accuracy


def main_train_and_eval(epochs, addi_features, number_of_categories, batch_size=32768, checkgpu=False):  # Perform a training
    if checkgpu:
        check_gpu()
    # Initialize-------------------------------------------------------------------
    general_info = f'E{epochs}-AF[{addi_features[0]}][{addi_features[1]}]-NoC{number_of_categories}'

    label_name = 'label_in'
    feature_names = ['day_type']
    for i in range(24):
        feature_names.append(f'time{i}')

    # custom settings
    for feature_name in addi_features:
        feature_names.append(feature_name)

    # Get the data----------------------------------------------------------------
    tensorflow_folder = get_value('tensorflow_folder')
    train = pd.read_csv(f'{tensorflow_folder}/train.csv')
    train_feature, train_label = prepare_data(train, label_name, feature_names, number_of_categories)

    eval = pd.read_csv(f'{tensorflow_folder}/eval.csv')
    eval_feature, eval_label = prepare_data(eval, label_name, feature_names, number_of_categories)

    # Build and train model------------------------------------------------------
    model = build_and_compile_model()
    train_history = model.fit(train_feature, train_label, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                              verbose=0, callbacks=[PrintDot()])
    # To plot the training history, use:
    # plot_train_history(train_history, general_info)
    # To save a model, use:
    # model.save('./model/model1')
    # To load a model, use:
    # To reconstructed_model = keras.models.load_model('./model/model1')

    # Evaluate the model---------------------------------------------------------
    try:
        evaluation, accuracy = get_eval_accuracy(model, eval_label, eval_feature, general_info)
        write_training_log(label_name, addi_features, epochs, batch_size, number_of_categories, evaluation, accuracy)
        print(f'Finished training and evaluation: {general_info}')
        return True
    except:
        return False


def get_feature_combinations():  # Get all the combinations for two weights
    tensorflow_folder = get_value('tensorflow_folder')
    try:
        columns = pd.read_csv(f'{tensorflow_folder}/train.csv').columns
    except:
        split_data()
        columns = pd.read_csv(f'{tensorflow_folder}/train.csv').columns
    population_weights = []
    stopvolume_weights = []
    for column in columns:
        if column[0:4] == 'popu':
            population_weights.append(column)
        elif column[0:4] == 'stop' or column[0:4] == 'surr':
            stopvolume_weights.append(column)

    feature_comb = []
    for population_weight in population_weights:
        for stopvolume_weight in stopvolume_weights:
            feature_comb.append([population_weight, stopvolume_weight])

    return feature_comb


def main_batch_train_and_eval():  # Main entrance of the current script, offers batch training using different feature combinations
    feature_combinations = get_feature_combinations()  # Use this if you want to try out all combinations of features.
    # feature_combinations = [
    #     ['population_weight_t1000b1000', 'stop_volume_weight_t1000b25']
    #     # ['population_weight_t1500b500', 'surrounding_stop_volume_weight_t2000b100']
    # ]

    cate = [3, 4, 5]  # Set the number of categories
    epochs = 100

    total = len(feature_combinations) * len(cate)

    all_done = False
    while not all_done:
        unsuccessful_count = 0
        count = 0
        for feature_combination in feature_combinations:
            for category in cate:
                duplicate = check_identical_param(epochs, feature_combination, category)
                if not duplicate:
                    success = main_train_and_eval(
                        epochs=epochs,
                        addi_features=feature_combination,
                        number_of_categories=category
                    )
                    if not success:
                        split_data()
                        print('Ran into error, splitting the data into train and eval again.')
                        unsuccessful_count += 1
                else:
                    print('Duplicate initial setting, continue.')
                count += 1
                print(f'FINISHED: {count}/{total}\n')
        if unsuccessful_count == 0:
            all_done = True
        else:
            print(f'Training not completed due to error, restart unfinished training. Remaining: {unsuccessful_count}')
    print('All parameters have been tested successfully, program stops.')


if __name__ == "__main__":
    main_batch_train_and_eval()






