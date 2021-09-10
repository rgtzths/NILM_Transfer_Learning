
from os.path import join, isfile
from os import listdir
import math
import json

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from wt import get_discrete_features
from generate_timeseries import generate_appliance_timeseries, generate_main_timeseries

import utils
import plots

class MLP():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'MLP')
        #Percentage of values used as cross validation data from the training data.
        self.cv_split = params.get('cv_split', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_path',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 1)
        
        self.appliances = params["appliances"]

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 178,
            "batch_size": 1024,
            "epochs": 300,
            "n_nodes":256,
            "on_treshold" : 50,
            "feature_extractor": "",
            "wavelet": 'db4',
        }

        self.training_history_folder = params.get("training_history_folder", None)
        self.results_folder = params.get("results_folder", None)
        self.checkpoint_folder = params.get("checkpoint_folder", None)
        self.plots_folder = params.get("plots_folder", None)

        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)
        
        if self.results_folder is not None:
            utils.create_path(self.results_folder)
        
        if self.checkpoint_folder is not None:
            utils.create_path(self.checkpoint_folder)
        
        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

        #In case of existing a model path, load every model in that path.
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_data, cv_data=None):

        #For each appliance to be classified
        for app_name, data in train_data.items():

            if( self.verbose != 0):
                print("Preparing Dataset for %s" % app_name)

            appliance_model = self.appliances.get(app_name, {})

            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            timestep = appliance_model["timestep"]
            batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
            epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
            n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])
            feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])
            app_mean = appliance_model.get("mean", None)
            app_std = appliance_model.get("std", None)
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            transfer_path = appliance_model.get("transfer_path", None)
            mains_std = appliance_model.get("mains_std", None)
            mains_mean = appliance_model.get("mains_mean", None)

            if feature_extractor == "wt":
                if self.verbose > 0:
                    print("Using Discrete Wavelet Transforms as Features")
                wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                X_train, mean, std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
                X_train, mains_mean, mains_std = get_discrete_features(X_train*std + mean, len(data["mains"][0].columns.values), wavelet)
                appliance_model["mains_mean"] = mains_mean
                appliance_model["mains_std"] = mains_std
                if cv_data is not None:
                    X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, mean, std)[0]
                    X_cv = get_discrete_features(X_cv*std + mean, len(data["mains"][0].columns.values), wavelet, mains_mean, mains_std)[0]

            else:
                if self.verbose > 0:
                    print("Using the Timeseries as Features")
                if mains_mean is None:
                    X_train, mains_mean, mains_std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
                    appliance_model["mains_mean"] = mains_mean
                    appliance_model["mains_std"] = mains_std
                else:
                    X_train = generate_main_timeseries(data["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]

                X_train = X_train.reshape(X_train.shape[0], -1)

                if cv_data is not None:
                    X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]
                    X_cv = X_cv.reshape(X_cv.shape[0], -1)

            if app_mean is None:
                y_train, app_mean, app_std = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap)
                appliance_model["mean"] = app_mean
                appliance_model["std"] = app_std
            else:
                y_train = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap, app_mean, app_std)[0]

            binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean])
            
            positives = np.where(binary_y == 1)[0]

            train_n_activatons = positives.shape[0]
            train_on_examples = train_n_activatons / y_train.shape[0]
            train_off_examples = (y_train.shape[0] - train_n_activatons) / y_train.shape[0]

            if cv_data is not None:
                y_cv = generate_appliance_timeseries(cv_data[app_name]["appliance"], False, timewindow, timestep, overlap, app_mean, app_std)[0]
            
                binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_cv*app_std) + app_mean])
                
                cv_positives = np.where(binary_y == 1)[0]

                cv_n_activatons = cv_positives.shape[0]
                cv_on_examples = cv_n_activatons / y_cv.shape[0]
                cv_off_examples = (y_cv.shape[0] - cv_n_activatons) / y_cv.shape[0]

            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(X_train.shape[0]))
                print("Nº of activations: ", str(train_n_activatons))
                print("On Percentage: ", str(train_on_examples))
                print("Off Percentage: ", str(train_off_examples))
                if cv_data is not None:
                    print("-"*5 + "Cross Validation Info" + "-"*5)
                    print("Nº of examples: ", str(X_cv.shape[0]))
                    print("Nº of activations: ", str(cv_n_activatons))
                    print("On Percentage: ", str(cv_on_examples))
                    print("Off Percentage: ", str(cv_off_examples))
                print("-"*10)
                print("Mains Mean: ", str(mains_mean))
                print("Mains Std: ", str(mains_std))
                print(app_name + " Mean: ", str(app_mean))
                print(app_name + " Std: ", str(app_std))
            
            if app_name in self.model:
                if self.verbose > 0:
                    print("Starting from previous step")
                model = self.model[app_name]
            else:
                if transfer_path is None:
                    if self.verbose > 0:
                        print("Creating new model")
                    model = self.create_model(n_nodes, (X_train.shape[1],))       
                else:
                    if self.verbose > 0:
                        print("Starting from pre-trained model")
                    model = self.create_transfer_model(transfer_path, (X_train.shape[1],), n_nodes)
            
            if self.verbose != 0:
                print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            verbose = 1 if self.verbose >= 1 else 0

            checkpoint = ModelCheckpoint(
                    self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                    monitor='val_loss', 
                    verbose=verbose, 
                    save_best_only=True, 
                    mode='min'
            )
            
            if cv_data is not None:
                history = model.fit(X_train, 
                        y_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(X_cv, y_cv),
                        verbose=verbose
                        )        
            else:
                history = model.fit(X_train, 
                    y_train,
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=False,
                    callbacks=[checkpoint],
                    validation_split=self.cv_split,
                    verbose=verbose
                    ) 
            
            history = json.dumps(history.history)

            if self.training_history_folder is not None:
                f = open(self.training_history_folder + "history_"+app_name.replace(" ", "_")+".json", "w")
                f.write(history)
                f.close()
            
            if self.plots_folder is not None:
                utils.create_path(self.plots_folder + "/" + app_name + "/")
                plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + app_name + "/")

            #Gets the trainning data score
            if cv_data is not None:
                X = np.concatenate((X_train, X_cv), axis=0)
                y = np.concatenate((y_train, y_cv), axis=0)
            else:
                X = X_train
                y = y_train

            model.load_weights(self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5")

            if transfer_path is not None:
                model.summary()
                for layer in model.layers:
                    layer.trainable = True

                model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer=Adam(1e-5))
                model.summary()
                model.fit(X, 
                        y,
                        epochs=10, 
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=verbose
                        )

            #Stores the trained model.
            self.model[app_name] = model

            pred = self.model[app_name].predict(X) * app_std + app_mean

            train_rmse = math.sqrt(mean_squared_error(y * app_std + app_mean, pred))
            train_mae = mean_absolute_error(y * app_std + app_mean, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("RMSE: ", train_rmse )
                print("MAE: ", train_mae )
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X_train.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(train_n_activatons)+ "\n")
                f.write("On Percentage: "+ str(train_on_examples)+ "\n")
                f.write("Off Percentage: "+ str(train_off_examples)+ "\n")
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5+ "\n")
                    f.write("Nº of examples: "+ str(X_cv.shape[0])+ "\n")
                    f.write("Nº of activations: "+ str(cv_n_activatons)+ "\n")
                    f.write("On Percentage: "+ str(cv_on_examples)+ "\n")
                    f.write("Off Percentage: "+ str(cv_off_examples)+ "\n")
                f.write("-"*10+ "\n")
                f.write("Mains Mean: " + str(mains_mean) + "\n")
                f.write("Mains Std: " + str(mains_std) + "\n")
                f.write(app_name + " Mean: " + str(app_mean) + "\n")
                f.write(app_name + " Std: " + str(app_std) + "\n")
                f.write("Train RMSE: "+str(train_rmse)+ "\n")
                f.write("Train MAE: "+str(train_mae)+ "\n")
                f.close()

    def disaggregate_chunk(self, test_mains, app_name):

        test_predictions_list = []
        appliance_powers_dict = {}
        
        if self.verbose > 0:
            print("Preparing the Test Data for %s" % app_name)

        appliance_model = self.appliances[app_name]

        timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
        overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
        timestep = appliance_model["timestep"]
        feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])
        app_mean = appliance_model["mean"]
        app_std = appliance_model["std"]
        mains_std = appliance_model["mains_std"]
        mains_mean = appliance_model["mains_mean"]
        
        if feature_extractor == "wt":
            if( self.verbose > 0):
                print("Using Discrete Wavelet Transforms as Features")
            X_test, mean, std = generate_main_timeseries(test_mains, timewindow, timestep, overlap)
            wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
            X_test = get_discrete_features(X_test*std + mean, len(test_mains[0].columns.values), wavelet, mains_mean, mains_std)[0]
        else:
            if( self.verbose > 0):
                print("Using the Timeseries as Features")
            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, mains_mean, mains_std)[0]
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        if( self.verbose == 2):
            print("Nº of examples", X_test.shape[0])

        if self.verbose > 0:
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
        
        pred = self.model[app_name].predict(X_test).flatten()* app_std + app_mean
        pred = np.where(pred > 0, pred, 0)

        appliance_powers_dict[app_name] = pd.Series(pred)
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        
        #For each appliance trained store its model
        for app in self.model:
            self.model[app].save(join(folder_name, app.replace(" ", "_")+ ".h5"))

            app_params = self.appliances[app]
            app_params["mean"] = float(app_params["mean"])
            app_params["std"] = float(app_params["std"])
            params_to_save = {}

            
            app_params['mains_mean'] = [float(x) for x in app_params['mains_mean']]
            app_params['mains_std'] = [float(x) for x in app_params['mains_std']]

            params_to_save['appliance_params'] = app_params

            f = open(join(folder_name, app.replace(" ", "_") + ".json"), "w")
            f.write(json.dumps(params_to_save))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and ".h5" in f ]
        for app in app_models:
            app_name = app.split(".")[0].replace("_", " ")
            self.model[app_name] = load_model(join(folder_name, app))

            f = open(join(folder_name, app_name.replace(" ", "_") + ".json"), "r")

            model_string = f.read().strip()
            params_to_load = json.loads(model_string)
            self.appliances[app_name] = params_to_load['appliance_params']


    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(int(n_nodes/8), activation='relu'))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(int(n_nodes/8), activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def create_transfer_model(self, transfer_path, input_shape, n_nodes=256):

        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(int(n_nodes/8), activation='relu'))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(int(n_nodes/8), activation='relu'))
        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)
        
        for layer in model.layers[1:]:
            layer.trainable = False

        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        
        return model
