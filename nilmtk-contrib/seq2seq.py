from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import json
from os.path import join, isfile
from os import listdir

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

import utils
import plots

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class Seq2Seq(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "Seq2Seq"
        self.models = OrderedDict()
        self.file_prefix = params.get('file_prefix', "")
        self.verbose =  params.get('verbose', 1)
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.batch_size = params.get('batch_size',512)
        self.on_treshold = params.get('on_treshold', 50)
        self.appliances = params.get('appliances', {})
        self.load_model_path = params.get('load_model_path',None)

        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

        self.training_history_folder = params.get("training_history_folder", None)
        self.plots_folder = params.get("plots_folder", None)
        self.results_folder = params.get("results_folder", None)
        
        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)

        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

        if self.results_folder is not None:
            utils.create_path(self.results_folder)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_data, cv_data=None):
        if self.verbose > 0:
            print("...............Seq2Seq partial_fit running...............")

        for appliance_name, data in train_data.items():

            appliance_model = self.appliances.get(appliance_name, {})

            if appliance_model.get("mean", None) is None:
                self.set_appliance_params(train_data)

            if appliance_model.get("mains_mean", None) is None:
                self.set_mains_params(train_data)
                
            appliance_model = self.appliances[appliance_name]

            transfer_path = appliance_model.get("transfer_path", None)
            on_treshold = appliance_model.get("on_treshold", 50)
            mean = appliance_model["mean"]
            std = appliance_model["std"]
            mains_mean = appliance_model["mains_mean"]
            mains_std = appliance_model["mains_std"]

            train_main, train_appliance = self.call_preprocessing(data["mains"], data["appliance"], appliance_name, 'train')

            train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
            train_appliance = pd.concat(train_appliance, axis=0).values.reshape((-1, self.sequence_length))

            if cv_data is not None:
                cv_main, cv_appliance = self.call_preprocessing(cv_data[appliance_name]["mains"], cv_data[appliance_name]["appliance"], appliance_name, 'train')
                
                cv_main = pd.concat(cv_main, axis=0).values.reshape((-1,self.sequence_length,1))
                cv_appliance = pd.concat(cv_appliance, axis=0).values.reshape((-1, self.sequence_length))

            binary_y = np.array([ 1 if x[1 + self.sequence_length // 2] > on_treshold else 0 for x in train_appliance*std + mean])
            
            train_positives = np.where(binary_y == 1)[0]

            train_n_activatons = train_positives.shape[0]
            train_on_examples = train_n_activatons / train_appliance.shape[0]
            train_off_examples = (train_appliance.shape[0] - train_n_activatons) / train_appliance.shape[0]

            if cv_data is not None:
                binary_y = np.array([ 1 if x[1 + self.sequence_length // 2] > on_treshold else 0 for x in cv_appliance*std + mean])
                    
                cv_positives = np.where(binary_y == 1)[0]

                cv_n_activatons = cv_positives.shape[0]
                cv_on_examples = cv_n_activatons / cv_appliance.shape[0]
                cv_off_examples = (cv_appliance.shape[0] - cv_n_activatons) / cv_appliance.shape[0]

            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(train_appliance.shape[0]))
                print("Nº of activations: ", str(train_n_activatons))
                print("On Percentage: ", str(train_on_examples))
                print("Off Percentage: ", str(train_off_examples))
                if cv_data is not None:
                    print("-"*5 + "Cross Validation Info" + "-"*5)
                    print("Nº of examples: ", str(cv_appliance.shape[0]))
                    print("Nº of activations: ", str(cv_n_activatons))
                    print("On Percentage: ", str(cv_on_examples))
                    print("Off Percentage: ", str(cv_off_examples))
                print("-"*10)
                print("Mains Mean: ", str(mains_mean))
                print("Mains Std: ", str(mains_std))
                print(appliance_name + " Mean: ", str(mean))
                print(appliance_name + " Std: ", str(std))

            if appliance_name not in self.models:
                if transfer_path is not None:
                    if( self.verbose != 0):
                        print("Using transfer learning for ", appliance_name)
                    self.models[appliance_name] = self.create_transfer_model(transfer_path)
                else:
                    if( self.verbose != 0):
                        print("First model training for", appliance_name)
                    self.models[appliance_name] = self.return_network()
            else:
                if( self.verbose != 0):
                    print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]

            filepath = self.file_prefix + "{}.h5".format("_".join(appliance_name.split()))

            verbose = 1 if self.verbose >= 1 else 0

            checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=verbose,save_best_only=True,mode='min')

            if cv_data is not None:
                history = model.fit(train_main, 
                        train_appliance,
                        epochs=self.n_epochs, 
                        batch_size=self.batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(cv_main, cv_appliance),
                        verbose=verbose
                        )        
            else:
                history = model.fit(train_main, 
                    train_appliance,
                    epochs=self.n_epochs, 
                    batch_size=self.batch_size,
                    shuffle=False,
                    callbacks=[checkpoint],
                    validation_split=0.15,
                    verbose=verbose
                    )

            history = json.dumps(history.history)
            
            if self.training_history_folder is not None:
                f = open(self.training_history_folder + "history_"+appliance_name.replace(" ", "_")+".json", "w")
                f.write(history)
                f.close()

            if self.plots_folder is not None:
                utils.create_path(self.plots_folder + "/" + appliance_name + "/")
                plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + appliance_name + "/")

            #Gets the trainning data score
            #Concatenates training and cross_validation
            if cv_data is not None:
                X = np.concatenate((train_main, cv_main), axis=0)
                y = np.concatenate((train_appliance, cv_appliance), axis=0)
            else:
                X = train_main
                y = train_appliance

            model.load_weights(filepath)

            if transfer_path is not None:
                model.summary()
                for layer in model.layers:
                    layer.trainable = True

                model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer=Adam(1e-5))
                model.summary()
                model.fit(X, 
                        y,
                        epochs=10, 
                        batch_size=self.batch_size,
                        shuffle=False,
                        verbose=verbose
                        )

            self.models[appliance_name] = model

            y = np.array([x[1 + len(x)//2] for x in  y])
            pred = self.models[appliance_name].predict(X)

            #####################
            # This block is for creating the average of predictions over the different sequences
            # the counts_arr keeps the number of times a particular timestamp has occured
            # the sum_arr keeps the number of times a particular timestamp has occured
            # the predictions are summed for  agiven time, and is divided by the number of times it has occured
                
            l = self.sequence_length
            n = len(pred) + l - 1
            sum_arr = np.zeros((n))
            counts_arr = np.zeros((n))
            for i in range(len(pred)):
                sum_arr[i:i + l] += pred[i].flatten()
                counts_arr[i:i + l] += 1
            for i in range(len(sum_arr)):
                sum_arr[i] = sum_arr[i] / counts_arr[i]
            #################
            
            pred = mean + sum_arr.flatten() * std
            n = self.sequence_length
            units_to_pad = n // 2
            pred = pred[units_to_pad:-units_to_pad]

            train_rmse = math.sqrt(mean_squared_error(y * std + mean, pred))
            train_mae = mean_absolute_error(y * std + mean, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("RMSE: ", train_rmse )
                print("MAE: ", train_mae )
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + appliance_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(train_appliance.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(train_n_activatons)+ "\n")
                f.write("On Percentage: "+ str(train_on_examples)+ "\n")
                f.write("Off Percentage: "+ str(train_off_examples)+ "\n")
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5+ "\n")
                    f.write("Nº of examples: "+ str(cv_appliance.shape[0])+ "\n")
                    f.write("Nº of activations: "+ str(cv_n_activatons)+ "\n")
                    f.write("On Percentage: "+ str(cv_on_examples)+ "\n")
                    f.write("Off Percentage: "+ str(cv_off_examples)+ "\n")
                f.write("-"*10+ "\n")
                f.write("Mains Mean: " + str(mains_mean) + "\n")
                f.write("Mains Std: " + str(mains_std) + "\n")
                f.write(appliance_name + " Mean: " + str(mean) + "\n")
                f.write(appliance_name + " Std: " + str(std) + "\n")
                f.write("Train RMSE: "+str(train_rmse)+ "\n")
                f.write("Train MAE: "+str(train_mae)+ "\n")
                f.close()
  
    def disaggregate_chunk(self, test_mains, app_name):
        test_predictions = []
        disggregation_dict = {}


        test_main_list = self.call_preprocessing(test_mains, app_df_list=None, appliance_name=app_name, method='test')
        test_main = pd.concat(test_main_list, axis=0).values.reshape((-1,self.sequence_length,1))

        app_mean = self.appliances[app_name]['mean']
        app_std = self.appliances[app_name]['std']

        prediction = self.models[app_name].predict(test_main, batch_size=self.batch_size)

        #####################
        # This block is for creating the average of predictions over the different sequences
        # the counts_arr keeps the number of times a particular timestamp has occured
        # the sum_arr keeps the number of times a particular timestamp has occured
        # the predictions are summed for  agiven time, and is divided by the number of times it has occured
            
        l = self.sequence_length
        n = len(prediction) + l - 1
        sum_arr = np.zeros((n))
        counts_arr = np.zeros((n))
        for i in range(len(prediction)):
            sum_arr[i:i + l] += prediction[i].flatten()
            counts_arr[i:i + l] += 1
        for i in range(len(sum_arr)):
            sum_arr[i] = sum_arr[i] / counts_arr[i]
        #################

        prediction = app_mean + sum_arr.flatten() * app_std
        prediction = np.where(prediction>0, prediction,0)

        disggregation_dict[app_name] = pd.Series(prediction)
        results = pd.DataFrame(disggregation_dict,dtype='float32')

        test_predictions.append(results)
        return test_predictions
    
    def save_model(self, folder_name):
        
        #For each appliance trained store its model
        for app in self.models:
            self.models[app].save(join(folder_name, app.replace(" ", "_")+ ".h5"))

            app_params = self.appliances[app]
            app_params["mean"] = float(app_params["mean"])
            app_params["std"] = float(app_params["std"])
            app_params['mains_mean'] = float(app_params['mains_mean'])
            app_params['mains_std'] = float(app_params['mains_std'])
            params_to_save = {}
            params_to_save['appliance_params'] = app_params

            f = open(join(folder_name, app.replace(" ", "_") + ".json"), "w")
            f.write(json.dumps(params_to_save))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and ".h5" in f ]
        for app in app_models:
            app_name = app.split(".")[0].replace("_", " ")
            self.models[app_name] = load_model(join(folder_name, app))

            f = open(join(folder_name, app_name.replace(" ", "_") + ".json"), "r")

            model_string = f.read().strip()
            params_to_load = json.loads(model_string)
            self.appliances[app_name] = params_to_load['appliance_params']

    def return_network(self):

        model = Sequential()
        # 1D Conv
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=2))
        model.add(Conv1D(30, 8, activation='relu', strides=2))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(self.sequence_length))
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model
    def create_transfer_model(self, transfer_path):
        # Model architecture
        model = Sequential()
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=1))
        model.add(Conv1D(30, 8, activation='relu', strides=2))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())

        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)

        for layer in model.layers:
            layer.trainable = False
        
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(self.sequence_length))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def call_preprocessing(self,  mains_lst, app_df_list, appliance_name, method):
        mains_mean = self.appliances[appliance_name]['mains_mean']
        mains_std = self.appliances[appliance_name]['mains_std']
        app_mean = self.appliances[appliance_name]['mean']
        app_std = self.appliances[appliance_name]['std']

        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - mains_mean) / mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))

            processed_app_dfs = []
            for app_df in app_df_list:                    
                new_app_readings = app_df.values.flatten()
                new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                new_app_readings = (new_app_readings - app_mean) / app_std
                processed_app_dfs.append(pd.DataFrame(new_app_readings))

            return processed_mains_lst, processed_app_dfs

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - mains_mean) / mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self,train_data):
        for app_name, data in train_data.items():
            app = self.appliances.get(app_name, None)
            if app is None:
                self.appliances[app_name] = {}

            l = np.array(pd.concat(data["appliance"],axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliances[app_name]['mean'] = app_mean
            self.appliances[app_name]['std'] = app_std

    def set_mains_params(self, train_data):
        for app_name, data in train_data.items():
            app = self.appliances.get(app_name, None)
            if app is None:
                self.appliances[app_name] = {}

            l = np.array(pd.concat(data["mains"],axis=0))
            mains_mean = np.mean(l)
            mains_std = np.std(l)
            if mains_std<1:
                mains_std = 100
            self.appliances[app_name]['mains_mean'] = mains_mean
            self.appliances[app_name]['mains_std'] = mains_std

