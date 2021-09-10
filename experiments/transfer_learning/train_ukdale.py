from nilmtk.api import API

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from utils import create_path
from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

ukdale_dataset = '../../../datasets/ukdale/ukdale.h5'#Path to the REFIT dataset.

no_transfer_path = "./base_train/refit_train/models/"

def run_fridge(base_path, timestep, epochs, batch_size, sequence_length, model_path):
    fridge = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['fridge'],
        "use_activations" : True,
        "appliances_params" : {
            "fridge" : {
                "min_off_time" : 12,
                "min_on_time" : 60,
                "number_of_activation_padding": 50,
                "min_on_power" : 50
            }
        },
        'methods': {
            'DAE':DAE({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "results_folder" : base_path + "results/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                #"load_model_path" : no_transfer_path + "DAE/",
                "appliances" : {
                    "fridge" : {
                        "on_threshold" : 50,
                        'transfer_path': model_path + "DAE/fridge.h5"
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "results_folder" : base_path + "results/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                #"load_model_path" : no_transfer_path + "Seq2Point/",
                "appliances" : {
                    "fridge" : {
                        "on_threshold" : 50,
                        'transfer_path': model_path + "Seq2Point/fridge.h5"
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                "verbose" : 2,
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "results_folder" : base_path + "results/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                #"load_model_path" : no_transfer_path + "Seq2Seq/",
                "appliances" : {
                    "fridge" : {
                        "on_threshold" : 50,
                        'transfer_path': model_path + "Seq2Seq/fridge.h5"
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_path" : no_transfer_path + "ResNet/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 64,
                        'on_treshold' : 50,
                        'transfer_path': model_path + "ResNet/fridge.h5"
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_path" : no_transfer_path + "DeepGRU/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 128,
                        'on_treshold' : 50,
                        'transfer_path': model_path + "DeepGRU/fridge.h5"
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_path" : no_transfer_path + "MLP/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP/fridge.h5"
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_path" : no_transfer_path + "MLP_Raw/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 50,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP_Raw/fridge.h5"
                    }     
                },
            }),
        },
        'train': {   
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-06-30",
                            'end_time': "2014-07-15"
                        }         
                    }
                },
            }
        },
        'cross_validation': {    
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-08-08"
                        }       
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        1: {
                            'start_time': "2013-05-01",
                            'end_time': "2013-05-15",
                        },
                        2: {
                            'start_time': "2013-07-01",
                            'end_time': "2013-07-08",
                        }  
                    }
                },
            },
            'metrics':['mae', 'rmse',  'nrmse']
        }
    }

    ### Training and testing fridge ####
    results = API(fridge)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_kettle(base_path, timestep, epochs, batch_size, sequence_length, model_path):
    kettle = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['kettle'],
        "use_activations" : True,
        "appliances_params" : {
            "kettle" : {
                "min_off_time" : 0,
                "min_on_time" : 12,
                "number_of_activation_padding": 10,
                "min_on_power" : 2000
            }
        },
        'methods': {
            'DAE':DAE({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "results_folder" : base_path + "results/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                #"load_model_path" : no_transfer_path + "DAE/",
                "appliances" : {
                    "kettle" : {
                        "on_threshold" : 2000,
                        'transfer_path': model_path + "DAE/kettle.h5"
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "results_folder" : base_path + "results/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                #"load_model_path" : no_transfer_path + "Seq2Point/",
                "appliances" : {
                    "kettle" : {
                        "on_threshold" : 2000,
                        'transfer_path': model_path + "Seq2Point/kettle.h5"
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                "verbose" : 2,
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "results_folder" : base_path + "results/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                #"load_model_path" : no_transfer_path + "Seq2Seq/",
                "appliances" : {
                    "kettle" : {
                        "on_threshold" : 2000,
                        'transfer_path': model_path + "Seq2Seq/kettle.h5"
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_path" : no_transfer_path + "ResNet/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 64,
                        'on_treshold' : 2000,
                        'transfer_path': model_path + "ResNet/kettle.h5"
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_path" : no_transfer_path + "DeepGRU/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 128,
                        'on_treshold' : 2000,
                        'transfer_path': model_path + "DeepGRU/kettle.h5"
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_path" : no_transfer_path + "MLP/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 2000,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP/kettle.h5"
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_path" : no_transfer_path + "MLP_Raw/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 2000,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP_Raw/kettle.h5"
                    }     
                },
            }),
        },
        'train': {   
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-06-30",
                            'end_time': "2014-08-15"
                        } 
                                
                    }
                },
            }
        },
        'cross_validation': {    
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-08-15",
                            'end_time': "2014-11-12"
                        } 
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-10",
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse',  'nrmse']
        }
    }
    ### Training and testing kettle ####
    results = API(kettle)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_microwave(base_path, timestep, epochs, batch_size, sequence_length, model_path):
    microwave = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['microwave'],
        "use_activations" : True,
        "appliances_params" : {
            "microwave" : {
                "min_off_time" : 30,
                "min_on_time" : 12,
                "number_of_activation_padding": 7,
                "min_on_power" : 200
            }
        },
        'methods': {
            'DAE':DAE({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "results_folder" : base_path + "results/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                #"load_model_path" : no_transfer_path + "DAE/",
                "appliances" : {
                    "microwave" : {
                        "on_threshold" : 200,
                        'transfer_path': model_path + "DAE/microwave.h5"
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "results_folder" : base_path + "results/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                #"load_model_path" : no_transfer_path + "Seq2Point/",
                "appliances" : {
                    "microwave" : {
                        "on_threshold" : 200,
                        'transfer_path': model_path + "Seq2Point/microwave.h5"
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                "verbose" : 2,
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "results_folder" : base_path + "results/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                #"load_model_path" : no_transfer_path + "Seq2Seq/",
                "appliances" : {
                    "microwave" : {
                        "on_threshold" : 200,
                        'transfer_path': model_path + "Seq2Seq/microwave.h5"
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_path" : no_transfer_path + "ResNet/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 64,
                        'on_treshold' : 200,
                        'transfer_path': model_path + "ResNet/microwave.h5"
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_path" : no_transfer_path + "DeepGRU/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 128,
                        'on_treshold' : 200,
                        'transfer_path': model_path + "DeepGRU/microwave.h5"
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_path" : no_transfer_path + "MLP/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 200,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP/microwave.h5"
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_path" : no_transfer_path + "MLP_Raw/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 200,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP_Raw/microwave.h5"
                    }     
                },
            }),
        },
        'train': {   
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-06-30",
                            'end_time': "2014-11-01"
                        }       
                    }
                },
            }
        },
        'cross_validation': {    
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-11-01",
                            'end_time': "2014-11-12"
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        1: {
                            'start_time': "2013-03-18",
                            'end_time': "2015-04-25",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-10",
                        }
                    }
                },
            },
            'metrics':['mae', 'rmse',  'nrmse']
        }
    }

    ### Training and testing microwave ####
    results = API(microwave)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_dish_washer(base_path, timestep, epochs, batch_size, sequence_length, model_path):
    dish_washer = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['dish washer'],
        "use_activations" : True,
        "appliances_params" : {
            "dish washer" : {
                "min_off_time" : 1800,
                "min_on_time" : 1800,
                "number_of_activation_padding": 250,
                "min_on_power" : 10
            }
        },
        'methods': {
            'DAE':DAE({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "results_folder" : base_path + "results/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                #"load_model_path" : no_transfer_path + "DAE/",
                "appliances" : {
                    "dish washer" : {
                        "on_threshold" : 10,
                        'transfer_path': model_path + "DAE/dish_washer.h5"
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "results_folder" : base_path + "results/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                #"load_model_path" : no_transfer_path + "Seq2Point/",
                "appliances" : {
                    "dish washer" : {
                        "on_threshold" : 10,
                        'transfer_path': model_path + "Seq2Point/dish_washer.h5"
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                "verbose" : 2,
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "results_folder" : base_path + "results/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                #"load_model_path" : no_transfer_path + "Seq2Seq/",
                "appliances" : {
                    "dish washer" : {
                        "on_threshold" : 10,
                        'transfer_path': model_path + "Seq2Seq/dish_washer.h5"
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_path" : no_transfer_path + "ResNet/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 64,
                        'on_treshold' : 10,
                        'transfer_path': model_path + "ResNet/dish_washer.h5"
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_path" : no_transfer_path + "DeepGRU/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 128,
                        'on_treshold' : 10,
                        'transfer_path': model_path + "DeepGRU/dish_washer.h5"
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_path" : no_transfer_path + "MLP/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 10,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP/dish_washer.h5"
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_path" : no_transfer_path + "MLP_Raw/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 10,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP_Raw/dish_washer.h5"
                    }     
                },
            }),
        },
        'train': {   
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-06-30",
                            'end_time': "2014-11-01"
                        }        
                    }
                },
            }
        },
        'cross_validation': {    
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-11-01",
                            'end_time': "2014-11-12"
                        }          
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        1: {
                            'start_time': "2013-04-01",
                            'end_time': "2013-12-01",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-08-17",
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse',  'nrmse']
        }
    }

    ### Training and testing dish washer ####
    results = API(dish_washer)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_washing_machine(base_path, timestep, epochs, batch_size, sequence_length, model_path):
    washing_machine = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['washing machine'],
        "use_activations" : True,
        "appliances_params" : {
            "washing machine" : {
                "min_off_time" : 160,
                "min_on_time" : 1800,
                "number_of_activation_padding": 200,
                "min_on_power" : 20
            }
        },
        'methods': {
            'DAE':DAE({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "results_folder" : base_path + "results/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                #"load_model_path" : no_transfer_path + "DAE/",
                "appliances" : {
                    "washing machine" : {
                        "on_threshold" : 20,
                        'transfer_path': model_path + "DAE/washing_machine.h5"
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                "verbose" : 2,
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "results_folder" : base_path + "results/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                #"load_model_path" : no_transfer_path + "Seq2Point/",
                "appliances" : {
                    "washing machine" : {
                        "on_threshold" : 20,
                        'transfer_path': model_path + "Seq2Point/washing_machine.h5"
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                "verbose" : 2,
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "results_folder" : base_path + "results/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                #"load_model_path" : no_transfer_path + "Seq2Seq/",
                "appliances" : {
                    "washing machine" : {
                        "on_threshold" : 20,
                        'transfer_path': model_path + "Seq2Seq/washing_machine.h5"
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_path" : no_transfer_path + "ResNet/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 64,
                        'on_treshold' : 20,
                        'transfer_path': model_path + "ResNet/washing_machine.h5"
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_path" : no_transfer_path + "DeepGRU/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 128,
                        'on_treshold' : 20,
                        'transfer_path': model_path + "DeepGRU/washing_machine.h5"
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_path" : no_transfer_path + "MLP/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 20,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP/washing_machine.h5"
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_path" : no_transfer_path + "MLP_Raw/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 20,
                        "n_nodes" : 1024,
                        'transfer_path': model_path + "MLP_Raw/washing_machine.h5"
                    }     
                },
            }),
        },
        'train': {   
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-06-30",
                            'end_time': "2014-11-01"
                        } 
                    }
                },
            }
        },
        'cross_validation': {    
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        5: {
                            'start_time': "2014-11-01",
                            'end_time': "2014-11-12"
                        }          
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': ukdale_dataset,
                    'buildings': {
                        1: {
                            'start_time': "2013-03-18",
                            'end_time': "2013-08-18",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-10",
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse',  'nrmse']
        }
    }

    ### Training and testing washing machine ####
    results = API(washing_machine)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

if __name__ == "__main__":
    '''
    If you want to run the experiments without doing transfer learning
    you need to uncoment the  (#"load_model_path" : ...) lines.
    '''

    epochs = 30
    batch_size = 512
    sequence_length = 299
    timestep = 6
    

    #base_path = "./no_transfer_results/ukdale/"
    base_path = "./transfer_results/ukdale/"
    models_path = "./base_train/refit_train/models/"
    
    for i in range(1):
        run_fridge(base_path, timestep, epochs, batch_size, sequence_length, models_path)
        run_microwave(base_path, timestep, epochs, batch_size, sequence_length, models_path)
        run_dish_washer(base_path, timestep, epochs, batch_size, sequence_length, models_path)
        run_kettle(base_path, timestep, epochs, batch_size, sequence_length, models_path)
        run_washing_machine(base_path, timestep, epochs, batch_size, sequence_length, models_path)