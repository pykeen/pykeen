from keen.constants import KG_MODEL_PAPER_INFO_MAPPING


def print_section_divider():
    print('--------------------------------------------------------------------------------------------------------')


def print_welcome_message():
    print('#################################################')
    print('#\t\t\t\t\t\t#')
    print('#\t\tWelcome to KEEN\t\t\t#')
    print('#\t\t\t\t\t\t#')
    print('##################################################')
    print()


def print_intro():
    print("The interactive command line interface will assist you to configure your experiment.")
    print()
    print("KEEN can be run in two modes: \n\t 1.) Training mode \n\t 2.) Hyper-parameter optimization (HPO) mode")
    print()
    print("In training mode KEEN trains a model based on a set of user-defined hyper-parameters.")
    print()
    print("In HPO mode KEEN the user defines for each hyper-parameter a set of possible values, and KEEN runs \n"
          "Random Search to determine the most appropriate set of hyper-parameter values")


def print_training_set_message():
    print("Here you are asked to provide the path to your training data.")
    print()
    print("The training file must be a TSV file (.tsv) containing three columns:")
    print("\t Column 1: Contains the subjects of the triples.")
    print("\t Column 2: Contains the predicates of the triples.")
    print("\t Column 3: Contains the objects of the triples.")
    print()
    print("Example of a valid path: /Users/david/data/corpora/fb15k/fb_15k_train.tsv")
    print()


def print_execution_mode_message():
    print("Here you are asked, whether to run KEEN in training mode or in hyper-parameter search mode.")
    print()
    print("In training mode, you are asked to provide for each hyper-parameter one single value.")
    print()


def print_model_selection_message():
    print("Here you are asked to choose from of the models provided in KEEN.")
    print("Depending on which model you select, KEEN will assist you to configure the required hyper-parameters.")
    print()

def print_training_embedding_dimension_message():
    print("Here you are asked to specify the embedding dimension to use for learning the entities and relations\n"
          "of the knowedge graph. The embedding dimension must be a positive integer e.g. 20.")
    print()

def print_trans_e_embedding_dimension_info_message():
    print("In TransE the embedding dimension of entiteis and relations are the same.")
    print()

def print_training_margin_loss_message():
    print("Here you are asked to specify the value of the margin loss used for the margin-ranking-loss function.")
    print("The margin ranking loss is a float value. An example for the margin loss is a value of 1")
    print()


def print_scoring_fct_message():
    print("Here you asked to specify the norm used as scoring function.")
    print("The norm should be a positive integer value such as a value of 1")
    print()

def print_entity_normalization_message():
    print("Here you asked to specify the norm used to normalize the entities.")
    print("The norm should be a positive integer value such as a value of 1")
    print()

def print_learning_rate_message():
    print("Here you are asked to specify the learning rate.")
    print("The learning rate should be a positive float value such as 0.01")
    print()

def print_batch_size_message():
    print("Here you are asked to specify the batch size.")
    print("Typical batch sizes are 32,64 and 128")
    print()

def print_number_epochs_message():
    print("Here you are asked to specify the number of epochs.")
    print("The number of epochs defines how often to iterte over the whole training set during the training the model.")
    print()

def ask_for_evlauation_message():
    print("Here you can specify whether you want to evaluate your model after training or not.")
    print()

def test_set_message():
    print("Here you can specify whether you provide a test set yourself, or whether the test set\n"
          "should be randomly extracted from training set.")
    print()
