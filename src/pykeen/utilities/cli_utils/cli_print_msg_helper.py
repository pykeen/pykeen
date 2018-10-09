def print_section_divider():
    print('--------------------------------------------------------------------------------------------------------')


def print_welcome_message():
    print('#################################################')
    print('#\t\t\t\t\t\t#')
    print('#\t\tWelcome to pykeen\t\t\t#')
    print('#\t\t\t\t\t\t#')
    print('#################################################')
    print()


def print_intro():
    print("The interactive command line interface will assist you to configure your experiment.")
    print()
    print("pykeen can be run in two modes: \n\t 1.) Training mode \n\t 2.) Hyper-parameter optimization (HPO) mode")
    print()
    print("In training mode pykeen trains a model based on a set of user-defined hyper-parameters.")
    print()
    print("In HPO mode the user defines for each hyper-parameter a set of possible values, and pykeen runs \n"
          "Random Search to determine the most appropriate set of hyper-parameter values")


def print_existing_config_message():
    print(
        "Here you are asked whether you have already an exisiing configuration that was created for a previous experiment.\n"
        "The configuration is saved as a JSON file (.json)\n")
    print("Example of a valid path: /Users/david/data/configuration.json")
    print()


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
    print("Here you are asked, whether to run pykeen in training mode or in hyper-parameter search mode.")
    print()
    print("In training mode, you are asked to provide for each hyper-parameter one single value.")
    print()


def print_model_selection_message():
    print("Here you are asked to choose from of the models provided in pykeen.")
    print("Depending on which model you select, pykeen will assist you to configure the required hyper-parameters.")
    print()


def print_training_embedding_dimension_message():
    print("Here you are asked to specify the embedding dimension to use for learning the entities and relations\n"
          "of the knowedge graph. The embedding dimension must be a positive integer e.g. 20.")
    print()


def print_embedding_dimension_info_message():
    print("For the selcted model the embedding dimension of entiteis and relations are the same.")
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
    print("The norm should be a positive integer value such as a value of 2")
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


def print_ask_for_evlauation_message():
    print("Here you can specify whether you want to evaluate your model after training or not.")
    print()


def print_test_set_message():
    print("Here you can specify whether you provide a test set yourself, or whether the test set\n"
          "should be randomly extracted from the training set.")
    print()


def print_test_ratio_message():
    print('Here you are asked to select the ratio of the training set that should be used as a test\n'
          '(e.g. 0.5 meaning half of the training set is used as a test set):')
    print()


def print_filter_negative_triples_message():
    print('Here you are asked to specify whether you want to filter negative triples out during evaluation.\n'
          'Filtered evaluation is more expressive, for further information we refer to \n'
          'Bordes et al. \"Translating embeddings for modeling multi-relational data.\"')
    print()


def print_output_directory_message():
    print('Here you are asked to specify the path to your output directory.\n'
          'In the output directory, the trained model, the learned embeddings, the evaluation results,\n'
          'the configuration are exported.\n')
    print("Example of a valid path: /Users/david/output_direc")
    print()


def print_trans_h_soft_constraints_weight_message():
    print('In TransH, soft constraints are introduced and incorporated into the loss function.\n'
          'The authors proposed to weight the soft constraints.\n'
          'For further information we refer to Wang, Zhen, et al. \"Knowledge Graph Embedding by Translating on Hyperplanes.\"\n'
          'Here you are asked, to specify the weight value for the soft constraints.\n')
    print("Example of a valid path: /Users/david/output_direc")
    print()


def print_entities_embedding_dimension_message():
    print(
        "Here you are asked to specify the embedding dimension to use for learning the entities of the knowedge graph.\n"
        "The embedding dimension must be a positive integer e.g. 20.")
    print()

def print_relations_embedding_dimension_message():
    print(
        "Here you are asked to specify the embedding dimension to use for learning the relations of the knowedge graph.\n"
        "The embedding dimension must be a positive integer e.g. 30.")
    print()

def print_conv_width_height_message():
    print("In ConvE, the input is of the CNN is the embedding of the head and of the relation.\n"
          "Those are transformed into an \"image\" representation. The constraint is that height*width must equal\n"
          "to the embedding dimension. If the embedding dimension is for example 100, then valied values for\n"
          "height and width are 5 and 20 since 5*20 = 100.")
    print()

def print_conv_input_channels_message():
    print("Here you are asked to provide the number of input channels for the convolutional layer.\n"
          "The number of input channels is an positive integer.")
    print()

def print_conv_kernel_height_message():
    print("Here you are asked to provide the height of the convolution kernel.\n"
          "Important note: The kernel height must be smaller or equal to the input height, specified before.")
    print()

def print_conv_kernel_width_message():
    print("Here you are asked to provide the width of the convolution kernel.\n"
          "Important note: The kernel width must be smaller or equal to the input width, specified before.")
    print()

def print_input_dropout_message():
    print("Here you are asked to provide the dropout rate for the input layer.\n"
          "The dropout rate must be a value between 0 and 1")
    print()

def print_hpo_embedding_dimensions_message():
    print("Here you are asked to specify a list of embedding dimensions for the entities and relations. \n"
          "The embedding dimensions need to be positive integers. Please separate your input by a \',\':\n"
          "50,100,200\n"
          "You can also provide just a single value, in this case the \'\' is not required."
          "")
    print()

def print_hpo_margin_losses_message():
    print("Here you are asked to specify a list of margin losses used for the margin-ranking-loss function.")
    print("The margin ranking losses need to be float values. Please separate your input by a \',\':\n"
          "0.5,1,2.4\n")
    print()

def print_hpo_scoring_fcts_message():
    print("Here you asked to specify a list of norms used as scoring function.")
    print("The norms should be positive integer values. Please separate your input by a \',\':\n"
          "1,2,3\n")
    print()


