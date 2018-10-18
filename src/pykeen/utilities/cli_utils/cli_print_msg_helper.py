# -*- coding: utf-8 -*-

"""PyKEEN's command line interface helper."""

import click


def print_section_divider():
    click.secho(
        '--------------------------------------------------------------------------------------------------------')


def print_welcome_message():
    click.secho('#################################################')
    click.secho("#\t\tWelcome to " + click.style("PyKEEN", bold=True) + "\t\t#")
    click.secho('#################################################')


def print_intro():
    click.secho("This interface will assist you to configure your experiment.")
    click.secho("")
    click.secho(
        "PyKEEN can be run in two modes: \n"
        "1.) Training mode: PyKEEN trains a model based on a set of user-defined hyper-parameters.\n"
        "2.) Hyper-parameter optimization mode: "
        "Apply Random Search to determine the most appropriate set of hyper-parameter values"
    )


def print_existing_config_message():
    click.secho(
        "Here you are asked whether you have already an exisiing configuration that was created for a previous experiment.\n"
        "The configuration is saved as a JSON file (.json)\n")
    click.secho("Example of a valid path: /Users/david/data/configuration.json")
    click.secho("")


def print_training_set_message():
    click.secho(click.style("Current Step: Please provide the path to your training data.", fg='blue'))
    click.secho("")
    click.secho("The training file must be a TSV file (.tsv) containing three columns:")
    click.secho("Column 1: Contains the subjects of the triples.")
    click.secho("Column 2: Contains the predicates of the triples.")
    click.secho("Column 3: Contains the objects of the triples.")
    click.secho("")
    click.secho("Example of a valid path: \"/Users/david/data/corpora/fb15k/fb_15k_train.tsv\"")
    click.secho("")


def print_execution_mode_message():
    click.secho(click.style(
        "Current Step: Please choose the execution mode: training mode (1) or hyper-parameter search mode (2)",
        fg='blue'))
    click.secho("")


def print_model_selection_message():
    click.secho(click.style("Current Step: Please choose one of the provide models.", fg='blue'))
    click.secho(
        "Depending on which model you select, PyKEEN will assist you to configure the required hyper-parameters.")
    click.secho("")


def print_training_embedding_dimension_message():
    click.secho(click.style(
        "Current Step: Please specify the embedding dimension to use for learning the entities and relations.",
        fg='blue'))
    click.secho("The embedding dimension must be a positive integer e.g. 20.")
    click.secho("")


def print_embedding_dimension_info_message():
    click.secho("For the selected model the embedding dimension of entities and relations are the same.")
    click.secho("")


def print_training_margin_loss_message():
    click.secho(click.style(
        "Current Step: Please specify the value of the margin loss used for the margin-ranking-loss function.",
        fg='blue'))
    click.secho("The margin ranking loss is a float value. An example for the margin loss is a value of 1")
    click.secho("")


def print_scoring_fct_message():
    click.secho(click.style(
        "Current Step: Please specify the norm used as scoring function.", fg='blue'))
    click.secho("The norm should be a positive integer value such as a value of 1")
    click.secho("")


def print_entity_normalization_message():
    click.secho(click.style("Current Step: Please specify the norm used to normalize the entities.", fg='blue'))
    click.secho("The norm should be a positive integer value such as a value of 2")
    click.secho("")


def print_learning_rate_message():
    click.secho(click.style("Current Step: Please specify the learning rate.", fg='blue'))
    click.secho("The learning rate should be a positive float value such as 0.01")
    click.secho("")


def print_batch_size_message():
    click.secho(click.style("Current Step: Please specify the batch size.", fg='blue'))
    click.secho("Typical batch sizes are 32,64 and 128")
    click.secho("")


def print_number_epochs_message():
    click.secho(click.style("Current Step: Please specify the number of epochs", fg='blue'))
    click.secho(
        "The number of epochs defines how often to iterte over the whole training set during the training the model.")
    click.secho("")


def print_ask_for_evlauation_message():
    click.secho(click.style("Current Step: Please specify the number of epochs", fg='blue'))
    click.secho("Here you can specify whether you want to evaluate your model after training or not.")
    click.secho("")


def print_test_set_message():
    click.secho(
        click.style("Current Step: Please specify whether you provide a test set yourself, or whether the test set\n"
                    "should be randomly extracted from the training set.", fg='blue'))
    click.secho("")


def print_test_ratio_message():
    click.secho(
        click.style("Current Step: Please specify the ratio of the training set that should be used as a test set",
                    fg='blue'))
    click.secho('For example 0.5 means half of the training set is used as a test set.')
    click.secho("")


def print_filter_negative_triples_message():
    click.secho(
        click.style("Current Step: Please specify whether you want to filter negative triples out during evaluation.",
                    fg='blue'))
    click.secho('Filtered evaluation is more expressive, for further information we refer to \n'
                'Bordes et al. \"Translating embeddings for modeling multi-relational data.\"')
    click.secho("")


def print_output_directory_message():
    click.secho(
        click.style("Current Step: Please specify the path to your output directory.", fg='blue'))
    click.secho("Example of a valid path: /Users/david/output_direc")
    click.secho("")


def print_trans_h_soft_constraints_weight_message():
    click.secho(
        click.style("Current Step: Please specify the weight value for the soft constraints.", fg='blue'))
    click.secho('In TransH, soft constraints are introduced and incorporated into the loss function.\n'
                'For further information we refer to Wang, Zhen, et al. \"Knowledge Graph Embedding by Translating on Hyperplanes')
    click.secho("")


def print_entities_embedding_dimension_message():
    click.secho(
        click.style(
            "Current Step: Please specify the embedding dimension to use for learning the entities of the knowedge graph.",
            fg='blue'))
    click.secho("")


def print_relations_embedding_dimension_message():
    click.secho(
        click.style(
            "Current Step: Please specify the embedding dimension to use for learning the relations of the knowedge graph.",
            fg='blue'))
    click.secho("")


def print_conv_width_height_message():
    click.secho("In ConvE, the input is of the CNN is the embedding of the head and of the relation.\n"
                "Those are transformed into an \"image\" representation. The constraint is that height*width must equal\n"
                "to the embedding dimension. If the embedding dimension is for example 100, then valied values for\n"
                "height and width are 5 and 20 since 5*20 = 100.")
    click.secho("")


def print_conv_input_channels_message():
    click.secho(
        click.style("Current Step: Please specify the number of input channels for the convolutional layer.",
                    fg='blue'))
    click.secho("")


def print_conv_kernel_height_message():
    click.secho(click.style("Current Step: Please specify the height of the convolution kernel.", fg='blue'))
    click.secho("Important note: The kernel height must be smaller or equal to the input height, specified before.")
    click.secho("")


def print_conv_kernel_width_message():
    click.secho(click.style("Current Step: Please specify the width of the convolution kernel.", fg='blue'))
    click.secho("Important note: The kernel width must be smaller or equal to the input width, specified before.")
    click.secho("")


def print_input_dropout_message():
    click.secho(click.style("Current Step: Please specify the dropout rate for the input layer.", fg='blue'))
    click.secho("The dropout rate must be a value between 0 and 1")
    click.secho("")


def print_hpo_embedding_dimensions_message():
    click.secho(
        click.style("Current Step: Please specify a list of embedding dimensions for the entities and relations.",
                    fg='blue'))
    click.secho("You can also provide just a single value, in this case the \'\' is not required.")
    click.secho("")


def print_hpo_margin_losses_message():
    click.secho(
        click.style("Current Step: Please provide a list of margin losses to use for the margin-ranking-loss function",
                    fg='blue'))
    click.secho("The margin ranking losses need to be float values. Please separate your input by a \',\':\n"
                "0.5,1,2.4\n")
    click.secho("")


def print_hpo_scoring_fcts_message():
    click.secho(click.style("Current Step: Please provide a list of norms used as scoring function", fg='blue'))
    click.secho("The norms should be positive integer values. Please separate your input by a \',\':\n"
                "1,2,3\n")
    click.secho("")


def print_hpo_entity_normalization_norms_message():
    click.secho(
        click.style("Current Step: Please provide a list of norms used used for normalizing the entities.", fg='blue'))
    click.secho("The norms should be positive integer values. Please separate your input by a \',\':\n"
                "1,2,3\n")
    click.secho("")


def print_hpo_learning_rates_message():
    click.secho(click.style("Current Step: Please provide a list of learning rates", fg='blue'))
    click.secho("The learning rates need to be float values. Please separate your input by a \',\':\n"
                "0.1,0.01,0.001\n")
    click.secho("")


def print_hpo_batch_sizes_message():
    click.secho(click.style("Current Step: Please provide a list of batch sizes.", fg='blue'))
    click.secho("The batch sizes should be positive integer values. Please separate your input by a \',\':\n"
                "1,2,3\n")
    click.secho("")


def print_hpo_epochs_message():
    click.secho(click.style("Current Step: Please provide a list of epochs.", fg='blue'))
    click.secho("The epochs should be positive integer values. Please separate your input by a \',\':\n"
                "1,2,3\n")
    click.secho("")


def print_hpo_iterations_message():
    click.secho(click.style("Current Step: Please provide a list of epochs.", fg='blue'))
    click.secho("The epochs should be positive integer values. Please separate your input by a \',\':\n"
                "1,2,3\n")
    click.secho("")

def print_hpo_trans_h_soft_constraints_weights_message():
    click.secho(
        click.style("Current Step: Please provide a list of weight values for the soft constraints.", fg='blue'))
    click.secho('In TransH, soft constraints are introduced and incorporated into the loss function.\n'
                'For further information we refer to Wang, Zhen, et al. \"Knowledge Graph Embedding by Translating on Hyperplanes')
    click.secho("")