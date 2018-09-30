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


if __name__ == '__main__':
    print_welcome_message()
    print_section_divider()
    print_intro()
    print_section_divider()
    print_training_set_message()
    print_section_divider()
    print_execution_mode_message()
    print_section_divider()
    print_model_selection_message()
