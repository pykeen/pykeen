
import matplotlib
import matplotlib.pyplot as plt
matplotlib.matplotlib_fname()
import numpy as np
import os
import json

def plot_losses(losses, output_direc):
    out_path = os.path.join(output_direc, 'losses.png')
    epochs = np.arange(len(losses))
    plt.title(r'Loss Per Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.plot(epochs, losses)
    plt.savefig(out_path)

if __name__ == '__main__':
    input_path = '/Users/mehdi/PycharmProjects/kg_embeddings_pipeline/data/experiments/server/28-09-2018_03:20:45/losses.json'
    output_direc = '/Users/mehdi/PycharmProjects/kg_embeddings_pipeline/data/experiments/server/28-09-2018_03:20:45'

    with open(input_path) as json_data:
        losses = json.load(json_data)

    plot_losses(losses=losses, output_direc=output_direc)
