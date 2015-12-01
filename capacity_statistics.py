
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from random import random
from random import shuffle
from matplotlib import cm

from collections import defaultdict

max_vehicle_size = defaultdict(list)

for (dirpath, dirname, filenames) in os.walk('vehicle_results'):
    for f in filenames:
        fpath = os.path.join(dirpath, f)

        capacities = map(int, f.split(".")[0].split("=")[1].split("_")[0].split("+"))
        # print fname
        
        with open(fpath, "r") as f:
            j = json.load(f, encoding="latin-1")
            
            # probability_matrix = np.array(j['task_distribution']['probability_matrix'])
            # distance_matrix = np.array(j['task_distribution']['distance_matrix'])
            # prob_distance = np.multiply(probability_matrix, distance_matrix)
            # prob_distance_avg = np.average(prob_distance)
            # sum_distance = np.sum(distance_matrix)

            max_size = max(capacities)

            for run in j["runs"]:
                max_vehicle_size[max_size].append(run["avg_task_cost_vs_number_of_task"])
                

# plot avg cost compared to number of tasks

# avg_cost = []

# for key, item in max_vehicle_size.iteritems():
#     avg_cost += [key, np.average(item, axis=0)[10]]

# plt.figure()
# for e in np.array(avg_cost).reshape(-1,2):

#     plt.scatter(e[0], e[1])
# plt.show()

cost_list = []

for vsize_values in max_vehicle_size.values():
    cost_list.append(np.array(vsize_values)[:,29])

plt.figure()
plt.boxplot(cost_list, positions=[5,10,15,20,25,30])
plt.savefig("output_vehicle_size/boxplot_size_vs_average_cost_for_30_tasks.png")


def plot(dict_name, dataset):
    fig = plt.figure()
    plt.title(dict_name)
    ax1 = fig.add_subplot(111)
    cs = colors.cnames.keys()

    cdict = {key: cs[i] for i, key in enumerate(max_vehicle_size.keys())}

    # To avoid one series totally overwriting another
    data_array = [(key, dataset[key][i]) for key in dataset.keys() for i in range(len(dataset[key]))]
    shuffle(data_array)

    max_key = max(dataset.keys())

    for key, item in data_array:

        # avg = np.average(np.array(item), axis=0)
        c = cdict[key]
        x = np.arange(len(item)) + 1

        ax1.plot(x, item, color=cm.jet(1.*key/max_key), alpha=0.2, lw=1, label=key)

    # Remove dulplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    
    leg = plt.legend(newHandles, newLabels)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
        legobj.set_alpha(1)

    plt.savefig("output_vehicle_size/%s.png" % dict_name)
    # plt.show()

plot("max_vehicle_size", max_vehicle_size)
