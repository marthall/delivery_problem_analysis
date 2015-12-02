
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from random import random
from random import shuffle
from matplotlib import cm
from math import log

from collections import defaultdict

max_vehicle_size = defaultdict(list)
max_vehicle_size_probdist_norm = defaultdict(list)
max_vehicle_size_probdist_norm_size_norm = defaultdict(list)
max_vehicle_size_probdist_norm_logsize_norm = defaultdict(list)


max_prob_dist_avg = 0
max_distance_avg = 0

for (dirpath, dirname, filenames) in os.walk('vehicle_results'):
    for f in filenames:
        fpath = os.path.join(dirpath, f)

        # print fname
        
        with open(fpath, "r") as f:
            j = json.load(f, encoding="latin-1")
            
            probability_matrix = np.array(j['task_distribution']['probability_matrix'])
            distance_matrix = np.array(j['task_distribution']['distance_matrix'])
            prob_distance = np.multiply(probability_matrix, distance_matrix)
            
            distance_avg = np.average(distance_matrix)
            prob_distance_avg = np.average(prob_distance)   

            if distance_avg > max_distance_avg:
                max_distance_avg = distance_avg

            if prob_distance_avg > max_prob_dist_avg:
                max_prob_dist_avg = prob_distance_avg

            for run in j["runs"]:
                max_size = max(j["vehicle_capacities"][:run["number_of_vehicles"]])

                max_vehicle_size[max_size].append(run["avg_task_cost_vs_number_of_task"])
                max_vehicle_size_probdist_norm[max_size].append(run["avg_task_cost_vs_number_of_task"] / (prob_distance_avg ** 0.9))
                max_vehicle_size_probdist_norm_size_norm[max_size].append(run["avg_task_cost_vs_number_of_task"] / (prob_distance_avg ** 0.9) * max_size**0.62)

                max_vehicle_size_probdist_norm_logsize_norm[max_size].append(run["avg_task_cost_vs_number_of_task"] / (prob_distance_avg ** 0.9) * log(max_size)**1.4)

print max_distance_avg
print max_prob_dist_avg

# plot avg cost compared to number of tasks

# avg_cost = []

# for key, item in max_vehicle_size.iteritems():
#     avg_cost += [key, np.average(item, axis=0)[10]]

# plt.figure()
# for e in np.array(avg_cost).reshape(-1,2):

#     plt.scatter(e[0], e[1])
# plt.show()


for task_no in [10,20,30]:

    cost_list = []

    for key in sorted(max_vehicle_size):
        cost_list.append(np.array(max_vehicle_size[key])[:,task_no-1])

    plt.figure()
    plt.boxplot(cost_list, positions=[5,10,15,20,25,30])
    plt.savefig("output_vehicle_size/boxplot_size_vs_average_cost_for_%d_tasks.png" % task_no)


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

    # Legend order
    newLabels, newHandles = zip(*sorted(zip(map(int, newLabels), newHandles)))
    
    leg = plt.legend(newHandles, newLabels)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
        legobj.set_alpha(1)

    plt.savefig("output_vehicle_size/%s.png" % dict_name)
    # plt.show()

plot("max_vehicle_size", max_vehicle_size)
plot("max_vehicle_size_probdist_norm", max_vehicle_size_probdist_norm)
plot("max_vehicle_size_probdist_norm_size_norm", max_vehicle_size_probdist_norm_size_norm)
plot("max_vehicle_size_probdist_norm_logsize_norm", max_vehicle_size_probdist_norm_logsize_norm)
