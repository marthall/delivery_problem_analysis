
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from random import random
from random import shuffle

number_of_vehicles = {2: [], 3: [], 4: [], 5: []}
countries = {"switzerland": [], "england": [], "the_netherlands": [], "france": []}

countries_prob_dist = {"switzerland": [], "england": [], "the_netherlands": [], "france": []}
countries_norm_dist = {"switzerland": [], "england": [], "the_netherlands": [], "france": []}

cost_vs_dist = np.array([])
cost_vs_probdist = np.array([])


for (dirpath, dirname, filenames) in os.walk('data'):
    for f in filenames:
        fpath = os.path.join(dirpath, f)

        country, _ = f.split(".")
        country = country.split("_")
        country = "_".join(country[:-2])
        # print fname
        
        f = open(fpath)
        j = json.load(f, encoding="latin-1")
        runs = j["runs"]

        probability_matrix = np.array(j['task_distribution']['probability_matrix'])
        distance_matrix = np.array(j['distance_matrix'])

        prob_distance = np.multiply(probability_matrix, distance_matrix)

        prob_distance_avg = np.average(prob_distance)

        sum_distance = np.sum(distance_matrix)



        for run in runs:
            number_of_vehicles[run["number_of_vehicles"]].append(run["avg_task_cost_vs_number_of_task"])
            countries[country].append(run["avg_task_cost_vs_number_of_task"])
            
            relative_cost = np.array(run["avg_task_cost_vs_number_of_task"]) / (prob_distance_avg ** 0.9)
            countries_prob_dist[country].append(relative_cost)
            
            norm_sum = np.array(run["avg_task_cost_vs_number_of_task"]) / sum_distance
            countries_norm_dist[country].append(norm_sum)

            # cost_vs_dist = np.concatenate((cost_vs_dist, [run["avg_task_cost_vs_number_of_task"][20], sum_distance]))
            # cost_vs_probdist = np.concatenate((cost_vs_probdist, [run["avg_task_cost_vs_number_of_task"][20], prob_distance_avg]))

# print number_of_vehicles




# plot avg cost compared to number of tasks

def plot(dict_name, dataset, average=False):
    fig = plt.figure()
    plt.title(dict_name)
    ax1 = fig.add_subplot(111)
    cs = colors.cnames.keys()
    cdict = {"france": cs[0], "england": cs[1], "the_netherlands": cs[2], "switzerland": cs[3]}

    # To avoid one series totally overwriting another
    data_array = [(key, dataset[key][i]) for key in dataset.keys() for i in range(len(dataset[key]))]
    shuffle(data_array)

    for key, item in data_array[:100]:

        avg = np.average(np.array(item), axis=0)

        c = cdict[key]
        # print key, c

        x = np.arange(len(item)) + 1
        if random() < 0.00:
            ax1.plot(x, item, color=c, alpha=0.6, lw=1, label=key)
        else:
            ax1.plot(x, item, color=c, alpha=0.5, label=key)
        # ax1.scatter(x, item, color=c, s=2, alpha=0.3)

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

    plt.savefig("output/%s.png" % dict_name)
    # plt.show()

# plot("number_of_vehicles", number_of_vehicles)
plot("countries_prob_dist", countries_prob_dist)
plot("countries_norm_dist", countries_norm_dist)
# plot("countries", countries)

# print f
    # for f in filenames:
    #   print dirpath + f
    # f = open(filename)
    # j = json.load(f)
    # print j


data = np.concatenate(countries_prob_dist.values())
dy, dx = data.shape


# print averages.shape

# print dy, dx


# print curve



### Exponential fit
from scipy.optimize import curve_fit

xdata = np.arange(data.shape[1]) + 1
ydata = np.average(data, axis=0)

# def func(x, a, b, c):
#   return a * np.exp(-b * x) + c

# popt, pcov = curve_fit(func, xdata, ydata)

# print popt, pcov

std = np.std(data, axis=0)

minerr = ydata - std
maxerr = ydata + std

strategy = np.concatenate((np.repeat(np.array(maxerr[9]), 9), maxerr[9:]))

plt.figure()
# plt.plot(xdata, func(xdata, popt[0], popt[1], popt[2]))
plt.plot(xdata, ydata, label="mean")
plt.plot(xdata, minerr, label="mean - std")
plt.plot(xdata, maxerr, label="mean + std")
plt.plot(xdata, strategy, lw=3, ls="--", c="black", label="strategy")

box = plt.boxplot(data)

plt.legend()
plt.axis([0, 30, 0, maxerr[0]])
plt.savefig('output/relative_cost_boxplot.png')


# plt.figure()
# for cost, dist in np.reshape(cost_vs_dist, (-1, 2)):
#   plt.scatter(dist, cost)

# # plt.show()


# plt.figure()
# for cost, dist in np.reshape(cost_vs_probdist, (-1, 2)):
#   plt.scatter(dist, cost)
# plt.xlabel("prob*dist")
# plt.ylabel("avg. cost")
# # plt.show()
