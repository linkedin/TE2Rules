import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=16)

intrees = pd.read_csv('./results/global_result/intrees/experiment.csv')
defrag = pd.read_csv('./results/global_result/defrag/experiment.csv')
te2rules = pd.read_csv('./results/global_result/te2rules/experiment.csv')
file_extension = ".jpg"

linestyle = {3:"-", 5:"--"}
color = {100:"r", 200:"g", 500:"b"}

X_metric = "num_stages"
Y_metric = "time"
for dataset in ["compas", "bank", "adult"]:
    te2rules_dataset = te2rules[te2rules.dataset == dataset]
    te2rules_dataset = te2rules_dataset[te2rules_dataset.jaccard_threshold == 0.2]            
    assert(len(te2rules_dataset) == 3*3*2)

    plt.figure()
    plt.title(dataset)        
    for num_trees in [100, 200, 500]:
        for depth in [3, 5]:
            exp_name = str("(") + str(num_trees) + ", " + str(depth) + ")"

            te2rules_to_plot = te2rules_dataset[te2rules_dataset.num_trees == num_trees]            
            te2rules_to_plot = te2rules_to_plot[te2rules_dataset.depth == depth]            
            plt.plot(te2rules_to_plot[X_metric], te2rules_to_plot[Y_metric], 
                     marker='o', 
                     linestyle=linestyle[depth], 
                     color=color[num_trees], 
                     label=exp_name)
            plt.yscale("log")
            plt.ylabel("runtime (seconds)")
            plt.xlabel("stages")
            plt.xticks([1,2,3])
    plt.legend()
    plt.savefig("./plots/scalability/" + dataset + "_time" + file_extension)
    plt.close()

X_metric = "num_stages"
Y_metric = "test_fidelity_positive"
for dataset in ["compas", "bank", "adult"]:
    te2rules_dataset = te2rules[te2rules.dataset == dataset]
    te2rules_dataset = te2rules_dataset[te2rules_dataset.jaccard_threshold == 0.2]            
    assert(len(te2rules_dataset) == 3*3*2)

    plt.figure()        
    plt.title(dataset)        
    for num_trees in [100, 200, 500]:
        for depth in [3, 5]:
            exp_name = str("(") + str(num_trees) + ", " + str(depth) + ")"

            te2rules_to_plot = te2rules_dataset[te2rules_dataset.num_trees == num_trees]            
            te2rules_to_plot = te2rules_to_plot[te2rules_dataset.depth == depth]            
            plt.plot(te2rules_to_plot[X_metric], te2rules_to_plot[Y_metric], 
                     marker='o', 
                     linestyle=linestyle[depth], 
                     color=color[num_trees], 
                     label=exp_name)
            plt.ylabel("fidelity (positives)")
            plt.xlabel("number of stages")
            plt.xticks([1,2,3])
            plt.ylim([0, 1])
    plt.legend()
    plt.savefig("./plots/scalability/" + dataset + "_fidelity_positive" + file_extension)
    plt.close()

