import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 10
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=16)

intrees = pd.read_csv('./results/global_result/intrees/experiment.csv')
defrag = pd.read_csv('./results/global_result/defrag/experiment.csv')
te2rules = pd.read_csv('./results/global_result/te2rules/experiment.csv')
file_extension = ".jpg"


for dataset in ["compas", "bank", "adult"]:
    for depth in [3, 5]:
        plt.figure()
        plt.title(dataset)        
        plt.xscale("log")            
        for num_trees in [100, 200, 500]:
            exp_name = "_".join([str(dataset), str(depth)])
            color = {100:"r", 200:"g", 500:"b"}

            te2rules_to_plot = te2rules[te2rules.dataset == dataset]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.num_trees == num_trees]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.depth == depth]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.jaccard_threshold == 0.2]            
            assert(len(te2rules_to_plot) == 3)
            te2rules_1_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 1]            
            te2rules_2_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            
            te2rules_3_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            

            intrees_to_plot = intrees[intrees.dataset == dataset]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.num_trees == num_trees]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.depth == depth]
            assert(len(intrees_to_plot) == 1)

            defrag_to_plot = defrag[defrag.dataset == dataset]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.num_trees == num_trees]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.depth == depth]
            assert(len(defrag_to_plot) == 3)

            X_metric = "num_rules"
            Y_metric = "test_fidelity_positive"
            plt.scatter(te2rules_to_plot[X_metric], te2rules_to_plot[Y_metric], marker='^', c=color[num_trees], label="te2rules")
            plt.scatter(intrees_to_plot[X_metric], intrees_to_plot[Y_metric], marker='o', c=color[num_trees], label="intrees")
            plt.scatter(defrag_to_plot[X_metric], defrag_to_plot[Y_metric], marker='+', c=color[num_trees], label="defragtrees")
        plt.ylim([0, 1])
        plt.xlabel("number of rules")
        plt.ylabel("fidelity (positives)")
        plt.savefig("./plots/performance/fidelity_positive/" + exp_name + file_extension)
        plt.close()

for dataset in ["compas", "bank", "adult"]:
    for depth in [3, 5]:
        plt.figure()
        plt.title(dataset)        
        plt.xscale("log")            
        for num_trees in [100, 200, 500]:
            exp_name = "_".join([str(dataset), str(depth)])
            color = {100:"r", 200:"g", 500:"b"}

            te2rules_to_plot = te2rules[te2rules.dataset == dataset]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.num_trees == num_trees]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.depth == depth]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.jaccard_threshold == 0.2]            
            assert(len(te2rules_to_plot) == 3)
            te2rules_1_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 1]            
            te2rules_2_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            
            te2rules_3_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            

            intrees_to_plot = intrees[intrees.dataset == dataset]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.num_trees == num_trees]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.depth == depth]
            assert(len(intrees_to_plot) == 1)

            defrag_to_plot = defrag[defrag.dataset == dataset]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.num_trees == num_trees]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.depth == depth]
            assert(len(defrag_to_plot) == 3)

            X_metric = "num_rules"
            Y_metric = "test_fidelity_total"
            plt.scatter(te2rules_to_plot[X_metric], te2rules_to_plot[Y_metric], marker='^', c=color[num_trees])
            plt.scatter(intrees_to_plot[X_metric], intrees_to_plot[Y_metric], marker='o', c=color[num_trees])
            plt.scatter(defrag_to_plot[X_metric], defrag_to_plot[Y_metric], marker='+', c=color[num_trees])
        plt.ylim([0, 1])
        plt.xlabel("number of rules")
        plt.ylabel("fidelity (overall)")
        plt.savefig("./plots/performance/fidelity_total/" + exp_name + file_extension)
        plt.close()

for dataset in ["compas", "bank", "adult"]:
    for depth in [3, 5]:
        plt.figure()
        plt.title(dataset)        
        plt.xscale("log")            
        for num_trees in [100, 200, 500]:
            exp_name = "_".join([str(dataset), str(depth)])
            color = {100:"r", 200:"g", 500:"b"}

            te2rules_to_plot = te2rules[te2rules.dataset == dataset]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.num_trees == num_trees]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.depth == depth]
            te2rules_to_plot = te2rules_to_plot[te2rules_to_plot.jaccard_threshold == 0.2]            
            assert(len(te2rules_to_plot) == 3)
            te2rules_1_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 1]            
            te2rules_2_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            
            te2rules_3_to_plot = te2rules_to_plot[te2rules_to_plot.num_stages == 2]            

            intrees_to_plot = intrees[intrees.dataset == dataset]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.num_trees == num_trees]
            intrees_to_plot = intrees_to_plot[intrees_to_plot.depth == depth]
            assert(len(intrees_to_plot) == 1)

            defrag_to_plot = defrag[defrag.dataset == dataset]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.num_trees == num_trees]
            defrag_to_plot = defrag_to_plot[defrag_to_plot.depth == depth]
            assert(len(defrag_to_plot) == 3)

            X_metric = "time"
            Y_metric = "test_fidelity_positive"
            plt.scatter(te2rules_to_plot[X_metric], te2rules_to_plot[Y_metric], marker='^', c=color[num_trees])
            plt.scatter(intrees_to_plot[X_metric], intrees_to_plot[Y_metric], marker='o', c=color[num_trees])
            plt.scatter(defrag_to_plot[X_metric], defrag_to_plot[Y_metric], marker='+', c=color[num_trees])
        plt.ylim([0, 1])
        plt.xlabel("runtime (seconds)")
        plt.ylabel("fidelity (positives)")
        plt.savefig("./plots/performance/time/" + exp_name + file_extension)
        plt.close()
