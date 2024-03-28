################ ##
## Lorenzo Dall'Olio
## 15 dec 2022
## perform clustering on lower dimensional embedding of the soft thresholded correlation matrix from vst transofrmed data
################ ##


import os
import umap
import random
import hdbscan

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


sns.set(style="darkgrid")
plt.rcParams.update({'font.size': 18})


# QUASTIONS:
#    - make many functions or leave the script pipeline-styled

# =============================================================================
# COLORMAP for 20+ clusters cases
# =============================================================================

    
def make_discrete_large_colormap(n_colors=len(list(mcolors.CSS4_COLORS.keys())),
                                 random_state=42, start_with_black=False,
                                 remove_light_colors_threshold=0.5):
    """
    The function `make_discrete_large_colormap` generates a discrete colormap with more than 20 colors,
    potentially starting with black and removing light colors based on a threshold.
    
    :param n_colors: The `n_colors` parameter in the `make_discrete_large_colormap` function specifies
    the number of colors to include in the colormap that will be generated. By default, it is set to the
    total number of colors available in the CSS4_COLORS dictionary
    :param random_state: The `random_state` parameter is used to set the seed for the random number
    generator. Setting a specific `random_state` ensures reproducibility of the random shuffling of
    colors in the colormap creation process, defaults to 42 (optional)
    :param start_with_black: The `start_with_black` parameter is a boolean flag that determines whether
    the colormap should start with the color black as the first color. If set to `True`, the colormap
    will begin with black to represent noise or background. If set to `False`, the colormap will start
    with an empty list of, defaults to False (optional)
    :param remove_light_colors_threshold: The `remove_light_colors_threshold` parameter in the
    `make_discrete_large_colormap` function is used to determine the threshold for removing light colors
    from the colormap. Colors with a Euclidean distance from white (1, 1, 1) below this threshold will
    be removed from the list of
    :return: The function `make_discrete_large_colormap` returns a matplotlib ListedColormap object with
    a specified number of colors, where the colors are selected from the CSS4_COLORS dictionary. The
    function may remove light colors based on a threshold value and can start with black if specified.
    """

    if start_with_black:
        vals = [(0.0, 0.0, 0.0)]  # start with black for noise
    else:
        vals = []

    colori = mcolors.CSS4_COLORS
    keep_only = list(colori.keys())  # use all colors

    random.seed(random_state)
    random.shuffle(keep_only)

    for i in keep_only:
        k = 1-np.asarray(mcolors.hex2color(colori[i]))
        if np.sum(k**2)**0.5 < remove_light_colors_threshold:
            keep_only.remove(i)

    for i in keep_only[:n_colors-1]:
        vals.append(mcolors.hex2color(colori[i]))

    return mcolors.ListedColormap(vals)


newcmp = make_discrete_large_colormap(start_with_black=True,
                                      random_state=42)


# %%


# =============================================================================
# LOAD DATA
# =============================================================================

working_directory = "./"  # change it with the correct location

db = pd.read_csv(working_directory+"dataset_vst.csv", index_col=0)



metadata = pd.read_csv(working_directory+"metadata.csv", index_col=0)
genes_names = pd.read_csv(working_directory+"gene_names.csv", index_col=0)
results_storage_dir = working_directory+"results/"

metadata = metadata.loc[db.index, :]

save_plots = True
convert_gene_names = False  




if not os.path.exists(results_storage_dir):
    # Create a new directory because it does not exist
    os.makedirs(results_storage_dir)
    print("Created directory to store saved plots")



# %%


# =============================================================================
# SOFT THRESHOLD SELECTION
# =============================================================================


def soft_threshold_selection(db, soft_threshold_candidates=np.arange(1, 21),
                             correlation_method='spearman', fit_threshold=0.75,
                             selection_method='first_relevant_local_maximum',
                             make_summary_plot=False, show_best_fit_plot=False,
                             results_storage_dir='./',
                             return_correlation_matrix=True, save_resulting_plots=False):
    """
    The function `soft_threshold_selection` selects the best fitting epsilon parameter for matrix
    exponentiation to fit the scale-free network assumption using various input parameters and methods.
    
    :param db: The `db` parameter in the `soft_threshold_selection` function is expected to be a
    DataFrame containing the data for which you want to select the best value for the epsilon parameter.
    This DataFrame is used to calculate correlations and perform calculations within the function
    :param soft_threshold_candidates: The `soft_threshold_candidates` parameter in the
    `soft_threshold_selection` function is a numpy array containing the range of values to be considered
    as candidates for the epsilon parameter in matrix exponentiation. By default, it is set to
    `np.arange(1, 21)`, which means it will consider
    :param correlation_method: The `correlation_method` parameter in the `soft_threshold_selection`
    function specifies the method used to calculate the correlation between variables. In this function,
    the default method is set to 'spearman', which calculates the Spearman correlation coefficient. This
    method is often used when the relationship between variables may, defaults to spearman (optional)
    :param fit_threshold: The `fit_threshold` parameter in the `soft_threshold_selection` function is
    used to specify the threshold value for the goodness of fit (R^2 score) of the scale-free fit. This
    threshold is used to determine which epsilon value provides a satisfactory fit to the data. If the
    R^2
    :param selection_method: The `selection_method` parameter in the `soft_threshold_selection` function
    determines how the best epsilon value is selected. There are three options for this parameter:,
    defaults to first_relevant_local_maximum (optional)
    :param make_summary_plot: The `make_summary_plot` parameter in the `soft_threshold_selection`
    function determines whether a summary plot showing the selection process for the epsilon value will
    be generated. If set to `True`, the function will create a plot displaying the R^2 scores for
    different epsilon values and highlight the selected epsilon value, defaults to False (optional)
    :param show_best_fit_plot: The `show_best_fit_plot` parameter in the `soft_threshold_selection`
    function determines whether to display a plot showing the best fit for the selected epsilon value.
    If set to `True`, the function will generate a plot comparing the node degree distribution with the
    fitted power-law function based on the selected epsilon, defaults to False (optional)
    :param results_storage_dir: The `results_storage_dir` parameter in the `soft_threshold_selection`
    function is a string that specifies the directory where any resulting plots or files should be
    saved. This parameter allows you to define the location where the function will store any output
    files, such as summary plots or best fit plots, generated during, defaults to ./ (optional)
    :param return_correlation_matrix: The `return_correlation_matrix` parameter in the
    `soft_threshold_selection` function determines whether the function should return the correlation
    matrix along with the selected epsilon value or just the epsilon value, defaults to True (optional)
    :param save_resulting_plots: The `save_resulting_plots` parameter in the `soft_threshold_selection`
    function determines whether the resulting plots generated during the execution of the function
    should be saved to a file or not. If `save_resulting_plots` is set to `True`, the plots will be
    saved in the specified `results, defaults to False (optional)
    :return: The function `soft_threshold_selection` returns either the selected epsilon value and the
    correlation matrix (if `return_correlation_matrix` is True) or just the selected epsilon value (if
    `return_correlation_matrix` is False).
    """

    # fitting general power-law function for degree distribution
    def degree_distr(x, cost, N):
        y = cost*x**N
        return y

    r2 = []
    pars = []

    cor = np.abs(db.corr(method=correlation_method))

    # order and discard doubles
    candidates = np.unique(soft_threshold_candidates)

    for i in candidates:
        print(i)
        corr = cor**i

        ics = np.linspace(np.min(corr.sum()), 
                          np.max(corr.sum()), 
                          int(len(corr)**.5))
        a = np.histogram(corr.sum(axis=1), bins=ics)
        # change bins extremes with bins center 50->49 values
        ics = np.mean((ics[1:], ics[:-1]), axis=0)
        parameters, covariance = curve_fit(degree_distr, ics, a[0], p0=[np.mean(a[0]), -2.5],
                                           bounds=([0., -np.inf], [+np.inf, -0.]))

        r2.append(r2_score(a[0], degree_distr(ics, *parameters)))
        pars.append(parameters)

    r2 = np.asarray(r2)

    if selection_method.lower() == 'first_relevant_local_maximum':
        indx_partial = np.where(r2[:-1] > r2[1:])[0]
        if len(indx_partial) == 0:
            indx = -1
        else:
            indx = np.where(r2 > fit_threshold)[0]
            indx = list(set(indx).intersection(set(indx_partial)))
            if len(indx) == 0:
                indx = -1
            else:
                indx = np.min(indx)
    elif selection_method.lower() == 'max':
        indx = np.argmax(r2)

    else:
        # select first point above fit_threshold
        indx = np.where(r2 > fit_threshold)[0][0]

    epsilon = candidates[indx]  # automatic epsilon selection

    if make_summary_plot:
        # Fit summary plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.scatter(candidates, r2)
        ax.scatter(epsilon, r2[indx], s=10, label='selected epsilon')
        ax.plot([np.min(candidates), np.max(candidates)], [fit_threshold, fit_threshold],
                'r-.', label='fit R^2 = '+str(fit_threshold))
        ax.set_xlabel("Epsilon value")
        ax.set_ylabel("R^2 score of scale free fit")
        ax.set_title("Epsilon selection plot")
        ax.legend()
        fig.tight_layout()

        if save_resulting_plots:
            fig.savefig(results_storage_dir +
                        "soft_threshold_selection.png", facecolor='w')

    if show_best_fit_plot:
        corr = cor**epsilon

        ics = np.linspace(np.min(corr.sum()), np.max(
            corr.sum()), int(len(corr)**.5))
        a = np.histogram(corr.sum(axis=1), bins=ics)
        # change bins extremes with bins center 50->49 values
        ics = np.mean((ics[1:], ics[:-1]), axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(a[1][:-1], a[0], label='node degree distribution')
        ax.plot(ics, pars[indx][0]*ics**pars[indx][1],
                label=str(pars[indx][0])+"*X^"+str(pars[indx][1]))
        ax.legend()
        ax.set_title(correlation_method+" correlation ^ " +
                     str(epsilon)+" degree distribution")
        ax.set_xlabel("Node degree")
        ax.set_ylabel("counts")
        fig.tight_layout()

        if save_resulting_plots:
            fig.savefig(results_storage_dir +
                        "soft_threshold_best_fit.png", facecolor='w')

    if return_correlation_matrix:
        return epsilon, cor
    else:
        return epsilon


# use 7 for WB and 6 for PBMC
epsilon, correlation_matrix = soft_threshold_selection(db, save_resulting_plots=save_plots,
                                                       show_best_fit_plot=True,
                                                       soft_threshold_candidates=7)

correlation_matrix_soft_thresholded = correlation_matrix**epsilon

print("selected epsilon parameter:", epsilon)


# %%


# DIMENSIONALITY REDUCTION (if needed)

nn = 50  # number of neighbors for UMAP should stay in between 30 and 300 based on the data

mapper = umap.UMAP(n_components=2,
                   n_neighbors=nn,
                   metric='precomputed',  # jaccard for boolean data
                   init='spectral',
                   min_dist=0.0,
                   random_state=42,
                   low_memory=True,   # False gives faster but heavier computation
                   verbose=51)        # maximum verbosity & printable to file if needed

# use whichever metric you prefer
embed = mapper.fit_transform(1-correlation_matrix_soft_thresholded)



plt.scatter(*embed.T, s=1)
plt.colorbar()

np.save(results_storage_dir+"embedding.npy", embed)

# %%


# CLUSTERING
################ ##
## Lorenzo Dall'Olio
## 15 dec 2022
## perform clustering on lower dimensional embedding of the soft thresholded correlation matrix from vst transofrmed data
################ ##
# fine tune the following parameters by verifying the lower dimensional embedding plot
clusterer = hdbscan.HDBSCAN(min_cluster_size=20,
                            min_samples=5,
                            prediction_data=True,
                            cluster_selection_epsilon=0.0,
                            alpha=1.0,
                            cluster_selection_method='eom',
                            approx_min_span_tree=True,
                            algorithm='best',
                            gen_min_span_tree=True)


clusterer.fit(embed)

clusters = clusterer.labels_+1  # cluister 0 will be noisecluster



fig, ax = plt.subplots(1, 1)
a = ax.scatter(*embed.T, s=1, c=clusters, cmap=newcmp)
fig.colorbar(a)
ax.grid()
print(np.unique(clusters, return_counts=True))


if save_plots:
    plt.savefig(results_storage_dir+"clusters.png", facecolor='w')


# %%


# =============================================================================
# Save .csv file genes names per cluster
# =============================================================================


if not os.path.exists(results_storage_dir+"genes lists/"):
    print("\n-->creating directory", results_storage_dir+"genes lists/")
    os.makedirs(results_storage_dir+"genes lists/")


for i in np.unique(clusters):
    mask = correlation_matrix_soft_thresholded[clusters == i].index
    if convert_gene_names:
        genes_names.loc[mask].to_csv(results_storage_dir+"genes lists/module_"+str(i)+"_genes.tsv",
                                     header=None, sep='\t')
    else:
        mask.to_frame().to_csv(results_storage_dir+"genes lists/module_"+str(i)+"_genes.tsv",
                               header=None, sep='\t')

# save reference list for all used genes
mask = correlation_matrix_soft_thresholded.index
if convert_gene_names:
    genes_names.loc[mask].to_csv(results_storage_dir+"genes lists/reference_genes_list.tsv",
                                 header=None, sep='\t')
else:
    mask.to_frame().to_csv(results_storage_dir+"genes lists/reference_genes_list.tsv",
                           header=None, sep='\t')


# %%

