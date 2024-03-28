from skopt.space import Real, Integer  # , Categorical
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from sklearn.cross_decomposition import PLSRegression
from skopt import gp_minimize
from skopt.plots import plot_objective
from skopt.utils import use_named_args
import os
import umap
import random
import hdbscan
import requests

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_samples
from statsmodels.stats.multitest import multipletests


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
    """Build a discrete colormap with more than 20 colors. In case analysis
    identifies 20+ clusters.

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
                                      random_state=123456789)


# %%


# =============================================================================
# LOAD DATA
# =============================================================================

working_directory = "/home/lyro/Github/RIOT-Net/old/"

db = pd.read_csv(working_directory+"dataset_vst.csv", index_col=0)

clusters = pd.DataFrame(index=db.columns)


for a in range(13):
    gens = pd.read_csv(working_directory+"/results/genes lists/module_"+str(a)+"_genes.tsv",
                       sep='\t', index_col=0)
    clusters.loc[gens.index, 'cluster'] = a


db = db.loc[:, clusters.index]
clusters = np.hstack(np.asarray(clusters))

# db = pd.read_csv("/home/lyro/Scrivania/pSS/normalized_counts/normalized_counts_WB_vst.csv",
#                  index_col=0)

# raw_db = pd.read_csv(working_directory+"dataset_raw.csv", index_col=0).T.loc[db.index, db.columns]

# median of DESEq2

# gene_geometric_means = st.mstats.gmean(raw_db)
# size_factors = raw_db/gene_geometric_means  # nans are not a big issue here
# normalization_factors = size_factors.median(axis=1)
# normalized_db = np.divide(raw_db.T, normalization_factors).T

metadata = pd.read_csv(working_directory+"metadata.csv", index_col=0)
genes_names = pd.read_csv(working_directory+"gene_names.csv", index_col=0)
results_storage_dir = working_directory+"results/"

metadata = metadata.loc[db.index, :]

save_plots = False
convert_gene_names = True




if not os.path.exists(results_storage_dir):
    # Create a new directory because it does not exist
    os.makedirs(results_storage_dir)
    print("Created directory to store saved plots")


# dbT3 = db.loc[db.index[['T3' in a for a in db.index]].values[:], :]

# lst0 = [a[:-1] for a in dbT0.index]

# intersection01 = list(set(lst0).intersection(lst1))


# lst0 = np.asarray([a[:-1] if a[-1]=='0' else '' for a in db.index])
# lst0 = lst0[lst0 !='']

# lst3 = np.asarray([a[:-1] if a[-1]=='3' else '' for a in db.index])
# lst3 = lst3[lst3 !='']

# intersection = list(set(lst0).intersection(lst3))
# dbT0 = db.loc[[a+'0' for a in intersection], :]
# dbT3 = db.loc[[a+'3' for a in intersection], :]
# dbT0.index = intersection
# dbT3.index = intersection
# db_delta = dbT3 - dbT0
# db_delta.index = [a+'3' for a in intersection]
# dbT3.index = [a+'3' for a in intersection]
# dbT0.index = [a+'0' for a in intersection]


# %%


# =============================================================================
# SOFT THRESHOLD SELECTION
# =============================================================================


def soft_threshold_selection(db, soft_threshold_candidates=np.arange(1, 21),
                             correlation_method='spearman', fit_threshold=0.85,
                             selection_method='first_relevant_local_maximum',
                             make_summary_plot=False, show_best_fit_plot=False,
                             results_storage_dir='./',
                             return_correlation_matrix=True, save_resulting_plots=False):
    """Function to selecti which value best fits epsilon parameter for matrix
    epxonentiation to fit scale-free network assumption."""

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

        ics = np.linspace(np.min(corr.sum()), np.max(
            corr.sum()), int(len(corr)**.5))
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

nn = 100  # int(min(len(correlation_matrix_soft_thresholded)//5, 30))

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


plt.scatter(*embed.T, s=1, c=clusters, cmap='tab20')
plt.colorbar()

# %%


# CLUSTERING


bayes_space = hdbscan.HDBSCAN()

space = [Integer(2, 100, name='min_cluster_size'),
         Integer(1, 100, name='min_samples'),
         Real(1e-15, 2., "log-uniform", name='cluster_selection_epsilon'),
         Real(0.1, 2, "uniform", name='alpha')]
# Categorical(['eom', 'leaf'], name='cluster_selection_method')]

# The decorator below enables the objective function
# to receive the parameters as keyword arguments.


@use_named_args(space)
def objective(**params):
    '''
    Scitkit Learn Optimize requires an objective function to minimize.
    We use the average of cross-validation mean absolute errors as
    the objective function (also called cost function in optimization)
    '''
    clustering = hdbscan.HDBSCAN(prediction_data=True, approx_min_span_tree=False,
                                 algorithm='best', gen_min_span_tree=True)

    clustering.set_params(min_cluster_size=int(params['min_cluster_size']),
                          min_samples=int(params['min_samples']),
                          cluster_selection_epsilon=params['cluster_selection_epsilon'],
                          alpha=params['alpha'],
                          cluster_selection_method='eom')  # params['cluster_selection_method'])

    clustering.fit(embed)

    return -clustering.relative_validity_


res_gp = gp_minimize(objective, space, n_calls=50, random_state=42,
                     n_jobs=-1, verbose=True, initial_point_generator='lhs')

_ = plot_objective(res_gp, n_points=100, size=12/len(res_gp.x))
plt.tight_layout()


clusterer = hdbscan.HDBSCAN(min_cluster_size=int(res_gp.x[0]),
                            min_samples=int(res_gp.x[1]),
                            prediction_data=True,
                            cluster_selection_epsilon=res_gp.x[2],
                            alpha=res_gp.x[3],
                            cluster_selection_method='eom',
                            approx_min_span_tree=True,
                            algorithm='best',
                            gen_min_span_tree=True)


clusterer.fit(embed)

clusters = clusterer.labels_+1


# # assign noise patients to coherent nearest neighbors
# nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree',
#                         metric='euclidean').fit(transformed[clusterer.labels_!=-1])
# distances, indices = nbrs.kneighbors(transformed[clusterer.labels_==-1])
# # dists = np.median(distances.T[-1])
# best_assignment = [genetic_db['cluster'].loc[index].mode()[0] for index in indices]


# color_palette = sns.color_palette('Paired', max(control_db['cluster'])+1)
# cluster_colors = [color_palette[x] if x >= 0
#                   else (0.5, 0.5, 0.5)
#                   for x in clusterer.labels_]
# cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                          zip(cluster_colors, clusterer.probabilities_)]

# plt.hist(clusterer.probabilities_, bins=100, alpha=0.5)

# soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

# plt.scatter(*transformed.T, s=1, c=cluster_member_colors)

fig, ax = plt.subplots(1, 1)
a = ax.scatter(*embed.T, s=1, c=clusters, cmap='tab20')
fig.colorbar(a)
# ax.grid()
print("\nClusters' Range: ["+str(min(clusters)), str(max(clusters))+"]")
print(pd.value_counts(clusters), '\n', clusterer.relative_validity_)


# %%

clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

clusterer.condensed_tree_.plot()
# clusterer.condensed_tree_.plot(select_clusters=True)


# %%


# =============================================================================
# UMAP embedding
# =============================================================================

mapper = umap.UMAP(n_components=2,        # we suggest 2 or 3 to visualize it with plots
                   # use more than default value (15) if data size allows it
                   n_neighbors=30,
                   metric='precomputed',  #
                   init='spectral',       # use 'spectral' if possible
                   min_dist=0.0,          # use < 0.1 to compact clusters
                   random_state=42,       # fix it to a constant value
                   # use True to reduce memory reuirements, False to reduce Time requirements, based on your priorities
                   low_memory=True,
                   verbose=51)

embedding = mapper.fit_transform(1-correlation_matrix_soft_thresholded)


# %%


# =============================================================================
# HDBSCAN clustering
# =============================================================================
plt.rcParams.update({'font.size': 20})


# explain how to select parameters -> analysis section of paper
# it is time consumping, so: ideas to improve it
# TODO!!! tune this parameters depending on your specific dataset
min_s = 25  # 25
# silhouettes = []
# n_clusters = []
# davies_b = []
# trials = np.arange(2, 101, dtype=int)  # use arange since it must be integer
# for i in trials: # np.arange(2, 51):
#     print(i)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=int(i), min_samples=int(i),
#                                 cluster_selection_epsilon=0.00,
#                                 cluster_selection_method='eom')

#     clusters = clusterer.fit_predict(embedding)+1  # cluster 0 will now correspond to noise cluster
#     mask = clusters != -2
#     silhouettes.append(silhouette_score(embedding[mask], clusters[mask]))
#     davies_b.append(davies_bouldin_score(embedding[mask], clusters[mask]))
#     n_clusters.append(max(clusters))


# max_size = 30
# min_size = 10

# window = 5
# start = min_size#  window//2
# end = max_size #  35 #  -window//2
# smooth_silhouettes = np.convolve(silhouettes, np.ones((window))/window, mode='same')[start:end]
# smooth_davies = np.convolve(davies_b, np.ones((window))/window, mode='same')[start:end]


# score = smooth_silhouettes/np.std(smooth_silhouettes) - smooth_davies/np.std(smooth_davies)

# fig, ax = plt.subplots(4, 1, figsize=(18, 9), sharex=True)
# ax[0].plot(trials[start:end], smooth_silhouettes)
# ax[1].plot(trials[start:end], smooth_davies)
# ax[2].plot(trials[start:end], score)
# ax[3].plot(trials[start:end], n_clusters[start:end])


# min_s = int(trials[np.argmax(score)])+start
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_s,
                            min_samples=min_s,
                            cluster_selection_epsilon=0.0,
                            cluster_selection_method='eom')
# cluster 0 will now correspond to noise cluster
clusters = clusterer.fit_predict(embedding)+1

# plotting HDBSCAN clusters on UMAP embedding
fig = plt.figure(figsize=(16, 9))

if mapper.n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)

for cluster in np.unique(clusters):
    ax.scatter(*embedding[clusters == cluster].T, s=1,
               color=newcmp.colors[cluster], label=cluster)

plt.xlabel("UMAP 1 (a.u.)")
plt.ylabel("UMAP 2 (a.u.)")
plt.title("UMAP embedding + HDBSCAN clustering\nfound "+str(max(clusters)) +
          " clusters\n noise: "+str(np.where(clusters == 0, 1, 0).sum())+" points" +
          " using min_sample = "+str(min_s))

lgnd = plt.legend(loc='best',
                  ncol=3, fancybox=True, shadow=True,
                  title='Gene Module', markerscale=20)

plt.tight_layout()

save_plots = False
if save_plots:
    plt.savefig(results_storage_dir+"clusters.png", facecolor='w')


# sil_scores = silhouette_samples(embedding, clusters)

# silhouette_plots = pd.DataFrame(np.vstack([clusters, sil_scores]).T, columns=['cluster', 'silhouette_score'])

# silhouette_plots.sort_values(by=['cluster', 'silhouette_score'], ascending=False)

# avg_cluster_sil_scores = pd.DataFrame(sil_scores).groupby(clusters).mean()


# %%


# =============================================================================
# FIND TARGET GENES IN UMAP EMBEDDING
# =============================================================================

interesting_genes_coded = []
drugs = ['Leflunomide', 'Hydroxychloroquine']

for drug in drugs:
    interesting_genes = set(pd.read_csv(
        working_directory+"interesting_genes_"+drug+".csv").loc[:, drug])

    # filter to only genes still in the analysis
    interesting_genes = interesting_genes.intersection(
        set(genes_names.loc[db.columns, 'Name']))
    interesting_genes_coded += [genes_names[genes_names.Name == x].index[0]
                                for x in genes_names.loc[db.columns, 'Name'] if x in interesting_genes]

interesting_genes_coded = np.unique(interesting_genes_coded)

plot_color = np.where(
    [x in interesting_genes_coded for x in db.columns], 'green', 'lightgrey')


# plots
fig = plt.figure(figsize=(16, 8))

if mapper.n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)


ax.scatter(*embedding.T, s=5, alpha=0.5, c=plot_color, cmap='Greens',
           vmax=4, vmin=-np.log10(0.05))
ax.scatter(*embedding[plot_color == 'green'].T, s=5, alpha=0.5, c=plot_color[plot_color == 'green'], cmap='Greens',
           vmax=4, vmin=-np.log10(0.05))

ax.set_xlabel("UMAP 1 (a.u.)")
ax.set_ylabel("UMAP 2 (a.u.)")
ax.set_title("UMAP embedding + Targeted genes \nfound "
             + str(len(interesting_genes_coded))+" genes targeted by the drug: "+str(drugs))
# fig.colorbar(a).set_label("-Log10( adjusted p-value )")
fig.tight_layout()
if save_plots:
    fig.savefig(results_storage_dir+"_"+str(drugs) +
                "_targeted_genes.png", facecolor='w')


# problem: no IGG on STRING db,

# %%


# # =============================================================================
# # Enrichment test for target genes in clusters
# # =============================================================================

# target_genes = np.where(plot_color=='green', 1, 0)
# target_genes = pd.DataFrame(target_genes, index=db.columns, columns=['target_gene'])

# tgenes_per_cluster = target_genes.groupby(clusters).mean()
# sizes = target_genes.groupby(clusters).size()

# pvals = []
# for i in np.unique(clusters):
#     if i == 0: continue

#     res = []
#     for j in range(10000):
#         res.append(target_genes.sample(sizes[i], random_state=j).mean().values[0])
#     res = np.asarray(sorted(res))
#     pvals.append(np.where(res >= tgenes_per_cluster.iloc[i, -1], 1, 0).mean())

#     if pvals[-1] < 0.05/len(np.unique(clusters)):
#         print("\nCluster", i, "has a SIGNIFICANT one-sided pvalue of", pvals[-1], "of being more",
#               "enriched in target genes than a randomly sampled gene group of the",
#               "exact same size")
#     else:
#         print("\nCluster", i, "has a one-sided pvalue of", pvals[-1], "of being more",
#               "enriched in target genes than a randomly sampled gene group of the",
#               "exact same size")


# %%


# =============================================================================
# Hypergeometric enrichment test for target genes in clusters
# =============================================================================


target_genes = np.where(plot_color != 'lightgrey', 1, 0)
target_genes = pd.DataFrame(
    target_genes, index=db.columns, columns=['target_gene'])

tgenes_per_cluster = target_genes.groupby(clusters).sum()
sizes = target_genes.groupby(clusters).size()

pvals = []

for i in np.unique(clusters):
    if i == 0:
        continue  # we are not interested in noise cluster

    random_var = st.hypergeom(len(db.columns),  # how many genes in total
                              # how many interesting genes
                              len(interesting_genes_coded),
                              sizes[i])  # how many genes in the sampled subset, equal to the size of cluster i

    # how unlikely it is to find "tgenes_per_cluster" interesting genes in the
    # hypergeometric distribution we have just defined
    pvals.append(1-random_var.cdf(tgenes_per_cluster.loc[i].values[0]))


# find significant pvalues & print summary

multiple_tests_corrected_results = multipletests(
    pvals, alpha=0.05, method='fdr_bh')

interesting_clusters = []

with open(results_storage_dir+'Enrichment.txt', 'w') as f:
    print("Hypergeometric enrichment test results over", max(clusters),
          "clusters:\n\n", file=f)

if min(multiple_tests_corrected_results[1]) < .05:

    for i, adjusted_p in enumerate(multiple_tests_corrected_results[1]):
        if adjusted_p < 0.05:
            with open(results_storage_dir+'Enrichment.txt', 'a') as f:
                interesting_clusters.append(i+1)
                print("Cluster", i+1, "contains", sizes[i+1], "genes, among which the following list of",
                      tgenes_per_cluster.loc[i +
                                             1].values[0], "interesting genes is found:\n",
                      genes_names.loc[[x for x in db.columns[clusters == i+1]
                                       if x in interesting_genes_coded], "Name"].values,
                      "\nTherefore it could be a good candidate for deeper analysis, since its enrichment in interesting genes has a random chance of",
                      adjusted_p *
                      100, "% (below 5% after multiple test FDR correction) considering random subsamples of the same size.\n\n",
                      file=f)
else:
    with open(results_storage_dir+'Enrichment.txt', 'a') as f:
        print("No significantly enriched cluster was found.", file=f)

print("interesting clusters:", interesting_clusters)


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


# =============================================================================
# AUTOMATIC OVERREPRESENTATION ANALYSIS USING PANTHER DB ONLINE API
# =============================================================================


if len(interesting_clusters):

    print("Overrepresentation analysis using PANTHER online DB over ",
          len(interesting_clusters),
          "clusters (~2 minutes per interesting cluster)")

    # For deeper information check out PANTHER Tools - Enrichment (Overrepresentation) at
    # http://pantherdb.org/services/tryItOut.jsp?url=%2Fservices%2Fapi%2Fpanther

    test_type = ''  # either FISHER(default) or BINOMIAL
    # correction for multiple tests, either FDR(default), BONFERRONI, NONE
    correction = ''
    # ''.join(e+',' for e in genes_names.loc[db.columns[:], 'Name'])[:-1]
    genes_ref_list = ''
    # bad request, maybe url length, 65410 not ok, 65408 ok (255.5*256)
    annot_dbs = requests.get("http://pantherdb.org/services/oai/pantherdb/supportedannotdatasets").json()[
        'search']['annotation_data_sets']['annotation_data_type']
    # ''.join(e+',' for e in db.columns[clusters==module][:2])[:-1]+\

    for module in interesting_clusters:
    # for module in np.unique(clusters):

        module_genes = ''.join(
            e+',' for e in genes_names.loc[db.columns[clusters == module], 'Name'])[:-1]

        if not os.path.exists(results_storage_dir+"/module "+str(module)+" PANTHER analysis/"):
            print("\n-->creating directory", results_storage_dir +
                  "module "+str(module)+" PANTHER analysis/")
            os.makedirs(results_storage_dir+"module " +
                        str(module)+" PANTHER analysis/")

        for annot_db in annot_dbs:

            # if annot_db['id'] != 'GO:0003674': continue
            panther_url_request = "http://pantherdb.org/services/oai/pantherdb/enrich/overrep?geneInputList=" +\
                                  module_genes+"&organism=9606&" +\
                                  "&annotDataSet="+annot_db['id'] +\
                                  "&enrichmentTestType="+test_type.upper() +\
                                  "&correction="+correction.upper()  # +"refInputList="+genes_ref_list+"&refOrganism=9606"

            a = requests.get(panther_url_request)

            if a.status_code == 200:

                infodb = pd.DataFrame(a.json()['results']['result'])

                # filtering only significantly enriched terms
                infodb.drop(
                    index=infodb[infodb.loc[:, 'plus_minus'] != '+'].index, inplace=True)
                infodb.drop(
                    index=infodb[infodb.loc[:, 'fdr'] > 0.05].index, inplace=True)

                if len(infodb) == 0:
                    print("No significantly enriched element within cluster", module,
                          "for db:", annot_db['label'])
                    continue

                # sorting for increasing decreasing
                names = [x['label'] for x in infodb.loc[:, 'term']]
                infodb.sort_values(by='fdr', inplace=True)

                # names = np.asarray([name.split('(')[0] for name in infodb.iloc[:, 0]])
                # names = infodb.loc[:, 'description']
                n_genes = np.clip(infodb.loc[:, 'fold_enrichment'], 0, 100)

                info = -np.log10(infodb.loc[:, 'fdr'])

                how_many = np.min([20, len(infodb)])

                # actual plot
                fig, ax = plt.subplots(1, 1, figsize=(16, 11))

                ax.set_title("module "+str(module)+" " +
                             annot_db['label'], fontsize=18)

                a = ax.scatter(info[:how_many], 1+np.arange(how_many)
                               [::-1], s=200, c=n_genes[:how_many], cmap='rainbow')
                fig.colorbar(a).set_label("Fold Enrichment")
                ax.set_xlabel("Significance level {= -Log10(p values)}")
                plt.yticks(ticks=1+np.arange(how_many)
                           [::-1], labels=names[:how_many])
                fig.tight_layout()
                fig.savefig(results_storage_dir+"module "+str(module) +
                            " PANTHER analysis/"+ax.get_title()+".png")

                print("module:", module, annot_db['label'])
            else:
                print("\nSomething went wrong, request returned the code",
                      a.status_code, ", different from the normal flow of the",
                      "program, which should return 200.")

                if a.status_code == 400:
                    print("Probably the url for the request was too long",
                          "(more than 65408 characters). This causes a bad",
                          "request error, try not providing a reference list or",
                          "this specific cluster has too many genes")
                break
else:
    print("No intersting cluster was found, skipping overrepresentation analysis")


# %%


# interesting_clusters += [10]  # since cluster 10 is not available on STRING db
i = 1
# =============================================================================
# PLS/PLS2 for interesting clusters
# =============================================================================
# PLS-DA  <--
# SVD approach XTy
# --> X * left singular scores
# plot the 2 scores


# Questions: how to deal with generalization? can we use all point in fit ?
# passing 3 targets to PLS-DA? can be correct?
# how to combine loadings to find interesting genes?
# --- only biggest squared loadings from 1st component?
# --- sum squared loadings of 3 components?
# problem, most big clusters (even noise) have good score after PLS-DA,
# still no significantly different gene, even using deltas, only T3, only T0, for both,
# --- I believe it could still be interesting to present an overall cluster merged p-values,
# --- based on gene-wise Welch's T-test or similars.


use = db_delta

targetP = np.where(
    metadata.loc[use.index, 'Treatment'] == 'placebo medication', 1, 0)
targetNR = np.where(np.logical_and(metadata.loc[use.index, 'Responder'] == 'non-responder',
                                   metadata.loc[use.index, 'Treatment'] == 'Verum medication'), 1, 0)
targetR = np.where(metadata.loc[use.index, 'Responder'] == 'responder', 1, 0)
targets = np.vstack([targetP, targetNR, targetR]).T

color_code = targetP + 2*targetNR + 3*targetR


for i in interesting_clusters:
    mask = correlation_matrix_soft_thresholded[clusters == i].index

    # pls = PLSSVD(n_components=1)
    pls = PLSRegression(n_components=3)
    predicted_data = pls.fit_transform(use.loc[:, mask],
                                       targets)
    # st.pearsonr(np.asarray(predicted_data).T[1], target)
    print("cluster", i, "Score", pls.score(use.loc[:, mask], targets))

    order = np.argsort(pls.x_loadings_.T[0]**2)[:50]

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(*predicted_data[0].T, c=color_code, cmap='tab10')

    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    ax2.barh(np.arange(50), pls.x_loadings_.T[0][order]**2,
             tick_label=genes_names.loc[mask[order]].values.T[0])

    ax.grid()
    fig.tight_layout()


# we'll compute the SVD of the cross-covariance matrix = X.T.dot(Y)
# This matrix rank is at most min(n_samples, n_features, n_targets) so
# n_components cannot be bigger than that.


# %%


# =============================================================================
# DIFFERENTIAL GENE EXPRESSION try2
# =============================================================================

targets_only = False
# -/-/-/-/-/-/CMAS/-/-/-/IGKV3D-15,TRBV29-1/-/EEF2K

use = db_delta
analysis = 'treated'

interesting_clusters = [10]

if len(interesting_clusters):

    treated = use[metadata.loc[use.index, 'Treatment']
                  == 'Verum medication'].index
    placebo = use[metadata.loc[use.index, 'Treatment']
                  == 'placebo medication'].index
    responders = use[metadata.loc[use.index, 'Responder'] == 'responder'].index
    non_responders = use.loc[treated,
                             :][metadata.loc[treated, 'Responder'] == 'responder'].index

    genes = db.columns[[
        cluster in interesting_clusters for cluster in clusters]]

    pvals = []

    if targets_only:
        genes = genes[[gene in interesting_genes_coded for gene in genes]]

    # if not os.path.exists(results_storage_dir+"/DEG/"):
    #     print("\n-->creating directory", results_storage_dir+"DEG/")
    #     os.makedirs(results_storage_dir+"DEG/")

    for gene in genes:
        if analysis.lower() == 'treated':
            pvals.append(st.ttest_ind(use.loc[treated, gene],
                                      use.loc[placebo, gene],
                                      equal_var=False)[1])

        else:
            pvals.append(st.ttest_ind(use.loc[responders, gene],
                                      use.loc[non_responders, gene],
                                      equal_var=False)[1])

    multiple_tests_corrected_results = multipletests(
        pvals, alpha=0.05, method='fdr_bh')

    deg = []

    # with open(results_storage_dir+'Enrichment.txt', 'w') as f:
    #     print("Hypergeometric enrichment test results over", max(clusters),
    #           "clusters:\n\n", file=f)

    if min(multiple_tests_corrected_results[1]) < .05:

        for i, adjusted_p in enumerate(multiple_tests_corrected_results[1]):
            if adjusted_p < 0.05:
                # with open(results_storage_dir+'Enrichment.txt', 'a') as f:
                #     interesting_clusters.append(i+1)
                print("\nGene", genes_names.loc[genes[i]].Name, "from cluster",
                      clusters[db.columns == genes[i]][0],
                      "is candidate for being Differentially Expressed, with an FDR of",
                      adjusted_p)
                deg.append(genes[i])

    print("\mFound a total of", len(deg),
          "possible Differentially Expressed Genes.")


# %%


# =============================================================================
# SINGLE GENE COMPLETE PLOT
# =============================================================================


# gene = expr.columns[np.argsort(np.abs(encv.coef_))[::-1][4]]
# gene = db.columns[np.argmin(adjusted_p[1])]

# for gene in deg:
# mask[order][-6]  # use negative since rank is opposite
gene = genes[np.argsort(multiple_tests_corrected_results[1])[0]]
print("\nPlots for", genes_names.loc[gene].values[0])
expr = db.copy()

condition = metadata.loc[expr.index, "Responder"]
condition[metadata.loc[expr.index, "Treatment"]
          == 'placebo medication'] = "placebo"
time = metadata.loc[expr.index, "Visitnumber"]

code = metadata.loc[expr.index, "Code"]
target = metadata.loc[expr.index, "deltaESSDAI"]

plot_db = pd.concat([expr.loc[:, gene], condition, time],
                    axis=1)

plot_db.columns = [gene, "condition", "time"]
# plot_db.columns = [geneinfo.loc[gene].index[0], "condition", "time"]
# plot_db.sort_values(by="Time", inplace=True)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Usual boxplot

sns.boxplot(data=plot_db, x='time', y=plot_db.columns[0], ax=ax,
            hue='condition',
            order=["Baseline", "8 weeks", "16 weeks", "24 weeks"],
            # order=["8 weeks", "16 weeks", "24 weeks"],
            hue_order=["placebo", "non-responder", "responder"],
            showmeans=True, meanprops={"marker": "+",
                                       "markeredgecolor": "black",
                                       "markersize": "15"})

# Add jitter with the swarmplot function
sns.swarmplot(x='time', y=plot_db.columns[0], data=plot_db, ax=ax,
              hue_order=["placebo", "non-responder", "responder"],
              order=["Baseline", "8 weeks", "16 weeks", "24 weeks"], hue='condition')

name = genes_names.loc[plot_db.columns[0]].values[0]
ax.set_title(name+" normalized expression over time for different groups in WB (cluster " +
             str(clusters[db.columns == gene][0])+")")
fig.tight_layout()
# fig.savefig(results_storage_dir+"DEG/"+name+" normalized expression over time in WB.png")


# %%


# =============================================================================
# SILHOUETTE SCORE
# =============================================================================

# (optional, but strongly suggested!) --> remove noise for silhouette score
remove_noise = True


if remove_noise:
    # put "features" here if you want
    X = embedding[np.where(clusters != 0)[0]]
    cluster_labels = clusters[np.where(clusters != 0)[0]]

else:
    X = embedding  # and also here
    cluster_labels = clusters

n_clusters = len(np.unique(cluster_labels))


sample_silhouette_values = silhouette_samples(X, cluster_labels)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))

# The silhouette coefficient can range from -1 to +1
ax1.set_xlim([-1, 1])

# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = np.mean(sample_silhouette_values)
silhouette_std = np.std(sample_silhouette_values)

y_lower = 10


# Aggregate the silhouette scores for samples belonging to
# cluster i, and sort them
avg_silhouette_per_cluster = []
std_of_silhouette_per_cluster = []
for i in np.unique(cluster_labels):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    avg_silhouette_per_cluster.append(np.mean(ith_cluster_silhouette_values))
    std_of_silhouette_per_cluster.append(np.std(ith_cluster_silhouette_values))

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = newcmp(i)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.8)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples


ax1.set_title("The silhouette plot for the various clusters",
              fontsize=14, fontweight='bold')
ax1.set_xlabel("The silhouette coefficient values", fontsize=15)
ax1.set_ylabel("Cluster label", fontsize=15)

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="black", linestyle="--",  linewidth=2,
            label="average silhouette score = "+str(silhouette_avg)[:6])
ax1.axvline(x=silhouette_std, color="red", linestyle="--",  linewidth=2,
            label="std dev of silhouette scores = "+str(silhouette_std)[:6])

ax1.set_yticks([])  # Clear the yaxis labels / ticks
x_ticks = np.linspace(np.min(sample_silhouette_values),
                      np.max(sample_silhouette_values), 10)
ax1.set_xticks(x_ticks)
ax1.legend(loc=2, fontsize=13)


ax2.barh(np.arange(0, n_clusters), avg_silhouette_per_cluster,
         height=.9, color="royalblue", label="average silhouette per cluster")
ax2.barh(np.arange(0, n_clusters), std_of_silhouette_per_cluster,
         height=0.5, color="lightcoral", label="std of silhouette per cluster")
ax2.axvline(x=np.mean(avg_silhouette_per_cluster), color="black", linestyle="-.", linewidth=1.5,
            label="average over clusters average silhouette scores = "+str(np.mean(avg_silhouette_per_cluster))[:6])
ax2.axvline(x=np.mean(std_of_silhouette_per_cluster), color="red", linestyle="-.", linewidth=1.5,
            label="average over clusters std dev. silhouette scores = "+str(np.mean(std_of_silhouette_per_cluster))[:6])

ax2.set_title("The average silhouette score for each cluster",
              fontsize=14, fontweight='bold')
ax2.set_xlabel("The average silhouette coefficient values", fontsize=15)
ax2.set_ylabel("Cluster label", fontsize=15)
ax2.legend(loc=3, fontsize=13)


fig.suptitle(("%d clusters" % n_clusters),
             fontsize=15, fontweight='bold')
fig.tight_layout(pad=0.6)
fig.savefig(results_storage_dir+"silhouette_plot_report.png", facecolor='w')

print("Found", n_clusters, "clusters and",
      np.where(clusters == 0, 1, 0).sum(), "noise points,",
      "\nthe average silhouette score is:", silhouette_avg,
      "\nthe standard deviation of silhouette scores is:", silhouette_std,
      "\nthe average over clusters average silhouette scores is:", np.mean(
          avg_silhouette_per_cluster),
      "\nthe standard deviation over clusters average silhouette scores is:", np.std(
          avg_silhouette_per_cluster),
      "\nthe average over clusters standard deviation silhouette scores is:", np.mean(
          std_of_silhouette_per_cluster),
      "\nthe standard dev. over clusters standard dev. silhouette scores is:", np.std(std_of_silhouette_per_cluster))


# %%


# =============================================================================
# DGE plots
# =============================================================================

# check 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html#rc356e4bdcefb-8'
# for further details regarding the selected method. Choose between 'fisher',
# 'pearson', 'mudholkar_george', 'tippett', 'stouffer'
# we strongly suggest pearson to try prevent big clusters being significant
# mostly due to their sample size
method_to_combine_pvalues = 'pearson'
for file in os.listdir(working_directory):
    if file.startswith("DEG.~"):

        print("Making DGE plot on UMAP embedding for", file)
        # find design formula, automatically embedded in filename
        formula = file.split('.')[1]
        target_column = formula.split('~')[-1]  # exclude starting '~'
        # find last addend (the one DGE analysis was performed on)
        target_column = target_column.split('+')[-1]

        differentially_expressed_genes_report = pd.read_csv(working_directory+file,
                                                            index_col=0)

        number_of_differentially_expressed_genes = np.where(
            differentially_expressed_genes_report.loc[db.columns, "padj"] < 0.05, 1, 0).sum()

        # information of p-values plot
        plot_color = -np.log10(np.where(differentially_expressed_genes_report.loc[db.columns, "padj"] >= 0.05, 1,
                                        differentially_expressed_genes_report.loc[db.columns, "padj"]))

        fig = plt.figure(figsize=(20, 8))

        if mapper.n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        a = ax.scatter(*embedding.T, s=5, alpha=0.5, c=plot_color,
                       cmap='Greens', vmax=4, vmin=-np.log10(0.05))
        ax.set_xlabel("UMAP 1 (a.u.)")
        ax.set_ylabel("UMAP 2 (a.u.)")
        ax.set_title("UMAP embedding + Differentially Expressed Genes \nfound "
                     + str(number_of_differentially_expressed_genes)+" DEG among different groups in "+target_column+" column")
        fig.colorbar(a).set_label("-Log10( adjusted p-value )")
        fig.tight_layout()
        if save_plots:
            fig.savefig(results_storage_dir+"DEG_"+formula +
                        "_pvalues.png", facecolor='w')

        # Log2FoldChange plot
        plot_color = differentially_expressed_genes_report.loc[db.columns,
                                                               "log2FoldChange"]

        fig = plt.figure(figsize=(20, 8))

        if mapper.n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        a = ax.scatter(*embedding.T, s=10, alpha=0.8,
                       c=plot_color, cmap='RdBu_r', vmin=-3, vmax=3)

        ax.set_xlabel("UMAP 1 (a.u.)")
        ax.set_ylabel("UMAP 2 (a.u.)")
        ax.set_title("UMAP embedding + Differentially Expressed Genes \nfound "
                     + str(number_of_differentially_expressed_genes)+" DEG among different groups in "+target_column+" column")
        fig.colorbar(a).set_label(
            "Log2FoldChange (sign depend on the contrast variable used during DGE analysis)")
        fig.tight_layout()
        if save_plots:
            fig.savefig(results_storage_dir+"DEG_"+formula +
                        "_foldchange.png", facecolor='w')

        # combine p-values within the same cluster to see if a cluster has overall significance
        aggregator = differentially_expressed_genes_report.loc[db.columns, "padj"].groupby(
            clusters)
        aggregated_pvalues = aggregator.agg(
            st.combine_pvalues, method=method_to_combine_pvalues)

        # combine p-values within the same cluster to see if a cluster has overall significance
        aggregated_Log2FoldChanges = differentially_expressed_genes_report.loc[db.columns, "log2FoldChange"].groupby(
            clusters).mean()

        with open(results_storage_dir+'DGE_analysis_cluster-wise_report_for'+formula+'.txt', 'w') as f:
            f.write("Considering the design formula: "+formula+" the following " +
                    "results were obtained combining single gene FDR corrected " +
                    "p-values within the same cluster with "+method_to_combine_pvalues +
                    " method.\nCluster\tcombined p-val\taverage Log2FoldChange")
            for i in aggregated_pvalues.index:
                if i == 0:
                    continue  # we are not interested in noise cluster's properties
                if aggregated_pvalues.loc[i][1] < 0.05:
                    print("cluster", i, "is significantly differentially expressed regarding",
                          target_column, "variable")
                f.write('\n'+str(i)+'\t'+str(aggregated_pvalues.loc[i][1])+'\t' +
                        str(aggregated_Log2FoldChanges.loc[i]))


# %%


ft = 18

scores = pd.read_csv("/home/lyro/Scrivania/pSS/scores.csv",
                     index_col="Code").fillna(0).drop(columns="Treatment").replace('', 0)

scores_correlation_targets = ['delta_cESSDAI', 'delta_ESSPRI', 'delta_UWS',
                              'delta_IgG', 'delta_RF', 'delta_Schirmer',
                              'delta_MXA', 'delta_Gal9', 'delta_CXCL10',
                              'STAR_tot', 'CRESS_tot']

hm = scores.loc[:, scores_correlation_targets].corr(method='spearman')

scores_dicotomic_targets = []  # ['STAR_Responder', 'CRESS_Responder']


threshold = 0.05   # remove multi test corrections


baseline = ['Visitnumber.BL']
endline = ['Visitnumber24wks']
visit = ['Visitnumber']
dropouts = ['Drop_out']
correlation_targets = ['deltaESSDAI']


# include also age?
# correlation_targets = ['ESSDAI.lymfneutr', 'deltaESSDAI',
#                        'ESSPRItotal', 'ESSPRIdryness', 'ESSPRIfatigue', 'ESSPRIpain',
#                        'VASocular', 'VASoral', 'GApat', 'GAphys', 'Schirmer', 'UWS',
#                        'SWS', 'sIgG', 'ESR', 'IFNscore_PBMC', 'IFNscore_WB', 'MxA',
#                        'MFIgenfat', 'MFIphysfat', 'MFImentfat', 'MFIredmot',
#                        'MFIredact', 'SF36physfunct', 'SF36socfunct', 'SF36rolephys',
#                        'SF36roleemot', 'SF36menthealth', 'SF36vitality', 'SF36pain',
#                        'SF36genhealth', 'SF36healthchange', 'ESSDAI_const_score',
#                        'lumIL31', 'lumCXCL10', 'lumCXCL13', 'lumsPD1', 'lumGal9',
#                        'ClinESSDAI', 'C3', 'C4', 'RF']


eigendb = db.copy()
eigendb.drop(inplace=True, columns=db.columns)

standardize_features = False
make_deltas = True
show_genes_significance = True
time_trend = True
save = True
# interesting_clusters = [1, 4, 6, 7, 8, 11]  # WB s7
# interesting_clusters = [5, 9]               # PBMC s6
interesting_clusters = []                     # None
interesting_clusters = np.unique(clusters)  # All
# interesting_clusters = [9]                     # interesting


# WB: P 5  N 7  R 8   /// ALL  P 8  N 10  R 11

p_values_targets, p_values_covariates = [],  []
statements_targets, statements_covariates = [], []
headers = []
tests_per_cluster = len(correlation_targets)+len(scores_correlation_targets) +\
    len(scores_dicotomic_targets)+2  # tr/pl and res/non-res


heatmap = pd.DataFrame(index=correlation_targets+scores_correlation_targets,
                       columns=np.arange(1, max(clusters)+1))

for cluster in np.unique(clusters):
    if cluster == 0:
        continue
    print(cluster)

    cluster_genes = db.columns[clusters == cluster]

    pca = PCA(n_components=1, random_state=42)
    if standardize_features:
        pca.fit((db.loc[:, cluster_genes]-db.loc[:,
                cluster_genes].mean())/db.loc[:, cluster_genes].std())
    else:
        pca.fit(db.loc[:, cluster_genes])

    expression0 = (dbT0.loc[:, cluster_genes]-dbT0.loc[:,
                   cluster_genes].mean())*pca.components_[0]
    expression0 = expression0.sum(axis=1)

    expression3 = (dbT3.loc[:, cluster_genes]-dbT3.loc[:,
                   cluster_genes].mean())*pca.components_[0]
    expression3 = expression3.sum(axis=1)

    int0 = [a + '0' for a in list(intersection)]
    int3 = [a + '3' for a in list(intersection)]

    eigendb['eigengene'+str(cluster)] = (db.loc[:,
                                                cluster_genes]*pca.components_[0]).sum(axis=1)

    if make_deltas:
        expression = expression3.loc[int3].values - \
            expression0.loc[int0].values  # T3-T0 for deltas
        expression = pd.DataFrame(expression, index=int0)
    else:
        expression = pd.DataFrame(
            expression0, index=expression0.index)  # only T0 for baseline

    headers.append("\n\nModule "+str(cluster)+" :\n\t-containing a total of " +
                   str(len(cluster_genes))+" genes.\n\t-1st eigenvector explained variance: " +
                   str(pca.explained_variance_ratio_[0]*100)+" %")

    # test for Treatment
    target_T = metadata.loc[expression.index].Treatment
    treated = target_T == "Verum medication"
    placebo = target_T == "placebo medication"
    pval_T = st.ttest_ind(expression[treated], expression[placebo],
                          equal_var=False, random_state=42)[1]

    p_values_targets.append(pval_T[0])
    statements_targets.append(
        "\t-significantly different expression between Treated/Placebo FDR: ")

    # test for Responder condition
    target_R = metadata.loc[expression.index].Responder
    responders = (target_R == 'responder')*treated
    non_responders = (target_R == 'non-responder')*treated
    pval_R = st.ttest_ind(expression[responders], expression[non_responders],
                          equal_var=False, random_state=42)[1]

    p_values_targets.append(pval_R[0])
    statements_targets.append(
        "\t-significantly different expression between Responder/Non_responder FDR: ")

    if make_deltas:
        # time related conditions
        bl = metadata.loc[int0].set_index(
            'Code', drop=True).select_dtypes(['number'])
        el = metadata.loc[int3].set_index(
            'Code', drop=True).select_dtypes(['number'])
        delta = el-bl

        metadata.fillna(0, inplace=True)
        delta.fillna(0, inplace=True)
        pos = metadata.loc[expression.index, "Code"]
        for col in correlation_targets:
            if delta.loc[pos, col].var() == 0:
                p_values_covariates.append(1.)
                statements_covariates.append("skipped")
                continue

            p_val = st.spearmanr(expression.iloc[:, 0], delta.loc[pos, col],
                                 nan_policy='omit')
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significant correlated with delta "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")
            heatmap.loc[col, cluster] = p_val[0]*100

        # scores continuous variables
        for col in scores_correlation_targets:

            p_val = st.spearmanr(expression.iloc[:, 0], scores.loc[pos, col],
                                 nan_policy='omit')
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significant correlated with "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")
            heatmap.loc[col, cluster] = p_val[0]*100

        # scores discrete variables
        for col in scores_dicotomic_targets:
            groups = np.unique(scores.loc[pos, col])
            p_val = st.ttest_ind(expression.loc[pos[(scores.loc[pos, col] == groups[0]).values].index],
                                 expression.loc[pos[(
                                     scores.loc[pos, col] == groups[1]).values].index],
                                 equal_var=False, random_state=42)
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significantly different between "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")

    else:
        metadata.fillna(0, inplace=True)
        pos = metadata.loc[expression.index, "Code"]
        for col in correlation_targets:
            if metadata.loc[expression.index, col].var() == 0:
                p_values_covariates.append(1.)
                statements_covariates.append("skipped")
                continue

            p_val = st.spearmanr(expression.values, metadata.loc[expression.index, col],
                                 nan_policy='omit')
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significant correlated with delta "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")
            heatmap.loc[col, cluster] = p_val[0]*100

        # scores continuous variables
        for col in scores_correlation_targets:
            p_val = st.spearmanr(expression.iloc[:, 0], scores.loc[pos, col],
                                 nan_policy='omit')
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significant correlated with "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")
            heatmap.loc[col, cluster] = p_val[0]*100

        # scores discrete variables
        for col in scores_dicotomic_targets:
            groups = np.unique(scores.loc[pos, col])
            p_val = st.ttest_ind(expression.loc[pos[(scores.loc[pos, col] == groups[0]).values].index],
                                 expression.loc[pos[(
                                     scores.loc[pos, col] == groups[1]).values].index],
                                 equal_var=False, random_state=42)
            p_values_covariates.append(p_val[1])
            statements_covariates.append("\t-significantly different between "+str(col) +
                                         " corr: "+str(p_val[0]*100)+" %   FDR: ")

    # PLOTS
    if cluster in interesting_clusters:  # use -2 to make it impossible and skip plots
        expression_labels = expression.copy()
        expression_labels[responders] = "responders"
        expression_labels[non_responders] = "non responders"
        expression_labels[placebo] = "placebo"

        plot_db = pd.concat([expression, expression_labels], axis=1)
        plot_db.columns = ["eigengene expression", "condition"]
        plot_db.sort_values(by="condition", inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Usual boxplot
        sns.boxplot(x='condition', y='eigengene expression', data=plot_db, ax=ax,
                    order=["placebo", "non responders", "responders"],
                    showmeans=True, meanprops={"marker": "+",
                                               "markeredgecolor": "black",
                                               "markersize": "15"})

        # Add jitter with the swarmplot function
        sns.swarmplot(x='condition', y='eigengene expression', data=plot_db,
                      color="red", ax=ax,
                      order=["placebo", "non responders", "responders"])

        ax.set_title("Module "+str(cluster)+" eigengene expression")
        if True:
            fig.savefig("/home/lyro/Scrivania/pSS/results/module "+str(cluster) +
                        " eigengene average expression.png")

        if show_genes_significance:

            # show the N most relevant genes within module
            n = np.min([30, len(cluster_genes)])

            significances = 100*pca.components_[0]**2
            order_s = np.argsort(np.abs(significances))[::-1][:n]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.barh(y=np.arange(n)[::-1],
                    width=significances[order_s],
                    color=np.where(pca.components_[0][order_s] > 0., 'darkred',
                                   'darkblue'),
                    tick_label=[a for a in genes_names.loc[cluster_genes].iloc[order_s].Name])
            ax.legend(loc='lower right')
            handles, labels = ax.legend().axes.get_legend_handles_labels()
            handles.append(Patch(facecolor='darkred', edgecolor='darkred'))
            labels.append("positive expression")
            handles.append(Patch(facecolor='darkblue', edgecolor='darkblue'))
            labels.append("negative expression")
            ax.legend(handles, labels, loc='lower right', fontsize=ft)
            ax.set_xlabel("normalized genes weights in 1st eigengene (%)",
                          fontsize=ft)

            fig.suptitle("cluster "+str(cluster)+" eigengene composition")
            fig.tight_layout()
            if save:
                fig.savefig("/home/lyro/Scrivania/pSS/results/module " +
                            str(cluster)+" genes significance.png")

            # print(np.cumsum(np.abs(significances)))
            # print(geneinfo.loc[cluster_genes].iloc[order_s])

            interesting_genes = np.asarray(cluster_genes)  # [order_s]

            degrees = correlation_matrix_soft_thresholded[clusters == cluster].sum(
                axis=1)
            order_d = np.argsort(degrees)[::-1][:n].values

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.barh(y=np.arange(n)[::-1],
                    width=degrees[order_d],
                    color='darkgreen',
                    tick_label=[a for a in genes_names.loc[cluster_genes].iloc[order_s].Name])
            ax.legend(loc='lower right')

            fig.suptitle("cluster "+str(cluster)+" gene degrees")
            fig.tight_layout()
            if save:
                fig.savefig("/home/lyro/Scrivania/pSS/results/module " +
                            str(cluster)+" hub genes.png")

            # print([a[0] for a in geneinfo.loc[cluster_genes].iloc[order_d].index])
            # print([a[1] for a in geneinfo.loc[cluster_genes].iloc[order_d].index])

        if time_trend:

            expr = db.loc[:, cluster_genes]*pca.components_[0]
            expr = expr.sum(axis=1)

            t_labels = metadata.loc[expr.index, "Treatment"]
            r_labels = metadata.loc[expr.index, "Responder"]
            expression_labels = r_labels
            expression_labels[t_labels != 'Verum medication'] = 'placebo'

            expression_times = metadata.loc[expr.index, "Visitnumber"]

            plot_db = pd.concat([expr, expression_labels, expression_times],
                                axis=1)
            plot_db.columns = ["eigengene expression", "condition", "time"]
            # plot_db.sort_values(by="Time", inplace=True)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # Usual boxplot

            sns.boxplot(x='time', y='eigengene expression', data=plot_db, ax=ax,
                        hue='condition',
                        order=["Baseline", "8 weeks", "16 weeks", "24 weeks"],
                        hue_order=["placebo", "non-responder", "responder"],
                        showmeans=True, meanprops={"marker": "+",
                                                   "markeredgecolor": "black",
                                                   "markersize": "15"})

            # Add jitter with the swarmplot function
            # sns.swarmplot(x='time', y='eigengene expression', data=plot_db, ax=ax,
            #               order=["Baseline", "8 weeks", "16 weeks", "24 weeks"], hue='condition')

            ax.set_title("Module "+str(cluster) +
                         " eigengene expression over time for different groups")
            if True:
                fig.savefig("/home/lyro/Scrivania/pSS/results/module " +
                            str(cluster)+" eigengene expression over time.png")


# find significant pvalues & print summary

significants_targets = multipletests(
    p_values_targets, alpha=0.05, method='fdr_bh')
significants_covariates = multipletests(
    p_values_covariates, alpha=0.05, method='fdr_bh')


with open(results_storage_dir+'Eigengenes results.txt', 'w') as f:
    for i in np.arange(len(statements_targets)+len(statements_covariates)):
        pos1 = i // tests_per_cluster
        pos2 = i % tests_per_cluster

        if pos2 == 0:
            print(headers[pos1], file=f)

        if pos2 < 2:
            idx = pos1*2+pos2
            if significants_targets[0][idx]:
                print(statements_targets[idx],
                      significants_targets[1][idx], file=f)
        else:
            idx = pos1*(tests_per_cluster-2)+pos2-2
            if significants_covariates[0][idx]:
                print(statements_covariates[idx],
                      significants_covariates[1][idx], file=f)


heatmap = heatmap.astype(float)


# %%  CORRELATION HEATMAP W COVARIATES


sns.set(font_scale=1.5)

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap, vmin=-100, vmax=100, cmap='RdBu_r', cbar_kws={"label": "Spearman correlation (%)",
                                                                   "orientation": "vertical",
                                                                   "pad": 0.05,
                                                                   "fraction": 0.045},
            linewidths=0.1, linecolor='k', cbar=True, square=True)
plt.xlabel("Cluster")
plt.title("Correlation between cluster eignengene expression and clinical features\n WB data",
          fontsize=18)

plt.tight_layout()
plt.savefig("/home/lyro/Scrivania/pSS/results/correlations.png")
