################ ##
## Lorenzo Dall'Olio
## 15 dec 2022
## Various Plot regarding the interesting genes clusters
################ ##


import os
import random
import requests

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
                                      random_state=123456789)

# =============================================================================
# LOAD DATA
# =============================================================================

working_directory = "./"

db = pd.read_csv(working_directory+"dataset_vst.csv", index_col=0)

clusters = pd.DataFrame(index=db.columns)


for a in range(13):
    gens = pd.read_csv(working_directory+"/results/genes lists/module_"+str(a)+"_genes.tsv",
                       sep='\t', index_col=0)
    clusters.loc[gens.index, 'cluster'] = a



db = db.loc[:, clusters.index]
clusters = np.hstack(np.asarray(clusters))

metadata = pd.read_csv(working_directory+"metadata.csv", index_col=0)
genes_names = pd.read_csv(working_directory+"gene_names.csv", index_col=0)
results_storage_dir = working_directory+"results/"

embed = np.load(results_storage_dir+"embedding.npy")


metadata = metadata.loc[db.index, :]

save_plots = False
convert_gene_names = True


# =============================================================================
# FIND TARGET GENES IN UMAP EMBEDDING
# =============================================================================

interesting_genes_coded = []
drugs = ['1', '2', 'etc']  # put here interesting drugs

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
