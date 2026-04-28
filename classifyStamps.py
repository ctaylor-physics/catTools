import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
from utils import STYLE_PATH, calcFigSize, get_colors

import glob
from cosmic_utils import look_for_combs, log_with_pandas
from scipy.stats import median_abs_deviation
from seticore import viewer
from cosmic_database_analysis import sarfi

from blri.dsp import upchannelise


PLOT_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc'
STAMP_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/final_10pc_candidates'

def view_png(filename):
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return

def copy_to_new_directory(df_column,DIR):
    os.makedirs(DIR, exist_ok=True)
    for file in df_column:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(DIR, os.path.basename(file)))
    return 

def sort_plots(node = 1):

    # Load in all the storage 1 files
    storage1 = pd.read_csv(os.path.join(PLOT_PATH, f'uniqueFinalHits10pc_storage{node}.csv'))
    plot_diagnostics = pd.read_csv(os.path.join(PLOT_PATH, f'diagnostic_plots_s{node}', 'stamp_diagnostics.csv'))
    plot_files = sorted(glob.glob(os.path.join(PLOT_PATH, f'diagnostic_plots_s{node}', '*.png')))
    show_antennas_plots = sorted(glob.glob(os.path.join(PLOT_PATH, f'diagnostic_plots_s{node}', '*show_antennas.png')))

    newcolumn = []
    for i,row in plot_diagnostics.iterrows():
        substring = f"{row.source_name}_id{row.id}"
        for value in filter(lambda x: substring in x, show_antennas_plots):
            newcolumn.append(value) 

    plot_diagnostics['show_ants_plot'] = newcolumn


    # Plots that do not have SARFI according to our tests:
    craig_sarfi = plot_diagnostics[plot_diagnostics.signal_score == True]
    only_sarfi = plot_diagnostics[plot_diagnostics.sarfi_score.isna()]
    both_sarfi = plot_diagnostics[(plot_diagnostics.signal_score == True) & (plot_diagnostics.sarfi_score.isna())]

    print('craig', len(craig_sarfi))
    print('sarfi', len(only_sarfi))
    print('both', len(both_sarfi))

    # missing = plot_diagnostics[~plot_diagnostics.show_ants_plot.isin(both_sarfi.show_ants_plot)]
    # missing_fname_sort = missing..sort_values(by='show_ants_plot')
    # copy_to_new_directory(missing.show_ants_plot, os.path.join(PLOT_PATH, f'missing_s{node}'))
    # Note my manual inspection was sorted using both_sarfi.sort_values(by='show_ants_plot'), then saved to both_sarfi_classifier.csv
    return


def plot_useful_classes(obs_info_classified):
    """
    Plot, in different colors, the hits that I deem to be worthy of investigation
    """
    # Format plot
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)
    # unique_labels = obs_info_classified.classification.unique()
    useful_labels = ['stepped  candidate', 'candidate', 'non-uniform slope', ]
    # colors = get_colors(len(useful_labels), cmap='jet')

    fig, ax = plt.subplots(1,1,figsize=figs,constrained_layout=True)

    for i,label in enumerate(useful_labels):
        plotdf = obs_info_classified[obs_info_classified.classification == label]
        ax.scatter(plotdf.signal_frequency, plotdf.signal_drift_rate, label=label, marker = '.', s=3)
    ax.set_xlabel('Signal_Frequency (MHz)')
    ax.set_ylabel('Signal_Drift_Rate')
    plt.legend(bbox_to_anchor = (1.05,0.5), loc='center left')
    plt.show()
    return

def plot_individual_classes(obs_info_classified):
    """
    Iteratively plot all my classifications in individual plots .
    """
    # Format plot
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)
    unique_labels = obs_info_classified.classification.unique()
    colors = get_colors(len(unique_labels), cmap='jet')

    for i,label in enumerate(unique_labels):
        if label == 'candidate':
            mark = 'D'
            colors[i] = 'k'
        else:
            mark = '.'
        plotdf = obs_info_classified[obs_info_classified.classification == label]
        fig, ax = plt.subplots(1, 1, figsize=figs, constrained_layout=True)
        ax.scatter(plotdf.signal_frequency, plotdf.signal_drift_rate,
                color = colors[i], label=label, marker = mark, s=2)
        ax.set_xlabel('Signal_Frequency (MHz)')
        ax.set_ylabel('Signal_Drift_Rate')
        plt.legend(bbox_to_anchor = (1.05,0.5), loc='center left')
        plt.show()
        return


if os.path.exists(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_classified.csv")):
    obs_info_classified = pd.read_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_classified.csv"))
else:
    # Get Classifications and Combine into single table
    classified_s1 = pd.read_csv(os.path.join(PLOT_PATH, 'both_sarfi_classifier_s1_label.csv')).drop('Column1', axis=1)
    classified_s2 = pd.read_csv(os.path.join(PLOT_PATH, 'both_sarfi_classifier_s2_label.csv')).drop('Column1', axis=1)
    classified_stamps = pd.concat((classified_s1, classified_s2)).sort_values(by='id')

    # Recombine the observation info
    obs_info_s1 = pd.read_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_storage1.csv")).drop('Unnamed: 0', axis=1)
    obs_info_s2 = pd.read_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_storage2.csv")).drop('Unnamed: 0', axis=1)
    obs_info = pd.concat((obs_info_s1, obs_info_s2)).sort_values(by='id')

    # Gather those that I classified using pd.merge to overlap the two mismatched catalogs
    obs_info_classified = pd.merge(obs_info, classified_stamps, on='id', how='inner', suffixes=('_oi', "_sc"))
    obs_info_classified.to_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_classified.csv"))
    

# Could add: 3478160727866058368_id1511193325_2494.11892MHz_show_antennas.png - non-uniform
        # 4330690742322011520_id1299722665_3413.33708MHz_show_antennas.png - weird half hit
        # 4330690742322011520_id1299722673_3413.54607MHz_show_antennas.png - weird half hit

try:
    obs_info_final = pd.read_csv(os.path.join(PLOT_PATH, 'uniqueFinalHits10pc_shortlist.csv'))
except:
    obs_info_final = obs_info_classified[(obs_info_classified.classification == 'candidate') |
                                         (obs_info_classified.classification == 'stepped  candidate') |
                                         (obs_info_classified.classification == 'non-uniform slope')]
    obs_info_final.to_csv(os.path.join(PLOT_PATH, 'uniqueFinalHits10pc_shortlist.csv'))

plot_useful_classes(obs_info_classified)

copy_to_new_directory(obs_info_final.show_ants_plot, os.path.join(PLOT_PATH, 'final_stamp_plots'))
