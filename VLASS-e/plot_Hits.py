import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from utils import STYLE_PATH, calcFigSize, get_colors

RFI_PATH = "/home/cat-work/work/SETI/cosmicStellarHosts/Full_Crickets_CleanedUp.pkl"

axes_labels = {'signal_frequency': "Signal Frequency (MHz)",
                'signal_drift_rate': "Signal Drift Rate (Hz)",
                'signal_snr': "Signal Signal-to-Noise"}

def plot_src_data(filenames, outfile=None):

    # filenames = sorted(glob.glob(filenames))
    colors = get_colors(len(filenames), cmap='jet')

    # Format plot
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)

    x_axis_data = ['signal_frequency', 'signal_drift_rate']
    y_axis_data = ['signal_snr']
    # Go
    fig, ax = plt.subplots(len(y_axis_data),len(x_axis_data), figsize=figs, constrained_layout=True)
    total_hits = 0
    for i in range(len(filenames)):
        print(f"Plotting {filenames[i]}")
        obs_data = pd.read_pickle(filenames[i])
        total_hits += obs_data.shape[0]
        print(f"({obs_data.shape[0]}, {total_hits})")
        sources = obs_data.source_name.unique()
        
        for k in range(len(x_axis_data)):
            ax[k].set_title(f"{x_axis_data[k].partition('_')[2]}-{y_axis_data[0].partition('_')[2]}")
            ax[k].set_xlabel(axes_labels[x_axis_data[k]])
            ax[k].set_ylabel(axes_labels[y_axis_data[0]])
            ax[k].scatter(obs_data[x_axis_data[k]], obs_data[y_axis_data[0]], color=colors[i], s=1, label=str(sources)) #maybe sources[0]


    # Plot RFI Bars:
    # rfi = pd.read_pickle(RFI_PATH)
    # for _,row in rfi.iterrows():
    #     ax[0].axvspan(
    #         row['start_frequency'], 
    #         row['stop_frequency'],
    #         color='gray',
    #         alpha=0.1,
    #     )


    # Temporary axes limits:
    ax[0].set_xlim(2000, 4000)
    ax[1].set_xlim(-50, 50)
    ax[-1].legend(frameon=True, facecolor='lightgrey',framealpha=0.5, fontsize=4)
    

    if args.save_filepath:
        plt.savefig(args.save_filepath)
    plt.show()

    return

def main(args):
    plot_src_data(args.filenames, outfile=args.save_filepath)
    return
    


if __name__ == "__main__":
    
# This should probably move to a different function where you can pick what to plot with args
    parser = argparse.ArgumentParser(
            description='This code is designed to apply a set of filters to incoming COSMIC-VLASS data to identify interesting candidate signals. ',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filenames', type=str, nargs='+',
                        help='filename path pattern to be globbed')
    parser.add_argument('-s', '--save_filepath', type=str,
                        help='output file path for plot')
    # parser.add_argument('-x', '--x_axis', nargs='+', type=str, default='signal_frequency', 
    #                     help='what variable to plot on x-axis, max 4 inputs'
    # parser.add_argument('-y', '--y_axis', type=str, default='signal_snr', 
    #                     help='target or list of targets to be processed. You can also specify "all" to process each source in the csv independently (Use with caution as many Obs Ids have lots of sources within!). ')

    args = parser.parse_args()
    # print(args)
    main(args)

# if args.plot:
#     t_unique = pd.read_pickle('/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/ObsId_31524/src2824770686019003904_100snr_50dr_unique.pkl')
#     plot_src_data(t_unique)

"""
This is how I plotted the rfi regions last time. 
In principle it should be even easier than this with just frequencies instead of dates. 
The rectangle collection I expect is the most useful part. 

def rfiLines():
    day = "2024/07/11 "
    rfi =  [day + r for r in ["00:38:19", "01:28:35","13:48:21", "15:00:00", "22:38:36", "23:58:35"]]
    sun =  [day + s for s in ["15:58:39", "18:28:46"]]
    lightning = [day + l for l in ["18:28:48", "20:08:40"]]


    #conv to dt
    rfi_dt = np.array([datetime.strptime(t, '%Y/%m/%d %H:%M:%S') for t in rfi])
    sun_dt = np.array([datetime.strptime(t, '%Y/%m/%d %H:%M:%S') for t in sun])
    ltg_dt = np.array([datetime.strptime(t, '%Y/%m/%d %H:%M:%S') for t in lightning])

    return mdates.date2num(rfi_dt), mdates.date2num(sun_dt), mdates.date2num(ltg_dt)
rfi, sun, ltg = rfiLines()
# Flag RFI
Rects = []
for i in range(3):
    anchor = rfi[2*i]
    extnt = -1*(anchor-rfi[(2*i)+1])
    Rects.append(Rectangle((anchor,-0.2), height=1.4, width = extnt, alpha=0.5))

pc = PatchCollection(Rects, edgecolor='k', facecolor='grey', alpha=0.5)
axs.add_collection(pc)
"""