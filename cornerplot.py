import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from tqdm import tqdm


def CornerPlot(dfs, df_names, corner_params, weights=None, bandwidth_fac=1, thresh=[68,95], downsample=False, prior=None, \
                cuts=None, limits=None, plot_limits=None, labels=None, ticks=None, title=None, Nbins=50, plot_pts=False, plot_hist=False, print_credible=90, colormap=None, save=False):
    """
    Makes a sexy cornerplot.

    :dfs: list of pandas dataframes that have the data you wish to plot
    :df_names: list that specifies the label for each dataframe in the legend
    :corner_params: names of the series you wish the use in the corner plot
    :weights: name of the series that contains weights for each sample
    :bandwidth_fac: multiplicative factor to the bandwidth of the KDE (values >1 make KDE smoother)
    :downsample: can specify an int which allows you to downsample the data for the 2D densities to speed things up
    :prior: dataframe with the prior, if you want to plot this too; keys should match those in :corner_params:
    :cuts: dict allowing you to slice the data by specifying a tuple of upper and lower bounds for each parameter
    :limits: dict of tuples for the limits of each parameter for the KDE evaluation (so it doesn't go outside the bounds)
    :plot_limits: dict of tuples for the plotting limits of each parameter
    :labels: dict of labels to use for annotating :corner_params:
    :ticks: dict of ticks to use for each parameter
    :title: suptitle of the corner plot
    :Nbins: number of bins (in each dimension) to use for the marginalized histograms and for constructing the 2d density
    :plot_pts: boolean that determines whether to plot scatterplot points
    :plot_hist: boolean that determines whether histogram is plotted behind marginalized KDEs
    :print_credible: prints the median and symmetric credible interval provided above the marginalized distributions if not False
    :colormap: allows for user-input color maps
    :save: saves as `cornerplot.png` if True, or at specific path if string is provided
    """
    # set up figure and axes
    fig = plt.figure(figsize=(7*len(corner_params),6*len(corner_params)))
    gs = gridspec.GridSpec(len(corner_params),len(corner_params))
    gs.update(wspace=0.05, hspace=0.05)

    # get colormap
    stock_colors = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
    colors = colormap if colormap is not None else stock_colors

    # store all the marginal distribution axes
    marg_axs = [fig.add_subplot(gs[i,i]) for i in range(len(corner_params))]

    # dict for storing joint axes, where first key is the xidx and second key is the yidx
    joint_axs = {}

    # if None is provided for dict arguments (cuts, limits, plot_limits, labels, ticks), create dict of all None values
    _cuts = {key: None for key in corner_params} if cuts==None else cuts
    _limits = {key: None for key in corner_params} if limits==None else limits
    _plot_limits = {key: None for key in corner_params} if plot_limits==None else plot_limits
    _labels = {key: None for key in corner_params} if labels==None else labels
    _ticks = {key: None for key in corner_params} if ticks==None else ticks
    for param in corner_params:
        if param not in _cuts.keys():
            _cuts[param] = None
        if param not in _limits.keys():
            _limits[param] = (None,None)
        if param not in _plot_limits.keys():
            _plot_limits[param] = (None,None)
        if param not in _labels.keys():
            _labels[param] = None
        if param not in _ticks.keys():
            _ticks[param] = None

    # loops over each dataframe
    for df_idx, df in tqdm(enumerate(dfs), total=len(dfs)):

        # cut the data, if specified
        for param in _cuts.keys():
            if _cuts[param] is not None:
                if _cuts[param][0] is not None:
                    df = df.loc[(df[param]>=_cuts[param][0])]
                if _cuts[param][1] is not None:
                    df = df.loc[(df[param]<=_cuts[param][1])]

        # if number to downsample is specified, downsample the data
        if downsample != False:
            if len(df) > downsample:
                df = df.sample(downsample, replace=False)

        # get weights, if provided
        _weights = None if weights==None else np.asarray(df[weights])

        # loop over all parameters
        for idx, param in enumerate(corner_params):

            # store data for given parameter
            param_data = np.asarray(df[param])

            ### PLOT MARGINALIZED DISTRIBUTIONS ###
            sns.kdeplot(data=param_data, ax=marg_axs[idx], weights=_weights, bw_adjust=bandwidth_fac, \
                        gridsize=1000, clip=(_limits[param]), color=colors[df_idx], lw=2, vertical=False, label=df_names[df_idx])
            if plot_hist==True:
                _ = marg_axs[idx].hist(param_data, density=True, weights=_weights, histtype='step', color=colors[df_idx], bins=Nbins, \
                            alpha=0.4, orientation="vertical", label=None)

            # plot prior distributions, if provided
            if (prior is not None) and (df_idx==len(dfs)-1):
                if prior[param] is not None:
                    prior_data = np.asarray(prior[param])
                    sns.kdeplot(data=prior_data, ax=marg_axs[idx], bw_adjust=bandwidth_fac, \
                                gridsize=1000, color='k', lw=1, linestyle=':', vertical=False, label='prior')
                    if plot_hist==True:
                        _ = marg_axs[idx].hist(prior_data, density=True, weights=_weights, histtype='step', color='k', bins=Nbins, \
                                    alpha=0.4, linestyle=':', lw=1, orientation="vertical")

            # plot median and credible range for the specified threshold values
            median = np.median(param_data)
            marg_axs[idx].axvline(median, color=colors[df_idx], linestyle=':')
            for tidx, t in enumerate(thresh):
                cred_low = np.percentile(param_data, (100-t)/2.0)
                cred_high = np.percentile(param_data, 100 - (100-t)/2.0)
                marg_axs[idx].axvline(cred_low, color=colors[df_idx], alpha=0.6, ymin=0.95, lw=3/(tidx+1))
                marg_axs[idx].axvline(cred_high, color=colors[df_idx], alpha=0.6, ymin=0.95, lw=3/(tidx+1))


            # adjust plot limits if provided
            marg_axs[idx].set_xlim(_plot_limits[param])

            # adjust ticks if provided
            if _ticks[param] is not None:
                marg_axs[idx].set_xticks(_ticks[param])
                marg_axs[idx].set_yticklabels([])

            # remove labels for marginalized axes except for last one
            if param==corner_params[-1]:
                if _labels[param] is not None:
                    xlbl = _labels[param]
                else:
                    xlbl = param
                marg_axs[idx].set_xlabel(xlbl)
            else:
                marg_axs[idx].set_xticklabels([])
            marg_axs[idx].get_yaxis().set_visible(False)
            marg_axs[idx].set_ylabel([])

            # if print_credible, print the median and provided credible interval above the marginalized
            # NOTE: this only prints info for the first dataframe provided!
            if print_credible is not False and df_idx==0:
                median = np.median(param_data)
                cred_low = np.percentile(param_data, (100-print_credible)/2.0)
                cred_high = np.percentile(param_data, 100 - (100-print_credible)/2.0)
                marg_axs[idx].set_title(r'$%0.2f ^{+%0.2f} _{-%0.2f}$' % (median, cred_high-median, median-cred_low), pad=15)

            # setup joint axes
            for joint_idx, joint_param in enumerate(corner_params):
                if joint_idx <= idx:
                    continue

                # grab axes, or setup axes if this is the first instance
                ax_key = str(idx)+str(joint_idx)
                if ax_key not in joint_axs.keys():
                    joint_ax = fig.add_subplot(gs[joint_idx,idx])
                else:
                    joint_ax = joint_axs[ax_key]

                # store data for the joint parameter
                joint_param_data = np.asarray(df[joint_param])

                # plot median value
                median_joint = np.median(joint_param_data)
                joint_ax.scatter(median, median_joint, marker='X', color=colors[df_idx], s=75)

                ### PLOT JOINT DISTRIBUTIONS ###
                thresholds = [1-t/100.0 for t in thresh[::-1]]

                sns.kdeplot(x=param_data, y=joint_param_data, ax=joint_ax, weights=_weights, bw_adjust=bandwidth_fac, \
                            levels=thresholds, clip=(_limits[param],_limits[joint_param]), colors=[colors[df_idx]], \
                            cmap=None, shade=False, alpha=0.7, linewidths=np.linspace(1,3,len(thresholds)))
                if plot_pts==True:
                    joint_ax.scatter(param_data, joint_param_data, \
                                 color=colors[df_idx], s=0.1, marker='.', alpha=0.3, rasterized=True)

                # adjust plot limits, if provided
                joint_ax.set_xlim(_plot_limits[param])
                joint_ax.set_ylim(_plot_limits[joint_param])

                # adjust ticks, if provided
                if _ticks[param] is not None:
                    joint_ax.set_xticks(_ticks[param])
                if _ticks[joint_param] is not None:
                    joint_ax.set_yticks(_ticks[joint_param])

                # adjust labels, if provided, and remove ticks from middle plots
                xlbl = _labels[param] if _labels[param] is not None else param
                ylbl = _labels[joint_param] if _labels[joint_param] is not None else joint_param
                if joint_idx==len(corner_params)-1:
                    joint_ax.set_xlabel(xlbl)
                else:
                    joint_ax.set_xticklabels([])
                if idx==0:
                    joint_ax.set_ylabel(ylbl)
                else:
                    joint_ax.set_yticklabels([])

                # save this updated ax in the joint_axs dict
                joint_axs[ax_key] = joint_ax

    # add legend
    marg_axs[0].legend(loc='upper left', bbox_to_anchor=(len(corner_params)-0.95,1.0), prop={'size':30})

    # add title
    if title is not None:
        plt.suptitle(title, fontsize=30)

    # save figure
    if save is not False:
        if save == True:
            plt.savefig('./corner_plot.png', dpi=300)
        else:
            plt.savefig(save, dpi=300)


