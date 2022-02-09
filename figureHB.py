#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Figure configuration class for scientific publication
#
#
# Authors: Hao BAI
# Date: 12/08/2020
#
# Version:
#   - 0.0: Initial version
#
# Comments:
#   - suggest to use this package with `figure.mplstyle`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#!------------------------------------------------------------------------------
#!                                          MODULES PRÉREQUIS
#!------------------------------------------------------------------------------
#*============================== Modules Personnels ============================

#*============================== Modules Communs ==============================
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import warnings



#!------------------------------------------------------------------------------
#!                          CONSTANT AND DEFAULT VALUE
#!------------------------------------------------------------------------------
GOLDEN_RATIO = (1+5**0.5)/2
TEXTWIDTH = 434.90039 / 72  # unit: pt -> inch
TEXTHEIGHT = 700.50687 / 72  # unit: pt -> inch
DEFAULTWIDTH = TEXTWIDTH
DEFAULTHEIGHT = TEXTWIDTH / GOLDEN_RATIO
SMALL = 8  # small fontsize (unit: pt)
LEFT = 0.097  # figure left spacing
RIGHT = 0.983  # figure right spacing
BOTTOM = 0.123  # figure bottom spacing
TOP = 0.985  # figure top spacing



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                  FUNCTIONS
#!------------------------------------------------------------------------------
def align_yaxis(axes, left=0.10, width=0.882, labelx=-0.085, labely=0.5):
    ''' Align yaxis, ylabel and yticklabel
    '''
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes, ]
    for ax in axes:
        origin = ax.get_position()
        ax.set_position([left, origin.y0, width, origin.height])
        ax.yaxis.set_label_coords(labelx, labely)


def draw(figure, toPDF, transparent=False):
    ''' Render figure to screen or file
    '''
    if toPDF != [] and toPDF[0]:
        # export figure to pdf file
        if ".pdf" not in toPDF[1]:
            filename = toPDF[1] + ".pdf"
        else:
            filename = toPDF[1]
        plt.savefig(filename, transparent=transparent)
    else:
        # show figure on screen
        # plt.tight_layout()
        plt.show()
    plt.close()


def adjust_figure(figure, **kwargs):
    ''' Adjust figure properties
    '''
    # figure size
    if "width" in kwargs:
        w = kwargs.get("width")
        kwargs.pop("width")
    else:
        w = DEFAULTWIDTH
    if "height" in kwargs:
        h = kwargs.get("height")
        kwargs.pop("height")
    else:
        h = DEFAULTHEIGHT
    figure.set_size_inches(w, h)
    # figure margins
    figure.subplots_adjust(**kwargs)
    # figure.tight_layout(pad=1.1)


def adjust_axis(axis, title="", loc=0, minor=[True, False, False, False],
    xlim=[], ylim=[], xlabel="Time (s)", ylabel="", xVisible=True, 
    yVisible=True, seperator=True):
    ''' Adjust axis properties
    '''
    axis.set_title(title)
    if isinstance(loc, int):
        axis.legend(loc=loc)

    if minor[0]: # X ticks minorlocator
        axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if minor[1]: # X minor axis
            axis.grid(which="minor", axis="x", color="tab:gray", linestyle=":")
    if minor[2]: # Y ticks minorlocator
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        if minor[3]: # Y minor axis
            axis.grid(which="minor", axis="y", color="tab:gray", linestyle=":")
    if minor != [False, False, False, False]:
        axis.grid(which="major")

    if xlim != []:
        if len(xlim) == 2:
            axis.set_xlim(xlim[0], xlim[1])
        if len(xlim) == 3:
            axis.set_xlim(xlim[0], xlim[1])
            axis.set_xticks(np.arange(xlim[0], xlim[1]+xlim[2], xlim[2]))
    if ylim != []:
        if len(ylim) == 2:
            axis.set_ylim(ylim[0], ylim[1])
        if len(ylim) == 3:
            axis.set_ylim(ylim[0], ylim[1])
            axis.set_yticks(np.arange(ylim[0], ylim[1]+ylim[2], ylim[2]))

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if not xVisible: plt.setp(axis.get_xticklabels(), visible=False)
    if not yVisible: plt.setp(axis.get_yticklabels(), visible=False)

    # use " " as thousand seperator
    if seperator:
        axis.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x,pos: format(int(x), ",")
            .replace(",", " ")))


def autolabel(axis, rects, xpos="center", fontSize="medium"):
    '''
    Attach a text label above each bar in *rects*, displaying its height.
    INPUT
        rects: an object of matplotlib.axes.Axes.bar
        xpos: indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {"center", "right", "left"}.
    '''
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {"center": "center", "right": "left", "left": "right"}
    offset = {"center": 0.5, "right": 0.57, "left": 0.43}  # x_txt = x + w*off
    for rect in rects:
        height = rect.get_height()
        if height != 0:
            axis.text(rect.get_x()+rect.get_width()*offset[xpos], 1.002*height,
                "{}".format(height), ha=ha[xpos], va="bottom", size=fontSize)


def zoom_outside(srcax, roi, dstax, color="cyan", linestyle="-", linewidth=0.8, 
    roiKwargs={}, arrowKwargs={}):
    ''' Create a zoomed subplot outside the original subplot

    srcax: matplotlib.axes
        Source axis where locates the original chart
    dstax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", linestyle),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    # draw a rectangle on original chart
    srcax.add_patch(Rectangle([roi[0], roi[1]], roi[2]-roi[0], roi[3]-roi[1], 
                            **roiKwargs))
    # get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]
    dstCorners = dstax.get_position().corners()
    srcBB = srcax.get_position()
    dstBB = dstax.get_position()
    # find corners to be linked
    if srcBB.max[0] <= dstBB.min[0]: # right side
        if srcBB.min[1] < dstBB.min[1]: # upper
            corners = [1, 2]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [0, 1]
        else:
            corners = [0, 3] # lower
    elif srcBB.min[0] >= dstBB.max[0]: # left side
        if srcBB.min[1] < dstBB.min[1]:  # upper
           corners = [0, 3]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [2, 3]
        else:
            corners = [1, 2]  # lower
    elif srcBB.min[0] == dstBB.min[0]: # top side or bottom side
        if srcBB.min[1] < dstBB.min[1]:  # upper
            corners = [0, 2]
        else:
            corners = [1, 3] # lower
    else:
        RuntimeWarning("Cannot find a proper way to link the original chart to "
                       "the zoomed chart! The lines between the region of "
                       "interest and the zoomed chart wiil not be plotted.")
        return
    # plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        srcax.annotate('', xy=srcCorners[corners[k]], xycoords="data",
            xytext=dstCorners[corners[k]], textcoords="figure fraction",
            arrowprops=arrowKwargs)


def set_axis_sci(ax, axis="y", scilimits=(3, 3), textx=0.01, texty=0.95):
    ax.ticklabel_format(axis=axis, style="sci", scilimits=scilimits)
    if axis == "x": # hide default offset
        ax.xaxis.get_offset_text().set_visible(False)
    elif axis == "y":
        ax.yaxis.get_offset_text().set_visible(False)
    if scilimits[0] < 0 or scilimits[0] > 9:
        exponent = "{" + str(scilimits[0]) + "}"
    else:
        exponent = str(scilimits[0])
    ax.text(textx, texty, r"$\times10^{}$".format(exponent),
        transform=ax.transAxes)


def parallelplot(axis, ys, color, axislabels=None, bezier_curve=False):
    ''' Parallel plot
        ys: list or np.ndarray
            (N*M) array where N is the number of data, M is the number of axis
        color: list or np.ndarray
            (N,) array indicates the color to use line by line
        ref: https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    '''
    # read data
    ys, color, N = np.array(ys), np.array(color), len(color)

    # organize the data
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    # ymins = ymins - dys * 0.05  # add 5% padding below and above
    # ymaxs = ymaxs + dys * 0.05
    # dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # prepare the subplot
    axes = [axis] + [axis.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="y", labelsize=SMALL)
        if ax != axis:  # hide left axis
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
        if i == 0:
            ax.set_yticks(range(3, 27, 2))
        if i == 1:
            ax.set_yticks([])
        if i == 2:
            ax.set_yticks(range(0, 360, 10))
    
    # adjust the style of subplot
    axis.set_xlim(0, ys.shape[1] - 1)
    axis.set_xticks(range(ys.shape[1]))
    if not axislabels is None:
        axis.set_xticklabels(axislabels)  # fontsize=14
    axis.tick_params(axis='x', which='major', pad=7)
    axis.spines['right'].set_visible(False)
    axis.xaxis.tick_top()

    if bezier_curve == True:
        # draw bezier curves between the axes:
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        #   at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice
        from matplotlib.path import Path
        import matplotlib.patches as patches
        for j in range(N):
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                             np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: axis.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + \
                [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none',
                                      lw=1, edgecolor=color[j],)
            axis.add_patch(patch)
    else:
        # draw straight lines between the axes:
        for j in range(N):
            axis.plot(range(ys.shape[1]), zs[j, :], c=color[j],)
    return axis
