#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of useful functions commonly used in any project.

Created on Sun Apr  7 01:06:34 2019
@author: K. Masunaga, LASP CU Boulder (kei.masunaga@lasp.colorado.edu)
"""
import numpy as np
import os, fnmatch, glob
from itertools import chain
from datetime import datetime, timedelta
import calendar
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from time import strftime
from IPython import get_ipython
ipy = get_ipython()

###############################
#### Time conversion tools ####
###############################

def Dt2unix(timeDt):
    '''
    Converts datetime to unix time
    argment:
        timeDt: a datetime or a list of those
    returns:
        unix_time: a double precision unix time or a list of those
    '''
    if np.size(timeDt) == 1:
        if type(timeDt) is list or type(timeDt) is np.array:
            unix_time = calendar.timegm(timeDt[0].timetuple())
        else:
            unix_time = calendar.timegm(timeDt.timetuple())
    else:
        unix_time = [calendar.timegm(i.timetuple()) for i in timeDt]
    return unix_time


def unix2Dt(unix_time):
    '''
    Converts unix time to datetime
    argment:
        unix_time: a double precision unix time or a list of those
    returns:
        timeDt: a datetime or a list of those
    '''
    if np.size(unix_time) == 1:
        if type(unix_time) is float or type(unix_time) is np.float64:
            return datetime.utcfromtimestamp(unix_time)
        else:
            return datetime.utcfromtimestamp(unix_time[0])
    else:
        return [datetime.utcfromtimestamp(it) for it in unix_time]


def datenum_to_datetime(datenum):
    """
        Convert Matlab datenum into Python datetime.
        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=days) \
        - timedelta(days=366)


#########################################
#### Coordinate transformation tools ####
#########################################

def quaternion_rotation(q, v):

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    t2 = a*b
    t3 = a*c
    t4 = a*d
    t5 = -b*b
    t6 = b*c
    t7 = b*d
    t8 = -c*c
    t9 = c*d
    t10 = -d*d

    v1new = 2*((t8 + t10)*v1 + (t6 - t4)*v2 + (t3 + t7)*v3) + v1
    v2new = 2*((t4 + t6)*v1 + (t5 + t10)*v2 + (t9 - t2)*v3) + v2
    v3new = 2*((t7 - t3)*v1 + (t2 + t9)*v2 + (t5 + t8)*v3) + v3

    return v1new, v2new, v3new


######################
#### Useful tools ####
######################

def flatten_list(list_of_lists):
    flattened_list = list(chain.from_iterable(list_of_lists))
    return flattened_list

def listup_files(path):
    yield [os.path.abspath(p) for p in glob.glob(path)]

def file_search(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    if len(result) == 1:
        return result[0]
    else:
        return result


def get_increment_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_increment_path('/etc/issue')
    '/etc/issue-1'
    >>> get_increment_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    return new_fname


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def mergedict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value , dict1[key]]

    return dict3

def get_timeDt_mean(timeDt):
    if np.size(timeDt) > 1:
        timeDt_mean = unix2Dt(sum(Dt2unix(timeDt))/len(timeDt))
    elif np.size(timeDt) == 1:
        timeDt_mean = timeDt
    elif np.size(timeDt) == 0:
        timeDt_mean = None
    return timeDt_mean

def interpDt(Dt_new, Dt, y):
    utime_new = Dt2unix(Dt_new)
    utime = Dt2unix(Dt)
    y_new = np.interp(utime_new, utime, y)
    return y_new

def datenum_to_datetime(datenum):
    """
        Convert Matlab datenum into Python datetime.
        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=days) \
        - timedelta(days=366)


class NearestIndex:

    def __init__(self, arr, x):
        if np.size(x) > 1:
            self.idx = self.get_nearest_indice(arr, x)
        else:
            if type(x) is list or type(x) is np.array:
                x = x[0]
            self.idx = self.get_nearest_index(arr, x)

    def get_nearest_index(self, arr, x):
        delta = abs(arr - x)
        idx = np.where(delta == np.min(delta))[0][0]
        return idx

    def get_nearest_indice(self, arr, x):
        idx = np.array([self.get_nearest_index(arr, ix) for ix in x])
        return idx

class NearestDtIndex:

    def __init__(self, Dtarr, Dt):
        if np.size(Dt) > 1:
            self.idx = self.get_nearestDt_indice(Dtarr, Dt)
        else:
            if type(Dt) is list or type(Dt) is np.array:
                Dt = Dt[0]
            self.idx = self.get_nearestDt_index(Dtarr, Dt)

    def get_nearestDt_index(self, Dtarr, Dt):
        delta = np.array([abs((iDt - Dt).total_seconds()) for iDt in Dtarr])
        idx = np.where(delta == np.min(delta))[0][0]
        return idx

    def get_nearestDt_indice(self, Dtarr, Dt):
        idx = np.array([self.get_nearestDt_index(Dtarr, iDt) for iDt in Dt])
        return idx

def nn(arr, x):
    nearest = NearestIndex(arr, x)
    return nearest.idx

def nnDt(Dtarr, Dt):
    nearest = NearestDtIndex(Dtarr, Dt)
    return nearest.idx

class RunTime:
    '''
        Caclulate a runnning time of a code.
        The calculated time would be a time difference between
        RunTime.start() and RunTime.stop().
        If RunTime.start() is not set, the start time is defined when the RunTime object is called.
        Usage:
        from RunTime import RunTime
        RT = RunTime()
        RT.start()
        Write any code here.
        RT.stop()
        '''
    def __init__(self):
        self.startrun = time.time()

    def start(self):
        self.startrun = time.time()

    def stop(self):
        runtime = time.time() - self.startrun
        print ("Runtime:{0}".format(runtime) + "[sec]")
        print ("Runtime:{0}".format(runtime/60) + "[min]")

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class MyLogger:
    def __init__(self):
        self.isnotebook = self._isnotebook()

    def _isnotebook(self):
        return isnotebook()

    def start(self):
        ipy = get_ipython()
        now = datetime.now()
        this_month = '{:04d}{:02d}'.format(now.year, now.month, now.day)
        current_dir = ipy.magic('pwd')
        ldir = current_dir + '/log/' + this_month
        os.makedirs(ldir, exist_ok=True)
        if self.isnotebook:
            #fname = strftime('%Y-%m-%d') + '.log' #if notebook
            pass
        else:
            #fname = strftime('%Y-%m-%d-%H-%M-%S') + '.log' #if ipython
            fname = strftime('%Y-%m-%d') + '.log' #if ipython

        filename = os.path.join(ldir, fname)
        ipy.run_line_magic('logstart', '-o %s append' % filename)

    def stop(self):
        ipy.magic('logstop')
        pass


############################
#### plot related tools ####
############################

def copy_plot_width(ax_src, ax_dest):
    r'''
    copies the wifth of the plot using ax_src to the plot using ax_dest
    '''

    ll, bb, _, hh = ax_dest.get_position().bounds
    _, _, ww, _ = ax_src.get_position().bounds
    ax_dest.set_position([ll, bb, ww, hh])


class AnchoredHScaleBar(mpl.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = mpl.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **kwargs)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **kwargs)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **kwargs)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = mpl.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = mpl.offsetbox.VPacker(children=[size_bar,txt],
                                                 align="center", pad=ppad, sep=sep)
        mpl.offsetbox.AnchoredOffsetbox.__init__(self, loc,pad=pad,borderpad=borderpad,child=self.vpac, prop=prop,frameon=frameon)

def gcas():
    r'''
    Get axes (plural!!) from currently displayed figure.
    '''
    fig = plt.gcf()
    axes = fig.get_axes()
    return axes

def shift_grids(x, y):
    shifted_x = np.r_[x[0], x[1:] - np.diff(x)*0.5, x[-1]]
    shifted_y = np.r_[y[0], y[1:] - np.diff(y)*0.5, y[-1]]
    return shifted_x, shifted_y
