### Segmented Linear Regression ###
""" 
I found other ways of doing this but this turned out to be the best by far.
Someone could take take some time to turn this into a class but this is functional.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

# 3 status of trend
INC = 0
DEC = 1
FLAT = 2

L = 3

def cal_slope(series):
    yi = series.reshape(-1, 1)
    xi = np.array([i for i in range(len(series))]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(xi, yi)
    k = model.coef_[0][0]
    if k > 0.01:
        k = 1.0
    elif k < -0.01:
        k = -1.0
    else:
        k = 0.0
    return k

def cal_status(value):
    if value > 0.3:
        return INC
    elif value < -0.3:
        return DEC
    else:
        return FLAT
    
def smooth(series):
    for i in range(len(series)):
        if i > 0 and i < len(series)-1 and series[i] != series[i-1] and series[i-1] == series[i+1]:
            series[i] = series[i-1]
        elif i > 1 and i < len(series)-3 and series[i] == series[i+1] and series[i] != series[i-1] and \
             series[i-1] == series[i+2] and series[i-2] == series[i-1] and series[i+2] == series[i+3]:
            series[i] = series[i-1]
            series[i+1] = series[i-1]
    return series

def label_status(x, d=0.3):
    labels = []
    slopes = []
    for i in range(len(x)):
        if i == 0:
            slope = 0
        elif i == len(x) - 1:
            slope = slopes[-1]
        elif i - L <= 0:
            slope = cal_slope(x[0:i])
        # elif i + L >= len(x):
        #     pred = cal_slope(x[i:])
        else:
            slope = cal_slope(x[i-L : i+L]) if np.sum(x[i-L : i+L] > 0) > 0 else 0
            
        # weighted smoothing
        mean_slope = slope * d + slopes[-1] * (1-d) if i > 0 else slope
        slopes.append(mean_slope)
        labels.append(cal_status(mean_slope))
    labels = smooth(labels)
    return labels

def calculate_slope(series):
    yi = series.reshape(-1, 1)
    xi = np.array([i for i in range(len(series))]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(xi, yi)
    k = model.coef_[0][0]
    return k

def approximate_line(series):
    yi = series.reshape(-1, 1)
    xi = np.array([i for i in range(len(series))]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(xi, yi)
    slope = model.coef_[0][0]
    line = model.predict(xi).reshape(-1)
    return line, slope

def calculate_error(seq_i, seq_j, lbl_i, lbl_j):
    slope_i = calculate_slope(seq_i)
    slope_j = calculate_slope(seq_j)
    error = abs(slope_i - slope_j)
    if np.all(lbl_i[-int(len(lbl_i)*0.8):] == INC) and np.all(lbl_j[:int(len(lbl_j)*0.8)] == INC): error = 0
    if np.all(lbl_i[-int(len(lbl_i)*0.8):] == DEC) and np.all(lbl_j[:int(len(lbl_j)*0.8)] == DEC): error = 0
    return error

def bottom_up(series, labels, max_error, step=3, k=2):
    seg_list = []
    idx_list = []
    lbl_list = []
    for i in range(0, len(series)-1, step):
        if i+step >= len(series)-1: step = len(series)-i
        seg_list.append(series[i:i+step])
        lbl_list.append(np.array(labels[i:i+step]))
        idx_list.append((i, i+step))

    merge_cost = []
    for i in range(len(seg_list)-1):
        merge_cost.append(calculate_error(seg_list[i], seg_list[i+1], lbl_list[i], lbl_list[i+1]))

    while np.min(merge_cost) < max_error:
        i = np.argmin(merge_cost)
        seg_list[i] = np.hstack((seg_list[i], seg_list[i+1]))
        lbl_list[i] = np.hstack((lbl_list[i], lbl_list[i+1]))
        idx_list[i] = (idx_list[i][0], idx_list[i+1][1])
        del(seg_list[i+1])
        del(lbl_list[i+1])
        del(idx_list[i+1])
        del(merge_cost[i])
        if(len(seg_list)==k): break
        if i < len(seg_list)-1: merge_cost[i] = calculate_error(seg_list[i], seg_list[i+1], lbl_list[i], lbl_list[i+1])
        merge_cost[i-1] = calculate_error(seg_list[i-1], seg_list[i], lbl_list[i-1], lbl_list[i])
    return seg_list, idx_list
    
def combine_segment(x, label_list, seg_list, idx_list,):
    slope_list = []

    for seg, idx in zip(seg_list, idx_list):
        line, slope = approximate_line(seg)
        slope_list.append(slope)

    segments = []
    
    # upward segments
    slope_threshold = 0.005
    delta_threshold = 0.3
    segment = [0, 0]
    for idx, slope in zip(idx_list, slope_list):
        if slope > slope_threshold:
            segment[1] = idx[1]
        else:
            if segment[0] < segment[1] and np.max(x[segment[1]-3:segment[1]]) - np.min(x[segment[0]:segment[0]+3]) > delta_threshold:
                segments.append([segment[0], segment[1], INC])
            segment = [idx[1], idx[1]]
    if segment[0] < segment[1] and np.max(x[segment[1]-3:segment[1]]) - np.min(x[segment[0]:segment[0]+3]) > delta_threshold:
        segments.append([segment[0], segment[1], INC])
        
    # downward segments
    segment = [0, 0]
    for idx, slope in zip(idx_list, slope_list):
        if slope < -slope_threshold:
            segment[1] = idx[1]
        else:
            if segment[0] < segment[1]:
                segments.append([segment[0], segment[1], DEC])
            segment = [idx[1], idx[1]]
    if segment[0] < segment[1]:
        segments.append([segment[0], segment[1], DEC])
    
    segments.sort() 
    return segments