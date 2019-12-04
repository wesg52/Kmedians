import os
import pandas as pd
import numpy as np
import json
import sys

def xij_preprocess(state, state_df, n_distrs, radius):
    data_name = '_'.join([state, str(n_distrs), str(radius)])
    path = 'data/lengths/%s.json' % data_name
    if os.path.exists(path):
        return json.load(open(path, 'r'))

    tracts = list(state_df.index)
    x = state_df['x'].values
    y = state_df['y'].values
    pop = state_df['population'].values
    ideal_pop = radius * np.sum(pop) / n_distrs

    tract_lengths_dict = {}
    for ix, tract in enumerate(tracts):
        tract_lengths_dict[tract] = make_lengths_data(tracts, ix, pop, x, y, ideal_pop)
    json.dump(tract_lengths_dict, open(path, 'w'))

    return tract_lengths_dict


def make_lengths_data(tracts, t, pop, cent_x, cent_y, ideal_pop):
    pdist = np.round(vecdist(cent_y, cent_x, cent_y[t], cent_x[t])).flatten()
    if ideal_pop/2 > np.sum(pop) - 100:
        return {tracts[tr]: int(pdist[tr]) for tr in range(len(tracts))}
    search_range = [0, max(pdist)]
    search_radius = sum(search_range) / 2
    in_range = np.sum(pop[pdist < search_radius])
    iters = 0
    while in_range < ideal_pop or in_range > ideal_pop * 1.05:
        if in_range < ideal_pop:
            search_range[0] = sum(search_range) / 2
        else:
            search_range[1] = sum(search_range) / 2
        if iters > 30:
            break
        search_radius = sum(search_range) / 2
        in_range = np.sum(pop[pdist < search_radius])
        iters += 1
    tract_ix_in_range = np.argwhere(pdist < search_radius)
    return {tracts[tr]: int(pdist[tr]) for tr in tract_ix_in_range.flatten()}


def vecdist(s_lat, s_lng, e_lat, e_lng):
    # approximate radius of earth in km
    R = 6373

    s_lat = s_lat*np.pi/180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d))