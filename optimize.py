from gurobipy import *
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

def vecdist(s_lat, s_lng, e_lat, e_lng):

    # approximate radius of earth in km
    R = 6373

    s_lat = s_lat*np.pi/180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d))


def geo_to_euclidean_coords(lat, lon):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    R = 6373
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def euclidean_coords_to_geo(x, y, z):
    R = 6373
    lat = np.arcsin(z / R)
    lon = np.arctan2(y, x)
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return np.array([lon, lat]).T


def gap_callback(model, where):
    if where == GRB.Callback.BARRIER:
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        model._primalobj.append(primobj)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        model._dualobj.append(dualobj)

    if where == GRB.Callback.MIPSOL:
        lb = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        model._best_bound.append(lb)

def make_relaxed_model(config, pop_dict, lengths):
    n_districts = config['n_districts']
    alpha = config['cost_exponential']
    avg_pop = sum(pop_dict.values()) / n_districts
    pmin = round((1 - config['population_tolerance']) * avg_pop)
    pmax = round((1 + config['population_tolerance']) * avg_pop)

    kmed = Model('Kmedians')
    xs = {i: {} for i in lengths}
    ys = {}
    for i in lengths:
        ys[i] = kmed.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y%s' % i)
        for j in lengths[i]:
            xs[i][j] = kmed.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='x%s,%s' % (i, j))

    kmed.addConstr(quicksum(ys[i] for i in lengths) == n_districts,
                   name='n_centers')

    for j in pop_dict:
        kmed.addConstr(quicksum(xs[i][j] for i in lengths if j in xs[i]) == 1,
                       name='one_center_%s' % j)

    kmed.addConstrs(((xs[i][j] <= ys[i]) for i in lengths
                     for j in lengths[i]),
                    name='center_allocation')

    for i in lengths:
        kmed.addConstr(quicksum(pop_dict[j] * xs[i][j]
                                for j in lengths[i]) <= pmax * ys[i],
                       name='population_maximum')
        kmed.addConstr(quicksum(pop_dict[j] * xs[i][j]
                                for j in lengths[i]) >= pmin * ys[i],
                       name='population_minimum')

    kmed.setObjective(quicksum(xs[i][j] * int(lengths[i][j] ** alpha * pop_dict[j] / 1000)
                               for i in lengths
                               for j in lengths[i]),
                      GRB.MINIMIZE)
    kmed.update()

    return kmed, xs, ys

def get_seeds(config, pts, state_df, valid_centers, init='random'):
    n_distrs = config['n_districts']
    weights = state_df['population'].values + 1
    kmeans = KMeans(n_clusters=n_distrs, init=init, n_jobs=-1)\
        .fit(pts, sample_weight=weights).cluster_centers_
    kmeans = euclidean_coords_to_geo(kmeans[:, 0], kmeans[:, 1], kmeans[:, 2])
    centers = []
    epts = euclidean_coords_to_geo(pts[:, 0], pts[:, 1], pts[:, 2])
    for mean in kmeans:
        pdist = vecdist(mean[1], mean[0], epts[valid_centers, 1], epts[valid_centers, 0])
        center = np.argmin(pdist)
        centers.append(valid_centers[center])
    return centers

def solve_kmeans_heuristic(config, lengths, state_df):
    best_obj = 1e100
    best_sol = None
    num_infeasible = 0
    pop_dict = state_df['population'].to_dict()
    lat = state_df['y'].values
    lon = state_df['x'].values
    x, y, z = geo_to_euclidean_coords(lat, lon)
    tracts = list(state_df.index)
    valid_centers = [ix for ix, t in enumerate(tracts) if t in lengths]
    for i in range(config['kmeans_iterations']):
        centers = get_seeds(config, np.array([x, y, z]).T, state_df, valid_centers)
        center_lengths = {c: lengths[tracts[c]] for c in centers}
        transport, xs = make_transportation_problem(config, center_lengths, pop_dict)
        transport.Params.MIPGap = config['MIPGap_tol']
        transport.Params.TimeLimit = 5
        transport.optimize()
        try:
            obj = transport.objVal
        except AttributeError:
            num_infeasible += 1
            continue
        if obj >= best_obj:
            continue
        center_mapping = {center: i for i, center in enumerate(centers)}
        districting = {i: [j for j in xs[i] if xs[i][j].X > .5] for i in centers}
        district_mapping = {}
        for center, district in districting.items():
            for d in district:
                district_mapping[d] = center_mapping[center]
        best_sol = district_mapping
        best_obj = obj

    return best_sol, best_obj, num_infeasible



def make_transportation_problem(config, lengths, pop_dict):
    n_districts = config['n_districts']
    avg_pop = sum(pop_dict.values()) / n_districts
    pmin = round((1 - config['population_tolerance']) * avg_pop)
    pmax = round((1 + config['population_tolerance']) * avg_pop)
    alpha = config['cost_exponential']

    transport = Model('Transportation Problem')
    transport.Params.TimeLimit = 30
    xs = {}
    for i in lengths:
        xs[i] = {}
        for j in lengths[i]:
            xs[i][j] = transport.addVar(vtype=GRB.BINARY,
                                       name="x%s(%s)" % (i, j))

    for j in pop_dict:
        transport.addConstr(quicksum(xs[i][j] for i in xs if j in xs[i]) == 1,
                            name='exactlyOne')
    for i in xs:
        transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
                                    for j in xs[i]) >= pmin,
                           name='x%s_minsize' % j)
        transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
                                    for j in xs[i]) <= pmax,
                           name='x%s_maxsize' % j)

    transport.setObjective(quicksum(xs[i][j] * int(lengths[i][j] ** alpha * pop_dict[j] / 1000)
                               for i in lengths
                               for j in lengths[i]),
                      GRB.MINIMIZE)
    transport.update()

    return transport, xs
