from preprocess import *
from optimize import *
import random
import time
import networkx as nx


def run_trial(config):
    # Basic Trial Setup
    results = {}
    results['preprocess'] = {}
    results['barrier'] = {}
    results['spectral'] = {}
    results['MIP'] = {}
    results['kmeans'] = {}

    config['kmeans_iterations'] = random.randint(1, 20)
    results['config'] = config

    activation_threshold = config['spectral_threshold']
    n_districts = config['n_districts']
    xij_prune_radius = config['xij_prune_radius']
    yi_prune_ratio = config['yi_prune_ratio']
    state = config['state']

    # Load Data and Solve LP Relaxation
    data_path = os.path.join('data', 'tracts', '%s_tracts.csv' % state)
    state_df = pd.read_csv(data_path).set_index('GEOID')
    state_df.index = state_df.index.astype(str)

    tract_dist_dict = xij_preprocess(state, state_df, n_districts, xij_prune_radius)
    if yi_prune_ratio > 0:
        tract_dist_dict = {t: tract_dist_dict[t] for t in tract_dist_dict
                           if random.random() > yi_prune_ratio}
    results['preprocess']['n_centers'] = len(tract_dist_dict)
    n_vars = sum([len(tract_dist_dict[t]) + 1 for t in tract_dist_dict])
    results['preprocess']['n_variables'] = n_vars

    model_construction_start_t = time.time()
    pop_dict = state_df['population'].to_dict()
    pop_dict = {str(k): v for k, v in pop_dict.items()}
    model, xs, ys = make_relaxed_model(config, pop_dict, tract_dist_dict)
    results['preprocess']['time'] = time.time() - model_construction_start_t

    model.Params.TimeLimit = config['barrier_time_limit']
    model.Params.BarConvTol = config['barrier_convergence_tol']
    model.Params.Crossover = 0
    model.Params.Method = 2

    model._primalobj = []
    model._dualobj = []

    barrier_start_t = time.time()
    model.optimize(gap_callback)
    results['barrier']['time'] = time.time() - barrier_start_t
    results['barrier']['obj'] = model.objVal
    results['barrier']['ub'] = model._primalobj
    results['barrier']['lb'] = model._dualobj
    y_activation = {i: ys[i].X if ys[i].X > activation_threshold
                    else 0 for i in ys}
    results['barrier']['solution'] = y_activation

    # Spectral clustering on xij activations
    spectral_label_start_t = time.time()
    xij_activations = [('center' + i, j, {'weight': xs[i][j].X})
                       for i in xs for j in xs[i]
                       if xs[i][j].X > activation_threshold]
    B = nx.Graph(xij_activations)  # Bipartite graph
    A = nx.adjacency_matrix(B, weight='weight').toarray()
    sc = SpectralClustering(n_districts, affinity='precomputed',
                            n_init=10, n_jobs=-1)
    clustering = sc.fit(A)
    labels = clustering.labels_

    # Use clusters to pick centers
    cluster_map = {n: l for n, l in zip(list(B.nodes), labels)}
    n_clusters = config['n_districts']
    cluster_ys = {i: [] for i in range(n_clusters)}
    cluster_xs = {i: [] for i in range(n_clusters)}
    for node, cluster in cluster_map.items():
        if node[0:6] == 'center':
            cluster_ys[cluster].append(node[6:])
        else:
            cluster_xs[cluster].append(node)

    tracts = list(state_df.index)
    valid_centers = [ix for ix, t in enumerate(tracts) if t in tract_dist_dict]
    centers = []
    for i, cluster in cluster_ys.items():
        cluster_weights = np.array([y_activation[y] for y in cluster])
        cluster_weights /= sum(cluster_weights.flatten())
        cluster_positions = state_df.loc[cluster][['x', 'y']].values
        cluster_center = cluster_weights.dot(cluster_positions).flatten()
        pdist = vecdist(cluster_center[1], cluster_center[0],
                        state_df['y'].values[valid_centers],
                        state_df['x'].values[valid_centers])
        center = np.argmin(pdist)
        centers.append(tracts[valid_centers[center]])

    results['spectral']['time'] = time.time() - spectral_label_start_t
    results['spectral']['centers'] = centers
    results['spectral']['clusters'] = cluster_xs

    # Solve MIP transportation problem using spectral centers
    MIP_start_t = time.time()
    centers_dict = {i: tract_dist_dict[i] for i in centers}
    transport, xs = make_transportation_problem(config, centers_dict, pop_dict)
    transport.Params.MIPGap = config['MIPGap_tol']
    transport._best_bound = []
    transport.optimize(gap_callback)
    results['MIP']['time'] = time.time() - MIP_start_t
    try:
        center_mapping = {center: i for i, center in enumerate(centers)}
        districting = {i: [j for j in xs[i] if xs[i][j].X > .5] for i in centers}
        district_mapping = {}
        for center, district in districting.items():
            for d in district:
                district_mapping[d] = center_mapping[center]
        results['MIP']['obj'] = transport.objVal
        results['MIP']['sol'] = district_mapping
        results['MIP']['lb'] = transport._best_bound
    except AttributeError:
        results['MIP']['obj'] = np.nan
        results['MIP']['sol'] = None



    # Try Kmeans heuristic to pick centers
    kmeans_start_t = time.time()
    kresult = solve_kmeans_heuristic(config, tract_dist_dict, state_df)
    best_sol, best_obj, num_infeasible = kresult
    results['kmeans']['time'] = time.time() - kmeans_start_t
    results['kmeans']['best_sol'] = best_sol
    results['kmeans']['best_obj'] = best_obj
    results['kmeans']['num_infeasible'] = num_infeasible

    id = '_'.join([str(v) for k, v in config.items()]).replace('.', '')
    json.dump(results, open(os.path.join('results', id + '.json'), 'w'))


def run_experiment(experiment_config):
    # Build partial trail configs
    param_space = experiment_config['param_settings']
    trial_params = []
    for k, v in param_space.items():
        extended_params = []
        for p in v:
            if trial_params:
                for t in trial_params:
                    extended_params.append(t + [(k, p)])
            else:
                extended_params.append([(k, p)])
        trial_params = extended_params

    state_params = experiment_config['state_trials']

    config_params = []
    for s in state_params:
        for t in trial_params:
            config_params.append(s + t)

    if experiment_config['random_order']:
        random.shuffle(config_params)

    for params in config_params:
        trial_config = experiment_config['base_config'].copy()
        for k, v in params:
            trial_config[k] = v
        print(trial_config)
        run_trial(trial_config)



if __name__ == '__main__':
    param_setting = {
        'xij_prune_radius': [2, 4],
        'yi_prune_ratio': [0, .5],
        'cost_exponential': [1, 1.5, 2],
        'population_tolerance': [.01, .025, .05]
    }
    base_config = {
        'spectral_threshold': 1e-4,
        'barrier_time_limit': 1800,  # 30 minutes
        'barrier_convergence_tol': 1e-4,
        'MIPGap_tol': 1e-5

    }
    state_trials = json.load(open('state_trials.json', 'r'))

    experiment_config = {
        'param_settings': param_setting,
        'base_config': base_config,
        'state_trials': state_trials,
        'random_order': True
    }

    run_experiment(experiment_config)