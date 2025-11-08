import json
import numpy as np
from dimensions import estimators
from dimensions.estimators.mle import mle

def run_estimator(args, dataset, verbose=False, filename=None):         
    estimator = args.estimator
    if estimator == "mle":
        results = run_mle(args, dataset, verbose=verbose, filename=filename)
    elif estimator == "geomle":
        results = run_geomle(args, dataset)
    elif estimator == "twonn":
        results = run_twonn(args, dataset)
    elif estimator == "shortest-path":
        results = run_shortest_path(args, dataset)
    return results


def run_mle(args, dataset, verbose=False, filename=None):   
    if args.single_k:
        dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
    else:
        if args.average_inverse:
            indiv_est = mle(dataset, nb_iter=args.nb_iter, random_state=None, k1=args.k1, k2=args.k2, average=True, args=args)[0]
            dim = 1. / np.mean(1. / indiv_est)
        else:
            dim = mle(dataset, nb_iter=args.nb_iter, random_state=None, k1=args.k1, k2=args.k2, average=True, args=args)[0].mean()
        inv_mle_dim = None

    # Log and save results
    save_fp = filename
    save_dict = vars(args)
    if args.eval_every_k:
        for nk, k in enumerate(range(2, args.k1+1)):
            save_dict["k{}_dim".format(k)] = float(dim[nk])
            if verbose:
                print("k={}, Estimated dimension of inv mle: {}".format(k, inv_mle_dim[nk]))
            save_dict["k{}_inv_mle_dim".format(k)] = float(inv_mle_dim[nk])
    else:
        save_dict["dim"] = float(dim)
        if inv_mle_dim is not None:
            if verbose:
                print("Estimated dimension of inv mle: {}".format(inv_mle_dim))
            save_dict["inv_mle_dim"] = float(inv_mle_dim)
    with open(save_fp, 'a') as fh:
        json.dump(save_dict, fh)
    if verbose:
        print(save_dict)
    return save_dict


def run_geomle(args, dataset):
    dim_ = estimators.geomle(dataset, k1=args.k1, k2=args.k2, nb_iter1=args.nb_iter1, nb_iter2=args.nb_iter2,
                 degree=(1, 2), alpha=5e-3, ver='GeoMLE', random_state=None, debug=False, args=args)
    dim = dim_.mean()
    print("Estimated dimension: {}".format(dim))
    # Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


def run_twonn(args, dataset):
    dim = estimators.twonn(dataset, args=args)
    print("Estimated dimension: {}".format(dim))
    # Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


def run_shortest_path(args, dataset):
    results = estimators.shortest_path(dataset, args=args)
    dim = results['Dmin']
    print("Estimated dimension: {}".format(dim))
    # Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    save_dict["results"] = results
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict
