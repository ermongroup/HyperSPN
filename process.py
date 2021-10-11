import csv
import numpy as np
import pandas as pd

def process(dataset, run, h, N, modeltype, wd):
    savepath = 'output/%s_run=%u_h=%u_N=%u_%s_wd=%.5f' % (dataset, run, h, N, modeltype, wd)

    try:
        with open('%s.csv' % savepath, 'r+') as csvfile:
            try:
                reader = csv.reader(csvfile)
                fold, train, valid, test, parzen = zip(*[ [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])] for row in reader])
                train, valid, test, parzen = np.array(train), np.array(valid), np.array(test), np.array(parzen)
                idx = np.argmax(valid)
                
                return dataset, train[idx], valid[idx], test[idx], parzen[idx]
            
            except:
                print("error reading %s" % savepath)
    except:
        print("error finding %s" % savepath)
        
        
    return dataset, None, None, None, None

if __name__ == "__main__":
    datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 
            'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star',
            'dna', 'kosarek', 'msweb', 'book', 'tmovie',
            'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad',]

    datasets += ["amzn_apparel", "amzn_bath", "amzn_bedding", "amzn_carseats", "amzn_diaper", "amzn_feeding", "amzn_furniture", "amzn_gear", "amzn_gifts", "amzn_health", "amzn_media", "amzn_moms", "amzn_safety", "amzn_strollers", "amzn_toys"]

    datasets += ["toy"]
    
    data = []
    columns = ['dataset','test','parzen']
    index_keys = ('run', 'h', 'N', 'modeltype', 'wd')
    indices = []

    for dataset in datasets:
        kwargs_list = [
            (0,  5, 5, 'hyperspn', 0),
            (0, 10, 5, 'hyperspn', 0),
            (0, 20, 5, 'hyperspn', 0),

            (0, 5, 5, 'spn', 1e-3),
            (0, 5, 5, 'spn', 1e-4),
            (0, 5, 5, 'spn', 1e-5),
        ]

        for kwargs_values in kwargs_list:
            kwargs = dict(zip(index_keys, kwargs_values))
            _, train, valid, test, parzen = process(dataset, **kwargs)
            indices.append(kwargs_values)
            data.append((dataset, test, parzen))


    df = pd.DataFrame(data, columns=columns, index=pd.MultiIndex.from_tuples(indices,names=index_keys))
    verbose = False
    
    # sweep wd
    df_s = df.query('modeltype=="spn"')
    for wd in [1e-3, 1e-4, 1e-5]:
        dfq = df_s.query('wd==%.5f' % wd)
        print("weight decay: %s" % wd)
        print(dfq[['test', 'parzen']].mean())
        if verbose: print(dfq[['dataset', 'test', 'parzen']])
        print('')
    
    # sweep h
    df_h = df.query('modeltype=="hyperspn"')
    for h in [5, 10, 20]:
        dfq = df_h.query('h==%u' % h)
        print("h: %u" % h)
        print(dfq[['test', 'parzen']].mean())
        if verbose: print(dfq[['dataset', 'test', 'parzen']])
        print('')
    