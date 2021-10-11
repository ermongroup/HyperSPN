import pickle

path = "../models/%s/svhn/num_clusters_100/cluster_%u/record.pkl"
clusters = 100

names = ["einet_0_0", "einet_0_1"]
c_idx = [i for i in range(clusters)]

for name in names:
    avg_best = 0.0
    for c in c_idx:
        try:
            f = path % (name, c)
            l = pickle.load(open(f, 'rb'))

            best_idx = l['valid_ll'].index(max(l['valid_ll'])) # might throw value error. be careful since try will catch it so no error output
            test_ll = l['test_ll'][best_idx]
            print("%u %s %f %f" % (c, name, l['best_validation_ll'], test_ll))
            avg_best += test_ll

        except Exception as e:
            print(e)
            pass

    avg_best /= clusters
    print("%s %f" % (name, avg_best))