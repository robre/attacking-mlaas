from client import Client
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy
import random
import math
from operator import itemgetter

class Adversary:
    """
    This class simulates a special Client, an adversary. He tries to steal a model by sending specifically
    crafted polling requests to a server.
    """
    attack_types = {"SVR": {"linear": ["extraction", "adaptive retraining", "retraining"],
                            "polynomial": ["extraction", "adaptive retraining", "retraining"],
                            "quadratic": ["extraction", "adaptive retraining", "retraining"],
                            "rbf": ["adaptive retraining", "retraining"],
                            "sigmoid": ["adaptive retraining", "retraining"]},
                    "SVM": {"linear": ["lowd-meek", "adaptive retraining", "retraining"],
                            "polynomial": ["lowd-meek", "adaptive retraining", "retraining"],
                            "rbf": ["adaptive retraining", "retraining"],
                            "sigmoid": ["adaptive retraining", "retraining"]}}

    def __init__(self):
        """
        - select an attack type based on the attacked instance
        - create a model for multiple extraction tasks
        - attack structure: server, instance, kernel, class/reg, proposed attacktype, attacktype, ...
        - make created model pollable etc.
        """
        self.client = Client()

    def attack_svr(self, server, predictor_name, kernel_type, attack_type, dimension, query_budget, dataset=None, roundsize=5):
        if dataset is None and attack_type != "extraction" or len(dataset) < 2:
            print("[!] Dataset too small")
            print("[*] Aborting attack...")
            raise ValueError

        if not isinstance(dataset, list):
            dataset = dataset.tolist()
        if attack_type == "retraining":
            X = []
            y = []
            for datum in random.sample(dataset, query_budget):
                b = self.client.poll_server(server, predictor_name, [datum])
                X.append(datum)
                y.append(b)
            if kernel_type == "quadratic":
                my_model = svm.SVR(kernel="poly", degree=2)
            else:
                my_model = svm.SVR(kernel=kernel_type)


            my_model.fit(X, numpy.ravel(y))
            return my_model

        elif attack_type == "adaptive retraining":
            if len(dataset) >= query_budget > roundsize:

                pool = random.sample(dataset, query_budget)
                X = []
                y = []
                n = roundsize
                t = math.ceil(query_budget / n)

                for i in range(0, n):  # Initial training data for a basic start to train upon
                    a = pool.pop(0)
                    b = self.client.poll_server(server, predictor_name, [a])
                    X.append(a)
                    y.append(b)

                if kernel_type == "quadratic":
                    my_model = svm.NuSVR(kernel="poly", degree=2)
                else:
                    my_model = svm.NuSVR(kernel=kernel_type)
                for i in range(0, t - 1):  # perform t rounds minus the initial round.
                    #print(numpy.ravel(y))
                    my_model.fit(X, numpy.ravel(y))

                    if len(my_model.support_vectors_) == 0:
                        print("[!] NO SUPPORTVECTORS IN ROUND", i)
                        print("[*] Adding another round of random samples")
                        #print(my_model.support_)
                        #print(my_model.support_vectors_)
                        #print(my_model.dual_coef_)
                        for q in range(0, n):  # Initial training data for a basic start to train upon
                            if len(pool) == 0:
                                print("[!] Error: Not enough data")
                                raise IndexError
                            a = pool.pop(0)
                            b = self.client.poll_server(server, predictor_name, [a])
                            X.append(a)
                            y.append(b)
                        continue
                    print("Training Round", i, " of ", t-1)
                    pool, samples = self.get_furthest_samples(pool,
                                                              my_model.support_vectors_,
                                                              kernel_type,
                                                              my_model.coef0,
                                                              my_model.get_params()["gamma"],
                                                              my_model.get_params()["C"],
                                                              n,
                                                              my_model.dual_coef_)

                    for j in samples:
                        X.append(j)
                        y.append(self.client.poll_server(server, predictor_name, [j]))
                my_model.fit(X, numpy.ravel(y))
                return my_model
            else:
                print("[!] Error: either not enough data in data set, or query budget not bigger than round size.")
                print("[*] Aborting attack...")
                raise ValueError
        elif attack_type == "extraction":
            if kernel_type == "quadratic":
                # NOTE: KEEP IN MIND, IN THE IMPLEMENTATION THE VECTOR INDICES START AT 0, INSTEAD OF 1
                # Also DIMENSION - 1 is max index, not dimenstion itself.
                d_ = self.nCr(dimension, 2) + 2*dimension + 1  # d := Projection dimension
                if d_ > query_budget:
                    print("[!] Error: This algorithm will need", d_ ," queries.")
                    raise ValueError
                w_ = [0] * d_  # extracted weight vectors

                null_vector = [0] * dimension
                b_ = self.client.poll_server(server, predictor_name, [null_vector])[0]  # b' = w_d c +b
                for dim in range(dimension):
                    v_p = dim * [0] + [1] + (dimension - 1 - dim) * [0]
                    v_n = dim * [0] + [-1] + (dimension - 1 - dim) * [0]
                    f_v_p = self.client.poll_server(server, predictor_name, [v_p])[0] - b_
                    f_v_n = self.client.poll_server(server, predictor_name, [v_n])[0] - b_
                    w_[dimension - dim + 1 - 2] = (f_v_p + f_v_n) / 2
                    w_[d_ - dim - 2] = (f_v_p - f_v_n) / 2

                class QuadraticMockModel:
                    def __init__(self, d__, w__, b__):
                        self.dim = d__
                        self.w = w__
                        self.b = b__

                    def phi(self, x__):
                        vec = []
                        for i__ in x__[::-1]:
                            vec.append(i__**2)
                        for i__ in reversed(range(len(x__))):
                            for j__ in reversed(range(i__)):
                                vec.append(math.sqrt(2)*x__[i__]*x__[j__])
                        for i__ in x__[::-1]:
                            vec.append(i__)
                        vec.append(0)
                        return vec

                    def predict(self, arr):
                        rv = []
                        for v__ in arr:
                            val = numpy.dot(self.w, self.phi(v__)) + self.b
                            rv.append(val)
                        return rv

                if dimension <= 2:
                    return QuadraticMockModel(d_, w_, b_)
                for dim_i in range(dimension):
                    for dim_j in range(dim_i + 1, dimension):
                        #print(dim_i, dim_j)
                        v = dimension*[0]
                        v[dim_i], v[dim_j] = 1, 1
                        f_v = self.client.poll_server(server, predictor_name, [v])[0]
                        r = self.r_index(dim_i + 1, dim_j + 1, dimension) - 1
                        w_[r] = (f_v - w_[dimension - dim_i + 1 - 2] - w_[dimension - dim_j + 1 - 2] - w_[d_ - dim_i - 2] - w_[d_ - dim_j - 2] - b_) / math.sqrt(2)
                print("[+] w' extrahiert:", w_)

                return QuadraticMockModel(d_, w_, b_)

            if kernel_type != "linear":
                print("[!] Error: Unsupported Kernel for extraction attack.")
                raise ValueError
            d = [0] * dimension
            b = self.client.poll_server(server, predictor_name, [d])[0]
            w = []
            for j in range(0, dimension):
                x = j * [0] + [1] + (dimension - 1 - j) * [0]
                w.append(self.client.poll_server(server, predictor_name, [x])[0]-b)
            print("[+] Model parameters have been successfully extracted")
            print("[*] weight (w):", w)
            print("[*] bias   (b):", b)
            print("[*] Building mock model...")

            class LinearMockModel:
                def __init__(self, d__, w__, b__):
                    self.dim = d__
                    self.w = w__
                    self.b = b__

                def predict(self, arr):
                    rv = []
                    for v__ in arr:
                        val = numpy.dot(self.w, v__) + self.b
                        rv.append(val)
                    return rv

            return LinearMockModel(dimension, w, b)
        else:
            print("[!] Error: unknown attack type for svr")
            print("[*] Aborting attack...")
            raise ValueError

    def attack_svm(self, server, predictor_name, kernel_type, attack_type, dimension,  query_budget, dataset=None, roundsize=5):
        if dataset is None or len(dataset) < 2:
            print("[!] Dataset too small")
            print("[*] Aborting attack...")
            raise ValueError
        if not isinstance(dataset, list):
            dataset = dataset.tolist()
        if attack_type == "retraining":
            my_model = svm.SVC(kernel=kernel_type)
            X = []
            y = []

            for datum in random.sample(dataset, query_budget):
                b = self.client.poll_server(server, predictor_name, [datum])
                X.append(datum)
                y.append(b)
            my_model.fit(X, numpy.ravel(y))
            return my_model

        elif attack_type == "adaptive retraining":
            if len(dataset) >= query_budget > roundsize:
                pool = random.sample(dataset, query_budget)
                x = []
                y = []
                n = roundsize
                t = math.ceil(query_budget / n)
                for i in range(0, n):
                    a = pool.pop(0)
                    b = self.client.poll_server(server, predictor_name, [a])[0]
                    x.append(a)
                    y.append(b)

                while min(y) == max(y):
                    for i in range(0, n):
                        a = pool.pop(0)
                        b = self.client.poll_server(server, predictor_name, [a])[0]
                        x.append(a)
                        y.append(b)
                    t -= 1
                    print("[*] Additional initial random round had to be done due to no variance")
                my_model = svm.SVC(kernel=kernel_type)
                for i in range(0, t-1):

                    my_model.fit(x, numpy.ravel(y))
                    for j in range(0, n):
                        if not pool:
                            break
                        distances = my_model.decision_function(pool).tolist()
                        closest = pool.pop(distances.index(min(distances)))
                        x.append(closest)
                        y.append(self.client.poll_server(server, predictor_name, [closest])[0])
                my_model.fit(x, numpy.ravel(y))
                return my_model
            else:
                print("[!] Error: dataset to small or roundsize bigger than query_budget")
                raise ValueError
        elif attack_type == "lowd-meek":
            if len(dataset) != 2:
                print("[!] Error: For Lowd-Meek attack, please provide exactly a positive and a negative sample")
                raise ValueError
            elif kernel_type != "linear":
                print("[!] Error: Unsupported Kernel by lowd-meek attack")
                raise ValueError
            else:
                print("[*] Initiating lowd-meek attack.")
                epsilon = 0.01
                d = 0.01
                vector1 = dataset[0]
                vector2 = dataset[1]
                vector1_category = numpy.ravel(self.client.poll_server(server, predictor_name, [vector1]))
                vector2_category = numpy.ravel(self.client.poll_server(server, predictor_name, [vector2]))
                if vector1_category == vector2_category:
                    print("[!] Error: Provided Samples are in same category")
                    raise ValueError
                else:
                    if vector1_category == [0]:
                        print(vector1_category, "is 0")
                        negative_instance = vector1
                        positive_instance = vector2
                    else:
                        print(vector2_category, "is 0")
                        negative_instance = vector2
                        positive_instance = vector1

                    #sign_witness_p = positive_instance
                    sign_witness_n = negative_instance
                    print("[+] Positive and  Negative Instance confirmed.")
                    for feature in range(0, len(sign_witness_n)):
                        print("[*] Finding Signwitness. Checking feature", feature)
                        f = sign_witness_n[feature]
                        sign_witness_n[feature] = positive_instance[feature]
                        if numpy.ravel(self.client.poll_server(server, predictor_name, [sign_witness_n])) == [1]:
                            sign_witness_p = sign_witness_n.copy()

                            sign_witness_n[feature] = f
                            f_index = feature
                            print("[+] Sign Witnesses found with feature index:", f_index)
                            break

                    weight_f = 1 * (sign_witness_p[feature] - sign_witness_n[feature]) / abs(sign_witness_p[feature] - sign_witness_n[feature])
                    # Find Negative Instance of x with gap(x) < epsilon/4
                    delta = sign_witness_p[feature] - sign_witness_n[feature]

                    seeker = sign_witness_n
                    #seeker[feature] = sign_witness_p[feature] - delta
                    #print(sign_witness_p)
                    #print(sign_witness_n)
                    while True:
                        #print("S - ", seeker)
                        pred = self.client.poll_server(server, predictor_name, [seeker])
                        #print("p:", pred)
                        if pred == [1]:
                            #print("Positive. delta", delta)
                            delta = delta / 2
                            seeker[feature] = seeker[feature] - delta
                        else:
                            #print("Negative. delta", delta)
                            if abs(delta) < epsilon/4:
                                print("[+] found hyperplane crossing", seeker)
                                break
                            delta = delta / 2
                            seeker[feature] = seeker[feature] + delta
                    # seeker should be that negative instance now.
                    crossing = seeker.copy()
                    seeker[feature] += 1
                    classification = numpy.ravel(self.client.poll_server(server, predictor_name, [seeker]))

                    dooble = seeker.copy()  # dooble is negative instance

                    weight = [0]*len(dooble)
                    #print("Weight on initieal feature", weight_f)

                    for otherfeature in range(0, len(dooble)):
                        if otherfeature == feature:
                            weight[otherfeature] = weight_f
                            continue
                        # line search on the other features
                        dooble[otherfeature] += 1/d
                        if numpy.ravel(self.client.poll_server(server, predictor_name, [dooble])) == classification:
                            #print("DIDNOTCHANGE")
                            doox = dooble.copy()
                            dooble[otherfeature] -= 2/d
                            if numpy.ravel(self.client.poll_server(server, predictor_name, [dooble])) == classification:  # if even though added 1/d class stays the same -> weigh = 0
                                weight[otherfeature] = 0
                                dooble[otherfeature] = seeker[otherfeature]
                                #print("found weightless feature,", otherfeature)
                                continue
                            else:
                                distance_max = -1/d
                        else:

                            distance_max = 1/d

                        distance_min = 0
                        distance_mid = (distance_max + distance_min) / 2
                        dooble[otherfeature] = seeker[otherfeature] + distance_mid

                        while abs(distance_mid - distance_min) > epsilon / 4:

                            if numpy.ravel(self.client.poll_server(server, predictor_name, [dooble])) != classification:

                                distance_min = distance_min
                                distance_max = distance_mid
                                distance_mid = (distance_min + distance_max) / 2
                                dooble[otherfeature] = seeker[otherfeature] + distance_mid
                            else:
                                distance_min = distance_mid
                                distance_mid = (distance_min + distance_max) / 2
                                distance_max = distance_max
                                dooble[otherfeature] = seeker[otherfeature] + distance_mid
                        test = seeker[otherfeature]-dooble[otherfeature]
                        weight[otherfeature] = weight_f / test
                        continue
                    print("[+] Found Weights", weight)
                    a = -(weight[0] / weight[1])
                    intercept = crossing[1] - a * crossing[0]
                    print("[+] Found Intercept (2d)", intercept)

                    class LinearMockSVM:
                        def __init__(self, w__, b__):
                            self.w__ = w__
                            self.b__ = b__*w__[1]  # norm

                        def predict(self, val):
                            rv = []
                            for v in val:
                                #print(numpy.sign(numpy.dot(self.w__, v) - self.b__))
                                rv.append(0) if numpy.sign(numpy.dot(self.w__, v) - self.b__) == -1 else rv.append(1)
                            return rv
                    return LinearMockSVM(weight, intercept)
        else:
            print("Error: Unknown attack type")
            raise ValueError

    def attack(self, server, predictor_name, predictor_type, kernel_type, attack_type, dimension, query_budget, dataset=None, roundsize=5):
        self.client.reset_poll_count()
        random.seed()
        if attack_type not in self.attack_types[predictor_type][kernel_type]:
            print("[!] Error: Attack type not compatible with kernel type")
            print("[*] Aborting attack...")
            raise ValueError
        if predictor_type == "svr" or predictor_type == "SVR":
            return self.attack_svr(server, predictor_name, kernel_type, attack_type, dimension, query_budget, dataset=dataset, roundsize=roundsize)
        elif predictor_type == "svm" or predictor_type == "SVM":
            return self.attack_svm(server, predictor_name, kernel_type, attack_type, dimension, query_budget, dataset=dataset, roundsize=roundsize)
        else:
            return None

    def k(self, kernel_type, x_i, x_j, coef0=0, gamma=1):
        if gamma == 'auto':
            gamma = 1/len(x_i)
        if kernel_type == "linear":
            return numpy.dot(x_i, x_j)
        elif kernel_type == "quadratic":
            return (numpy.dot(x_i, x_j) + coef0)**2
        elif kernel_type == "rbf":
            return numpy.exp((numpy.linalg.norm(numpy.asarray(x_i) - numpy.asarray(x_j))**2)*(-1)*gamma)
        elif kernel_type == "sigmoid":
            return numpy.tanh(gamma*numpy.dot(x_i, x_j)+coef0)
        else:
            print("[!] Error: Unknown kernel type")
            raise ValueError

    def get_furthest_samples(self, pool, support_vectors, kernel_type, coef0, gamma, C, n, dual_coef):
        # for each sample in a pool calculate the distances from that sample to each support vector
        # pick the closest vector as the minimum distance
        # create

        ranking = [{"closest_support_vector_id": 0, "minimum_distance": 0, "sample_id": 0, "total_score": 0}]
        furthest_samples = []
        distances = []

        #print("SUPP VEC", support_vectors)

        for index, sample in enumerate(pool):
            sample_distances = []
            #print("SAMPLE", sample)
            for sv in support_vectors:
                distance = math.sqrt(self.k(kernel_type, sample, sample, coef0, gamma)+self.k(kernel_type, sv, sv, coef0, gamma)+2*self.k(kernel_type, sample, sv, coef0, gamma))
                sample_distances.append(distance)
            closest_index = sample_distances.index(min(sample_distances))
            #print("CLOSEST", sample_distances[closest_index])
            total_score = abs(sample_distances[closest_index] * dual_coef[0][closest_index] / C)
            distances.append((index, sample_distances, closest_index, total_score))
        #print(sorted(distances, key=lambda x: x[3]))
        indexes = []
        for sample in sorted(distances, key=lambda x: x[3])[:n]:
            furthest_samples.append(pool[sample[0]])
            indexes.append(sample[0])
        for index in sorted(indexes, reverse=True):
            del pool[index]

        return pool, furthest_samples

    def get_last_query_count(self):
        return self.client.poll_count

    def _model_factory(self, predictor_type, kernel, weights, intercept, point):
        pass

    def nCr(self, n, r):
        return math.factorial(n)//math.factorial(r)//math.factorial(n-r)

    def r_index(self, t, s, n):
        l = 0
        for i in range(1, n-s+1+1):
            l += n - i
        l += n-t+1
        return l

    def kernel_agnostic_attack(self, server, predictor_name, predictor_type, dimension, query_budget, dataset, test_set_X):
        kernels = ["linear", "rbf", "poly", "sigmoid"]
        models = []
        if not isinstance(dataset, list):
            dataset = dataset.tolist()
        dataset = random.sample(dataset, query_budget)  # all train on the same dataset
        for kernel in kernels:
            print("extracting as", kernel)
            if predictor_type == "SVR":
                model = self.attack_svr(server, predictor_name, kernel, "retraining", dimension,
                                        query_budget, dataset)
            elif predictor_type == "SVM":
                model = self.attack_svm(server, predictor_name, kernel, "retraining", dimension,
                                        query_budget, dataset)
            else:
                print("[!] Error: Specify predictor type")
                raise ValueError
            models.append(model)

        correct_predictions = self.client.poll_server(server, predictor_name, test_set_X)
        scores = []
        for model in models:
            if predictor_type == "SVM":
                scores.append(accuracy_score(correct_predictions, model.predict(test_set_X)))
            else:
                scores.append(mean_squared_error(correct_predictions, model.predict(test_set_X)))

        if predictor_type == "SVM":
            best = max(enumerate(scores), key=itemgetter(1))[0]
        else:
            best = min(enumerate(scores), key=itemgetter(1))[0]
        print(list(zip(kernels, scores)))
        print("Best Kernel: ", kernels[best])
        return models[best]






