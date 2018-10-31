""""
SVM & SVR Model extraction Simulator
(c) Robert Reith, TU Darmstadt
2018
"""

from sklearn import svm
from server import Server
from supervisor import Supervisor

import random
import math
import csv
import numpy
import matplotlib.pyplot as plt
import statistics

import sklearn.datasets

version = 1
debug = 0


def load_training_data_from_csv(filename, feature_start_index, feature_stop_index, long_index, lat_index, startline):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        line = 1
        X = []
        long = []
        lat = []
        for row in reader:
            if line < startline:
                line += 1
                continue
            integered = list(map(lambda x: int(x), row[feature_start_index:feature_stop_index]))
            longt = float(row[long_index])
            latt = float(row[lat_index])
            X.append(integered)
            long.append(longt)
            lat.append(latt)
            line += 1

    return {"x": X, "long": long, "lat": lat}


def pdf_report(training_data_name, training_data_X, training_data_y, attack_training_set, test_set,
               kernel_type, prediction_type, attack_type, bf_alpha, budget_factor_alpha_list, test_set_y=None):
    dimension = len(training_data_X[0])
    print("---------------------------------------------------------------------------")
    print("[*] Initiating Experiment")
    print("Parameters:")
    print("Training set:             ", training_data_name)
    print("Training data count:      ", len(training_data_X))
    print("Attack data count:        ", len(attack_training_set))
    print("Data dimension:           ", dimension)
    print("Test set size:            ", len(test_set))
    if kernel_type is not None:
        print("Kernel:                   ", kernel_type)
    print("Prediction type:          ", prediction_type)
    print("Attack type:              ", attack_type)
    if budget_factor_alpha_list:
        print("Budget Factor Alpha List: ", budget_factor_alpha_list[0], "..", budget_factor_alpha_list[-1])
    print("----------------------------------------------------")
    s = Supervisor()
    name = training_data_name

    print("[*] Training Server Model...")
    s.add_model(name, training_data_X, training_data_y, prediction_type, kernel_type)
    print("[*] Starting Test Attack...")

    query_budget = math.ceil(bf_alpha * (dimension + 1))
    if attack_type == "agnostic":
        kernel_type = None
    if True:
        run_time, queries, model = s.attack_with_metrics(name, kernel_type, attack_type, dimension,
                                                         query_budget, attack_training_set, roundsize=160, test_set=test_set)

        print("[*] Attack took ", run_time, " seconds and ", queries, " queries.")
        print("[*] Starting Prediction Comparison for test Attack... ")
        a = s.compare_predictions(name, test_set, correct_results=test_set_y, verbose=True)
    if attack_type == "lowd-meek":
        return a[0], queries, run_time
    if not budget_factor_alpha_list:
        return a[0], queries, run_time

    if attack_type not in ["extraction", "lowd-meek", "agnostic"]:
        print("[*] Running query mapping")
        s.plot_error_on_queries("mse", name, kernel_type, attack_type, dimension, budget_factor_alpha_list,
                                attack_training_set, test_set, 16)


def clean_main():
    print("SVM Model extractor")
    print("Version ", version)

    # 100 Lowd Meek Attacks on random models trained on datasets with 200 samples each and 2 features.
    #linear_svm_lowd_meek()
    #linear_svm_retraining()
    #test_kernel_agnostic()

    # RBF SVM Extraction on model trained with the breast cancer dataset with 500 samples and 30 dimensions. 100*(30+1) = 3100
    #rbf_svm("retraining")

    # RBF SVM Extraction On Model trained with 500 points and 2 features max q: 100* (2+1) = 300
    #rbf_svm_gen("adaptive retraining")

    #svr_1("rbf", "retraining")
    #svr_2("rbf", "adaptive retraining")
    #svr_3("quadratic", "extraction")
    svr_4("sigmoid", "retraining")
    #svr_gen("rbf", "adaptive retraining")
    #svr_gen_wifi("rbf", "adaptive retraining")

def linear_svm_lowd_meek():
    hh = []
    rt = []
    for i in range(1, 100):
        print(i)
        X, y = sklearn.datasets.make_blobs(n_samples=500, centers=2, random_state=i, n_features=1000)
        training_set_X = X[0:200]
        training_set_y = y[0:200]
        #attacker_training = X[200:500]
        for i in range(0, 200):
            a = y[i]
            b = y[i+1]
            if a != b:
                attacker_training = [X[i], X[i+1]]
                break

        test_set_X = X
        test_set_y = y
        jj, qq, rt_ = pdf_report("svm-blob", training_set_X, training_set_y, attacker_training, test_set_X, "linear", "SVM",
                            "lowd-meek", 100, [], test_set_y=test_set_y)
        rt.append(rt_)
        hh.append((jj, qq))
    print("Error Values For Predictions:")
    print(hh)
    print("Mean Query Amount")
    queries = list(zip(*hh))[1]
    print(sum(queries)/len(queries))
    print(rt)
    return

def linear_svm_retraining(adaptive=False):
    rt = []
    hh = []
    for i in range(1, 100):
        print(i)
        X, y = sklearn.datasets.make_blobs(n_samples=500, centers=2, random_state=i, n_features=10)
        training_set_X = X[0:200]
        training_set_y = y[0:200]
        attacker_training = X[200:500]

        test_set_X = X
        test_set_y = y
        jj, qq, rt_ = pdf_report("svm-blob", training_set_X, training_set_y, attacker_training, test_set_X, "linear",
                                 "SVM", "retraining", 0.5, [], test_set_y=test_set_y)
        rt.append(rt_)
        hh.append(jj)
    print("Runtimes mean", statistics.mean(rt))
    print("Accuracies", hh)
    print("Accuracy mean", statistics.mean(hh))


def rbf_svm_gen(attack_type):
    X, y = sklearn.datasets.make_blobs(n_samples=1500, centers=2, random_state=7, n_features=20)
    training_set_X = X[0:500]

    training_set_y = y[0:500]
    attacker_training = X[500:1000]
    test_set_X = X[1000:1500]
    test_set_y = y[1000:1500]
    data_amount = 101*(len(training_set_X[0]) + 1)
    attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("svm-blob-rbf", training_set_X, training_set_y, numpy.concatenate((attacker_training,attacker_generated)),
               test_set_X, "rbf", "SVM", attack_type, 40, range(1, 100, 1), test_set_y=test_set_y)


def rbf_svm(attack_type):
    data = sklearn.datasets.load_breast_cancer()  # 569 samples
    training_set_X = data.data[0:500]
    training_set_y = data.target[0:500]
    attacker_training = data.data[300:450]
    test_set_X = data.data[450:550]
    test_set_y = data.target[450:550]
    dimension = len(training_set_X[0])
    #print(test_set_y.tolist().count(1))
    data_amount = 101*(len(training_set_X[0]) + 1)
    attacker_generated = generate_positive_negative(training_set_X, training_set_y, data_amount)
    #attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("svm-cancer", training_set_X, training_set_y, attacker_generated,
               test_set_X, "rbf", "SVM", attack_type, 100, range(1, 100, 1), test_set_y=test_set_y)

def svr_1(kernel, attack_type):
    cali = sklearn.datasets.fetch_california_housing()
    training_set_X = cali.data[0:500]

    training_set_y = cali.target[0:500]
    attacker_training = cali.data[10000:11000]
    test_set_X = cali.data[15000:20000]
    test_set_y = cali.target[15000:20000]
    dimension = len(training_set_X[0])
    max_alpha = math.floor(len(attacker_training)/(dimension+1))

    pdf_report("svr-cali", training_set_X, training_set_y, attacker_training,
               test_set_X, kernel, "SVR", attack_type, 1, range(5, 100, 5), test_set_y=test_set_y)
    return

def svr_2(kernel, attack_type):
    boston = sklearn.datasets.load_boston()
    training_set_X = boston.data[0:100]
    training_set_y = boston.target[0:100]
    attacker_training = boston.data[100:500]
    test_set_X = boston.data[400:500]
    test_set_y = boston.data[400:500]
    dimension = len(training_set_X[0])
    max_alpha = math.floor(len(attacker_training)/(dimension+1))
    data_amount = 101 * (len(training_set_X[0]) + 1)
    attacker_generated = create_similar_data(training_set_X, data_amount)
    all_att_data = numpy.concatenate((attacker_training,attacker_generated))
    pdf_report("svr-boston", training_set_X, training_set_y, all_att_data,
               test_set_X, kernel, "SVR", attack_type, 5, range(5, 100, 5), test_set_y=test_set_y)
    return

def svr_3(kernel, attack_type):
    location_training_data = load_training_data_from_csv(
        r"C:\Users\Kolja\Documents\Uni\Bachelor\BA\STEALING SVM MODELS\Resources\Datasets\UJIIndoorLoc\1478167720_9233432_trainingData.csv",
        0, -9, -9, -8, 2)
    print("[+] Loaded (", len(location_training_data["x"]), ") total sets of data")
    training_set_X = location_training_data["x"][0:30]
    training_set_y = location_training_data["long"][0:30]
    attacker_training = location_training_data["x"][300:19000]
    test_set_X = location_training_data["x"][19000:]
    test_set_y = location_training_data["long"][19000:]
    #data_amount = 101 * (len(training_set_X[0]) + 1)
    #attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("UJIIndoorLoc", training_set_X, training_set_y, attacker_training,
               test_set_X, kernel, "SVR", attack_type, 1, [5, 20], test_set_y=test_set_y)

def svr_4(kernel, attack_type):
    location_training_data = load_training_data_from_csv(
        r"C:\Users\Kolja\Documents\Uni\Bachelor\BA\STEALING SVM MODELS\Resources\Datasets\IPIN 2016\1485881443_7042618_Train.csv",
        0, -9, -9, -8, 2)
    print("[+] Loaded (", len(location_training_data["x"]), ") total sets of data")
    training_set_X = location_training_data["x"][0:100]
    training_set_y = location_training_data["long"][0:100]
    attacker_training = location_training_data["x"][30:800]
    test_set_X = location_training_data["x"][800:]
    test_set_y = location_training_data["long"][800:]
    data_amount = 21 * (len(training_set_X[0]) + 1)
    attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("IPIN Tutorial", training_set_X, training_set_y, numpy.concatenate((attacker_training,attacker_generated)),
               test_set_X, kernel, "SVR", attack_type, 1, [1, 5, 20], test_set_y=test_set_y)


def svr_gen(kernel, attack_type):
    X, y = sklearn.datasets.make_regression(n_samples=1500, random_state=7, n_features=100)
    training_set_X = X[0:500]

    training_set_y = y[0:500]
    attacker_training = X[500:1000]
    test_set_X = X[1000:1500]
    test_set_y = y[1000:1500]
    print(min(y))
    print(max(y))
    data_amount = 101*(len(training_set_X[0]) + 1)
    attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("svr-gen", training_set_X, training_set_y, numpy.concatenate((attacker_training,attacker_generated)),
               test_set_X, kernel, "SVR", attack_type, 1, range(45,100,5), test_set_y=test_set_y)

def svr_gen_wifi(kernel, attack_type):
    X, y = sklearn.datasets.make_regression(n_samples=1500, random_state=7, n_features=4, noise=0, shuffle=True)
    training_set_X = X[0:500]

    training_set_y = y[0:500]
    attacker_training = X[500:1000]
    test_set_X = X[1000:1500]
    test_set_y = y[1000:1500]
    data_amount = 101*(len(training_set_X[0]) + 1)
    print(max(test_set_y))
    print(min(test_set_y))
    attacker_generated = create_similar_data(training_set_X, data_amount)
    pdf_report("svr-gen", training_set_X, training_set_y, attacker_training,
               test_set_X, kernel, "SVR", attack_type, 1, [1, 5, 20, 50], test_set_y=test_set_y)


def create_similar_data(initial_data_list, generate_amount):
    if not isinstance(initial_data_list, list):
        if isinstance(initial_data_list, tuple):
            initial_data_list = list(initial_data_list)
        else:
            initial_data_list = initial_data_list.tolist()
    dimension = len(initial_data_list[0])
    median = list(map(lambda x: statistics.median(x), zip(*initial_data_list)))
    mean = list(map(lambda x: statistics.mean(x), zip(*initial_data_list)))
    stdev = list(map(lambda x: statistics.stdev(x), zip(*initial_data_list)))
    rv = []
    random.seed()
    for i in range(generate_amount):
        dat = []
        for feature_index in range(dimension):
            generated_value = mean[feature_index] + random.choice([-1, 1]) * stdev[feature_index] / random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,20])
            dat.append(generated_value)
        rv.append(dat)
    return rv

def generate_positive_negative(initial_X, initial_y, total_generate_amount):
    each_amount = math.ceil(total_generate_amount/2)
    zip(initial_X, initial_y)
    positives, p_ = zip(*[item for item in zip(initial_X, initial_y) if item[1] == 1])
    negatives, n_ = zip(*[item for item in zip(initial_X, initial_y) if item[1] == 0])
    pos_generated = create_similar_data(positives, each_amount)
    neg_generated = create_similar_data(negatives, each_amount)
    alternated = [None] * (each_amount*2)
    alternated[::2] = pos_generated
    alternated[1::2] = neg_generated
    return alternated

def test_kernel_agnostic():
    boston = sklearn.datasets.load_boston()
    training_set_X = boston.data[0:100]
    training_set_y = boston.target[0:100]
    attacker_training = boston.data[100:500]
    test_set_X = boston.data[400:500]
    test_set_y = boston.data[400:500]
    dimension = len(training_set_X[0])
    max_alpha = math.floor(len(attacker_training) / (dimension + 1))
    pdf_report("agnostic attack", training_set_X, training_set_y, attacker_training, test_set_X, "rbf", "SVR", "agnostic", 20, [])

def main():
    qb1 = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    qb2 = range(1, 100)
    qb3 = range(10, 100)
    qb4 = range(10, 200, 5)
    qb5 = range(1, 250)
    qb6 = range(6, 500, 2)
    qb7 = range(5, 500)

    print("SVM Model extractor")
    print("Version ", version)
    """
    x = []
    y = []
    for i in range(0, 1000):
        datum = [i, random.choice([1, 3])]
        x.append([datum[0], datum[1] + 0.5*datum[0]])
        if datum[1] == 3:
            y.append(0)
        else:
            y.append(1)
    """
    s = Supervisor()
    X = []
    y = []
    m = 3
    b = 12
    for i in range(0, 1000):
        n = random.randrange(-50, 50, 1) / 100
        y.append( m * i + b + n )
        X.append([i])
    line_training_data = {"x": X, "y": y}
    line_training = "line-training-svr"
    line_trained_model = s.create_trained_model_from_training_data(line_training_data["x"][:500], line_training_data["y"][:500], "SVR", "linear")
    s.add_model(line_training, line_training_data, line_trained_model, "SVR")
    s.attack_model(line_training, "SVR", "linear", "adaptive retraining", 1, 0, 100, 100, line_training_data["x"][500:600])
    #s.compare_predictions(line_training, False, line_training_data["x"][600:700], "SVR")
    s.plot_mse_on_queries_with_dataset(line_training, "SVR", "linear", "adaptive retraining", 1, 0, 100, qb4,
                                       line_training_data["x"][500:1000])
    return
    X, y = sklearn.datasets.make_blobs(n_samples=60, centers=2, random_state=7)

    flat_training_data = {"x": X, "y": y}
    firstclass = y[0]
    for index, classf in enumerate(y):
        if classf != firstclass:
            secondindex = index
    posneg = [X[0], X[secondindex]]

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = numpy.linspace(xlim[0], xlim[1], 30)
    yy = numpy.linspace(ylim[0], ylim[1], 30)
    YY, XX = numpy.meshgrid(yy, xx)
    xy = numpy.vstack([XX.ravel(), YY.ravel()]).T


    s = Supervisor()
    """
    s.add_random_model("test_regression_linear", "SVR", "linear", 2, 100)
    s.attack_model("test_regression_linear", "SVR", "linear", "retraining", 2, 100)
    print(s.get_models())
    s.compare_random_predictions("test_regression_linear", 50, 2, 0)
    s.plot_mse_on_queries("test_regression_linear", "SVR", "linear", "retraining", 2, range(1, 101))
    """
    flat_trained_model = s.create_trained_model_from_training_data(flat_training_data["x"], flat_training_data["y"], "SVM", "linear")
    s.add_model("flat-linear-svm", flat_training_data, flat_trained_model, "SVM")
    print(s.get_models()["flat-linear-svm"]["original_model"].coef_[0])
    s.attack_model("flat-linear-svm", "SVM", "linear", "lowd-meek", 2, 0, 7, 100, posneg)
    #s.compare_predictions("flat-linear-svm", False, flat_training_data["x"][800:1000], "SVM")

    Z = s.get_models()["flat-linear-svm"]["original_model"].decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(s.get_models()["flat-linear-svm"]["original_model"].support_vectors_[:, 0], s.get_models()["flat-linear-svm"]["original_model"].support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    p,w,oneoff, doob, lastdoob, doox = s.get_models()["flat-linear-svm"]["extracted_model"]
    wr = s.get_models()["flat-linear-svm"]["original_model"].coef_[0]
    a = -w[0] / w[1]
    intercept = p[1] - a * p[0]
    inter = s.get_models()["flat-linear-svm"]["original_model"].intercept_
    print("correct a", -wr[0] / wr[1])
    print("correct intercept", inter)
    print("calculated with correct", p[1] - (-wr[0] / wr[1]) * p[0])
    print("with correct", (-wr[0] / wr[1])*p[0]+inter)
    print("a", a, "intercept", intercept)
    print("p", p)
    #xx = numpy.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx + intercept #/ w[1]
    plt.plot(xx, yy, 'k-', label='extracted')
    plt.plot(p[0], p[1], marker='o', color='r')
    plt.plot(oneoff[0], oneoff[1], marker='o', color='b')
    #for doo in doob:
    #    plt.plot(doo[0], doo[1], marker='o', color='g')
    plt.plot(lastdoob[0], lastdoob[1], marker='o', color='y')
    #plt.plot(doox[0], doox[1], marker='o', color='b')
    print("pred", s.get_models()["flat-linear-svm"]["original_model"].predict([doox]))
    print("pred", s.get_models()["flat-linear-svm"]["original_model"].predict([oneoff]))
    plt.show()

    return
    krebs = sklearn.datasets.load_breast_cancer()

    krebs_training_data = {"x": krebs.data, "y": krebs.target}
    krebs_trained_model = s.create_trained_model_from_training_data(krebs_training_data["x"], krebs_training_data["y"],
                                                                   "SVM", "rbf")
    print(krebs_training_data["x"][18:20])

    s.add_model("krebs-rbf",krebs_training_data, krebs_trained_model, "SVM")
    s.attack_model("krebs-rbf", "SVM", "rbf", "lowd-meek", 30, 0, 2000, 10, dataset=krebs_training_data["x"][18:20])
    #print(s.get_models())
    s.compare_predictions("krebs-rbf", False, krebs_training_data["x"][100:200], "SVM")




    #s.plot_mse_on_queries_with_dataset("krebs-rbf", "SVM", "rbf", "retraining", 30, 0, 2000, qb7,
    #                                   krebs_training_data["x"][0:500])
    #boston = sklearn.datasets.load_boston()

    #boston_training_data = {"x": boston.data, "y": boston.target}
    #boston_trained_model = s.create_trained_model_from_training_data(boston_training_data["x"], boston_training_data["y"],
    #                                                                 "SVR", "rbf")
    #s.add_model("boston-svr-rbf", boston_training_data, boston_trained_model, "SVR")
    #s.attack_model("boston-svr-rbf", "SVR", "rbf", "adaptive retraining", 13, 5, 50, 120, dataset=boston_training_data["x"][200:320])
    #s.compare_predictions("boston-svr-rbf", False, boston_training_data["x"][000:500], "SVR")
    #s.plot_mse_on_queries("boston-svr-rbf", "SVR" , "rbf", "adaptive retraining", 13, 5, 50, qb2)
    #print(boston.DESCR)
    return
    with open(r"C:\Users\Kolja\Documents\Uni\Bachelor\BA\STEALING SVM MODELS\Resources\Datasets\Alcala Tutorial 2017\1490779198_4046512_alc2017_training_set.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        line = 1
        X = []
        y1 = []
        y2 = []
        for row in reader:
            if line == 1:
                line += 1
                continue
            integered = list(map(lambda x: int(x), row[:-2]))
            floated = list(map(lambda x: float(x), row[-2:]))
            X.append(integered)
            y1.append(floated[0])
            y2.append(floated[1])
            line += 1
        print(len(X))
    location_training_data = {"x": X, "y": y1}
    #print(y1[500:])
    svr = svm.SVR(kernel="rbf")
    print(X[500:502])
    print(y1[500:502])
    svr.fit(X, y1)
    j = 12
    #for j in range(0, 100):
    #    print(svr.predict([X[j]]), y1[j])

    location_trained_model = s.create_trained_model_from_training_data(location_training_data["x"], location_training_data["y"], "SVR", "rbf")
    s.add_model("location-svr-rbf", location_training_data, location_trained_model, "SVR")
    #for j in range(0, 100):
    #    print(svr.predict([X[j]]), y1[j], s.get_models()["location-svr-rbf"]["original_model"].predict([X[j]]))
    s.attack_model("location-svr-rbf", "SVR", "rbf", "retraining", 13, 5, 50, 100, dataset=location_training_data["x"][200:400])
    s.compare_predictions("location-svr-rbf", False, location_training_data["x"][100:200], "SVR")
    s.plot_mse_on_queries_with_dataset("location-svr-rbf", "SVR", "rbf", "adaptive retraining", 13, 5, 50, qb4, location_training_data["x"][0:500])


if __name__ == "__main__":
    #main()
    clean_main()
