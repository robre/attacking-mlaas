from server import Server
from adversary import Adversary
import math
from sklearn import svm

import matplotlib.pyplot as plt
import statistics
import time

class Supervisor:
    """
    This Class delegates the whole experiment.
    It keeps track of actual functions that are mimicked by ML, of the ML models, and  the stolen Models
    in order to provide overview and comparison between those stages.
    This class is used to calculate errors and optimize the extraction algorithm.
    It can be seen as an omniscient class that knows of everything going on in this program
    """
    def __init__(self):
        """
        - create a server
        - generate training data for classification and regression models based on some function
        - train the classifiers and regressors
        - store the original function, the ml predictor and training data together for each instance
        - load server with a set of trained ml models
        - be able to compare an adversarys model to the correct function and initial model
        - calculate errors / different error types
        """
        self.server = Server()
        self.adversary = Adversary()
        self.models = {}  # {"Name": {"kernel_type": None, "predictor_type": None, "training_data": {"X": None, "y": None}, "original_model": None, "extracted_model": None}}

    def add_model(self, name, X, y, predictor_type, kernel_type):
        """
        Trains an ML model on the given data, adds it to the class internal database, and adds it to the server
        :param name: Name of the model
        :param X: TrainingData X
        :param y: TrainingData Targets
        :param predictor_type: SVM or SVR
        :param kernel_type: kernel to use.
        :return:
        """
        if kernel_type == "quadratic":
            kernel_type = "poly"
            deg = 2
        else:
            deg = 3
        if predictor_type == "SVM":
            original_model = svm.SVC(kernel=kernel_type, degree=deg)
            original_model.fit(X, y)
        elif predictor_type == "SVR":
            original_model = svm.SVR(kernel=kernel_type, degree=deg)
            original_model.fit(X, y)
        else:
            raise ValueError
        self.models[name] = {"training_data": {"X": X, "y": y},
                             "kernel_type": kernel_type,
                             "predictor_type": predictor_type,
                             "original_model": original_model,
                             "extracted_model": None}
        self.server.add_predictor(predictor_type, name, original_model)
        return self.models[name]

    def get_models(self):
        return self.models

    def compare_predictions(self, name, data, correct_results=None, verbose=False):
        """
        Compares predictions on dataset "data" between original and extracted model. Returns error percentage for svm
        and mean, mean squared and relative mean squared error for svr.
        :param name:
        :param data:
        :param correct_results:
        :param verbose:
        :return:
        """
        prediction_errors = []
        original_model = self.models[name]["original_model"]
        extracted_model = self.models[name]["extracted_model"]
        prediction_type = self.models[name]["predictor_type"]
        error_count = 0
        original_predictions = []
        for index, datum in enumerate(data):
            original_prediction = original_model.predict([datum])
            extracted_prediction = extracted_model.predict([datum])
            original_predictions.append(original_prediction[0])
            error = abs(original_prediction[0]-extracted_prediction[0])

            prediction_errors.append(error)

            if error != 0 and prediction_type == "SVM":
                error_count += 1
            if verbose:
                if error == 1:
                    pass
                    #print(original_prediction)
                    #print(extracted_prediction)
                    #print(datum.T[0],",",datum.T[1])
                elif 1==2:
                    print("data:", datum)
                    if correct_results is not None:
                        print("correct:             ", correct_results[index])
                    print("original prediction: ", original_prediction[0])
                    print("extracted prediction:", extracted_prediction[0])
                    print("error                ", error)
                    print("------------------------------")

        if prediction_type == "SVM":
            error_percentage = error_count*100/len(data)
            print("Total false Predictions", error_count, "out of", len(data))
            print("False Prediction Percentage:", error_percentage, "%")
            return error_percentage, 0, 0
        elif prediction_type == "SVR":
            mean_original = sum(original_predictions) / len(original_predictions)
            mean_error = sum(prediction_errors) / len(prediction_errors)
            mse = sum(map(lambda x: x ** 2, prediction_errors)) / len(prediction_errors)
            rmse = sum(map(lambda x: x ** 2, prediction_errors)) / sum(
                map(lambda x: (mean_original - x) ** 2, original_predictions))
            print("----------------------------------------")
            print("Mean Error:", mean_error)
            print("MSE:       ", mse)
            print("RMSE:      ", rmse)
            return mean_error, mse, rmse

    def attack_model(self, name, kernel_type, attack_type, dimension, query_budget, dataset=None, roundsize=5, test_set=None):
        if attack_type == "lowd-meek" or attack_type == "extraction":
            print("[*] Original Model Parameters")
            if kernel_type == "linear":
                print("w ", self.models[name]["original_model"].coef_)
                print("a ", -self.models[name]["original_model"].coef_[0][0]/self.models[name]["original_model"].coef_[0][1])
                print("b ",  -self.models[name]["original_model"].intercept_[0] / self.models[name]["original_model"].coef_[0][1])
            else:
                print("a ", self.models[name]["original_model"].dual_coef_)
            #print("sv", self.models[name]["original_model"].support_vectors_)
        predictor_type = self.models[name]["predictor_type"]
        if kernel_type is None and attack_type == "agnostic" and test_set is not None:
            self.models[name]["extracted_model"] = self.adversary.kernel_agnostic_attack(self.server, name,
                                                                                         predictor_type, dimension,
                                                                                         query_budget, dataset, test_set)
        else:
            self.models[name]["extracted_model"] = self.adversary.attack(self.server, name, predictor_type,
                                                                     kernel_type, attack_type, dimension,
                                                                     query_budget, dataset=dataset, roundsize=roundsize)
        return self.models[name]

    def attack_with_metrics(self, name, kernel_type, attack_type, dimension, query_budget, dataset=None, roundsize=5, test_set=None):
        start_time = time.time()
        self.attack_model(name, kernel_type, attack_type, dimension, query_budget,
                          dataset=dataset, roundsize=roundsize, test_set=test_set)
        queries = self.adversary.get_last_query_count()
        run_time = time.time() - start_time
        return run_time, queries, self.models[name]["extracted_model"]

    def suggest_attack_type(self, predictor_type, kernel_type):
        if predictor_type == "SVM" or predictor_type == "svm" or predictor_type == "SVC" or predictor_type == "svc":
            if kernel_type == "linear":
                return "lowd-meek"
            elif kernel_type == "polynomial":
                return "lowd-meek"
            else:
                return "adaptive retraining"
        elif predictor_type == "SVR" or predictor_type == "svr":
            if kernel_type == "linear":
                return "extraction"
            elif kernel_type == "polynomial":
                return "extraction"
            else:
                return "adaptive retraining"
        else:
            raise ValueError

    def get_error_on_queries(self, error_type, name, kernel_type, attack_type, dimension, budget_factor_alpha_list, training_data, test_data, roundsize=5):
        if not isinstance(test_data, list):
            test_data = test_data.tolist()
        if not isinstance(training_data, list):
            training_data = training_data.tolist()
        x_axis = budget_factor_alpha_list
        y_axis = []
        for budget_factor_alpha in budget_factor_alpha_list:
            print("Alpha:", budget_factor_alpha)
            query_budget = math.ceil(budget_factor_alpha * (dimension + 1))
            run_time, queries, model = self.attack_with_metrics(name, kernel_type, attack_type, dimension, query_budget, training_data, roundsize=roundsize)
            print("Queries", queries, " Runtime", run_time)
            errors = self.compare_predictions(name, test_data, verbose=False)
            if self.models[name]["predictor_type"] == "SVR":
                if error_type == "mean":
                    error_index = 0
                elif error_type == "mse":
                    error_index = 1
                elif error_type == "rmse":
                    error_index = 2
                y_axis.append(errors[error_index])
            else:
                y_axis.append(errors[0])
        return x_axis, y_axis

    def plot_error_on_queries(self, error_type, name, kernel_type, attack_type, dimension, budget_factor_alpha_list, training_data, test_data, roundsize=5):
        x_axis, y_axis = self.get_error_on_queries(error_type, name, kernel_type, attack_type, dimension, budget_factor_alpha_list, training_data, test_data, roundsize=roundsize)
        [print(i) for i in zip(x_axis, y_axis)]
        plt.plot(x_axis, y_axis)
        plt.show()
        return plt, x_axis, y_axis
