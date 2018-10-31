class Server:
    """
    This class simulates a MLaaS Provider.
    It orgranized multiple predictors, such as SVMs and SVRs and makes them accessible for polling
    The predictors are to be inserted as fully trained classes
    """

    poll_count = 0

    def __init__(self):
        self.predictors = {}

    def add_predictor(self, predictor_type, name, predictor):
        """
        Adds a new predictor to the server. A predictor is an sklearn.svm.SVC() or sklearn.svm.SVR() object,
        which is already fully trained (fitted) and ready for being polled
        :param predictor_type: SVM or SVR
        :param name: name by which this predictor shall be referred to. Unique as its the key to the dict
        :param predictor: the actual sklearn.svm.SVC() or sklearn.svm.SVR() object.
        :return: True
        """
        self.predictors[name] = {
            "predictor_type": predictor_type,
            "predictor": predictor}
        return True

    def del_predictor(self, name):
        """
        Delete a predictor by name
        :param name: name of the predictor to be deleted
        :return:s true
        """
        self.predictors[name] = None
        return True

    def poll_predictor(self, name, data):
        """
        returns a prediction result on each of a set of data
        :param name: name of the predictor that should be used to predict
        :param data: list of datasets
        :return: list of results
        """
        self.poll_count += len(data)
        predictor = self.get_predictor(name)
        return predictor.predict(data)

    def get_predictor(self, name):
        """
        returns a handler on a predictor from a name
        :param name: name of the predictor
        :return: predictor class or None
        """
        try:
            return self.predictors[name]["predictor"]
        except Exception as e:
            return None

    def get_predictor_list(self):
        s = ""
        for name in self.predictors.keys():
            s += "| " + name + " | " + self.predictors[name]["predictor_type"] + " |\n"
        return s
