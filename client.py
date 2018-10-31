import random


class Client:
    """
    This class simulates a Client who polls a Server for predictions on data
    Such as prediction of coordinates within a building by providing wifi AP data
    """
    def __init__(self):  #TODO
        pass
    poll_count = 0

    def poll_server(self, server, predictor_name, data):
        self.poll_count += 1
        return server.poll_predictor(predictor_name, data)

    def random_poll(self, server, predictor_name, dimension, rmin, rmax, step):
        """
        Polls the server with one instance of randomly generated data of a given dimension
        :param server:
        :param predictor_name:
        :param dimension:
        :param rmin:
        :param rmax:
        :param step:
        :return:
        """
        d = []
        random.seed()
        for i in range(0, dimension):
            d.append(random.randrange(rmin, rmax, step))
        return d, self.poll_server(server, predictor_name, [d])

    def reset_poll_count(self):
        self.poll_count = 0
