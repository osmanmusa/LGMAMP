class Config(object):
    def __init__(self, **kwargs):

        vars(self).update(kwargs)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def save_to_file(self):
        import pickle

        with open(self.config_file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self,file_path):
        import pickle

        with open(file_path, 'rb') as input:
            config = pickle.load(input)

        return config