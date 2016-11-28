import numpy


class MyRecomendationSystem:

    def __init__(self, trainset_data_representation):
        self.trainset_data_representation = trainset_data_representation

        self.row_sums_vector = self.trainset_data_representation[1:, 0]
        self.columnSumsVector = self.trainset_data_representation[0, 1:]
        self.row_averages_vector = self.row_sums_vector / len(self.row_sums_vector)
        self.column_averages_vector = self.columnSumsVector / len(self.columnSumsVector)
        self.column_averages_matrix = numpy.outer(numpy.ones(len(self.row_sums_vector)), self.column_averages_vector)
        self.row_averages_matrix = numpy.outer(self.row_averages_vector, numpy.ones(len(self.columnSumsVector)))
        self.processed_data_representation = self.column_averages_matrix + self.row_averages_matrix

    def get_result(self):
        return self.processed_data_representation
