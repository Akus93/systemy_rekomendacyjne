import numpy
from math import sqrt
from pprint import pprint
from progress.bar import Bar


class MyRecomendationSystem:
    """
    https://en.wikipedia.org/wiki/Collaborative_filtering#Memory-based
    """

    def __init__(self, trainset_data_representation, file_path):
        self.file_path = file_path

        self.users = set()
        self.movies = set()

        self.data_matrix = []
        self.data_dict = {}

        self.load_users_and_movies()
        self.create_initial_data_matrix()
        self.create_initial_data_dict()
        self.load_ratings_to_data_matrix()

        self.load_users_info('data/u.user')

        self.calculate_users_avg_rating()
        self.calculate_movies_avg_rating()

    def get_result(self):
        bar = Bar('Progress', max=len(self.movies))
        result_matrix = [['user/movie', ]]
        for user in self.users:
            result_matrix.append([user, ])
        for movie in self.movies:
            result_matrix[0].append(movie)
        for user in list(self.users)[:10]:
            for movie in list(self.movies)[:10]:
                result_matrix[user].append(1 if self.recomendation(user, movie) > 3 else 0)
                bar.next()
        return result_matrix

    def load_ratings_to_data_matrix(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                user, movie, rating, _ = line.split('\t')
                self.data_matrix[int(user)][int(movie)] = int(rating)

    def load_users_and_movies(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                user, movie, rating, _ = line.split('\t')
                self.users.add(int(user))
                self.movies.add(int(movie))

    def create_initial_data_matrix(self):
        self.data_matrix.append(['user/movie', ])
        for user in self.users:
            self.data_matrix.append([user, ])
        for movie in self.movies:
            self.data_matrix[0].append(movie)
        for user_index in range(len(self.users)):
            for movie_index in range(len(self.movies)):
                self.data_matrix[user_index + 1].append(0)

    def create_initial_data_dict(self):
        self.data_dict['users'] = {}
        self.data_dict['movies'] = {}
        for user in self.users:
            self.data_dict['users'][user] = {}
        for movie in self.movies:
            self.data_dict['movies'][movie] = {}

    def calculate_users_avg_rating(self):
        """Add to each user info about sum, number and avg of ratings"""
        for user in self.users:
            self.data_dict['users'][user]['ratings_sum'] = 0
            self.data_dict['users'][user]['ratings_num'] = 0
            for movie in self.movies:
                rating = self.data_matrix[user][movie]
                if rating:
                    self.data_dict['users'][user]['ratings_sum'] += rating
                    self.data_dict['users'][user]['ratings_num'] += 1
            self.data_dict['users'][user]['ratings_avg'] = \
                self.data_dict['users'][user]['ratings_sum'] / self.data_dict['users'][user]['ratings_num']

    def calculate_movies_avg_rating(self):
        """Add to each movie info about sum, number and avg of ratings"""
        for movie in self.movies:
            self.data_dict['movies'][movie]['ratings_sum'] = 0
            self.data_dict['movies'][movie]['ratings_num'] = 0
            for user in self.users:
                rating = self.data_matrix[user][movie]
                if rating:
                    self.data_dict['movies'][movie]['ratings_sum'] += rating
                    self.data_dict['movies'][movie]['ratings_num'] += 1
            self.data_dict['movies'][movie]['ratings_avg'] =\
                self.data_dict['movies'][movie]['ratings_sum'] / self.data_dict['movies'][movie]['ratings_num']

    def load_users_info(self, path):
        """Add to each user info about age, gender, occupation and zip_code"""
        with open(path, 'r') as file:
            for line in file:
                user, age, gender, occupation, zip_code = line.split('|')
                self.data_dict['users'][int(user)]['age'] = int(age)
                self.data_dict['users'][int(user)]['gender'] = gender
                self.data_dict['users'][int(user)]['occupation'] = occupation
                self.data_dict['users'][int(user)]['zip_code'] = zip_code.rstrip()

    def get_movies_rated_by_both(self, user_x, user_y):
        user_x_movies = [movie for movie in self.movies if self.data_matrix[user_x][movie]]
        user_y_movies = [movie for movie in self.movies if self.data_matrix[user_y][movie]]
        return list(set(user_x_movies) & set(user_y_movies))

    def pearson_correlation_similarity(self, user_x, user_y):
        movies_rated_by_both = self.get_movies_rated_by_both(user_x, user_y)
        numerator = 0
        denominator_one = 0
        denominator_two = 0
        for movie in movies_rated_by_both:
            numerator += (self.data_matrix[user_x][movie] - self.data_dict['users'][user_x]['ratings_avg']) *\
                         (self.data_matrix[user_y][movie] - self.data_dict['users'][user_y]['ratings_avg'])
            denominator_one += (self.data_matrix[user_x][movie] - self.data_dict['users'][user_x]['ratings_avg']) ** 2
            denominator_two += (self.data_matrix[user_y][movie] - self.data_dict['users'][user_y]['ratings_avg']) ** 2
        denominator = sqrt(denominator_one) * sqrt(denominator_two)
        if denominator:
            return numerator/denominator
        return 0

    def recomendation(self, user, movie):
        k_denominator = 0
        summation = 0
        users_set_without_user = self.users - {user}
        for other_user in users_set_without_user:
            similarity = self.pearson_correlation_similarity(user, other_user)
            k_denominator += abs(similarity)
            summation += similarity * (self.data_matrix[other_user][movie] - self.data_dict['users'][other_user]['ratings_avg'])

        k = 1 / k_denominator

        print('user={}, movie={}, user_avg_rate={}, k={}, summation={}, result={}'.format(user, movie, self.data_dict['users'][user]['ratings_avg'], k, summation, self.data_dict['users'][user]['ratings_avg'] + k * summation))

        return self.data_dict['users'][user]['ratings_avg'] + k * summation


if __name__ == '__main__':
    my_rec_system = MyRecomendationSystem(None, 'data/u.data')
    result = my_rec_system.get_result()
    print(result)
