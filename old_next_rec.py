from PvsRMeasurement import RecSystem
from math import sqrt


class MyNextRecSystem(RecSystem):

    def __init__(self, trainSet):
        self.trainSet = trainSet

        self.file_path = 'data/u.data'
        self.all_users = set()
        self.all_movies = set()
        self.all_votes = {}
        self.load_users_and_movies()
        self.create_all_votes_dict()
        self.load_ratings_to_all_votes_dict()

        self.users = set()
        self.movies = set()

        self.users_votes = {}

        self.inputDataProcessed = False

    def load_users_and_movies(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                user, movie, rating, _ = line.split('\t')
                self.all_users.add(user)
                self.all_movies.add(movie)

    def create_all_votes_dict(self):
        for user in self.all_users:
            self.all_votes[user] = {}
            for movie in self.all_movies:
                self.all_votes[user][movie] = 0

    def load_ratings_to_all_votes_dict(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                user, movie, rating, _ = line.split('\t')
                self.all_votes[user][movie] = int(rating)

    def calculate_users_avg_rating(self):
        for user in self.all_users:
            self.users_votes[user]['ratings_sum'] = 0
            self.users_votes[user]['ratings_num'] = 0
            for movie in self.all_movies:
                rating = self.users_votes[user][movie]
                if rating:
                    self.users_votes[user]['ratings_sum'] += rating
                    self.users_votes[user]['ratings_num'] += 1
            try:
                self.users_votes[user]['ratings_avg'] = \
                    self.users_votes[user]['ratings_sum'] / self.users_votes[user]['ratings_num']
            except ZeroDivisionError:
                self.users_votes[user]['ratings_avg'] = 0

    def processInputArray(self):

        for user in self.all_users:
            self.users_votes[user] = {}
            for movie in self.all_movies:
                self.users_votes[user][movie] = 0

        for tuple in self.trainSet:
            rate, user, movie = tuple
            self.users_votes[user][movie] = self.all_votes[user][movie]

        self.calculate_users_avg_rating()

        self.inputDataProcessed = True

    def getQueryFloatResult(self, queryTuple):
        user, movie = queryTuple
        return self.recomendation(user, movie)

    def get_movies_rated_by_both(self, user_x, user_y):
        user_x_movies = [movie for movie in self.movies if self.users_votes[user_x][movie]]
        user_y_movies = [movie for movie in self.movies if self.users_votes[user_y][movie]]
        return list(set(user_x_movies) & set(user_y_movies))

    def pearson_correlation_similarity(self, user_x, user_y):
        movies_rated_by_both = self.get_movies_rated_by_both(user_x, user_y)
        numerator = 0
        denominator_one = 0
        denominator_two = 0
        for movie in movies_rated_by_both:
            numerator += (self.users_votes[user_x][movie] - self.users_votes[user_x]['ratings_avg']) *\
                         (self.users_votes[user_y][movie] - self.users_votes[user_y]['ratings_avg'])
            denominator_one += (self.users_votes[user_x][movie] - self.users_votes[user_x]['ratings_avg']) ** 2
            denominator_two += (self.users_votes[user_y][movie] - self.users_votes[user_y]['ratings_avg']) ** 2
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
            if similarity:
                k_denominator += abs(similarity)
                summation += similarity * (self.users_votes[other_user][movie] - self.users_votes[other_user]['ratings_avg'])

        if k_denominator:
            k = 1 / k_denominator
        else:
            k = 0

        #  print('user={}, movie={}, user_avg_rate={}, k={}, summation={}, result={}'.format(user, movie, self.users_votes[user]['ratings_avg'], k, summation, self.users_votes[user]['ratings_avg'] + k * summation))

        return self.users_votes[user]['ratings_avg'] + k * summation
