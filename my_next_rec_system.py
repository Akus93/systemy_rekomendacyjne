from PvsRMeasurement import RecSystem
from math import sqrt


class MyNextRecSystem(RecSystem):

    def __init__(self, trainSet):
        self.trainSet = trainSet
        self.users = set()
        self.movies = set()
        self.votes = {}
        self.inputDataProcessed = False

    def processInputArray(self):
        print('processInputArray')
        self.load_users_and_movies()
        self.create_votes_dict()
        self.load_ratings_to_votes_dict()
        self.calculate_users_avg_rating()
        self.inputDataProcessed = True

    def getQueryFloatResult(self, queryTuple):
        user, movie = queryTuple
        return self.recomendation(user, movie)

    def load_users_and_movies(self):
        for row in self.trainSet:
            _, user, movie = row
            self.users.add(user)
            self.movies.add(movie)

    def create_votes_dict(self):
        for user in self.users:
            self.votes[user] = {}
            for movie in self.movies:
                self.votes[user][movie] = 0

    def load_ratings_to_votes_dict(self):
        for row in self.trainSet:
            rating, user, movie = row
            self.votes[user][movie] = int(rating)

    def calculate_users_avg_rating(self):
        for user in self.users:
            self.votes[user]['ratings_sum'] = 0
            self.votes[user]['ratings_num'] = 0
            for movie in self.movies:
                rating = self.votes[user][movie]
                if rating:
                    self.votes[user]['ratings_sum'] += rating
                    self.votes[user]['ratings_num'] += 1
            try:
                self.votes[user]['ratings_avg'] = \
                    self.votes[user]['ratings_sum'] / self.votes[user]['ratings_num']
            except ZeroDivisionError:
                self.votes[user]['ratings_avg'] = 0

    def get_movies_rated_by_both(self, user_x, user_y):
        user_x_movies = [movie for movie in self.movies if self.votes[user_x][movie]]
        user_y_movies = [movie for movie in self.movies if self.votes[user_y][movie]]
        return list(set(user_x_movies) & set(user_y_movies))

    def pearson_correlation_similarity(self, user_x, user_y):
        movies_rated_by_both = self.get_movies_rated_by_both(user_x, user_y)
        numerator = 0
        denominator_one = 0
        denominator_two = 0
        for movie in movies_rated_by_both:
            numerator += (self.votes[user_x][movie] - self.votes[user_x]['ratings_avg']) * \
                         (self.votes[user_y][movie] - self.votes[user_y]['ratings_avg'])
            denominator_one += (self.votes[user_x][movie] - self.votes[user_x]['ratings_avg']) ** 2
            denominator_two += (self.votes[user_y][movie] - self.votes[user_y]['ratings_avg']) ** 2
        denominator = sqrt(denominator_one) * sqrt(denominator_two)
        if denominator:
            return numerator / denominator
        return 0

    def recomendation(self, user, movie):
        if movie not in self.movies or user not in self.users:
            return 0
        k_denominator = 0
        summation = 0
        users_set_without_user = self.users - {user}
        for other_user in users_set_without_user:
            similarity = self.pearson_correlation_similarity(user, other_user)
            if similarity:
                k_denominator += abs(similarity)
                summation += similarity * (self.votes[other_user][movie] - self.votes[other_user]['ratings_avg'])
        if k_denominator:
            k = 1 / k_denominator
        else:
            k = 0

        # print('user={}, movie={}, user_avg_rate={}, k={}, summation={}, result={}'.format(user, movie, self.votes[user]['ratings_avg'], k, summation, self.votes[user]['ratings_avg'] + k * summation))

        return self.votes[user]['ratings_avg'] + k * summation
