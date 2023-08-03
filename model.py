from tqdm import tqdm
from statistics import mean
import numpy as np
import math
from scipy.stats import entropy
import random as rnd
from datetime import timedelta


# Given a descending sorted list returns
# the index corresponding to the h-index definition
# if a key function is provided it will applied to each element
# before comparisons.
def h_index_bisect(rsl, key=lambda x: x):

    lo = 0
    hi = len(rsl)

    while lo < hi:

        mid = (lo + hi) // 2
        value = key(rsl[mid])

        if value > mid:
            lo = mid + 1
        elif value < mid:
            hi = mid
        else:
            return mid

    return lo


def deltas(ls):
    if len(ls) == 0:
        return []
    d = []
    for n, k in enumerate(ls[1:], start=1):
        d.append(k - ls[n-1])
    return d

from scipy.stats import entropy
def custom_entropy(sequence):
    uniques_frq = {k: 0 for k in set(sequence)}
    #uniform = 1 / len(uniques_frq)
    #print(uniques_frq)
    base = len(uniques_frq)
    #print(base)
    distrib = []
    for n, x in enumerate(sequence, start=1):
        uniques_frq[x] += 1
        distrib.append(uniques_frq[x]/n)
    #print(uniques_frq)
    #print(distrib)
    return entropy(distrib, base=2)

from scipy.stats import entropy

def custom_entropy(sequence):
    uniques_frq = {k: 0 for k in set(sequence)}
    base = len(uniques_frq)
    distrib = []
    for n, x in enumerate(sequence, start=1):
        distrib.append(uniques_frq[x]/n)
        uniques_frq[x] += 1
    return entropy(distrib, base=base)




class HModel:

    def __init__(self) -> None:

        # For quick search by content id
        self.content_index = dict()
        # Map each user on shared contents
        self.author2contents = dict()
        # Map each user on his features
        self.author2features = dict()
        # Track content number
        self.n_contents = 0

        self.start_timestamp = None
        self.stop_timestamp = None

        self.weights = None

        self.rank = dict()

        self.author2time_features = dict()

    def fit(self, data, content_key, author_key, root_content_key, timestamp_key, misinf_key):
        """data is a list of any objects that represent a post or a document
           author_key is a function that applied to a data_list object return the auhtor name
           ... to be continued"""
        
        # Update the size of the training set
        self.n_contents = len(data)

        # Step 1: mapping authors to original contents
        for n, content in enumerate(data):

            # Values unpack
            content_id = content_key(content)
            author_id = author_key(content)
            root_content_id = root_content_key(content)
            timestamp = timestamp_key(content)
            is_misinf = misinf_key(content)

            if root_content_id == 'ORIGIN':

                content_track = {'reshares': 0,
                                 'self_reshares': 0,
                                 'timestamp': timestamp,
                                 'misinformation': is_misinf}

                try:
                    # Add the track for this content
                    self.author2contents[author_id][content_id] = content_track
                except KeyError:
                    # First time this author is encountered
                    self.author2contents[author_id] = {content_id: content_track}

            # Build the content index
            self.content_index[content_id] = (n, author_id, root_content_id, timestamp, is_misinf)
        
        #self.start_timestamp = list(self.content_index.values())[0][3]
        #self.stop_timestamp = list(self.content_index.values())[-1][3]

        # Step 2: updating reshares count
        for content_id, content_data in self.content_index.items():

            n, author_id, root_content_id, _, _ = content_data

            if root_content_id != 'ORIGIN': # if is a reshare

                try: # try to find the original author

                    root_author_id = self.content_index[root_content_id][1]

                except KeyError: # if never seen this content skip iteration

                    continue

                if root_author_id == author_id: # if is a self reshare
                    self.author2contents[root_author_id][root_content_id]['self_reshares'] += 1
                else:
                    self.author2contents[root_author_id][root_content_id]['reshares'] += 1

        # Step 3: feature extraction
        for author in self.author2contents.keys():

            author_shared_contents = self.author2contents[author].values()

            # Better solution: iterate over author_shared_contents once...

            # Counts
            author_total_contents = len(author_shared_contents)
            author_misinf_contents = sum([c['misinformation'] for c in author_shared_contents])
            author_inf_contents = author_total_contents - author_misinf_contents # Redundant to remove (?)

            # h-index (FIB and anti-FIB)
            misinf_reshares = [c['reshares'] for c in author_shared_contents if c['misinformation'] == 1]
            inf_reshares = [c['reshares'] for c in author_shared_contents if c['misinformation'] == 0]
            author_FIB = h_index_bisect(sorted(misinf_reshares, reverse=True))
            author_IB = h_index_bisect(sorted(inf_reshares, reverse=True))

            # fall-index
            if misinf_reshares == []:
                author_fall_index = 0
            else:
                try:
                    author_fall_index = author_FIB * sorted(misinf_reshares, reverse=True)[author_FIB-1]
                except:
                    print("error debug")
                    print(author_FIB)
                    print(misinf_reshares)



            # Self-reshares (how much the user is prone to self reshare posts)
            author_self_resharing = mean([c['self_reshares'] for c in author_shared_contents])

            # Influence
            author_influence_good = sum(inf_reshares)
            author_influence_bad = sum(misinf_reshares)

            # Time based features...
            #events = [self.start_timestamp] + [c['timestamp'] for c in author_shared_contents] + [self.stop_timestamp]
            #delays = [t.total_seconds() for t in deltas(events)]
            #author_continuity = (self.stop_timestamp - self.start_timestamp).total_seconds() / mean(delays)

            #author_shared_contents = [(n, c) for n, c in enumerate(author_shared_contents)]
            #author_shared_contents = sorted(author_shared_contents, key=lambda x: x[1]['reshares'], reverse=True)
            #author_reshares = [c['reshares'] for _, c in author_shared_contents]
            #penalized_reshares = [c['reshares'] * (c['self_reshares']+1) for _, c in author_shared_contents]
            #author_h_index = h_index_bisect([c['reshares'] for _, c in author_shared_contents])
            #sorted_contents = sorted(author_shared_contents, key=lambda x: x['reshares'], reverse=True)
            #hindex_full = h_index_bisect(sorted(author_reshares, reverse=True)) #, key=lambda x: x['reshares'])
            #hindex_penalty = h_index_bisect(sorted(penalized_reshares, reverse=True)) #, key=lambda x: x['reshares'])
            #author_boost = sum([c['self_reshares'] for _, c in author_shared_contents])
            
            # Take a list of 0s and 1s with lenght self.n_contents
            # If 1 then user posted something new, else 0.
            # The probability of posting is p = len(author_shared_contents) / self.n_contents
            # The probability of not posting is 1-p 
            # author_behavior = [0 for n in range(self.n_contents)] # Or [0] * self.n_contents
            # for _, c in author_shared_contents:
            #      author_behavior[c['progressive']] = 1
            #c_prob = len(author_shared_contents) / self.n_contents
            # print(author_behavior)
            #pk = [c_prob if x == 1 else 1 - c_prob for x in author_behavior]
            # print(pk)
            # post_entropy = entropy(pk, base=2)
            # print(post_entropy)
            # author_entropy = custom_entropy(author_behavior)

            # Add reliability: linked to data size for each user (more data more reliable score)

            #sorted_contents = sorted_contents[:hindex]  # Drop unnecessary contents (N.B. can generate empty list)
            #delta_values = deltas(sorted([c['progressive'] for c in sorted_contents]))
            #print([td.seconds for td in delta_values])
            #activity = sum([x / self.n_contents for x in sorted([c['progressive'] for c in sorted_contents])])
            #virality = hindex / len(sorted_contents)
            # frequency = len(author_shared_contents) / self.n_contents

            self.author2features[author] = {'BadBroadcaster': author_FIB,
                                            'GoodBroadcaster': author_IB,
                                            'total-shares': author_total_contents,
                                            'misinf-shares': author_misinf_contents,
                                            'inf-shares': author_inf_contents,
                                            'self-resharing': author_self_resharing,
                                            'fall-index': author_fall_index,
                                            'good-influence': author_influence_good,
                                            'bad-influence': author_influence_bad
                                            #'continuity': author_continuity,
                                            #'boost': author_boost,
                                            #'h-index-no-self': hindex_penalty,
                                            #'activity': activity,
                                            #'virality': virality,
                                            #'entropy': author_entropy,
                                            #'frequency': frequency,
                                            #'compound': frequency * hindex + activity,
                                            #'compound': frequency * author_h_index, #(very good)
                                            #'compound': hindex + activity ** virality #(good) 
                                            #'compound': frequency * (author_h_index + author_boost) # test
                                            }

        # Initialize the model weights
        self.weights_dim = len(self.author2features[author].values())
        self.weights = np.random.random(self.weights_dim)


    # Get the rank for detected authors
    def get_rank(self):

        for author, features in self.author2features.items():

            fvector = np.array(list(features.values()))
            #fvector /= np.linalg.norm(fvector)

            self.rank[author] = np.sum(fvector * self.weights)

        return dict(sorted(self.rank.items(), key=lambda x: x[1], reverse=True))


    # Get the rank for detected authors
    def get_time_rank(self):

        for author, features in self.author2time_features.items():

            fvector = np.array(list(features.values()))
            #fvector /= np.linalg.norm(fvector)

            self.rank[author] = np.sum(fvector * self.weights)

        return dict(sorted(self.rank.items(), key=lambda x: x[1], reverse=True))


    # Set the weights
    def set_weights(self, new_weights):

        self.weights = new_weights


    # Single aggregation for round trip time evaluation
    @staticmethod
    def standard_rtt(estimated, sampled, alpha=0.15):
        try:
            return (1 - alpha) * estimated + alpha * sampled
        except:
            print(estimated)
            print(sampled)
            raise ValueError


    # Single aggregation for round trip time evaluation (variance)
    @staticmethod
    def variance_rtt(estimated, sampled, delta=0.15):
        return estimated - delta * (estimated - sampled)


    @staticmethod
    def __merge_dict(author2estimated, author2sampled):

        # Update the estimated values if there are new sampled ones.
        for author, value in author2estimated.items():
            try:
                # new = HModel.standard_rtt(author2estimated[author], author2sampled[author])
                new = {k: HModel.variance_rtt(v, author2sampled[author][k]) for k, v in author2estimated[author].items()}
            except KeyError:
                # If the user have no activity in the next period the sampled value is considered 0
                #new = HModel.standard_rtt(author2estimated[author], 0.0)
                new = {k: HModel.variance_rtt(v, 0.0) for k, v in author2estimated[author].items()}

            author2estimated[author] = new

        # Add new user if any. The initial estimated value will be the sampled one.
        for author, value in author2sampled.items():
            try:
                author2estimated[author]
            except:
                author2estimated[author] = value


    def time_fit(self, data, content_key, author_key, root_content_key, timestamp_key, misinf_key, days_interval=5):

        # Sort by time
        time_data = sorted(data, key=timestamp_key)

        # Initial timestamp of current chunk
        initial_timestamp = timestamp_key(time_data[0])

        # Store the data chunk
        data_chunk = []

        for content in time_data:

            elapsed = timestamp_key(content) - initial_timestamp

            if elapsed / timedelta(days=1) > days_interval or content == time_data[-1]:
                # Evaluate chunk
                self.fit(data_chunk, content_key, author_key, root_content_key, timestamp_key, misinf_key)
                # Merge result (RTT)
                HModel.__merge_dict(self.author2time_features, self.author2features)
                # Reset the features dict
                self.author2features = dict()
                # Reset counter
                initial_timestamp = timestamp_key(content)
                # Init next chunk with this content
                data_chunk = [content]
            else:
                # Append this content to current chunk
                data_chunk.append(content)




# n_reshares: 3,  4, 10, 1, 0, 9, 3, 7, 9, 1,  2,  0,  0
# time_index: 0,  1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12  ## NOTE: should be done on the entire period not only the track

# sorted    : 10, 9,  9, 7, 4, 3, 3, 2, 1, 1,  0,  0,  0 -> hindex == 4
# time_index:  2, 5,  8, 7, ...                          -> sum == 22  (a greater value means more recent activity)
# deltas    :  2, 3,  3, 1, ...                          -> sum = 9 (a lower value mean high frequency activity)

# new_index = (hindex * time_sum) / delta_sum = (4 * 22) / 9 = 9,778



# Example usage:
if __name__ == "__main__":

    print("Testing...")

    # Toy example for dataset. If content_id == root_id then is original content else reshared
    toy = [{'content_id': "023", 'author_id': "794", 'root_content_id': '023', 'timestamp': None},
           {'content_id': "024", 'author_id': "431", 'root_content_id': '021', 'timestamp': None},
           {'content_id': "025", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
           {'content_id': "026", 'author_id': "431", 'root_content_id': '025', 'timestamp': None},
           {'content_id': "027", 'author_id': "431", 'root_content_id': '023', 'timestamp': None},
           {'content_id': "028", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
           {'content_id': "029", 'author_id': "794", 'root_content_id': '029', 'timestamp': None},
           {'content_id': "030", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
           {'content_id': "031", 'author_id': "431", 'root_content_id': '029', 'timestamp': None},
           {'content_id': "032", 'author_id': "794", 'root_content_id': '025', 'timestamp': None}]


    model = HModel()

    model.fit(toy, content_key=lambda x: x['content_id'],
              author_key=lambda x: x['author_id'],
              root_content_key= lambda x: x['root_content_id'],
              timestamp_key= lambda x: x['timestamp'])

    print(model.author2features)
    print(model.author2contents)


    # TODO: Old code. To be discarded.
    # toy = [{"author": 'pippo', 'interactions': 0},
    #        {"author": 'pluto', 'interactions': 10},
    #        {"author": 'kevin', 'interactions': 100},
    #        {"author": 'kevin', 'interactions': 12},
    #        {"author": 'pluto', 'interactions': 5},
    #        {"author": 'pluto', 'interactions': 7},
    #        {"author": 'kevin', 'interactions': 3},
    #        {"author": 'pippo', 'interactions': 19},
    #        {"author": 'pippo', 'interactions': 28},
    #        {"author": 'kevin', 'interactions': 0},     # pippo: 28, 19, 0 | pluto: 10, 7, 5 | kevin: 100, 12, 3, 0
    #        {"author": 'pippo', 'interactions': 18},
    #        {"author": 'pluto', 'interactions': 11},
    #        {"author": 'kevin', 'interactions': 20},
    #        {"author": 'kevin', 'interactions': 60},
    #        {"author": 'pluto', 'interactions': 0},
    #        {"author": 'pluto', 'interactions': 0},
    #        {"author": 'kevin', 'interactions': 0},
    #        {"author": 'pippo', 'interactions': 4},
    #        {"author": 'pippo', 'interactions': 16},
    #        {"author": 'kevin', 'interactions': 31}]

    # # gt: pippo: 28, 19, 18, 16, 4, 0 | pluto: 11, 10, 7, 5, 0, 0 | kevin: 100, 60, 31, 20, 12, 3, 0, 0
    # # pippo: 4, pluto: 4, kevin: 5

    # import random as rnd
    # from string import ascii_lowercase, digits
    # from scipy.stats import truncnorm

    # # Return true wih prob probability
    # def rand_chance(prob):
    #     return rnd.random() < prob

    # # Set the mean and standard deviation
    # mu1, sigma1 = 0.3, 0.4 # activity
    # mu2, sigma2 = 0.0, 0.4 # misinformation

    # # Set the lower and upper bounds
    # lower, upper = 0, 1

    # # Generate n author names, each with an activity index, chanches of spread misinformation
    # authid_len = 5
    # auth_count = 500
    # chars = ascii_lowercase + digits
    # lst = [''.join(rnd.choice(chars) for _ in range(authid_len)) for _ in range(auth_count)]
    # actividx = truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1).rvs(auth_count)
    # misinprob = truncnorm((lower - mu2) / sigma2, (upper - mu2) / sigma2, loc=mu2, scale=sigma2).rvs(auth_count)
    # authlist = list(zip(lst, actividx, misinprob))

    # N = 1000000
    # contents = []

    # # Generate n author names, each with an activity index, chanches of spread misinformation
    # print(f"Generating {N} synthetic contents...", flush=True)
    # for n in range(N):

    #     not_generated = True

    #     while not_generated:

    #         # Sample a single author name and with activity index probability create a single content
    #         author, act, mis = rnd.sample(authlist, 1)[0]

    #         if rnd.random() < act:

    #             content_id = str(n)
    #             try:
    #                 parent_id = content_id if rnd.random() < 0.7 else rnd.sample(contents, 1)[0]['content_id']
    #             except ValueError:
    #                 parent_id = content_id

    #             interactions = rnd.randint(0, int(N*act*rnd.random()))
    #             misinfection_lvl = 1 if rnd.random() < mis else 0

    #             #contents.append((content_id, author, parent_id, interactions, misinfection_lvl))
    #             contents.append({'content_id': content_id,
    #                              'author': author,
    #                              'parent_id': parent_id,
    #                              'interactions': interactions,
    #                              'misinfection_lvl': misinfection_lvl})

    #             not_generated = False

    # input(f"Fake contents generated and ready.\nPress any key to start algorithm...")

    # model = HModel()

    # model.fit(contents, lambda x: x['author'], lambda x: x['interactions'])
    # hindex_ranking = sorted(model.author2features.items(), key=lambda x: x[1]['h-index'], reverse=True)

    # [print(x) for x in hindex_ranking[:10]]

    # print('\n', model.author2content)
    # print('\n', model.author2features)