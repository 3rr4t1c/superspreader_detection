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

    def __init__(self,
                 content_key,   # Function, applied to a data object return the content id
                 author_key,    # Function, applied to a data object return the author id
                 root_content_key,  # Function, applied to a data object return the root content id
                 timestamp_key,     # Function, applied to a data object return the timestamp
                 misinf_key,    # Function, applied to a data object return the misinformation flag (binary)
                 time_delta=5,   # timedelta object
                 alpha=0.125    # Float
                 ) -> None:

        # Content features accessing functions
        self.content_key = content_key
        self.author_key = author_key
        self.root_content_key = root_content_key
        self.timestamp_key = timestamp_key
        self.misinf_key = misinf_key

        # Time training hyperparams
        self.time_delta = time_delta
        self.alpha = alpha

        # Content fast access
        self.content_index = None

        # Map each authors to their published contents
        self.author2contents = None

        # Map each author to their behavior features (e.g. FIB)
        self.author2features = None

        # Weights for compound metric (TODO: remove or refactor)
        self.feature_weights = [1, 0, 0, 0, 0, 0, 0, 0, 0]

        # Store the ranking by a defined metric
        self.output_rank = dict()


    # Main training function
    def fit(self, data):
        """data is a list of any objects that represent a post or a document
           author_key is a function that applied to a data_list object return the auhtor name
           ... to be continued"""

        # Init data structures
        self.content_index = dict()
        self.author2contents = dict()
        self.author2features = dict()

        # Step 1: map authors to original contents
        for content in data:

            # Values unpacking
            content_id = self.content_key(content)
            author_id = self.author_key(content)
            root_content_id = self.root_content_key(content)
            timestamp = self.timestamp_key(content)
            is_misinf = self.misinf_key(content)

            if root_content_id == 'ORIGIN':

                content_track = {'reshares': 0,
                                 'self-reshares': 0,
                                 'timestamp': timestamp,
                                 'misinformation': is_misinf}

                try:
                    # Add the track for this content
                    self.author2contents[author_id][content_id] = content_track
                except KeyError:
                    # First time this author is encountered: initialize
                    self.author2contents[author_id] = {content_id: content_track}

            # Build the content index
            self.content_index[content_id] = (author_id, root_content_id, timestamp, is_misinf)

        # NOTE: At this point, two index has been created.

        # Step 2: update reshares and self-reshares count
        for content_id, content_data in self.content_index.items():

            author_id, root_content_id, _, _ = content_data

            # If content is a reshare...
            if root_content_id != 'ORIGIN':

                try: # ...try to find the original author

                    root_author_id = self.content_index[root_content_id][0]

                except KeyError: # if never seen this content skip iteration

                    continue

                # If is a self-reshare...
                if root_author_id == author_id:
                    self.author2contents[root_author_id][root_content_id]['self-reshares'] += 1
                else:
                    self.author2contents[root_author_id][root_content_id]['reshares'] += 1

        # NOTE: At this point we have for each author all the original content they posted
        # and for each content they posted the count of reshares and self-reshares received

        # Step 3: feature extraction
        for author in self.author2contents.keys():

            author_shared_contents = self.author2contents[author].values()

            # Better solution: iterate over author_shared_contents once...

            # Counting stats
            author_total_contents = len(author_shared_contents)
            author_misinf_contents = sum([c['misinformation'] for c in author_shared_contents])
            author_non_misinf_contents = author_total_contents - author_misinf_contents

            # FIB and anti-FIB (h-index)
            misinf_reshares = [c['reshares'] for c in author_shared_contents if c['misinformation'] == 1]
            non_misinf_reshares = [c['reshares'] for c in author_shared_contents if c['misinformation'] == 0]
            author_FIB = h_index_bisect(sorted(misinf_reshares, reverse=True))
            author_anti_FIB = h_index_bisect(sorted(non_misinf_reshares, reverse=True))

            # Influence
            author_bad_influence = sum(misinf_reshares)
            author_non_bad_influence = sum(non_misinf_reshares)

            # Fall-index
            try:
                author_fall_index = author_FIB * sum(sorted(misinf_reshares, reverse=True)[:author_FIB-1])
            except IndexError:
                author_fall_index = 0

            # Self-reshares (how much the user is prone to self reshare posts)
            author_self_resharing = sum([c['self-reshares'] for c in author_shared_contents])

            # Add reliability: linked to data size for each user (more data more reliable score)

            self.author2features[author] = {'FIB-index': author_FIB,
                                            'anti-FIB-index': author_anti_FIB,
                                            'total-shares': author_total_contents,
                                            'misinf-shares': author_misinf_contents,
                                            'non-misinf-shares': author_non_misinf_contents,
                                            'self-resharing': author_self_resharing,
                                            'fall-index': author_fall_index,
                                            'bad-influence': author_bad_influence,
                                            'non-bad-influence': author_non_bad_influence}


    def set_weights(self, new_weights):
        assert len(new_weights) == len(self.feature_weights), 'Invalid weights!'
        self.feature_weights = new_weights


    # Get the rank for detected authors
    def get_rank(self, normalize=False):

        for author, features in self.author2features.items():

            fvector = np.array(list(features.values()), dtype=np.float32)
            fvector /= np.linalg.norm(fvector) if normalize else 1.0

            self.output_rank[author] = np.sum(fvector * self.feature_weights)

        return dict(sorted(self.output_rank.items(), key=lambda x: x[1], reverse=True))
        # TODO: Review this method


    # Single aggregation for round trip time evaluation
    def standard_rtt(self, estimated, sampled):
        return (1 - self.alpha) * estimated + self.alpha * sampled
        # return estimated - delta * (estimated - sampled) # OPTION 2


    # Estimate multiple authors features over time
    def __multi_rtt(self, author2estimated, author2sampled):

        # Update the estimated values if there are new sampled ones.
        for author, value in author2estimated.items():

            try:
                new = {k: self.standard_rtt(v, author2sampled[author][k]) for k, v in author2estimated[author].items()}
            except KeyError:
                # If the user have no activity in the next period the sampled value is considered 0
                new = {k: self.standard_rtt(v, 0.0) for k, v in author2estimated[author].items()}

            author2estimated[author] = new

        # Add new user if any. The initial estimated value will be the sampled one.
        for author, value in author2sampled.items():
            try:
                author2estimated[author]
            except:
                author2estimated[author] = value


    # Generate a time frame sequence. Each time frame is a list of events.
    # A time frame can be empty. A frame delta define a time frame size.
    def time_frames_generator(self, events, frame_delta):

        initial_timestamp = self.timestamp_key(events[0])

        time_frame = []

        for e in events:

            elapsed_time = self.timestamp_key(e) - initial_timestamp

            elapsed_frames = elapsed_time / frame_delta

            if elapsed_frames > 1:

                yield time_frame

                # Generate no activity frames if it happen
                for _ in range(math.floor(elapsed_frames) - 1):
                    yield []

                initial_timestamp = self.timestamp_key(e)

                time_frame = [e]

            else:

                time_frame.append(e)

            # Check and yield the last frame
            if e == events[-1]:
                yield time_frame


    # Time-aware fitting procedure: Splits the training period in sub-periods and aggregate the partial features
    def time_fit(self, data):

        # Init the time features accumulator
        author2time_features = dict()

        # Sort by time
        time_data = sorted(data, key=self.timestamp_key)

        for data_chunk in self.time_frames_generator(time_data, self.time_delta):

            # Evaluate chunk
            self.fit(data_chunk)

            # Merge result (Round Trip Time estimation formula)
            self.__multi_rtt(author2time_features, self.author2features)

            # Reset the main features dict
            self.author2features = dict()

        self.author2features = author2time_features



# Example usage:
if __name__ == "__main__":

    print("Testing...")

    # # Toy example for dataset. If content_id == root_id then is original content else reshared
    # toy = [{'content_id': "023", 'author_id': "794", 'root_content_id': '023', 'timestamp': None},
    #        {'content_id': "024", 'author_id': "431", 'root_content_id': '021', 'timestamp': None},
    #        {'content_id': "025", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
    #        {'content_id': "026", 'author_id': "431", 'root_content_id': '025', 'timestamp': None},
    #        {'content_id': "027", 'author_id': "431", 'root_content_id': '023', 'timestamp': None},
    #        {'content_id': "028", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
    #        {'content_id': "029", 'author_id': "794", 'root_content_id': '029', 'timestamp': None},
    #        {'content_id': "030", 'author_id': "999", 'root_content_id': '025', 'timestamp': None},
    #        {'content_id': "031", 'author_id': "431", 'root_content_id': '029', 'timestamp': None},
    #        {'content_id': "032", 'author_id': "794", 'root_content_id': '025', 'timestamp': None}]


    # model = HModel()

    # model.fit(toy, content_key=lambda x: x['content_id'],
    #           author_key=lambda x: x['author_id'],
    #           root_content_key= lambda x: x['root_content_id'],
    #           timestamp_key= lambda x: x['timestamp'])

    # print(model.author2features)
    # print(model.author2contents)


    # Random timestamp gen
    from random import randrange
    import datetime

    def random_date(start, l):
        current = start
        for _ in range(l):
            current += datetime.timedelta(minutes=randrange(60))
            yield current

    startDate = datetime.datetime(2022, 9, 20, 13, 00)
    events = []

    for x in random_date(startDate, 100):
        events.append((x, 'stub'))
        print(x.strftime("%d/%m/%y %H:%M"))
    print()

    # Time framer
    import math
    def time_frames(events, frame_delta, time_key):

        initial_timestamp = time_key(events[0])

        time_frame = []

        #print("\nEvents:\n", events)
        #print("\nFrame Delta:\n", frame_delta)

        for e in events:

            elapsed = time_key(e) - initial_timestamp

            #print("\nElapsed:\n", elapsed)

            if elapsed / frame_delta > 1:

                yield time_frame

                # No activity frames
                for _ in range(math.floor(elapsed/frame_delta) - 1):
                    yield []

                initial_timestamp = time_key(e)

                time_frame = [e]

            else:

                time_frame.append(e)

            if e == events[-1]:
                yield time_frame


    for f in time_frames(events, frame_delta=datetime.timedelta(minutes=45), time_key=lambda x: x[0]):
        print('!')
        print(f)