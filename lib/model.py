#from tqdm import tqdm
#from statistics import mean
#import statistics
import numpy as np
import math
#import random as rnd
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



class HModel:

    def __init__(self,
                 content_key,   # Function, applied to a data[i] object return the content id
                 author_key,    # Function, applied to a data[i] object return the author id
                 root_content_key,  # Function, applied to a data[i] object return the root content id
                 timestamp_key,     # Function, applied to a data[i] object return the timestamp
                 flag_key,    # Function, applied to a data[i] object return the flag value (e.g. misinformation)
                 delta=timedelta(days=5),   # timedelta object, the time frame size applied to data tweets
                 alpha=0.125,    # float, weight the importance of the past respect of the present in main time fit
                 enable_repost_count_scaling = False    # Enable the repost count scale. TODO: Explain why works better (?)
                 ) -> None:

        # Content features accessing functions
        self.content_key = content_key
        self.author_key = author_key
        self.root_content_key = root_content_key
        self.timestamp_key = timestamp_key
        self.flag_key = flag_key

        # Time training hyperparams
        self.alpha = alpha
        self.delta = delta

        # Content fast access
        self.content_index = {}

        # Map each authors to their published contents tracks
        self.author2contents = {}

        # Map each author to their behavior features (e.g. FIB)
        self.author2features = {}

        # Weights to linearly combine features
        self.feature_weights = None

        # Store the ranking by a defined metric
        self.output_rank = []

        # Track co-authors (followers) as the first authors that share a post
        self.author2coauthors = {}

        # EXPERIMENTAL: Enable repost count scaling.
        self.enable_repost_count_scaling = enable_repost_count_scaling


    # Getter for the learned rank
    def get_rank(self, normalize=False):

        self.output_rank = dict()

        for author, features in self.author2features.items():

            fvector = np.array(list(features.values()), dtype=np.float32)
            fvector /= np.linalg.norm(fvector) if normalize else 1.0

            self.output_rank[author] = np.sum(fvector * self.feature_weights[:-1]) + self.feature_weights[-1]

        self.output_rank = dict(sorted(self.output_rank.items(), key=lambda x: x[1], reverse=True))

        return self.output_rank


    # Setter for weights
    def set_weights(self, new_weights):
        assert len(new_weights) == len(self.feature_weights), 'Invalid weights lenght!'
        self.feature_weights = new_weights


    # Single aggregation for round trip time evaluation
    def standard_rtt(self, estimated, sampled):
        return (1 - self.alpha) * estimated + self.alpha * sampled
        # return estimated - delta * (estimated - sampled) # OPTION 2


    # Estimate multiple authors RTT over time
    def _multi_rtt(self, author2estimated, author2sampled):

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
            except KeyError:
                author2estimated[author] = value


    # Initialize counters for reposts
    def _fit_step_counters_init(self):

        for author in self.author2contents.keys():
            for track in self.author2contents[author].values():
                track['reposts_sampled_count'] = 0
                track['self_reposts_count'] = 0


    # Build A2C (author to contents) and CIDX (content index).
    # Require data chronologically sorted
    def _fit_step_index_and_track(self, data):

        # PRE: Chronologically sorted data sequence required
        for content in data:

            # Values unpacking
            content_id = self.content_key(content)
            author_id = self.author_key(content)
            root_content_id = self.root_content_key(content)
            timestamp = self.timestamp_key(content)
            flag = self.flag_key(content)

            # Add the content to the contents index for fast search
            self.content_index[content_id] = (author_id, root_content_id, timestamp, flag)

            if root_content_id == 'ORIGIN':

                content_track = {'reposts_total_count': 0,
                                 'reposts_sampled_count': 0,
                                 'reposts_used_count': 0,
                                 'reposts_cascade': [],
                                 'self_reposts_count': 0,
                                 'timestamp': timestamp,
                                 'flag': flag}

                try:
                    # Add the track for this content
                    self.author2contents[author_id][content_id] = content_track
                except KeyError:
                    # First time this author is encountered: initialize
                    self.author2contents[author_id] = {content_id: content_track}

            else:

                try: # ...try to resolve the original author

                    root_author_id = self.content_index[root_content_id][0]

                except KeyError: # if never seen this content skip iteration

                    continue # If data are chrono sorted and this happen it is not in the dataset, skip

                # If and author is found then a root content should exist
                root_content_track = self.author2contents[root_author_id][root_content_id]

                # If it is not a self reshare
                if root_author_id != author_id:

                    # Count as a reshare
                    root_content_track['reposts_sampled_count'] += 1

                    # Track the repost timeline
                    root_content_track['reposts_cascade'].append((content_id, author_id, timestamp))

                    # Assumption: the very first reposter should be a follower
                    if len(root_content_track['reposts_cascade']) == 1:

                        try:
                            # Enroll this user as a coauthor of this root_user
                            self.author2coauthors[root_author_id].add(author_id)
                        except KeyError:
                            # Handle non yet initialized authors
                            self.author2coauthors[root_author_id] = {author_id}

                else:
                    # Handle self reshare separately
                    root_content_track['self_reposts_count'] += 1


    # NOTE: At this point we have for each author all the original content they posted
    # and for each content they posted the count of reshares and self-reshares received


    # EXPERIMENTAL: enable to get better results. For further analysis.
    def _fit_step_repost_count_scaling(self, enabled=True):

        for author in self.author2contents.keys():
            for content in self.author2contents[author].values():
                if enabled:
                    content['reposts_used_count'] += content['reposts_total_count'] + content['reposts_sampled_count']
                    content['reposts_total_count'] += content['reposts_sampled_count']
                else:
                    content['reposts_used_count'] = content['reposts_sampled_count']


    # STEP 3: Extract features from tracked user's posts
    def _fit_step_feature_extraction(self):

        for author in self.author2contents.keys():

            # The list of stats for each shared content by current author (content id not included)
            author_shared_contents = self.author2contents[author].values()

            # Counts features init
            author_flagged_count = 0
            author_non_flagged_count = 0

            # Self reshares init
            author_self_resharing = 0

            # Influence init
            author_flagged_influence = 0
            author_non_flagged_influence = 0

            # FIB and anti-FIB init
            author_flagged_reshare_counts = []
            author_non_flagged_reshare_counts = []

            for c in author_shared_contents:

                # Contents count
                author_flagged_count += c['flag']
                author_non_flagged_count += (c['flag'] + 1) % 2

                # Self reshares
                author_self_resharing += c['self_reposts_count']

                # Influence
                author_flagged_influence += c['reposts_used_count'] if c['flag'] == 1 else 0
                author_non_flagged_influence += c['reposts_used_count'] if c['flag'] == 0 else 0

                # Lists with reshares count for each content
                if c['flag'] == 1:
                    author_flagged_reshare_counts.append(c['reposts_used_count'])
                else:
                    author_non_flagged_reshare_counts.append(c['reposts_used_count'])

            # FIB and anti_FIB finalization
            author_flagged_reshare_counts.sort(reverse=True)
            author_non_flagged_reshare_counts.sort(reverse=True)
            author_FIB = h_index_bisect(author_flagged_reshare_counts)
            author_anti_FIB = h_index_bisect(author_non_flagged_reshare_counts)

            try:
                # Fall-index: Should capture a relation between the total number of received reshares and FIB
                author_fall_index = author_FIB * sum(author_flagged_reshare_counts[:author_FIB-1])
            except IndexError:
                author_fall_index = 0

            try:
                author_coauthors = len(self.author2coauthors[author])
            except KeyError:
                author_coauthors = 0

            self.author2features[author] = {'FIB-index': author_FIB,
                                            'anti-FIB-index': author_anti_FIB,
                                            'flagged-influence': author_flagged_influence,
                                            'non-flagged-influence': author_non_flagged_influence,
                                            'flagged-count': author_flagged_count,
                                            'non-flagged-count': author_non_flagged_count,
                                            'self-resharing': author_self_resharing,
                                            'fall-index': author_fall_index,
                                            'co-authors': author_coauthors}


    # Automatically set the weights vector accordingly to the feature length
    def _auto_feature_size(self):

        fsize = len(list(self.author2features.values())[0]) + 1
        self.feature_weights = np.zeros(fsize, dtype=np.float32)
        self.feature_weights[0] = 1.0   # Set FIB as default feature


    # Core training function
    def fit(self, data, no_time_fit=True):

        content_stream = data

        if no_time_fit:
            content_stream = sorted(content_stream, key=self.timestamp_key)

        self._fit_step_counters_init()

        self._fit_step_index_and_track(content_stream)

        self._fit_step_repost_count_scaling(enabled=self.enable_repost_count_scaling)   # To be investigated

        self._fit_step_feature_extraction()

        if no_time_fit:
            self._auto_feature_size()


    # Generate a time frame sequence. Each time frame is a list of events.
    # A time frame can be empty. A frame delta define a time frame size.
    def time_frames_generator(self, events, frame_delta, timestamp_key, frame_no_activity=True):

        # No events mean one empty frame
        if not events:

            yield []

        else:

            initial_timestamp = timestamp_key(events[0])

            time_frame = []

            for e in events:

                current_time = timestamp_key(e)

                elapsed_time = current_time - initial_timestamp

                elapsed_frames = elapsed_time / frame_delta

                # If current event is out of this time frame
                if elapsed_frames > 1:
                    # Yield the current time frame
                    yield time_frame
                    # If requested, generate no activity time frames
                    if frame_no_activity:
                        for _ in range(math.floor(elapsed_frames) - 1):
                            yield []
                    # Set the new initial timestamp to the curent event time
                    initial_timestamp = current_time
                    # Init the new frame with the current event
                    time_frame = [e]
                else:
                    # Keep filling the current time frame
                    time_frame.append(e)

            # Yield the last frame
            yield time_frame


    def luceri_rank(self):

        for author, content2stats in self.author2contents.items():
            for contentID, stats in content2stats.items():
                single_post_rank = [repost[1] for repost in stats["reposts_cascade"]]


    # Time-aware fitting procedure: Splits the training period in sub-periods and aggregate the partial features
    def time_fit(self, data):

        # Init the time features accumulator
        temp_author2features = dict()

        # Sort by time
        time_data = sorted(data, key=self.timestamp_key)

        for data_chunk in self.time_frames_generator(time_data, self.delta, timestamp_key=self.timestamp_key):

            # Evaluate chunk
            self.fit(data_chunk, no_time_fit=False)

            # Merge result (Round Trip Time estimation formula)
            self._multi_rtt(temp_author2features, self.author2features)

            # Reset the main features dict (used for every frame)
            self.author2features = dict()

        # Save features to model instance variable
        self.author2features = temp_author2features

        # Detect and set feature vector size
        self._auto_feature_size()



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

    for x in random_date(startDate, 1000):
        events.append((x, 'stub'))
        print(x.strftime("%d/%m/%y %H:%M"))
    print()

    # Time framer
    import math
    def time_frames(events, frame_delta, time_key):

        if not events:

            yield []

        else:

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

            yield time_frame

    print('!')
    frame_list = []
    for f in time_frames(events, frame_delta=datetime.timedelta(minutes=15), time_key=lambda x: x[0]):
        print(f)
        frame_list += f
        print('!')

    print(len(frame_list))
    print(len(events))