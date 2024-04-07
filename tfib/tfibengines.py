import numpy as np
import math
from datetime import timedelta


def h_index_bisect(rsl, key=lambda x: x):
    """
        Binary search for quick h-index.

        Given a descending sorted list returns
        the index corresponding to the h-index definition
        if a key function is provided it will applied to each element
        before comparisons.
    """

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



class TFIBEngine:

    def __init__(self,
                 reshare_key,   # Function, applied to a data[i] object return the content id.
                 author_key,    # Function, applied to a data[i] object return the author id.
                 original_post_key,  # Function, applied to a data[i] object return the original content id.
                 original_author_key,   # Function, applied to a data[i] object return the original author id.
                 timestamp_key,     # Function, applied to a data[i] object return the timestamp.
                 label_key,    # Function, applied to a data[i] object return the flag value (e.g. misinformation).
                 score_threshold, # Set a threshold for score. Below that threshold contents are flagged with 1.
                 alpha=0.5,    # float, weight the importance of the past respect of the present in main time fit.
                 beta=0.5,       # float, weight the importance of deviation from estimated value.
                 gamma=1.0,       # float weight the final estimated value when using the Jacobson/Karels version.
                 delta=10.0,    # float indicating the interval in time units (days, hours, etc.) for each time slot.
                 use_original_rtt=False,    # Wheter to use standard RTT estimation formula or Jacobson/Karels version.
                 # enable_repost_count_scaling = False    # Enable the repost count scale. TODO: Explain why works better (?) Update (12.03.2024) Seems not working anymore, no changes if enabled.
                 ) -> None:

        # Content features accessing functions
        self.content_ID_key = reshare_key
        self.author_ID_key = author_key
        self.original_post_ID_key = original_post_key
        self.original_author_ID_key = original_author_key
        self.timestamp_key = timestamp_key
        self.label_key = label_key

        # Set the credibility threshold on the fly
        self.score_threshold = score_threshold

        # Time training perparams
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Select the formula to use for rtt
        self.use_original_rtt = use_original_rtt

        # Structures #

        # Map each authors to their published contents tracks
        self.author2tracks = {}

        # Map each author to their behavior features (e.g. FIB)
        self.author2features = {}

        # For TFIB estimation. Smoothed deviation of features.
        self.author2deviations = {}

        # Weights to linearly combine features
        self.feature_weights = None

        # Store the ranking by a defined metric
        self.output_rank = []

        # Track co-authors (followers) as the first authors that share a post
        # self.author2coauthors = {}

        # EXPERIMENTAL: Enable repost count scaling.
        # self.enable_repost_count_scaling = enable_repost_count_scaling


    # Time-aware fitting
    def time_fit(self, data):

        # Init the time features accumulator
        temp_author2features = {}

        # Sort by time
        time_sorted_data = sorted(data, key=self.timestamp_key)

        for data_chunk in self.time_slots_generator(time_sorted_data, key_fn=self.timestamp_key):

            # Evaluate chunk
            self.fit(data_chunk, time_fit=True)

            # Merge result (Round Trip Time estimation formula)
            self._multi_rtt(temp_author2features, self.author2features)

            # Reset the main features dict (used for every frame)
            self.author2features = dict()

            # try: # DEBUG (print the estimated index at each iteration)
            #     print(temp_author2features["1683455144"]["H-index"])
            # except KeyError:
            #     pass

        # Save features to model instance variable
        self.author2features = temp_author2features

        # Handle the different estimation methods
        if not self.use_original_rtt:
            # Finalize the JK algorithm
            for author, features in self.author2features.items():
                for fname, estimated in features.items():
                    deviation = self.author2deviations[author][fname]
                    features[fname] = estimated + self.gamma * deviation

        # Detect and set feature vector size
        self._auto_feature_size()


    def time_slots_generator(self, data, key_fn, allow_empty_slots=True):
        """
        Generate time slots based on a delta value.

        Args:
            data (iterable): Input data, should be sorted in ascending order based on key_fn.
            key_fn (function): A function to extract the timestamp from each element in data.
            delta (float, optional): Time interval for each time slot. Defaults to 1.0.
            allow_empty_slots (bool, optional): Whether to generate empty time slots. Defaults to True.

        Yields:
            list: Time slots.
        """

        # Initialize variables
        time_slot = []  # Current time slot
        clock = self.delta    # Clock indicating the end of current time slot

        # Iterate over the data
        for x in data:
            # Extract timestamp from the current element
            time_stamp = key_fn(x)

            # Check if the current element belongs to the current time slot
            if time_stamp <= clock:
                time_slot.append(x)  # Add the element to the current time slot
            else:
                # If the current element exceeds the current time slot, yield the current time slot
                yield time_slot
                time_slot = [x]  # Start a new time slot with the current element

                # Compute the number of empty time slots to be generated
                time_stamp_slot = time_stamp // self.delta  # Calculate the slot index of the current element
                current_slot = clock / self.delta           # Calculate the slot index of the current clock
                empty_slots = int(time_stamp_slot - current_slot)  # Compute the number of empty slots

                # Generate empty time slots if allowed
                if allow_empty_slots:
                    for _ in range(empty_slots):
                        yield []  # Yield an empty time slot

                # Update the clock to the end of the new time slot
                clock += self.delta * (1 + empty_slots)

        # Yield the final time slot if it's not empty
        if time_slot:
            yield time_slot


    # Core training function
    def fit(self, data, time_fit=False):

        content_stream = data

        if not time_fit: # If not time-fitting ensure data is sorted by time
            content_stream = sorted(content_stream, key=self.timestamp_key)

        self._fit_step_counters_init()

        self._fit_step_index_and_track(content_stream)

        # self._fit_step_repost_count_scaling(enabled=self.enable_repost_count_scaling)   # To be investigated

        self._fit_step_feature_extraction()

        if not time_fit:
            self._auto_feature_size()


    # Initialize counters for reshares
    def _fit_step_counters_init(self):

        for author in self.author2tracks.keys():
            for track in self.author2tracks[author].values():
                track["reposts_sampled_count"] = 0
                track["self_reshares_count"] = 0


    # Build A2C (author to contents)
    # Require data chronologically sorted
    def _fit_step_index_and_track(self, content_stream):

        for content in content_stream:

            # Values unpacking
            content_ID = self.content_ID_key(content)
            author_ID = self.author_ID_key(content)
            original_post_ID = self.original_post_ID_key(content)
            original_author_ID = self.original_author_ID_key(content)
            timestamp = self.timestamp_key(content)
            binary_flag = 1 if self.label_key(content) <= self.score_threshold else 0

            # # Handle contents that are original posts
            # is_original_post = original_post_ID == "ORIGIN"
            # if is_original_post:
            #     original_post_ID = content_ID
            #     original_author_ID = author_ID
            # TODO: Remove this code. We work with reshares only.

            try: # Get the contents published by this author
                post2track = self.author2tracks[original_author_ID]
            except KeyError: # First time see this author, init the contents index
                post2track = {}
                self.author2tracks[original_author_ID] = post2track

            try: # Get the track for this content
                root_content_track = post2track[original_post_ID]
            except KeyError: # New content found, init the track
                root_content_track = {
                    #'reposts_total_count': 0,
                    #'reposts_used_count': 0,
                    "reposts_sampled_count": 0,
                    "self_reshares_count": 0,
                    # "timestamp": timestamp,
                    "reshares_cascade": [],
                    "binary_flag": binary_flag
                }
                post2track[original_post_ID] = root_content_track

            # if it's a reshare
            # if not is_original_post:

            # if it's not a self-reshare
            if original_author_ID != author_ID:

                # Count as a reshare
                root_content_track["reposts_sampled_count"] += 1

                # Track the reshares cascade
                root_content_track["reshares_cascade"].append((content_ID, author_ID, timestamp))

                # # Assumption: the very first reposter could be a follower
                # if len(root_content_track['reshares_cascade']) == 1:

                #     try:
                #         # Enroll this user as a coauthor of this root_user
                #         self.author2coauthors[original_author_ID].add(author_ID)
                #     except KeyError:
                #         # Handle non yet initialized authors
                #         self.author2coauthors[original_author_ID] = {author_ID}

            else: # if it's a self-reshare count separately

                # Handle self reshare separately
                root_content_track["self_reshares_count"] += 1

        # NOTE: At this point we have for each author all the original contents they posted
        # and for each content they posted the count of reshares and self-reshares received


    # STEP 3: Extract features from tracked user's posts
    def _fit_step_feature_extraction(self):

        for author in self.author2tracks.keys():

            # The list of stats for each shared content by current author (content id not included)
            author_content_tracks = self.author2tracks[author].values()

            # Lower/Upper than threshold count
            author_flagged_count = 0
            author_non_flagged_count = 0

            # Self reshares count
            author_self_reshares = 0

            # Influence (total reshares) count
            author_flagged_influence = 0
            author_non_flagged_influence = 0

            # FIB and anti-FIB init
            author_flagged_reshares_seq = []
            author_non_flagged_reshares_seq = []

            for track in author_content_tracks:

                # Below threshold content
                if track["binary_flag"] == 1:

                    # Contents count
                    author_flagged_count += 1

                    # Influence
                    author_flagged_influence += track["reposts_sampled_count"]

                    # Lists with reshares count for each content
                    author_flagged_reshares_seq.append(track["reposts_sampled_count"])

                else:   # greater than score threshold

                    author_non_flagged_count += 1
                    author_non_flagged_influence += track["reposts_sampled_count"]
                    author_non_flagged_reshares_seq.append(track["reposts_sampled_count"])

                # self reshares total count
                author_self_reshares += track["self_reshares_count"]

            # H-index and anti H-index finalization
            author_flagged_reshares_seq.sort(reverse=True)
            author_non_flagged_reshares_seq.sort(reverse=True)
            author_H_index = h_index_bisect(author_flagged_reshares_seq)
            author_anti_H_index = h_index_bisect(author_non_flagged_reshares_seq)

            # Normalized H-index by total number of flagged posts
            all_posts_count = len(author_content_tracks)
            author_nH_index = author_H_index / all_posts_count
            author_anti_nH_index = author_anti_H_index / all_posts_count

            # try:
            #     author_coauthors = len(self.author2coauthors[author])
            # except KeyError:
            #     author_coauthors = 0

            # Add the extracted features for current author
            self.author2features[author] = {
                "H-index": author_H_index,
                "anti-H-index": author_anti_H_index,
                "flagged-influence": author_flagged_influence,
                "non-flagged-influence": author_non_flagged_influence,
                "flagged-count": author_flagged_count,
                "non-flagged-count": author_non_flagged_count,
                "self-reshares": author_self_reshares,
                "anti-normalized-H-index": author_anti_nH_index,
                "normalized-H-index": author_nH_index
            }

        # Set the feature size globally
        self.fsize = len(self.author2features[author]) + 1


    # Automatically set the weights vector accordingly to the feature length
    def _auto_feature_size(self):

        self.feature_weights = np.zeros(self.fsize, dtype=np.float32)
        self.feature_weights[0] = 1.0   # Set H-index as default feature


    # Estimate multiple authors RTT over time (variance version)
    def _multi_rtt(self, author2estimated, author2sampled):

        # Iterate over the sampled feature for current time slot
        for author, sampled_features in author2sampled.items():

            try:    # Try to update estimated features

                estimated_features = author2estimated[author]

                if not self.use_original_rtt:
                    estimated_deviations = self.author2deviations[author]

                for fname, estimated_value in estimated_features.items():

                    sampled_value = sampled_features[fname]

                    if self.use_original_rtt:
                        new_estimated = self._original_rtt(estimated_value, sampled_value)
                    else:
                        deviation = estimated_deviations[fname]
                        new_estimated, new_deviation = self._jk_rtt(estimated_value, sampled_value, deviation)
                        self.author2deviations[author][fname] = new_deviation

                    author2estimated[author][fname] = new_estimated

            except KeyError:    # If author not found: initialization needed

                # initialize estimated to sampled
                author2estimated[author] = sampled_features

                # initialize all deviations to zero
                if not self.use_original_rtt:
                    features_deviation = {k: 0.0 for k in sampled_features.keys()}
                    self.author2deviations[author] = features_deviation


    # Single aggregation for round trip time evaluation
    def _original_rtt(self, estimated, sampled):

        new_estimated = self.alpha * estimated + (1.0 - self.alpha) * sampled

        return new_estimated


    # Jacobson/Karels algorithm for better rtt estimation
    # Source > https://tcpcc.systemsapproach.org/algorithm.html
    # Source2 > http://isp.vsi.ru/library/Networking/TCPIPIllustrated/tcp_time.htm
    def _jk_rtt(self, estimated, sampled, deviation):

        # Compute the difference between the current sampled RTT and the estimated RTT
        difference = sampled - estimated

        # Update the estimated RTT using Jacobson/Karels algorithm
        new_estimated = estimated + self.alpha * difference

        # Update the deviation estimate using Jacobson/Karels algorithm
        new_deviation = deviation + self.beta * (abs(difference) - deviation)

        return new_estimated, new_deviation


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



    # # EXPERIMENTAL: enable to get better results. For further analysis.
    # def _fit_step_repost_count_scaling(self, enabled=True):

    #     for author in self.author2contents.keys():
    #         for content in self.author2contents[author].values():
    #             if enabled:
    #                 content['reposts_used_count'] += content['reposts_total_count'] + content['reposts_sampled_count']
    #                 content['reposts_total_count'] += content['reposts_sampled_count']
    #             else:
    #                 content['reposts_used_count'] = content['reposts_sampled_count']




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