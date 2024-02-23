import pandas as pd
import math


# TODO: Build a retweet network from DataFrame (returns the edgelist as a DataFrame). 
def get_retweet_network(retweets_df: pd.DataFrame,
                        rt_UserID_col:str,
                        userID_col: str,
                        rating_col:str,
                        low_cred_thr=None) -> pd.DataFrame:
    
    features = [rt_UserID_col, userID_col, rating_col]
    edge_list_df = retweets_df[features].copy()
    
    if low_cred_thr:
        # Keep rows with credibility under threshold (credibility retweet network)
        edge_list_df = edge_list_df[edge_list_df[rating_col] <= low_cred_thr]
    
    # Group by ID pairs (retweeter and tweeter = edge) and aggregate by counting
    edge_list_df = edge_list_df.groupby([rt_UserID_col, userID_col]).count()
    # Sort values for nice prininting and reset the index
    edge_list_df.sort_values(by=rating_col, ascending=False, inplace=True)
    edge_list_df.reset_index(inplace=True)
    # Change the columns names
    edge_list_df.rename(columns = {rt_UserID_col: "source",
                                   userID_col: "target",
                                   rating_col: "weight"}, inplace=True)
    
    return edge_list_df


# Given a network edgelist return the optimal order of removal
def get_optimal_ranking(edge_list):

    # Step 1: Calculate sum of outgoing links' weights for each source node
    outgoing_weights = edge_list.groupby("source")["weight"].sum().reset_index()
    outgoing_weights.rename(columns={"weight": "outgoing_weight"}, inplace=True)

    # Step 2: Calculate sum of incoming links' weights for each target node
    incoming_weights = edge_list.groupby("target")["weight"].sum().reset_index()
    incoming_weights.rename(columns={"weight": "incoming_weight"}, inplace=True)

    # Step 3: Merge results to get total weight for each node
    ranking_df = pd.merge(outgoing_weights, incoming_weights, left_on="source", right_on="target", how="outer")
    ranking_df["outgoing_weight"] = ranking_df["outgoing_weight"].fillna(0)
    ranking_df["incoming_weight"] = ranking_df["incoming_weight"].fillna(0)

    # Step 4: Merge the source and target columns and sort nodes
    ranking_df["node"] = ranking_df["source"].combine_first(ranking_df["target"])
    ranking_df.sort_values(by=['outgoing_weight', 'incoming_weight'], ascending=False, inplace=True)
    ranking_df.drop(["source", "target"], axis=1, inplace=True)
    ranking_df.reset_index(inplace=True, drop=True)

    return ranking_df[["node", "outgoing_weight", "incoming_weight"]]


# Dismantling procedure by given ranking.
# Remove from network_df nodes according to ranking_df order and save the remaining misinformation.
def network_dismantle(network_df: pd.DataFrame, ranking_df: pd.DataFrame) -> list:

    # print("Dismantling...")

    full_misinformation = network_df.weight.sum()

    dismantled_network_df = network_df.copy()

    network_sources = set(network_df[['source', 'target']].values.ravel(order='K')) #set(network_df.source.unique())

    track = [(None, 1.0)]

    for author, _ in ranking_df.itertuples():

        if author in network_sources:

            # The network without the current author as node.
            # NOTE: removing a node mean that all edges involving that node should be removed.
            author_mask = (dismantled_network_df.source != author) & (dismantled_network_df.target != author)
            dismantled_network_df = dismantled_network_df[author_mask]

            # The ratio of current remaining misinformation in the network
            remaining_misinformation = dismantled_network_df.weight.sum() / full_misinformation
            
            track.append((author, remaining_misinformation))

    # Adjust sequence to have same lenght
    n_residual_sources = dismantled_network_df.source.nunique()
    track += [track[-1] for _ in range(n_residual_sources)]

    return track


# nDCG is used in common Information Retrieval tasks for
# evaluate a IR model query result taking in account the order of retrieved documents
# and the relevance score.
def discounted_cumulative_gain(relevance_scores):
    return sum([rel / math.log(n+1, 2) for n, rel in enumerate(relevance_scores, start=1)])


def normalized_discounted_cumulative_gain(ranking: dict, optimal_ranking: dict) -> float:

    rels = []

    # Assign ground truth relevance values to current ranking
    for author in ranking.keys():

        try:
            rels.append(optimal_ranking[author])
        except KeyError:
            pass

    dcg = discounted_cumulative_gain(rels)
    idcg = discounted_cumulative_gain(sorted(rels, reverse=True))

    #print()
    #print('rels', rels)
    #print('DCG', dcg)
    #print('IDCG', idcg)
    try:
        result = dcg/idcg
    except ZeroDivisionError:
        result = 0

    return result


## CUSTOM SIMILARITY METRIC FOR RANKINGS ##

# Auxiliary function. Convert a ranking with scores to a position rank.
# Ties will be handled by assignig the same rank position. 
def to_sequential_rank(ranking: dict) -> dict:

    items = list(ranking.items())
    seq_ranking = [(items[0][0], 0)]

    for n, (k, v) in enumerate(items[1:], start=1):
        
        if v == items[n-1][1]:
            item = (items[n][0], seq_ranking[n-1][1])
        else:
            item = (items[n][0], seq_ranking[n-1][1] + 1)
            
        seq_ranking.append(item)
        
          
    return dict(seq_ranking)

# Auxiliary function. Invert the order of keys while keeping the values order.
# Basically generate the inverted ranking. {'a': 10, 'b': 7, 'c': 7} -> {'c': 10, 'b': 7, 'a': 7}
def invert_ranking(ranking: dict) -> dict:

    inverted_keys = list(ranking.keys())
    inverted_keys.reverse()
    values = ranking.values()

    return dict(zip(inverted_keys, values))


# Main function. Evaluate the error committed with the position in the ranking.
# A ground truth should be provided to evaluate against.
def ranking_similarity(ranking_GT, ranking):
    
    # Overlap
    common_items = set(ranking_GT.keys()).intersection(set(ranking.keys()))

    # Create overlapping ranking (drop items not in common)
    sort_key = lambda x: x[1]
    overlap_ranking_GT = dict(sorted([(k, ranking_GT[k]) for k in common_items], key=sort_key, reverse=True))
    overlap_ranking = dict(sorted([(k, ranking[k]) for k in common_items], key=sort_key, reverse=True))

    # convert to sequential rankings
    overlap_ranking_GT = to_sequential_rank(overlap_ranking_GT)
    overlap_ranking = to_sequential_rank(overlap_ranking)
    inverted_overlap_ranking_GT = invert_ranking(overlap_ranking_GT)

    # evaluate errors for values (v_rank - v_GT)
    distance = 0
    inverted_distance = 0
    for k, v in overlap_ranking.items():
        distance += (v - overlap_ranking_GT[k])**2 / math.log(v + 2, 2)
        inverted_distance += (v - inverted_overlap_ranking_GT[k])**2 / math.log(v + 2, 2)

    return 1 - (distance / inverted_distance)

## CUSTOM METRIC 2 UPDATED ##

import math

def ranking_normalizer(ascending_pair_list):

    output_ranking = []
    n = 0
    for i, (k, v) in enumerate(ascending_pair_list):
        
        if output_ranking and v != ascending_pair_list[i-1][1]:
            n += 1
            
        output_ranking.append((k, n))

    return output_ranking


def ranking_loss(true_ranking, test_ranking):

    # Find the overlapping items
    overlap_keys = set(true_ranking.keys()).intersection(set(test_ranking.keys()))

    # Re-build the rankings
    overlap_true = []
    overlap_test = []
    for k in overlap_keys:
        overlap_true.append((k, true_ranking[k]))
        overlap_test.append((k, test_ranking[k]))

    # To sorted pair list
    sort_fn = lambda x: x[1]
    sorted_true = sorted(overlap_true, reverse=True, key=sort_fn)
    sorted_test = sorted(overlap_test, reverse=True, key=sort_fn)

    # To normalized rank
    sorted_true_normalized = ranking_normalizer(sorted_true)
    sorted_test_normalized = ranking_normalizer(sorted_test)

    # Align with true (ground truth). If ground truth has tied items chains allow to not consider test ranks as errors
    # sorted_test_aligned = [(p[0], sorted_true_normalized[i][1]) for i, p in enumerate(sorted_test)]

    # Make a dict for fast access
    sorted_true_normalized = dict(sorted_true_normalized)

    loss = 0
    # den = 0
    # Evaluating the loss
    for k, v in sorted_test_normalized:
        
        true_v = sorted_true_normalized[k]  
        error = (true_v - v)**2
        # weight = 1 / math.log(v + 2, 2)
        loss += error #* weight
        #den += weight

    return loss #/ den


def invert_ranking(ranking_dict):
    
    max_value = max(ranking_dict.values())
    min_value = min(ranking_dict.values())
    
    result = {k: abs(v - max_value) + min_value for k, v in ranking_dict.items()}
    
    return result


def ranking_loss_normalized(true_ranking, test_ranking):

    loss = ranking_loss(true_ranking, test_ranking)
    inverted_true_ranking = invert_ranking(true_ranking)
    worse_loss = ranking_loss(true_ranking, inverted_true_ranking)
    
    return loss / worse_loss 

## END CUSTOM METRIC 2 ##


# Utility function. Convert a date time column to the float format.
def datetime_to_float(time_df, datetime_column, time_unit="second"):

    df = time_df.copy()

    time_rateo = 1

    if time_unit == "minute":
        time_rateo = 60
    elif time_unit == "hour":
        time_rateo = 3600
    elif time_unit == "day":
        time_rateo = 24 * 3600
    else:
        raise ValueError

    df["time_float"] = df[datetime_column] - df[datetime_column].min()
    df["time_float"] = df["time_float"].dt.total_seconds()
    df["time_float"] = df["time_float"] / time_rateo

    return df


# Example usage:
if __name__ == "__main__":

    import random as rnd
    from string import ascii_lowercase, digits

    authid_len = 5
    auth_count = 100
    chars = ascii_lowercase + digits
    lst = [''.join(rnd.choice(chars) for _ in range(authid_len)) for _ in range(auth_count)]

    rank = dict(sorted({k: rnd.random()*1000 for k in lst}.items(), key=lambda x: x[1], reverse=True))
    optimal = dict(sorted({k: rnd.random()*100 for k in lst}.items(), key=lambda x: x[1], reverse=True))

    print(rank)
    print(optimal)

    print(normalized_discounted_cumulative_gain(rank, optimal))
    print(normalized_discounted_cumulative_gain(optimal, optimal))