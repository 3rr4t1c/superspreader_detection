import pandas as pd
import math


# Build a retweet network from DataFrame (returns the edgelist as a DataFrame). 
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



# Network dismantling procedure
def network_dismantle(network_df, ranking):
    """
    Network dismantling procedure.
    
    Arguments:
        network_df (pandas.DataFrame): The network DataFrame as edgelist (source, target, weight)
        ranking (dict): A ranking with the format k: v where k is the item ID and v is the score

    Returns:
        list: A list of pairs (k, v) where k is the removed node name 
              and v is the remaining misinformation
    """

    # Convert the ranking format for fast computation
    ranking_df = pd.DataFrame.from_dict(ranking, orient="index", columns=["score"])
    ranking_df.sort_values(by="score")

    # The total misinformation retweeted
    total_misinformation = network_df.weight.sum()

    # Hold the network to be dismantled
    dismantled_network_df = network_df.copy()

    # Track the remaining misinformation
    misinformation_track = [("FULL", 1.0)]

    for i, (author, _) in enumerate(ranking_df.itertuples()):

        # Removing a non present node result in appending the same value. Handle this.
        if ((dismantled_network_df.source == author).any() |
            (dismantled_network_df.target == author).any()):

            # Build a boolean mask for edges
            remaining_edges_mask = ((
                dismantled_network_df.source != author) &
                (dismantled_network_df.target != author))

            # Remove all edges involving the current author (node)
            dismantled_network_df = dismantled_network_df[remaining_edges_mask]

            # The ratio of current remaining misinformation in the network
            remaining_misinformation = dismantled_network_df.weight.sum() / total_misinformation

            # Append data to the track list
            misinformation_track.append((author, remaining_misinformation))

    return misinformation_track


# # nDCG Remake (now with ties handling!)
# def discounted_cumulative_gain(relevance_scores: list) -> float:
#     """
#     Compute the Discounted Cumulative Gain for a list of scores.
#     If the list of scores contain blocks of equal subsequential scores,
#     they're considered to have same rank position and get the same discount.

#     Arguments:
#         relevance_scores (list): a list of numbers representig the relevance scores

#     Returns:
#         float: the computed discounted cumulative gain
#     """

#     dcg = 0
#     i = 1
#     prev_score = None

#     for score in relevance_scores:

#         if not prev_score or score != prev_score:
#             i += 1

#         dcg += score / math.log(i, 2)

#         prev_score = score

#     return dcg

def discounted_cumulative_gain(relevance_scores: list, k: int = None) -> float:
    """
    Compute the Discounted Cumulative Gain (DCG) for a list of relevance scores.
    
    DCG measures the ranking quality of the retrieved documents in a search engine
    or recommendation system. It considers both the relevance of the documents
    and their positions in the ranked list.

    If the list of scores contains blocks of equal subsequential scores,
    they're considered to have the same rank position and get the same discount.

    Arguments:
        relevance_scores (list): A list of numbers representing the relevance scores.
        k (int): The cutoff for considering only the top k scores. If None, consider all scores.

    Returns:
        float: The computed DCG.
    """

    # If k is not None, consider only the top k scores
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0  # Initialize discounted cumulative gain
    i = 1    # Initialize rank position
    prev_score = None  # Initialize previous score

    # Calculate DCG
    for score in relevance_scores:

        # If the current score is different from the previous score,
        # increment the rank position (ties handling).
        if not prev_score or score != prev_score:
            i += 1

        # Calculate discounted cumulative gain using the formula
        dcg += score / math.log(i, 2)

        # Update the previous score
        prev_score = score

    return dcg


def nDCG_loss(true_ranking:dict, test_ranking:dict, k:int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain based Loss.

    Arguments:
        true_ranking (dict): Pairs k: v where k is the item ID and v is the relevance score
        test_ranking (dict): Pairs k: v where k is the item ID and v is the relevance score

    Returns:
        float: The nDCG loss. A number that is equal to 0 if true_ranking == test_ranking
    """

    # Find the overlapping items
    overlap_keys = set(true_ranking.keys()).intersection(set(test_ranking.keys()))

    # Re-build the rankings
    overlap_true = []
    overlap_test = []
    for id in overlap_keys:
        overlap_true.append((id, true_ranking[id]))
        overlap_test.append((id, test_ranking[id]))

    # Quick access to score
    value_fn = lambda x: x[1]

    # Sort the normalized true pairlist by descending scores
    sorted_true = sorted(overlap_true, key=value_fn, reverse=True)
    sorted_test = sorted(overlap_test, key=value_fn, reverse=True)

    # Quick access to the true scores
    sorted_true_index = dict(sorted_true)

    # Re-build the test as a scramble of true scores
    build_test = [(id, sorted_true_index[id]) for id, _ in sorted_test]

    # Evaluate the gains
    dcg = discounted_cumulative_gain([value_fn(x) for x in build_test], k=k)
    idcg = discounted_cumulative_gain([value_fn(x) for x in sorted_true], k=k)

    return 1.0 - (dcg / idcg)



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

    print(nDCG_loss(rank, optimal))
    print(nDCG_loss(optimal, optimal))