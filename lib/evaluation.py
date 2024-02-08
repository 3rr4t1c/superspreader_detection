import pandas as pd
import math

# TODO: users with same position should be sorted by reshares (in-going links)
def get_optimal_rank(edge_list_df: pd.DataFrame) -> pd.DataFrame:

    sources_df = edge_list_df[['source', 'weight']].groupby('source').sum()
    target_df = edge_list_df[['target', 'weight']].groupby('target').sum()

    optimal_rank_df = sources_df.join(target_df, 'outer').fillna(0)

    # optimal_rank_df.rename(columns={'weight': 'Optimal'}, inplace=True)
    # optimal_rank_df.index.rename('author_id', inplace=True)

    # TOFIX: optimal_rank_df = optimal_rank_df.merge(known_users_df, how='right', on='author_id').fillna(0).astype(int)
    optimal_rank_df.sort_values(by='Optimal', ascending=False, inplace=True)
    optimal_rank_df.set_index('author_id', inplace=True)
    optimal_rank_df.index = optimal_rank_df.index.astype(str)

    return


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
    # Assign ground truth relevance values to current rank
    for author, _ in ranking.items():
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

    return dcg/idcg


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