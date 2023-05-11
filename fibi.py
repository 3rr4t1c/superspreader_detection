# Implementation of FIB (False Information Broadcaster) index

# Input: a list of tuples, each tuples represent a document (post, shared content, etc.)
# (content_id, author_id, parent_id, interactions_count, misinfection_lvl)
#
# content_id: is a unique identifier for this content or ACTIVITY_ID
# author_id: is a unique identifier for who shared this content
# parent_id: is the identifier of the source content, if this content is original then parent_id == post_id
# interaction_count: is an integer number counting any type of external interaction (retweet, likes, comment...) or the sum of multiple interactions
# misinfection_lvl: is a number that express how likely the content is carrying misinformation

# H-index algorithm with binary search implementation
def eval_binary_FIB_index(sorted_lst):

    start_index = 0
    end_index = len(sorted_lst)

    while start_index < end_index:

        mid_index = (end_index + start_index) // 2

        if sorted_lst[mid_index] < mid_index:
            end_index = mid_index
        elif sorted_lst[mid_index] > mid_index:
            start_index = mid_index + 1
        else:
            break

    return mid_index


def eval_FIB_index(content_list, misin_thr=60):

    author2index = dict()

    # Build authors activity track. NOTE: some attribute are still not used and stays for future experiments
    for content_id, author_id, parent_id, interactions_count, misinfection_lvl in content_list:
        # Collect tweet if is misinfected
        if misinfection_lvl <= misin_thr:
            try:
                author2index[author_id].append(interactions_count)
            except KeyError:
                author2index[author_id] = [interactions_count]

    # Build author to FIB-index map
    for author in author2index.keys():

        interlist = author2index[author]
        interlist.sort(reverse=True)
        author2index[author] = eval_binary_FIB_index(interlist)

    return author2index



# Example usage:
if __name__ == "__main__":

    print("Testing...")

    import random as rnd
    from string import ascii_lowercase, digits
    from scipy.stats import truncnorm

    # Return true wih prob probability
    def rand_chance(prob):
        return rnd.random() < prob

    # Set the mean and standard deviation
    mu1, sigma1 = 0.3, 0.4 # activity
    mu2, sigma2 = 0.0, 0.4 # misinformation

    # Set the lower and upper bounds
    lower, upper = 0, 1

    # Generate n author names, each with an activity index, chanches of spread misinformation
    authid_len = 5
    auth_count = 500
    chars = ascii_lowercase + digits
    lst = [''.join(rnd.choice(chars) for _ in range(authid_len)) for _ in range(auth_count)]
    actividx = truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1).rvs(auth_count)
    misinprob = truncnorm((lower - mu2) / sigma2, (upper - mu2) / sigma2, loc=mu2, scale=sigma2).rvs(auth_count)
    authlist = list(zip(lst, actividx, misinprob))

    N = 1000000
    contents = []
    
    # Generate n author names, each with an activity index, chanches of spread misinformation
    print(f"Generating {N} synthetic contents...", flush=True)
    for n in range(N):

        not_generated = True

        while not_generated:

            # Sample a single author name and with activity index probability create a single content
            author, act, mis = rnd.sample(authlist, 1)[0]

            if rnd.random() < act:

                content_id = str(n)
                try:
                    parent_id = content_id if rnd.random() < 0.7 else rnd.sample(contents, 1)[0][0]
                except ValueError:
                    parent_id = content_id
            
                interactions = rnd.randint(0, int(N*act*rnd.random()))
                misinfection_lvl = 1 if rnd.random() < mis else 0

                contents.append((content_id, author, parent_id, interactions, misinfection_lvl))
                not_generated = False

    # print(contents)
    input(f"Fake contents generated and ready.\nPress any key to start algorithm...")
    fibindexes = eval_FIB_index(contents)
    print("Done.", flush=True)
    # Print a rank
    print("First elements are:")
    [print(k, v) for k, v in sorted(fibindexes.items(), key=lambda x: x[1], reverse=True)[:10]]
