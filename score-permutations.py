import re

from tqdm import tqdm
import pandas as pd
import numpy as np

def get_optimal_position(data, movie):
    positions = []

    for _, row in data.iterrows():
        vote = np.argwhere(row == movie)[0][0]

        positions.append(vote)
    
    return np.mean(positions)

def get_score(data, perm):
    scores = []
    for week, movie in enumerate(perm):
        score = 0
        for _, row in data.iterrows():
            vote = np.argwhere(row == movie)[0][0]
            distance = np.round((abs(week - vote) / len(movie)) * 100)

            score += 100 - distance
        scores.append(score)

    return scores

def swap_values(perm, first, second):
    perm_copy = perm.copy()
    first_val = perm_copy[first]
    sec_val = perm_copy[second]

    perm_copy[second] = first_val
    perm_copy[first] = sec_val

    return perm_copy

data = pd.read_csv('results.csv', index_col=0)

dates = [re.search(r'\[([^\]]*)\]', x).group(1) for x in data.columns]
movies = data.iloc[0].values
perms = data.values
all_scores = []
max_score = 0
best_positions = []

for movie in movies:
    best_positions.append([movie, get_optimal_position(data, movie)])

best_positions = sorted(best_positions, key=lambda x: x[1])

best_positions = [x[0] for x in best_positions]

max_score = get_score(data, best_positions)

score = max_score

perm = best_positions.copy()

print("Starting best")
print(perm)
print(score)

for i in tqdm(range(1000)):
    if np.argmin(score) > 0:
        perm = swap_values(perm, np.argmin(score) - 1, np.argmin(score))
        score = get_score(data, perm)

        if score > max_score:
            print("New best")
            print(perm)
            print(score)
            best_positions = perm
            max_score = score

    if np.argmin(score) < len(perm) - 1:
        perm = swap_values(perm, np.argmin(score) + 1, np.argmin(score))
        score = get_score(data, perm)

        if score > max_score:
            print("New best")
            print(perm)
            print(score)
            best_positions = perm
            max_score = score

optimal_order = pd.DataFrame({'date': dates, 'movie': best_positions})

optimal_order.to_csv('optimal_order.csv', index=False)
# for perm in perms:
#     scores = get_score(data, perm)
#     # print(perm)
#     # print(scores)

#     if np.sum(scores) > max_score:
#         max_score = np.sum(scores)

#     all_scores.append(scores)

# for ix, score in enumerate(all_scores):
#     if np.sum(score) == max_score:
#         # print(score)
#         # print(data.iloc[ix])
#         perm = data.iloc[ix].values
#         # print(score)
#         # print(perm)

#         ranking = sorted(range(len(score)), key=lambda k: score[k])
#         print(ranking)

#         swap_values(perm, np.argmin(score) - 1, np.argmin(score))

#         print(perm)
#         print(get_score(data, perm))

