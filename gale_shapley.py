import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

def gale_shapley_doctors_propose(doctor_prefs, hospital_prefs):
    unmatched_doctors = [d for d in doctor_prefs]
    doctor_current_match = {d: None for d in doctor_prefs}
    hospital_current_match = {h: None for h in hospital_prefs}

    hospital_rankings = {h: {d: rank for rank, d in enumerate(prefs)} for h, prefs in hospital_prefs.items()}
    proposals = {d: 0 for d in doctor_prefs}
    proposal_count = 0

    while unmatched_doctors:
        doctor = unmatched_doctors.pop(0)
        hospital = doctor_prefs[doctor][proposals[doctor]]
        current = hospital_current_match[hospital]
        proposals[doctor] += 1
        proposal_count += 1

        if current is None:
            hospital_current_match[hospital] = doctor
            doctor_current_match[doctor] = hospital
        else:
            if hospital_rankings[hospital][doctor] < hospital_rankings[hospital][current]:
                hospital_current_match[hospital] = doctor
                doctor_current_match[doctor] = hospital
                doctor_current_match[current] = None
                unmatched_doctors.append(current)
            else:
                unmatched_doctors.append(doctor)

    return doctor_current_match, proposal_count

def generate_uniform_preferences(n):
    doctors = [f"D{i + 1}" for i in range(n)]
    hospitals = [f"H{i + 1}" for i in range(n)]

    doctor_prefs = {d: np.random.choice(hospitals, size = n, replace = False).tolist() for d in doctors}
    hospital_prefs = {h: np.random.choice(doctors, size = n, replace = False).tolist() for h in hospitals}

    return doctor_prefs, hospital_prefs

def generate_weighted_preferences(n):
    doctors = [f"D{i + 1}" for i in range(n)]
    hospitals = [f"H{i + 1}" for i in range(n)]

    popularity_scores = [2 ** i for i in range(n)] # [i + 1 for i in range(n)]
    doctor_popularity = np.array(np.random.permutation(popularity_scores) / sum(popularity_scores), dtype = float)
    hospital_popularity = np.array(np.random.permutation(popularity_scores) / sum(popularity_scores), dtype = float)

    doctor_prefs = {d: np.random.choice(hospitals, size = n, replace = False, p = hospital_popularity).tolist() for d in doctors}
    hospital_prefs = {h: np.random.choice(doctors, size = n, replace = False, p = doctor_popularity).tolist() for h in hospitals}

    return doctor_prefs, hospital_prefs

def average_proposals_vs_n(generate_preferences):
    ns = range(100, 1100, 100)
    average_proposals = []

    for n in ns:
        total_proposals = 0
        
        for _ in range(5):
            doctor_prefs, hospital_prefs = generate_preferences(n)
            _, proposals = gale_shapley_doctors_propose(doctor_prefs, hospital_prefs)
            total_proposals += proposals

        average_proposals.append(total_proposals / 5)

    plt.figure(figsize = (7,5))
    plt.plot(ns, average_proposals, marker = 'o', color = 'blueviolet', alpha = 0.8)
    plt.title('Average Number of Proposals vs N', fontsize = 14, pad = 10)
    plt.xlabel('Number of Doctors/Hospitals', fontsize = 11)
    plt.ylabel('Average Number of Proposals', fontsize = 11)
    plt.xticks(ns)
    plt.tight_layout()
    plt.savefig(f"{generate_preferences.__name__.split("_")[1]}_average_proposals_vs_n.png")
    plt.close()

def proposal_distribution(generate_preferences):
    n = 100
    proposal_counts = []

    for _ in range(1000):
        doctor_prefs, hospital_prefs = generate_preferences(n)
        _, proposals = gale_shapley_doctors_propose(doctor_prefs, hospital_prefs)
        proposal_counts.append(proposals)

    counts, bins = np.histogram(proposal_counts, bins = 25)
    normalized_counts = counts / counts.sum() * 100

    plt.figure(figsize = (7,5))
    plt.bar(bins[:-1], normalized_counts, width = np.diff(bins), color = 'blueviolet', alpha = 0.8, edgecolor = 'black')    
    plt.title(f"Proposal Distribution for N = {n}", fontsize = 14, pad = 10)
    plt.xlabel("Total Proposals", fontsize = 11)
    plt.ylabel("Frequency (%)", fontsize = 11)
    plt.yticks(range(int(np.ceil(normalized_counts.max())) + 1), [f"{y:.1f}" for y in range(int(np.ceil(normalized_counts.max())) + 1)])
    plt.tight_layout()
    plt.savefig(f"{generate_preferences.__name__.split("_")[1]}_proposal_distribution.png")
    plt.close()

def average_partner_ranks(generate_preferences):
    ns = range(100, 1100, 100)
    doctor_ranks = []
    hospital_ranks = []

    for n in ns:
        d_rank_sum = 0
        h_rank_sum = 0

        for _ in range(5):
            doctor_prefs, hospital_prefs = generate_preferences(n)
            match, _ = gale_shapley_doctors_propose(doctor_prefs, hospital_prefs)

            for d, h in match.items():
                d_rank_sum += doctor_prefs[d].index(h) + 1
                h_rank_sum += hospital_prefs[h].index(d) + 1

        doctor_ranks.append(d_rank_sum / (n * 5))
        hospital_ranks.append(h_rank_sum / (n * 5))

    plt.figure(figsize = (7,5))
    plt.plot(ns, doctor_ranks, label = "Doctors Average Rank", marker = 'o', color = 'blue', alpha = 0.5)
    plt.plot(ns, hospital_ranks, label = "Hospitals Average Rank", marker = 'x', color = 'red', alpha = 0.5)
    plt.title("Average Partner Ranks vs N", fontsize = 14, pad = 10)
    plt.xlabel("Number of Doctors/Hospitals", fontsize = 11)
    plt.ylabel("Average Rank of Matched Partner", fontsize = 11)
    plt.xticks(ns)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{generate_preferences.__name__.split("_")[1]}_average_partner_ranks_vs_n.png")
    plt.close()

def rank_distribution(generate_preferences):
    n = 100
    d_rank_freq = [0] * n
    h_rank_freq = [0] * n

    for _ in range(1000):
        doctor_prefs, hospital_prefs = generate_preferences(n)
        match, _ = gale_shapley_doctors_propose(doctor_prefs, hospital_prefs)

        for d, h in match.items():
            d_rank_freq[doctor_prefs[d].index(h)] += 1
            h_rank_freq[hospital_prefs[h].index(d)] += 1

    plt.figure(figsize = (7,5))
    plt.bar(range(1, n + 1), [x / 1000 for x in d_rank_freq], color = 'blue', alpha = 0.5, label = "Doctors")
    plt.bar(range(1, n + 1), [x / 1000 for x in h_rank_freq], color = 'red', alpha = 0.5, label = "Hospitals")
    plt.title(f"Partner Rank Distribution for N = {n}", fontsize = 14, pad = 10)
    plt.xlabel("Average Rank", fontsize = 11)
    plt.ylabel("Frequency (%)", fontsize = 11)
    plt.xticks(range(0, n + 10, 10))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{generate_preferences.__name__.split("_")[1]}_rank_distribution.png")
    plt.close()

uniform = generate_uniform_preferences
weighted = generate_weighted_preferences

for preferences in [uniform, weighted]:
    average_proposals_vs_n(preferences)
    proposal_distribution(preferences)
    average_partner_ranks(preferences)
    rank_distribution(preferences)
