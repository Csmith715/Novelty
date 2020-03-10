import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from statistics import mean

#cpc_entropy = pd.read_json('cpc_entropy.json')

def patent_entropy(cpc_list, year=2019):
    fn = 's3://aip-genome/entropy_files/ce_' + str(year) + '.json'
    cpc_entropy = pd.read_json(fn)
    pat_ents = []
    for c in cpc_list:
        if c not in cpc_entropy.entropies:
            p = math.log2(math.exp(1))
        else:
            p = cpc_entropy.entropies.loc[c]
        pat_ents.append(p)
    return(pat_ents)

def group_by_cpc(dataframe):
    # temp_df = dataframe[['patent_id', 'cpc_codes', 'priority_dt']].explode('cpc_codes')
    temp_df = dataframe[['patent_id', 'cpc_codes', 'priority_dt']].explode('cpc_codes').reset_index(drop=True).fillna("None")
    temp_df = temp_df[temp_df['cpc_codes'] != "None"]
    temp_df['cpc'] = [x.split('/')[0].replace(" ", "") for x in temp_df['cpc_codes']]
    temp_df = temp_df.drop('cpc_codes', axis=1)
    temp_df = temp_df.drop_duplicates()
    temp_df['priority_dt_month'] = pd.to_datetime(temp_df['priority_dt']).dt.to_period('M')
    temp_df['Quarter'] = pd.to_datetime(temp_df['priority_dt']).dt.to_period('Q')
    temp_df = temp_df.groupby(['patent_id', 'priority_dt_month', 'Quarter'])['cpc'].apply(list).reset_index(name='cpc_list')
    
    return temp_df

def get_score_from_bins(score=None, year=2019) -> int:
    fn = 'entropy_files/ce_' + str(year) + '.json'
    cpc_ent = pd.read_json(fn)
    d, bins = pd.qcut(cpc_ent.entropies, 5, retbins=True, labels=False)
    s = pd.cut([score], bins, labels=False, include_lowest=True)[0] + 1

    return(s)

def get_precedence_scores(dframe=None, for_year = 2019, seed_id=""):
    new_df = group_by_cpc(dframe)
    
    pat_seed_cpcs = new_df.cpc_list[new_df[new_df['patent_id'] == seed_id].index[0]]
    # Find the entropy of the technology by taking the average of all related patent entropies
    te = mean([mean(patent_entropy(x, for_year)) for x in new_df.cpc_list])
    # Return the individual entropy of the seed patent
    inde = mean(patent_entropy(pat_seed_cpcs))
    
    # Get binned scores
    tes = get_score_from_bins(te, for_year)
    indes = get_score_from_bins(inde, for_year)
    
    print('Patent Seed Entropy Precedence Score:  ', indes)
    print('Related Technology Entropy Precedence Score:  ', tes)

