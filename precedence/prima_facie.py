import sys, os, inspect
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import numpy as np
import re
import sqlalchemy
import psycopg2 as pg
import json
import segmented_regression as sr

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
USERNAME=os.getenv("PATENT_DB_USERNAME")
PASSWORD=os.getenv("PATENT_DB_PASSWORD")

def make_connectstring():
    """return an sql connectstring"""
    hostname = "genome.cjbyhapmrpxu.us-east-2.rds.amazonaws.com"
    port = "5354"
    db = "patent"
    uname=USERNAME
    password=PASSWORD
    prefix = 'postgresql'
    return f'{prefix}://{uname}:{password}@{hostname}:{port}/{db}'

def get_patent_info(sql="", return_df=False):
    connectstring = make_connectstring()
    engine = sqlalchemy.create_engine(connectstring, pool_size=10)
    df = pd.read_sql_query(sql, con=engine)
    return df

def get_family_patents(family_id):

    sql = f"""
        SELECT p.patent_id, 
            family_id,
            pa.name, 
            min(pc.priority_dt) as priority_dt,
            ARRAY_AGG(DISTINCT cpc.code) as cpc_codes
        FROM ifi.patent p
        left JOIN ifi.patentcpc cpc ON cpc.patent_pk = p.patent_pk
        left JOIN ifi.patentassignee pa ON pa.patent_pk = p.patent_pk
        left JOIN ifi.priorityclaims pc ON pc.patent_pk = p.patent_pk
        WHERE 
            p.family_id = '{family_id}'
        GROUP BY 
            1,2,3;
        """
    return get_patent_info(sql, return_df=True)

def get_patent(patent_ids, single=False):
    if single:
        operator = f"= '{patent_ids}'"
    else:
        operator = f"in {patent_ids}"
    sql = f"""
        SELECT p.patent_id, 
            family_id,
            pa.name, 
            min(pc.priority_dt) as priority_dt,
            ARRAY_AGG(DISTINCT cpc.code) as cpc_codes
        FROM ifi.patent p
        left JOIN ifi.patentcpc cpc ON cpc.patent_pk = p.patent_pk
        left JOIN ifi.patentassignee pa ON pa.patent_pk = p.patent_pk
        left JOIN ifi.priorityclaims pc ON pc.patent_pk = p.patent_pk
        WHERE 
            p.patent_id {operator}
        GROUP BY 
            1,2,3;
        """
    return get_patent_info(sql, return_df=True)

def get_sim_patents(seed_pat_id="",
                sim_pats_file = "",
                get_similar=False):

    if get_similar:
        from similarity_console import sc
        sc(source_type="patent", source_ids=seed_pat_id, file_path=sim_pats_file)

    # Get info on similar patents
    sim_pats_df = pd.read_csv(sim_pats_file).dropna()

    try:
        patent_ids = tuple(sim_pats_df['source_id'].tolist())
        sim_pats = get_patent(patent_ids=patent_ids, single=False)
        result = pd.merge(sim_pats, sim_pats_df, left_on="patent_id", right_on="source_id").drop("source_id", axis=1)
    except:
        patent_ids = tuple(sim_pats_df['patent_id'].tolist())
        sim_pats = get_patent(patent_ids=patent_ids, single=False)
        result = pd.merge(sim_pats, sim_pats_df, on="patent_id")

    # remove duplicate patent_ids to prevent over counting
    result.drop_duplicates("patent_id", keep="first", inplace=True)

    # remove same family as input seed
    family_id = result['family_id'][result['patent_id']==seed_pat_id]

    if family_id.shape[0] > 0:
        fid = family_id.iloc[0]
        rows_to_drop_famiy = result[(result['family_id']==fid)&(result['patent_id']!=seed_pat_id)].index.tolist()
        result = result.drop(rows_to_drop_famiy)

    # remove rows with None cpc_codes
    rows_to_drop_cpc = result[result['cpc_codes'].map(lambda d: None in d)].index.tolist()
    result = result.drop(rows_to_drop_cpc)

    result['priority_yr'] = [i.year for i in result['priority_dt']]
    result['cos_sim'] = 1 - result['score']
    result = result.fillna(0)
    result = result[result['name']!=0]
    result['cpc_codes'] = [[c.split('/')[0].replace(" ", "") for c in x] for x in result['cpc_codes']]

    return result

# this method is slightly faster than nltk's ngrams method
def ngrams(text, n=2):
    return zip(*[text[i:] for i in range(n)])

def cpc_parts_count(cpcs):
    sections = []
    classes = []
    subclasses = []
    groups = []
    if None in cpcs:
        cpcs.remove(None)
    for cpc in cpcs:
        sections.append(cpc[:1])
        classes.append(cpc[:3])
        subclasses.append(cpc[:4])
        groups.append(cpc)
    return len(list(set(sections))), len(list(set(classes))), len(list(set(subclasses))), len(list(set(groups))), len(list(groups))
    
def convert(row):
    cpc_features = cpc_parts_count(row['cpc_codes'])
    row['unique_cpc_sections'] = cpc_features[0]
    row['unique_cpc_classes'] = cpc_features[1]
    row['unique_cpc_subclasses'] = cpc_features[2]
    row['unique_cpc_groups'] = cpc_features[3]
    row['all_cpc_groups'] = cpc_features[4]
    row['cpc_diversity'] = row['unique_cpc_groups']/row['all_cpc_groups'] 
    return row

def contains(l,q):
    if q in l:
        return True
    else:
        return False

def col_norm(df, col):
    return (df[col]-df[col].min())/(df[col].max()-df[col].min())

def agg_yearly(df, seed_patent):
    """
    Input df should already be cleaned and reduced
    """

    # Get the priority year of the seed patent
    try:
        seed_year = df['priority_dt'][df['patent_id'] == seed_patent].iloc[0].year
    except:
        seed = get_patent(patent_ids=seed_patent, single=True)
        seed_year = seed['priority_dt'][0].year
    
    # Get the cpc's of the seed patent
    try:
        seed_cpcs = df['cpc_codes'][df['patent_id'] == seed_patent].tolist()[0]
    except:
        seed = get_patent(patent_ids=seed_patent, single=True)
        seed_cpcs = seed['cpc_codes'].tolist()[0]

    min_year = seed_year - 3
        
    # get the counts of applicants
    applicant_count = df[df['priority_yr']>min_year].groupby(['priority_yr'])["name"].nunique().to_frame()

    # get the counts of applications
    application_count = df[df['priority_yr']>min_year].groupby(['priority_yr'])["patent_id"].count().to_frame()

    # create a new dataframe with counts per year
    data = pd.merge(application_count,\
                    applicant_count,\
                    left_index=True,\
                    right_index=True,\
                    copy=False).reset_index()

    data.columns = ['priority_yr','applications','applicants']

    data = data.sort_values(by="priority_yr").reset_index(drop=True)
    data['delta_applications'] = data['applications'].diff().fillna(0)
    temp = pd.DataFrame(df.groupby(['priority_yr'])['cos_sim'].agg("mean"))
    data = pd.merge(data, temp, on="priority_yr")
    data.loc[data['priority_yr'] > seed_year, 'priority'] = 'after'
    data.loc[data['priority_yr'] <= seed_year, 'priority'] = 'before'

    agg_cpcs_yr = df[df['priority_yr']>min_year].groupby(['priority_yr'])["cpc_codes"].agg("sum").to_frame().reset_index()
    agg_data = agg_cpcs_yr.apply(lambda r: convert(r), axis=1)
    data = pd.merge(data, agg_data, on='priority_yr', copy=False)
    data = data.sort_values(by="priority_yr").reset_index(drop=True)
    data['applications_cumsum'] = data['applications'].cumsum().fillna(0)
    data['applicants_cumsum'] = data['applicants'].cumsum().fillna(0)
    data['adoption_quartiles'] = pd.qcut(data['applications_cumsum'], 4, labels=False)
    data['lct'] = data['delta_applications']/data['applications'].fillna(0)

    # Novelty in patent trajectory and cpc concentration
    all_years = data['priority_yr'].unique()

    # expand: make a wide dataframe with each cpc as a new column
    expanded_cpc = df['cpc_codes'][df['priority_yr']>min_year].apply(pd.Series) \
            .add_prefix("cpc_") \
            .merge(df[['name','priority_yr']], right_index = True, left_index = True) \
            .fillna(np.nan)
    value_vars = [i for i in expanded_cpc.columns if i != 'name' if i != 'priority_yr']

    # contract: make a long dataframe 
    melted = pd.melt(expanded_cpc,
        id_vars = ['name','priority_yr'],
        value_vars = value_vars,
        var_name='application_count')

    # if any values are NaN, drop it because it messes with the count
    melted = melted.dropna()

    melted = melted.groupby(['priority_yr','value','name'])['application_count'].count().to_frame().reset_index(drop=False)
    melted['section'] = melted.apply(lambda r: r['value'][0:1], axis=1)

    # condense: group by year and get counts
    section_df = melted.groupby(['priority_yr','section'])[['application_count','name']].agg(['sum','nunique']).reset_index(drop=False)
    section_df.columns = [''.join(col).strip() for col in section_df.columns.values]
    section_df.drop(['namesum','application_countnunique'], inplace=True, axis=1)
    section_df.columns = ['year','section','application_count','applicant_count']

    for year in all_years:
        section_max = section_df['application_count'][section_df['year']==year].max()
        section_total = section_df['application_count'][section_df['year']==year].sum()
        data.loc[data['priority_yr']==year,"section_concentration"] = section_max/section_total

    data.fillna(0, inplace=True)
    
    return data, section_df, seed_year, seed_cpcs

def get_indicators(sim_df=None, seed_patent=""):
    df = sim_df.copy()
    
    # get the top 1000 most similar patents
    if df.shape[0] > 1000:
        df = df.sort_values('cos_sim', ascending=False)[0:1001]
    else:
        df = df.sort_values('cos_sim', ascending=False)

    # Just in case
    # df = df[df['cos_sim'] >= 0.80] 

    # Get the aggregated data per year
    data, section_df, seed_year, seed_cpcs = agg_yearly(df, seed_patent)

    # normalize the columns we expect to use
    data["norm_cpc_diversity"] = col_norm(data, "cpc_diversity")

    # cpc diversity and concentration are based on a window of time around the priority year
    indicators = {}
    # initialize values with worse possible case
    indicators['cpc_diversity'] = 0
    indicators['section_concentration'] = 0
    indicators['recombination'] = 0
    indicators['adoption'] = 0
    indicators['maturity'] = 0

    all_years = data['priority_yr'].unique()
    x = data['norm_cpc_diversity'].values
    labels = sr.label_status(x)
    seg_list, idx_list = sr.bottom_up(x, labels, 0.03, k=1, step=4)
    segments = sr.combine_segment(x, labels, seg_list, idx_list)

    for seg in segments:
        if all_years[seg[0]] < seed_year < all_years[seg[1]-1]:
            if seg[2] == sr.DEC:
                indicators['cpc_diversity'] = 1
            temp = data[(data["priority_yr"]>=all_years[seg[0]])&(data["priority_yr"]<=all_years[seg[1]-1])].reset_index(drop=True)
            indicators['section_concentration'] = np.where(any(temp["section_concentration"] > 0.5), 1, 0).item()
   
    indicators['recombination'] = get_recombination_score(df[df['priority_yr']<=seed_year], seed_cpcs)

    # position along the adoption s-curve
    indicators['adoption'] = 1/(data['adoption_quartiles'][data['priority_yr']==seed_year].iloc[0] + 1)
    indicators['maturity'] = np.where(all(data["lct"][data['priority_yr']<=seed_year] >= 0), 1, 0).item()

    return indicators, data, section_df, seed_year

def get_recombination_score(sim_df, seed_cpcs):
    """
    Compares pairwise combinations of seed_cpcs to cpcs in the given dataframe

    """
    result = 1
    df = sim_df.copy()
  
    # Novelty in Recombinations 
    if len(seed_cpcs) > 1:
        # get all possible pair combinations
        seed_list = list((x, y) for x in seed_cpcs for y in seed_cpcs if y > x)
        # combine 
        for pat in df['patent_id'].unique().tolist():
            myList = []
            for cpc_list in df['cpc_codes'][df['patent_id']==pat]:
                if None not in cpc_list:
                    [myList.append(x) for x in cpc_list if x not in myList]
            unorderedPairs = list((x, y) for x in myList for y in myList if y > x)
            if len([i for i in seed_list if i not in unorderedPairs]) > 1:
                df.loc[df['patent_id']==pat, 'nr'] = 1
            else:
                df.loc[df['patent_id']==pat, 'nr'] = 0
    else:
        for pat in df['patent_id'].unique().tolist():
            myList = []
            for cpc_list in df['cpc_codes'][df['patent_id']==pat]:
                if None not in cpc_list:
                    [myList.append(x) for x in cpc_list if x not in myList]
            if len([i for i in seed_cpcs if i not in myList]) > 1:
                df.loc[df['patent_id']==pat, 'nr'] = 1
            else:
                df.loc[df['patent_id']==pat, 'nr'] = 0

    if 0 in df['nr'].tolist():
        result = 0

    return result

def get_forward_citations(seed_pat_id = "US-6619835-B2", get_counts_only=False):
    if get_counts_only:
        sql = f"""
            select count(patent_pk)
            from ifi.citations c
            where c.patent_id = '{seed_pat_id}'
            """
        counts_df = get_patent_info(sql, return_df=True)
        return counts_df['count'].values[0]
    
    else:
        sql = f"""
            select distinct(patent_pk)
            from ifi.citations c
            where c.patent_id = '{seed_pat_id}'
            """
        citing_patent_pks = get_patent_info(sql, return_df=True)

        citing_patent_pks = tuple(citing_patent_pks['patent_pk'].tolist())

        sql = f"""
            SELECT p.patent_id, 
                p.patent_pk,
                p.country, 
                p.publication_dt,
                ARRAY_AGG(DISTINCT cpc.code) as cpc_codes
            FROM ifi.patent p
            left JOIN ifi.patentcpc cpc ON p.patent_pk = cpc.patent_pk
            WHERE 
                p.patent_pk in {citing_patent_pks}
            GROUP BY
                p.patent_pk
            """
        
        citing_pats_df = get_patent_info(sql, return_df=True)
        citing_pats_df['pub_yr'] = [i.year for i in citing_pats_df['publication_dt']]

        citing_pats_df.sort_values('pub_yr', inplace=True)

        return citing_pats_df, citing_patent_pks

def get_backward_citations(seed_pat_id = "US-6619835-B2"):
    # get the patent_pk of the seed patent
    sql = f"""
        select p.patent_pk
        from ifi.patent p
        where p.patent_id = '{seed_pat_id}'
        """
    temp = get_patent_info(sql, return_df=True)
    seed_pat_pk = temp['patent_pk'].values[0]

    # use the seed patent_pk to get the cited patents from citations table
    sql = f"""
        select c.patent_id
        from ifi.citations c
        where c.patent_pk = {seed_pat_pk}
        """
    temp = get_patent_info(sql, return_df=True)

    cited_patent_ids = tuple(temp['patent_id'].tolist())

    if len(cited_patent_ids) == 0:
        print("No backward citations found")
        return
    elif len(cited_patent_ids) == 1:
        operator = f"= '{cited_patent_ids}'"
    else:
        operator = f"in {cited_patent_ids}"

    # get the cpc's of the cited patents
    sql = f"""
        SELECT p.patent_id, 
            ARRAY_AGG(DISTINCT cpc.code) as cpc_codes
        FROM ifi.patent p
        left JOIN ifi.patentcpc cpc ON p.patent_pk = cpc.patent_pk
        WHERE 
            p.patent_id {operator}
        GROUP BY
            p.patent_id
        """

    cited_pats_df = get_patent_info(sql, return_df=True)
  
    return cited_pats_df, cited_patent_ids