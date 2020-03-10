Install from the repo directory using:

```
$pip install -e .
```



Example usage for the entropy precedence score:

```
$python3 run_entropy.py --ids "US-6619835-B2" --file ~/Documents/chrism/cpc_patent_vectors/data/new_casio_wearable_sim_pats.zip --year 2019
```

After running the entropy precedence score, two scores will be returned. Similar to this:

Patent Seed Entropy Precedence Score:   5
Related Technology Entropy Precedence Score:   5

A scale of 1 to 5 has been applied to both scores. The higher the score, the more unique the patent and/or technology is deemed to be.

The entropy precedence score up to a specific year can also be determined by changing the default year in the example above. Please note that this range must be between 1991 and 2019.
