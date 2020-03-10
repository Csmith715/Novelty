import argparse
from precedence import prima_facie
from precedence import precedence_entropy
import json
parser = argparse.ArgumentParser()
parser.add_argument('--ids', required=True, help="Source IDs")
parser.add_argument('--download', required=False, help="Use similarity console to get similar patents", action="store_true")
parser.add_argument('--file', required=True, help="Local file path for similar patents")
parser.add_argument('--year', required=False, help="Desired year of returned entropy score")
    
if __name__ == "__main__":
    """
    Example usage:
    $python3 run_entropy.py --ids "US-6619835-B2" --file ~/Documents/chrism/cpc_patent_vectors/data/new_casio_wearable_sim_pats.zip
    """
    args = parser.parse_args()

    result, priority_year = prima_facie.get_patents(seed_pat_id = args.ids,
                                        sim_pats_file = args.file,
                                        get_similar=args.download)

    precedence.precedence_entropy.get_precedence_scores(result, year, seed_id = args.ids)
    #indicators, _ = prima_facie.get_indicators(result, priority_year)

    #print({"score": 1 + sum(indicators.values())})
