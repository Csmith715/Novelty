import argparse
from precedence import prima_facie
import json
parser = argparse.ArgumentParser()
parser.add_argument('--ids', required=True, help="Source IDs")
parser.add_argument('--download', required=False, help="Use similarity console to get similar patents", action="store_true")
parser.add_argument('--file', required=True, help="Local file path for similar patents")
    
if __name__ == "__main__":
    """
    Example usage:
    $python3 run.py --ids "US-6619835-B2" --file ~/Documents/chrism/cpc_patent_vectors/data/new_casio_wearable_sim_pats.zip
    """
    args = parser.parse_args()

    result = prima_facie.get_sim_patents(seed_pat_id = args.ids,
                                        sim_pats_file = args.file,
                                        get_similar=args.download)

    indicators, data, section_df, seed_year = prima_facie.get_indicators(result, args.ids)

    print({"score": sum(indicators.values())})