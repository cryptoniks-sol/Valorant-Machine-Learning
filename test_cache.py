import pickle
import os

def test_cache():
    cache_path = "cache/valorant_data_cache.pkl"
    if not os.path.exists(cache_path):
        print(f"Cache file not found at {cache_path}")
        return
        
    print(f"Loading cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
        
    print(f"Cache loaded! Type: {type(cache_data)}")
    print(f"Teams: {len(cache_data)}")
    
    # Print structure of first team
    if cache_data:
        first_team = next(iter(cache_data.keys()))
        print(f"\nFirst team: {first_team}")
        team_data = cache_data[first_team]
        print(f"Keys: {team_data.keys()}")
        
        if 'matches' in team_data:
            matches = team_data['matches']
            print(f"Matches: {len(matches)}")
            if matches:
                print(f"First match keys: {matches[0].keys()}")
                
        if 'stats' in team_data:
            print(f"Stats keys: {team_data['stats'].keys()}")
            
    # Count total matches
    total_matches = sum(len(team_data.get('matches', [])) for team_data in cache_data.values())
    print(f"\nTotal matches in cache: {total_matches}")

if __name__ == "__main__":
    test_cache()