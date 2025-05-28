
# QUICK FIX: Add this to your train.py file

def build_training_dataset_fixed(team_data_collection):
    """Fixed version of build_training_dataset with better opponent matching."""
    import difflib
    
    # Handle enhanced cache structure
    if isinstance(team_data_collection, dict) and 'teams' in team_data_collection:
        actual_teams = team_data_collection['teams']
    else:
        actual_teams = team_data_collection
    
    print(f"Building dataset from {len(actual_teams)} teams...")
    
    X, y = [], []
    successful_matches = 0
    
    for team_name, team_data in actual_teams.items():
        # Extract team stats and matches
        if 'stats' in team_data:
            team_stats = team_data['stats']
            matches = team_data.get('matches', [])
        else:
            team_stats = team_data
            matches = team_data.get('matches', [])
        
        for match in matches:
            opponent_name = match.get('opponent_name', '')
            if not opponent_name:
                continue
            
            # Find opponent with fuzzy matching
            best_match = None
            best_score = 0
            
            for potential_opponent in actual_teams.keys():
                score = difflib.SequenceMatcher(None, 
                    opponent_name.lower(), potential_opponent.lower()).ratio()
                if score > best_score:
                    best_match = potential_opponent
                    best_score = score
            
            # Use match if similarity > 50%
            if best_score > 0.5 and best_match in actual_teams:
                opponent_data = actual_teams[best_match]
                opponent_stats = opponent_data.get('stats', opponent_data)
                
                # Prepare features
                features = prepare_simple_features(team_stats, opponent_stats)
                if features:
                    X.append(features)
                    y.append(1 if match.get('team_won', False) else 0)
                    successful_matches += 1
    
    print(f"Created {successful_matches} training samples")
    return X, y

def prepare_simple_features(team1_stats, team2_stats):
    """Simple feature preparation that should always work."""
    try:
        features = {}
        
        # Basic features that should always be available
        features['win_rate_diff'] = team1_stats.get('win_rate', 0.5) - team2_stats.get('win_rate', 0.5)
        features['recent_form_diff'] = team1_stats.get('recent_form', 0.5) - team2_stats.get('recent_form', 0.5)
        
        # Match count features
        t1_matches = team1_stats.get('matches', 0)
        t2_matches = team2_stats.get('matches', 0)
        
        if isinstance(t1_matches, list):
            t1_count = len(t1_matches)
        else:
            t1_count = t1_matches if isinstance(t1_matches, (int, float)) else 10
            
        if isinstance(t2_matches, list):
            t2_count = len(t2_matches)
        else:
            t2_count = t2_matches if isinstance(t2_matches, (int, float)) else 10
        
        features['match_count_diff'] = t1_count - t2_count
        features['total_matches'] = t1_count + t2_count
        
        # Score features
        features['score_diff'] = team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)
        
        return features
    except Exception as e:
        print(f"Feature preparation error: {e}")
        return None

# TO USE: Replace your build_training_dataset call with build_training_dataset_fixed
