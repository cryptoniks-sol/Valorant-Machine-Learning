"""
Diagnostic script to identify and fix training dataset issues.
Run this script to diagnose why build_training_dataset is returning 0 samples.
"""

import pickle
import json
import os
from collections import Counter
import difflib

def diagnose_cache_structure(cache_path="cache/valorant_data_cache.pkl"):
    """Diagnose the structure of your cached data."""
    
    print("🔍 DIAGNOSING CACHE STRUCTURE")
    print("="*50)
    
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Cache loaded successfully")
        print(f"Cache type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            
            # Check for enhanced cache structure
            if 'teams' in data:
                print("📊 Enhanced cache structure detected")
                teams_data = data['teams']
                print(f"Number of teams: {len(teams_data)}")
                
                # Sample team analysis
                sample_team_name = list(teams_data.keys())[0]
                sample_team_data = teams_data[sample_team_name]
                
                print(f"\n📋 Sample team analysis: {sample_team_name}")
                print(f"Team data keys: {list(sample_team_data.keys())}")
                
                if 'matches' in sample_team_data:
                    matches = sample_team_data['matches']
                    print(f"Number of matches: {len(matches)}")
                    
                    if matches:
                        sample_match = matches[0]
                        print(f"Sample match keys: {list(sample_match.keys())}")
                        print(f"Sample opponent: {sample_match.get('opponent_name', 'NOT FOUND')}")
                
                return teams_data
            
            else:
                print("📊 Direct team collection structure detected")
                print(f"Number of teams: {len(data)}")
                
                # Sample team analysis
                sample_team_name = list(data.keys())[0]
                sample_team_data = data[sample_team_name]
                
                print(f"\n📋 Sample team analysis: {sample_team_name}")
                print(f"Team data keys: {list(sample_team_data.keys())}")
                
                return data
        
        else:
            print(f"❌ Unexpected cache structure: {type(data)}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return None

def analyze_opponent_matching(team_data):
    """Analyze opponent matching issues."""
    
    print("\n🎯 ANALYZING OPPONENT MATCHING")
    print("="*50)
    
    all_team_names = set(team_data.keys())
    all_opponent_names = set()
    team_match_counts = {}
    opponent_match_attempts = {}
    
    print(f"Total teams in cache: {len(all_team_names)}")
    
    # Collect all opponent names and match counts
    for team_name, team_info in team_data.items():
        matches = []
        
        # Extract matches from different possible structures
        if isinstance(team_info, dict):
            if 'matches' in team_info:
                matches = team_info['matches']
            elif 'stats' in team_info and isinstance(team_info['stats'], dict):
                # Sometimes matches might be in stats
                matches = team_info['stats'].get('matches', [])
        
        team_match_counts[team_name] = len(matches)
        
        for match in matches:
            opponent_name = match.get('opponent_name', '')
            if opponent_name:
                all_opponent_names.add(opponent_name)
                opponent_match_attempts[opponent_name] = opponent_match_attempts.get(opponent_name, 0) + 1
    
    print(f"Teams with matches: {len([t for t, c in team_match_counts.items() if c > 0])}")
    print(f"Unique opponent names found: {len(all_opponent_names)}")
    print(f"Total match records: {sum(team_match_counts.values())}")
    
    # Find matching statistics
    exact_matches = 0
    partial_matches = 0
    no_matches = 0
    
    matching_analysis = {}
    
    for opponent_name in list(all_opponent_names)[:100]:  # Analyze first 100 opponents
        best_match = None
        best_score = 0
        
        # Try exact match
        if opponent_name in all_team_names:
            exact_matches += 1
            matching_analysis[opponent_name] = ('exact', opponent_name, 1.0)
            continue
        
        # Try fuzzy matching
        for team_name in all_team_names:
            score = difflib.SequenceMatcher(None, opponent_name.lower(), team_name.lower()).ratio()
            if score > best_score:
                best_match = team_name
                best_score = score
        
        if best_score > 0.6:
            partial_matches += 1
            matching_analysis[opponent_name] = ('fuzzy', best_match, best_score)
        else:
            no_matches += 1
            matching_analysis[opponent_name] = ('none', None, 0)
    
    print(f"\n📊 MATCHING STATISTICS (first 100 opponents):")
    print(f"Exact matches: {exact_matches}")
    print(f"Fuzzy matches (>60% similarity): {partial_matches}")
    print(f"No matches found: {no_matches}")
    
    # Show problematic opponents
    problematic_opponents = [
        (opp, analysis) for opp, analysis in matching_analysis.items() 
        if analysis[0] == 'none'
    ]
    
    if problematic_opponents:
        print(f"\n❌ PROBLEMATIC OPPONENTS (first 10):")
        for opponent, (match_type, match, score) in problematic_opponents[:10]:
            attempts = opponent_match_attempts.get(opponent, 0)
            print(f"  '{opponent}' (appears in {attempts} matches)")
    
    # Show successful fuzzy matches
    fuzzy_matches = [
        (opp, analysis) for opp, analysis in matching_analysis.items() 
        if analysis[0] == 'fuzzy'
    ]
    
    if fuzzy_matches:
        print(f"\n✅ SUCCESSFUL FUZZY MATCHES (first 5):")
        for opponent, (match_type, match, score) in fuzzy_matches[:5]:
            print(f"  '{opponent}' -> '{match}' (score: {score:.2f})")
    
    return matching_analysis

def suggest_fixes(team_data, matching_analysis):
    """Suggest specific fixes for the training dataset issue."""
    
    print("\n🔧 SUGGESTED FIXES")
    print("="*50)
    
    total_opponents = len(matching_analysis)
    exact_matches = len([a for a in matching_analysis.values() if a[0] == 'exact'])
    fuzzy_matches = len([a for a in matching_analysis.values() if a[0] == 'fuzzy'])
    no_matches = len([a for a in matching_analysis.values() if a[0] == 'none'])
    
    match_rate = (exact_matches + fuzzy_matches) / total_opponents if total_opponents > 0 else 0
    
    print(f"Current opponent matching rate: {match_rate:.1%}")
    
    if match_rate < 0.3:
        print("\n❌ CRITICAL ISSUE: Very low opponent matching rate")
        print("Recommended actions:")
        print("1. Use the enhanced build_training_dataset_enhanced function")
        print("2. Lower the similarity threshold to 0.4 or 0.3")
        print("3. Check if team names in cache match opponent names in matches")
        
    elif match_rate < 0.6:
        print("\n⚠️  MODERATE ISSUE: Moderate opponent matching rate")
        print("Recommended actions:")
        print("1. Use fuzzy matching with similarity threshold 0.5")
        print("2. Implement team name normalization")
        
    else:
        print("\n✅ GOOD: High opponent matching rate")
        print("The issue might be in feature preparation or data structure.")
    
    # Specific code recommendations
    print(f"\n📝 CODE MODIFICATIONS NEEDED:")
    print("="*40)
    
    print("1. Replace the build_training_dataset function with:")
    print("   build_training_dataset_enhanced (see provided artifact)")
    
    print("\n2. Update the training section in main() to:")
    print("""
    # Replace this line:
    X, y = build_training_dataset(team_data_collection)
    
    # With this:
    X, y = build_training_dataset_enhanced(team_data_collection, debug=True)
    """)
    
    print("\n3. If you're still getting 0 samples, try this debugging version:")
    print("""
    # Add this before building dataset:
    print("Debug: Team data structure check")
    sample_team = list(team_data_collection.keys())[0]
    print(f"Sample team: {sample_team}")
    print(f"Sample team data keys: {list(team_data_collection[sample_team].keys())}")
    
    if 'matches' in team_data_collection[sample_team]:
        matches = team_data_collection[sample_team]['matches']
        print(f"Sample matches count: {len(matches)}")
        if matches:
            print(f"Sample match: {matches[0]}")
    """)

def generate_fix_script():
    """Generate a ready-to-use fix script."""
    
    fix_script = '''
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
'''
    
    with open('training_fix.py', 'w') as f:
        f.write(fix_script)
    
    print(f"\n💾 Generated fix script: training_fix.py")
    print("You can copy the functions from this file into your main script.")

def main():
    """Main diagnostic function."""
    print("🚀 VALORANT TRAINING DATASET DIAGNOSTICS")
    print("="*60)
    
    # Step 1: Diagnose cache structure
    team_data = diagnose_cache_structure()
    
    if not team_data:
        print("❌ Cannot proceed without valid cache data")
        return
    
    # Step 2: Analyze opponent matching
    matching_analysis = analyze_opponent_matching(team_data)
    
    # Step 3: Suggest fixes
    suggest_fixes(team_data, matching_analysis)
    
    # Step 4: Generate fix script
    generate_fix_script()
    
    print(f"\n🎯 SUMMARY")
    print("="*30)
    print("1. Run this diagnostic to understand your data structure")
    print("2. Use the enhanced build_training_dataset_enhanced function")
    print("3. If still having issues, use the simple fix from training_fix.py")
    print("4. Monitor the debug output to see where samples are being lost")

if __name__ == "__main__":
    main()