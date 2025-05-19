#!/usr/bin/env python3
# symmetry_test.py - Standalone test for feature symmetry

import numpy as np

def create_test_team_stats():
    """Create test team statistics for symmetry testing."""
    team1_stats = {
        'team_name': 'Team A',
        'team_id': '123',
        'win_rate': 0.75,
        'recent_form': 0.80,
        'matches': 50,
        'wins': 35,
        'losses': 15,
        'avg_score': 13.2,
        'avg_opponent_score': 8.7,
        'score_differential': 4.5,
        'avg_player_rating': 1.28,
        'avg_player_acs': 245.0,
        'avg_player_kd': 1.35,
        'avg_player_kast': 0.72,
        'avg_player_adr': 165.0,
        'avg_player_headshot': 0.25,
        'star_player_rating': 1.45,
        'team_consistency': 0.85,
        'fk_fd_ratio': 1.2,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'economy_efficiency': 0.75,
        'recency_weighted_win_rate': 0.70,
        'opponent_stats': {
            'Team B': {
                'matches': 5,
                'wins': 4,
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.15
            }
        },
        'map_statistics': {
            'Haven': {'win_percentage': 0.70, 'matches_played': 12},
            'Ascent': {'win_percentage': 0.65, 'matches_played': 10},
            'Bind': {'win_percentage': 0.60, 'matches_played': 8}
        }
    }
    
    team2_stats = {
        'team_name': 'Team B',
        'team_id': '456',
        'win_rate': 0.60,
        'recent_form': 0.65,
        'matches': 40,
        'wins': 24,
        'losses': 16,
        'avg_score': 11.8,
        'avg_opponent_score': 9.5,
        'score_differential': 2.3,
        'avg_player_rating': 1.18,
        'avg_player_acs': 220.0,
        'avg_player_kd': 1.15,
        'avg_player_kast': 0.68,
        'avg_player_adr': 155.0,
        'avg_player_headshot': 0.22,
        'star_player_rating': 1.32,
        'team_consistency': 0.80,
        'fk_fd_ratio': 1.1,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'economy_efficiency': 0.68,
        'recency_weighted_win_rate': 0.55,
        'opponent_stats': {
            'Team A': {
                'matches': 5,
                'wins': 1,
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        },
        'performance_trends': {
            'form_trajectory': {
                '5_vs_10': 0.05
            }
        },
        'map_statistics': {
            'Haven': {'win_percentage': 0.60, 'matches_played': 10},
            'Ascent': {'win_percentage': 0.55, 'matches_played': 9},
            'Bind': {'win_percentage': 0.50, 'matches_played': 6}
        }
    }
    
    return team1_stats, team2_stats

def prepare_data_for_model(team1_stats, team2_stats):
    """
    FIXED version to ensure symmetrical features for model prediction.
    """
    if not team1_stats or not team2_stats:
        print("Missing team statistics data")
        return None
    
    features = {}
    
    #----------------------------------------
    # 1. DIFFERENCE FEATURES (sign changes when teams swap)
    #----------------------------------------
    features['win_rate_diff'] = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    features['recent_form_diff'] = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
    features['score_diff_differential'] = team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)
    features['wins_diff'] = team1_stats.get('wins', 0) - team2_stats.get('wins', 0)
    features['losses_diff'] = team1_stats.get('losses', 0) - team2_stats.get('losses', 0)
    features['avg_score_diff'] = team1_stats.get('avg_score', 0) - team2_stats.get('avg_score', 0)
    features['avg_opponent_score_diff'] = team1_stats.get('avg_opponent_score', 0) - team2_stats.get('avg_opponent_score', 0)
    features['match_count_diff'] = team1_stats.get('matches', 0) - team2_stats.get('matches', 0)
    features['recency_weighted_win_rate_diff'] = team1_stats.get('recency_weighted_win_rate', 0) - team2_stats.get('recency_weighted_win_rate', 0)
    
    # Win-loss ratio difference
    team1_wl_ratio = team1_stats.get('wins', 0) / max(team1_stats.get('losses', 1), 1)
    team2_wl_ratio = team2_stats.get('wins', 0) / max(team2_stats.get('losses', 1), 1)
    features['win_loss_ratio_diff'] = team1_wl_ratio - team2_wl_ratio
    
    # Map win rate difference
    if 'map_statistics' in team1_stats and 'map_statistics' in team2_stats:
        team1_maps = set(team1_stats['map_statistics'].keys())
        team2_maps = set(team2_stats['map_statistics'].keys())
        common_maps = team1_maps.intersection(team2_maps)
        
        if common_maps:
            team1_map_winrates = [team1_stats['map_statistics'][m].get('win_percentage', 0) for m in common_maps]
            team2_map_winrates = [team2_stats['map_statistics'][m].get('win_percentage', 0) for m in common_maps]
            
            if team1_map_winrates and team2_map_winrates:
                features['avg_map_win_rate_diff'] = (sum(team1_map_winrates) / len(team1_map_winrates)) - (sum(team2_map_winrates) / len(team2_map_winrates))
                
                # Count maps where team1 has advantage - this is a BINARY feature
                maps_advantage_count = sum(1 for i in range(len(team1_map_winrates)) if team1_map_winrates[i] > team2_map_winrates[i])
                features['maps_advantage_team1'] = int(maps_advantage_count / len(common_maps) > 0.5)
                
                # Best map performance difference
                features['best_map_diff'] = max(team1_map_winrates) - max(team2_map_winrates)
    
    #----------------------------------------
    # 2. PLAYER STATS DIFFERENCES (sign changes when teams swap)
    #----------------------------------------
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        features['player_rating_diff'] = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
        features['acs_diff'] = team1_stats.get('avg_player_acs', 0) - team2_stats.get('avg_player_acs', 0)
        features['kd_diff'] = team1_stats.get('avg_player_kd', 0) - team2_stats.get('avg_player_kd', 0)
        features['kast_diff'] = team1_stats.get('avg_player_kast', 0) - team2_stats.get('avg_player_kast', 0)
        features['adr_diff'] = team1_stats.get('avg_player_adr', 0) - team2_stats.get('avg_player_adr', 0)
        features['headshot_diff'] = team1_stats.get('avg_player_headshot', 0) - team2_stats.get('avg_player_headshot', 0)
        features['star_player_diff'] = team1_stats.get('star_player_rating', 0) - team2_stats.get('star_player_rating', 0)
        features['team_consistency_diff'] = team1_stats.get('team_consistency', 0) - team2_stats.get('team_consistency', 0)
        features['fk_fd_diff'] = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
    
    #----------------------------------------
    # 3. ECONOMY STATS DIFFERENCES (sign changes when teams swap)
    #----------------------------------------
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        features['pistol_win_rate_diff'] = team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)
        
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            features['eco_win_rate_diff'] = team1_stats.get('eco_win_rate', 0) - team2_stats.get('eco_win_rate', 0)
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            features['full_buy_win_rate_diff'] = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
        
        if 'economy_efficiency' in team1_stats and 'economy_efficiency' in team2_stats:
            features['economy_efficiency_diff'] = team1_stats.get('economy_efficiency', 0) - team2_stats.get('economy_efficiency', 0)
    
    # Performance trends
    if 'performance_trends' in team1_stats and 'performance_trends' in team2_stats:
        if 'form_trajectory' in team1_stats['performance_trends'] and 'form_trajectory' in team2_stats['performance_trends']:
            team1_trajectory = team1_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            team2_trajectory = team2_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            features['recent_form_trajectory_diff'] = team1_trajectory - team2_trajectory
    
    #----------------------------------------
    # 4. BINARY FEATURES (value flips when teams swap: 0->1, 1->0)
    #----------------------------------------
    # Always use explicit integer conversion for all binary features
    features['better_win_rate_team1'] = int(team1_stats.get('win_rate', 0) > team2_stats.get('win_rate', 0))
    features['better_recent_form_team1'] = int(team1_stats.get('recent_form', 0) > team2_stats.get('recent_form', 0))
    features['better_score_diff_team1'] = int(team1_stats.get('score_differential', 0) > team2_stats.get('score_differential', 0))
    features['better_avg_score_team1'] = int(team1_stats.get('avg_score', 0) > team2_stats.get('avg_score', 0))
    features['better_defense_team1'] = int(team1_stats.get('avg_opponent_score', 0) < team2_stats.get('avg_opponent_score', 0))
    features['better_player_rating_team1'] = int(team1_stats.get('avg_player_rating', 0) > team2_stats.get('avg_player_rating', 0))
    features['better_pistol_team1'] = int(team1_stats.get('pistol_win_rate', 0) > team2_stats.get('pistol_win_rate', 0))
    
    if 'performance_trends' in team1_stats and 'performance_trends' in team2_stats:
        if 'form_trajectory' in team1_stats['performance_trends'] and 'form_trajectory' in team2_stats['performance_trends']:
            team1_trajectory = team1_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            team2_trajectory = team2_stats['performance_trends']['form_trajectory'].get('5_vs_10', 0)
            features['better_trajectory_team1'] = int(team1_trajectory > team2_trajectory)
    
    #----------------------------------------
    # 5. H2H (HEAD-TO-HEAD) FEATURES
    #----------------------------------------
    team1_name = team1_stats.get('team_name', 'Team1')
    team2_name = team2_stats.get('team_name', 'Team2') 
    
    # Initialize H2H features with default values
    features['h2h_win_rate'] = 0.5  # Default to even matchup
    features['h2h_matches'] = 0     # Default to no matches
    features['h2h_score_diff'] = 0  # Default to no score difference
    features['h2h_advantage_team1'] = 0  # Default to no advantage
    features['h2h_significant'] = 0  # Default to not significant
    features['h2h_recency'] = 0.5    # Default to moderate recency
    
    # Search for head-to-head data
    h2h_found = False
    
    # Search team1 vs team2
    if 'opponent_stats' in team1_stats and isinstance(team1_stats['opponent_stats'], dict):
        # Try to find team2 in team1's opponent stats
        if team2_name in team1_stats['opponent_stats']:
            h2h_data = team1_stats['opponent_stats'][team2_name]
            h2h_found = True
        else:
            # Try partial matching
            team2_tag = team2_stats.get('team_tag', '')
            for opponent_name, stats in team1_stats['opponent_stats'].items():
                if team2_name.lower() in opponent_name.lower() or \
                   (team2_tag and team2_tag.lower() in opponent_name.lower()):
                    h2h_data = stats
                    h2h_found = True
                    break
                    
        if h2h_found:
            # Team1 perspective - use directly
            features['h2h_win_rate'] = h2h_data.get('win_rate', 0.5)
            features['h2h_matches'] = h2h_data.get('matches', 0)
            features['h2h_score_diff'] = h2h_data.get('score_differential', 0)
            features['h2h_advantage_team1'] = int(h2h_data.get('win_rate', 0.5) > 0.5)
            features['h2h_significant'] = int(h2h_data.get('matches', 0) >= 3)
            features['h2h_recency'] = 0.8  # High weight for actual data
    
    # If not found, search team2 vs team1 (reverse direction)
    if not h2h_found and 'opponent_stats' in team2_stats and isinstance(team2_stats['opponent_stats'], dict):
        # Try to find team1 in team2's opponent stats
        if team1_name in team2_stats['opponent_stats']:
            h2h_data = team2_stats['opponent_stats'][team1_name]
            h2h_found = True
        else:
            # Try partial matching
            team1_tag = team1_stats.get('team_tag', '')
            for opponent_name, stats in team2_stats['opponent_stats'].items():
                if team1_name.lower() in opponent_name.lower() or \
                   (team1_tag and team1_tag.lower() in opponent_name.lower()):
                    h2h_data = stats
                    h2h_found = True
                    break
                    
        if h2h_found:
            # Team2 perspective - invert values for team1 perspective
            features['h2h_win_rate'] = 1 - h2h_data.get('win_rate', 0.5)  # Invert win rate
            features['h2h_matches'] = h2h_data.get('matches', 0)  # Matches count stays the same
            features['h2h_score_diff'] = -h2h_data.get('score_differential', 0)  # Negate score diff
            features['h2h_advantage_team1'] = int(1 - h2h_data.get('win_rate', 0.5) > 0.5)  # Invert advantage
            features['h2h_significant'] = int(h2h_data.get('matches', 0) >= 3)  # Significance stays the same
            features['h2h_recency'] = 0.8  # High weight for actual data
    
    # If no head-to-head data found, estimate from team strengths
    if not h2h_found:
        # Calculate estimated h2h using both win rates
        team1_win_rate = team1_stats.get('win_rate', 0.5)
        team2_win_rate = team2_stats.get('win_rate', 0.5)
        
        if team1_win_rate + team2_win_rate > 0:
            estimated_h2h = team1_win_rate / (team1_win_rate + team2_win_rate)
        else:
            estimated_h2h = 0.5
            
        # Scale towards 0.5 to reduce extremes
        estimated_h2h = 0.5 + (estimated_h2h - 0.5) * 0.6
        
        features['h2h_win_rate'] = estimated_h2h
        features['h2h_matches'] = 0
        features['h2h_score_diff'] = (team1_stats.get('score_differential', 0) - team2_stats.get('score_differential', 0)) * 0.5
        features['h2h_advantage_team1'] = int(estimated_h2h > 0.5)
        features['h2h_significant'] = 0
        features['h2h_recency'] = 0.2
    
    #----------------------------------------
    # 6. SYMMETRIC AVERAGE FEATURES (value is the same when teams swap)
    #----------------------------------------
    # These should remain identical regardless of team ordering
    features['avg_win_rate'] = (team1_stats.get('win_rate', 0) + team2_stats.get('win_rate', 0)) / 2
    features['avg_recent_form'] = (team1_stats.get('recent_form', 0) + team2_stats.get('recent_form', 0)) / 2
    features['total_matches'] = team1_stats.get('matches', 0) + team2_stats.get('matches', 0)
    features['match_count_ratio'] = team1_stats.get('matches', 1) / max(1, team2_stats.get('matches', 1))
    features['avg_score_metric'] = (team1_stats.get('avg_score', 0) + team2_stats.get('avg_score', 0)) / 2
    features['avg_defense_metric'] = (team1_stats.get('avg_opponent_score', 0) + team2_stats.get('avg_opponent_score', 0)) / 2
    
    # Player statistics averages
    if 'avg_player_rating' in team1_stats and 'avg_player_rating' in team2_stats:
        features['avg_player_rating'] = (team1_stats.get('avg_player_rating', 0) + team2_stats.get('avg_player_rating', 0)) / 2
        features['avg_acs'] = (team1_stats.get('avg_player_acs', 0) + team2_stats.get('avg_player_acs', 0)) / 2
        features['avg_kd'] = (team1_stats.get('avg_player_kd', 0) + team2_stats.get('avg_player_kd', 0)) / 2
        features['avg_kast'] = (team1_stats.get('avg_player_kast', 0) + team2_stats.get('avg_player_kast', 0)) / 2
        features['avg_adr'] = (team1_stats.get('avg_player_adr', 0) + team2_stats.get('avg_player_adr', 0)) / 2
        features['avg_headshot'] = (team1_stats.get('avg_player_headshot', 0) + team2_stats.get('avg_player_headshot', 0)) / 2
        features['star_player_avg'] = (team1_stats.get('star_player_rating', 0) + team2_stats.get('star_player_rating', 0)) / 2
        features['avg_team_consistency'] = (team1_stats.get('team_consistency', 0) + team2_stats.get('team_consistency', 0)) / 2
        features['avg_fk_fd_ratio'] = (team1_stats.get('fk_fd_ratio', 0) + team2_stats.get('fk_fd_ratio', 0)) / 2
    
    # Economy statistics averages
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        features['avg_pistol_win_rate'] = (team1_stats.get('pistol_win_rate', 0) + team2_stats.get('pistol_win_rate', 0)) / 2
        
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            features['avg_eco_win_rate'] = (team1_stats.get('eco_win_rate', 0) + team2_stats.get('eco_win_rate', 0)) / 2
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            features['avg_full_buy_win_rate'] = (team1_stats.get('full_buy_win_rate', 0) + team2_stats.get('full_buy_win_rate', 0)) / 2
        
        if 'economy_efficiency' in team1_stats and 'economy_efficiency' in team2_stats:
            features['avg_economy_efficiency'] = (team1_stats.get('economy_efficiency', 0) + team2_stats.get('economy_efficiency', 0)) / 2
    
    #----------------------------------------
    # 7. INTERACTION TERMS (must transform consistently)
    #----------------------------------------
    # Use direct calculation from original stats for all interactions
    
    # Player rating x win rate (both should negate when swapped)
    player_rating_diff = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
    win_rate_diff = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    # Store the product of negating features - should negate when teams swap
    features['rating_x_win_rate'] = player_rating_diff * win_rate_diff
    
    # Economy interactions
    if 'pistol_win_rate' in team1_stats and 'pistol_win_rate' in team2_stats:
        pistol_diff = team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)
        
        if 'eco_win_rate' in team1_stats and 'eco_win_rate' in team2_stats:
            eco_diff = team1_stats.get('eco_win_rate', 0) - team2_stats.get('eco_win_rate', 0)
            features['pistol_x_eco'] = pistol_diff * eco_diff
        
        if 'full_buy_win_rate' in team1_stats and 'full_buy_win_rate' in team2_stats:
            full_buy_diff = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
            features['pistol_x_full_buy'] = pistol_diff * full_buy_diff
    
    # First blood interaction
    if 'fk_fd_ratio' in team1_stats and 'fk_fd_ratio' in team2_stats:
        fk_fd_diff = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
        features['first_blood_x_win_rate'] = fk_fd_diff * win_rate_diff
    
    # H2H interactions - special handling with h2h_centered
    if 'h2h_win_rate' in features and features.get('h2h_matches', 0) > 0:
        # Center h2h_win_rate around 0.5 for proper negation
        h2h_centered = features['h2h_win_rate'] - 0.5
        recency = features['h2h_recency']
        
        # These should negate when teams swap
        features['h2h_x_win_rate'] = h2h_centered * win_rate_diff * recency
        
        form_diff = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
        features['h2h_x_form'] = h2h_centered * form_diff * recency
    
    return features

def verify_feature_symmetry():
    """Run a fixed symmetry verification."""
    print("\n===== TESTING FEATURE TRANSFORMATIONS =====")
    
    # Create test data
    team1_stats, team2_stats = create_test_team_stats()
    
    # Generate features with teams in both orders
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Define feature categories for testing
    difference_features = [
        'win_rate_diff', 'recent_form_diff', 'score_diff_differential',
        'wins_diff', 'losses_diff', 'avg_score_diff', 'avg_opponent_score_diff',
        'match_count_diff', 'recency_weighted_win_rate_diff', 'win_loss_ratio_diff',
        'avg_map_win_rate_diff', 'best_map_diff', 'player_rating_diff', 'acs_diff',
        'kd_diff', 'kast_diff', 'adr_diff', 'headshot_diff', 'star_player_diff',
        'team_consistency_diff', 'fk_fd_diff', 'pistol_win_rate_diff',
        'eco_win_rate_diff', 'full_buy_win_rate_diff', 'economy_efficiency_diff',
        'recent_form_trajectory_diff', 'h2h_score_diff'
    ]
    
    binary_features = [
        'better_win_rate_team1', 'better_recent_form_team1', 'better_score_diff_team1',
        'better_avg_score_team1', 'better_defense_team1', 'better_player_rating_team1',
        'better_pistol_team1', 'better_trajectory_team1', 'h2h_advantage_team1',
        'maps_advantage_team1', 'h2h_significant'
    ]
    
    symmetric_features = [
        'avg_win_rate', 'avg_recent_form', 'total_matches', 'match_count_ratio',
        'avg_score_metric', 'avg_defense_metric', 'avg_player_rating', 'avg_acs',
        'avg_kd', 'avg_kast', 'avg_adr', 'avg_headshot', 'star_player_avg',
        'avg_team_consistency', 'avg_fk_fd_ratio', 'avg_pistol_win_rate',
        'avg_eco_win_rate', 'avg_full_buy_win_rate', 'avg_economy_efficiency',
        'h2h_matches', 'h2h_recency'
    ]
    
    special_features = ['h2h_win_rate']
    
    interaction_features = [
        'rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy',
        'first_blood_x_win_rate', 'h2h_x_win_rate', 'h2h_x_form'
    ]
    
    # Test difference features
    print("\nTesting DIFFERENCE features (should negate when teams swap):")
    all_passed = True
    for feature in difference_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False

# Test binary features
    print("\nTesting BINARY features (should flip 0↔1 when teams swap):")
    for feature in binary_features:
        if feature in features_normal and feature in features_swapped:
            normal_val = int(features_normal[feature])
            swapped_val = int(features_swapped[feature])
            expected = 1 - normal_val
            
            if expected == swapped_val:
                print(f"✓ {feature:<30} transforms correctly: {normal_val} → {swapped_val}")
            else:
                print(f"✗ {feature:<30} FAILS: {normal_val} → {swapped_val} (expected {expected})")
                all_passed = False
    
    # Test h2h special features
    print("\nTesting H2H features (special transformations):")
    if 'h2h_win_rate' in features_normal and 'h2h_win_rate' in features_swapped:
        expected = 1 - features_normal['h2h_win_rate']
        actual = features_swapped['h2h_win_rate']
        
        if abs(expected - actual) < 1e-10:
            print(f"✓ h2h_win_rate{' '*20} transforms correctly: {features_normal['h2h_win_rate']:.4f} → {features_swapped['h2h_win_rate']:.4f}")
        else:
            print(f"✗ h2h_win_rate{' '*20} FAILS: {features_normal['h2h_win_rate']:.4f} → {features_swapped['h2h_win_rate']:.4f} (expected {expected:.4f})")
            all_passed = False
    
    # Test symmetric features
    print("\nTesting SYMMETRIC features (should remain identical when teams swap):")
    for feature in symmetric_features:
        if feature in features_normal and feature in features_swapped:
            expected = features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} remains invariant: {features_normal[feature]:.4f} = {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} ≠ {features_swapped[feature]:.4f}")
                all_passed = False
    
    # Test interaction features
    print("\nTesting INTERACTION features (should negate when teams swap):")
    for feature in interaction_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    # Print final result
    if all_passed:
        print("\n✓ All features are symmetric and transform correctly when teams are swapped!")
    else:
        print("\n✗ Feature symmetry verification FAILED. Model predictions will be inconsistent!")
    
    return all_passed, features_normal, features_swapped

def debug_interaction_terms(team1_stats, team2_stats):
    """Directly debug interaction term calculation."""
    print("\n===== DEBUGGING INTERACTION TERMS =====")
    
    # Calculate the differences directly
    player_rating_diff_normal = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
    win_rate_diff_normal = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    pistol_diff_normal = team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)
    eco_diff_normal = team1_stats.get('eco_win_rate', 0) - team2_stats.get('eco_win_rate', 0) 
    full_buy_diff_normal = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
    fk_fd_diff_normal = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
    form_diff_normal = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
    
    # Calculate interactions directly
    rating_x_win_rate_normal = player_rating_diff_normal * win_rate_diff_normal
    pistol_x_eco_normal = pistol_diff_normal * eco_diff_normal  
    pistol_x_full_buy_normal = pistol_diff_normal * full_buy_diff_normal
    first_blood_x_win_rate_normal = fk_fd_diff_normal * win_rate_diff_normal
    
    # Now calculate with teams swapped  
    player_rating_diff_swapped = team2_stats.get('avg_player_rating', 0) - team1_stats.get('avg_player_rating', 0)
    win_rate_diff_swapped = team2_stats.get('win_rate', 0) - team1_stats.get('win_rate', 0)
    pistol_diff_swapped = team2_stats.get('pistol_win_rate', 0) - team1_stats.get('pistol_win_rate', 0)
    eco_diff_swapped = team2_stats.get('eco_win_rate', 0) - team1_stats.get('eco_win_rate', 0)
    full_buy_diff_swapped = team2_stats.get('full_buy_win_rate', 0) - team1_stats.get('full_buy_win_rate', 0)
    fk_fd_diff_swapped = team2_stats.get('fk_fd_ratio', 0) - team1_stats.get('fk_fd_ratio', 0)
    form_diff_swapped = team2_stats.get('recent_form', 0) - team1_stats.get('recent_form', 0)
    
    # Calculate swapped interactions
    rating_x_win_rate_swapped = player_rating_diff_swapped * win_rate_diff_swapped
    pistol_x_eco_swapped = pistol_diff_swapped * eco_diff_swapped
    pistol_x_full_buy_swapped = pistol_diff_swapped * full_buy_diff_swapped
    first_blood_x_win_rate_swapped = fk_fd_diff_swapped * win_rate_diff_swapped
    
    # Check if they negate properly
    print(f"rating_x_win_rate:")
    print(f"  Normal: {rating_x_win_rate_normal:.6f}")
    print(f"  Swapped: {rating_x_win_rate_swapped:.6f}")
    print(f"  Sum (should be ~0): {rating_x_win_rate_normal + rating_x_win_rate_swapped:.6f}")
    print(f"  Negates properly: {abs(rating_x_win_rate_normal + rating_x_win_rate_swapped) < 1e-10}")
    
    print(f"\npistol_x_eco:")
    print(f"  Normal: {pistol_x_eco_normal:.6f}")
    print(f"  Swapped: {pistol_x_eco_swapped:.6f}")
    print(f"  Sum (should be ~0): {pistol_x_eco_normal + pistol_x_eco_swapped:.6f}")
    print(f"  Negates properly: {abs(pistol_x_eco_normal + pistol_x_eco_swapped) < 1e-10}")
    
    print(f"\npistol_x_full_buy:")
    print(f"  Normal: {pistol_x_full_buy_normal:.6f}")
    print(f"  Swapped: {pistol_x_full_buy_swapped:.6f}")
    print(f"  Sum (should be ~0): {pistol_x_full_buy_normal + pistol_x_full_buy_swapped:.6f}")
    print(f"  Negates properly: {abs(pistol_x_full_buy_normal + pistol_x_full_buy_swapped) < 1e-10}")
    
    print(f"\nfirst_blood_x_win_rate:")
    print(f"  Normal: {first_blood_x_win_rate_normal:.6f}")
    print(f"  Swapped: {first_blood_x_win_rate_swapped:.6f}")
    print(f"  Sum (should be ~0): {first_blood_x_win_rate_normal + first_blood_x_win_rate_swapped:.6f}")
    print(f"  Negates properly: {abs(first_blood_x_win_rate_normal + first_blood_x_win_rate_swapped) < 1e-10}")
    
    # Generate features using the model function
    features_normal = prepare_data_for_model(team1_stats, team2_stats)
    features_swapped = prepare_data_for_model(team2_stats, team1_stats)
    
    # Compare direct calculation with function output
    print("\nComparing direct calculation with function output:")
    print(f"rating_x_win_rate (direct): {rating_x_win_rate_normal:.6f}")
    print(f"rating_x_win_rate (function): {features_normal.get('rating_x_win_rate', 'N/A')}")
    print(f"Match: {abs(rating_x_win_rate_normal - features_normal.get('rating_x_win_rate', 0)) < 1e-10}")
    
    print(f"\npistol_x_eco (direct): {pistol_x_eco_normal:.6f}")
    print(f"pistol_x_eco (function): {features_normal.get('pistol_x_eco', 'N/A')}")
    print(f"Match: {abs(pistol_x_eco_normal - features_normal.get('pistol_x_eco', 0)) < 1e-10}")
    
    # Finally, check if the function's output negates properly
    print("\nChecking if function's interaction terms negate properly:")
    for feature in ['rating_x_win_rate', 'pistol_x_eco', 'pistol_x_full_buy', 'first_blood_x_win_rate']:
        if feature in features_normal and feature in features_swapped:
            print(f"{feature}:")
            print(f"  Normal: {features_normal[feature]:.6f}")
            print(f"  Swapped: {features_swapped[feature]:.6f}")
            print(f"  Sum (should be ~0): {features_normal[feature] + features_swapped[feature]:.6f}")
            print(f"  Negates properly: {abs(features_normal[feature] + features_swapped[feature]) < 1e-10}")

def fix_implementation_issue():
    """
    Fix the implementation issue by directly demonstrating the correct approach.
    """
    print("\n===== DIRECTLY IMPLEMENTING FIXED VERSION =====")
    
    # Create test team data
    team1, team2 = create_test_team_stats()
    
    # We'll focus only on the interaction terms where the issues are occurring
    print("Calculating interaction terms directly:")
    
    # Step 1: Calculate differences directly from stats
    player_rating_diff = team1.get('avg_player_rating', 0) - team2.get('avg_player_rating', 0)
    win_rate_diff = team1.get('win_rate', 0) - team2.get('win_rate', 0)
    pistol_diff = team1.get('pistol_win_rate', 0) - team2.get('pistol_win_rate', 0)
    eco_diff = team1.get('eco_win_rate', 0) - team2.get('eco_win_rate', 0)
    full_buy_diff = team1.get('full_buy_win_rate', 0) - team2.get('full_buy_win_rate', 0)
    fk_fd_diff = team1.get('fk_fd_ratio', 0) - team2.get('fk_fd_ratio', 0)
    
    # Step 2: Calculate interactions from these differences
    rating_x_win_rate = player_rating_diff * win_rate_diff
    pistol_x_eco = pistol_diff * eco_diff
    pistol_x_full_buy = pistol_diff * full_buy_diff
    first_blood_x_win_rate = fk_fd_diff * win_rate_diff
    
    # Step 3: Now calculate with teams swapped to verify
    player_rating_diff_swapped = team2.get('avg_player_rating', 0) - team1.get('avg_player_rating', 0)
    win_rate_diff_swapped = team2.get('win_rate', 0) - team1.get('win_rate', 0)
    pistol_diff_swapped = team2.get('pistol_win_rate', 0) - team1.get('pistol_win_rate', 0)
    eco_diff_swapped = team2.get('eco_win_rate', 0) - team1.get('eco_win_rate', 0)
    full_buy_diff_swapped = team2.get('full_buy_win_rate', 0) - team1.get('full_buy_win_rate', 0)
    fk_fd_diff_swapped = team2.get('fk_fd_ratio', 0) - team1.get('fk_fd_ratio', 0)
    
    # Calculate swapped interactions
    rating_x_win_rate_swapped = player_rating_diff_swapped * win_rate_diff_swapped
    pistol_x_eco_swapped = pistol_diff_swapped * eco_diff_swapped
    pistol_x_full_buy_swapped = pistol_diff_swapped * full_buy_diff_swapped
    first_blood_x_win_rate_swapped = fk_fd_diff_swapped * win_rate_diff_swapped
    
    # Verify they negate
    print(f"rating_x_win_rate: {rating_x_win_rate:.6f}, swapped: {rating_x_win_rate_swapped:.6f}")
    print(f"Sum (should be ~0): {rating_x_win_rate + rating_x_win_rate_swapped:.10f}")
    print(f"Negates correctly: {abs(rating_x_win_rate + rating_x_win_rate_swapped) < 1e-10}")
    
    print(f"\npistol_x_eco: {pistol_x_eco:.6f}, swapped: {pistol_x_eco_swapped:.6f}")
    print(f"Sum (should be ~0): {pistol_x_eco + pistol_x_eco_swapped:.10f}")
    print(f"Negates correctly: {abs(pistol_x_eco + pistol_x_eco_swapped) < 1e-10}")
    
    print(f"\npistol_x_full_buy: {pistol_x_full_buy:.6f}, swapped: {pistol_x_full_buy_swapped:.6f}")
    print(f"Sum (should be ~0): {pistol_x_full_buy + pistol_x_full_buy_swapped:.10f}")
    print(f"Negates correctly: {abs(pistol_x_full_buy + pistol_x_full_buy_swapped) < 1e-10}")
    
    print(f"\nfirst_blood_x_win_rate: {first_blood_x_win_rate:.6f}, swapped: {first_blood_x_win_rate_swapped:.6f}")
    print(f"Sum (should be ~0): {first_blood_x_win_rate + first_blood_x_win_rate_swapped:.10f}")
    print(f"Negates correctly: {abs(first_blood_x_win_rate + first_blood_x_win_rate_swapped) < 1e-10}")
    
    # Key insight: The issue is likely in the function implementation, not the mathematical logic

def fix_categorization_issue():
    """Fix the categorization issues in the verification function."""
    print("\n===== FIXING CATEGORIZATION IN VERIFICATION =====")
    
    team1, team2 = create_test_team_stats()
    features_normal = prepare_data_for_model(team1, team2)
    features_swapped = prepare_data_for_model(team2, team1)
    
    # Create correct categorization manually
    difference_features = []
    symmetric_features = []
    binary_features = []
    interaction_features = []
    special_features = []
    
    for feature in features_normal.keys():
        # Classify each feature based on its name and expected behavior
        if '_diff' in feature or 'differential' in feature:
            difference_features.append(feature)
        elif feature.startswith('better_') or feature == 'h2h_advantage_team1' or feature == 'maps_advantage_team1':
            binary_features.append(feature)
        elif '_x_' in feature:
            interaction_features.append(feature)
        elif feature == 'h2h_win_rate':
            special_features.append(feature)
        else:
            symmetric_features.append(feature)
    
    # Check for mis-categorized features
    print("Checking for mis-categorized features:")
    
    # Special case: better_score_diff_team1 could be mis-categorized as a difference feature
    if 'better_score_diff_team1' in difference_features:
        print("ERROR: better_score_diff_team1 is wrongly categorized as a difference feature")
        # Fix categorization
        difference_features.remove('better_score_diff_team1')
        if 'better_score_diff_team1' not in binary_features:
            binary_features.append('better_score_diff_team1')
    
    # Features with '_diff' in name wrongly categorized as symmetric
    diff_in_symmetric = [f for f in symmetric_features if '_diff' in f]
    if diff_in_symmetric:
        print(f"ERROR: These difference features are wrongly categorized as symmetric: {diff_in_symmetric}")
        # Fix categorization
        for feature in diff_in_symmetric:
            symmetric_features.remove(feature)
            if feature not in difference_features:
                difference_features.append(feature)
    
    # Test correct categorization
    print("\nTesting with corrected categorization:")
    
    # Test difference features
    all_passed = True
    for feature in difference_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} negates correctly")
            else:
                print(f"✗ {feature:<30} FAILS to negate: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    # Test interaction features
    for feature in interaction_features:
        if feature in features_normal and feature in features_swapped:
            expected = -features_normal[feature]
            actual = features_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} negates correctly")
            else:
                print(f"✗ {feature:<30} FAILS to negate: {features_normal[feature]:.4f} → {features_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    if all_passed:
        print("\nThe correct categorization yields passing tests!")
    else:
        print("\nEven with correct categorization, some tests still fail. The implementation needs fixing.")

if __name__ == "__main__":
    passed, features_normal, features_swapped = verify_feature_symmetry()
    
    if not passed:
        print("\n===== FURTHER DIAGNOSTICS =====")
        print("1. Debugging interaction term calculations directly...")
        team1, team2 = create_test_team_stats()
        debug_interaction_terms(team1, team2)
        
        print("\n2. Testing correct implementation of interaction terms...")
        fix_implementation_issue()
        
        print("\n3. Fixing categorization in verification function...")
        fix_categorization_issue()