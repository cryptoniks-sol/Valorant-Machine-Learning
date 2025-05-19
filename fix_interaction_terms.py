#!/usr/bin/env python3
# fix_interaction_terms.py

def calculate_interactions_correctly(team1_stats, team2_stats):
    """
    Correctly calculate interaction terms to ensure they negate when teams are swapped.
    This is a focused fix for the interaction term issue.
    """
    # Calculate differences directly from stats
    interactions = {}
    
    # Player rating x win rate
    player_rating_diff = team1_stats.get('avg_player_rating', 0) - team2_stats.get('avg_player_rating', 0)
    win_rate_diff = team1_stats.get('win_rate', 0) - team2_stats.get('win_rate', 0)
    interactions['rating_x_win_rate'] = player_rating_diff * win_rate_diff
    
    # Economy interactions
    pistol_diff = team1_stats.get('pistol_win_rate', 0) - team2_stats.get('pistol_win_rate', 0)
    eco_diff = team1_stats.get('eco_win_rate', 0) - team2_stats.get('eco_win_rate', 0)
    full_buy_diff = team1_stats.get('full_buy_win_rate', 0) - team2_stats.get('full_buy_win_rate', 0)
    
    interactions['pistol_x_eco'] = pistol_diff * eco_diff
    interactions['pistol_x_full_buy'] = pistol_diff * full_buy_diff
    
    # First blood interactions
    fk_fd_diff = team1_stats.get('fk_fd_ratio', 0) - team2_stats.get('fk_fd_ratio', 0)
    interactions['first_blood_x_win_rate'] = fk_fd_diff * win_rate_diff
    
    # H2H interactions - calculate from scratch
    # First, get the correct h2h win rate
    h2h_win_rate = 0.5  # Default
    
    # Get team names for lookup
    team1_name = team1_stats.get('team_name', 'Team1')
    team2_name = team2_stats.get('team_name', 'Team2')
    
    # Look up head-to-head data (same logic as in prepare_data_for_model)
    if 'opponent_stats' in team1_stats and team2_name in team1_stats['opponent_stats']:
        h2h_win_rate = team1_stats['opponent_stats'][team2_name].get('win_rate', 0.5)
        h2h_recency = 0.8
    elif 'opponent_stats' in team2_stats and team1_name in team2_stats['opponent_stats']:
        # Invert for team1 perspective
        h2h_win_rate = 1 - team2_stats['opponent_stats'][team1_name].get('win_rate', 0.5)
        h2h_recency = 0.8
    else:
        # Estimate based on win rates
        team1_win_rate = team1_stats.get('win_rate', 0.5)
        team2_win_rate = team2_stats.get('win_rate', 0.5)
        total = team1_win_rate + team2_win_rate
        if total > 0:
            h2h_win_rate = 0.5 + 0.6 * (team1_win_rate / total - 0.5)
        h2h_recency = 0.2
    
    # Center h2h_win_rate around 0.5 for proper negation
    h2h_centered = h2h_win_rate - 0.5
    form_diff = team1_stats.get('recent_form', 0) - team2_stats.get('recent_form', 0)
    
    interactions['h2h_x_win_rate'] = h2h_centered * win_rate_diff * h2h_recency
    interactions['h2h_x_form'] = h2h_centered * form_diff * h2h_recency
    
    return interactions

def test_fixed_interactions():
    """Test that the fixed interaction calculations work correctly."""
    # Create test data
    team1 = {
        'team_name': 'Team A',
        'avg_player_rating': 1.28, 
        'win_rate': 0.75,
        'recent_form': 0.80,
        'pistol_win_rate': 0.68,
        'eco_win_rate': 0.35,
        'full_buy_win_rate': 0.62,
        'fk_fd_ratio': 1.2,
        'opponent_stats': {
            'Team B': {
                'win_rate': 0.8,
                'score_differential': 3.5
            }
        }
    }
    
    team2 = {
        'team_name': 'Team B',
        'avg_player_rating': 1.18,
        'win_rate': 0.60,
        'recent_form': 0.65,
        'pistol_win_rate': 0.55,
        'eco_win_rate': 0.30,
        'full_buy_win_rate': 0.58,
        'fk_fd_ratio': 1.1,
        'opponent_stats': {
            'Team A': {
                'win_rate': 0.2,
                'score_differential': -3.5
            }
        }
    }
    
    # Calculate interactions both ways
    interactions_normal = calculate_interactions_correctly(team1, team2)
    interactions_swapped = calculate_interactions_correctly(team2, team1)
    
    # Test that they negate properly
    all_passed = True
    print("\n===== TESTING FIXED INTERACTION CALCULATIONS =====")
    
    for feature, value in interactions_normal.items():
        if feature in interactions_swapped:
            expected = -value
            actual = interactions_swapped[feature]
            
            if abs(expected - actual) < 1e-10:
                print(f"✓ {feature:<30} transforms correctly: {value:.4f} → {interactions_swapped[feature]:.4f}")
            else:
                print(f"✗ {feature:<30} FAILS: {value:.4f} → {interactions_swapped[feature]:.4f} (expected {expected:.4f})")
                all_passed = False
    
    if all_passed:
        print("\n✓ All fixed interaction calculations are symmetric and negate correctly!")
    else:
        print("\n✗ Fixed interaction calculations still have issues.")
    
    return interactions_normal, interactions_swapped

if __name__ == "__main__":
    test_fixed_interactions()