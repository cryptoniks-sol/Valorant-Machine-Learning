#!/usr/bin/env python3
"""
Test script to find ONLY Valorant matches from PandaScore API
"""

import json
import time
import requests
from datetime import datetime, timezone

def test_valorant_only_endpoints(api_token):
    """Test different ways to get ONLY Valorant matches"""
    
    base_url = "https://api.pandascore.co"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    
    print("🎯 TESTING VALORANT-ONLY API ENDPOINTS")
    print("=" * 80)
    
    # Different approaches to get Valorant matches
    test_endpoints = [
        {
            "name": "Direct Valorant Matches Endpoint",
            "url": f"{base_url}/valorant/matches",
            "params": {"per_page": 50}
        },
        {
            "name": "Valorant Upcoming Matches",
            "url": f"{base_url}/valorant/matches/upcoming",
            "params": {"per_page": 50}
        },
        {
            "name": "General Matches with Valorant videogame filter",
            "url": f"{base_url}/matches",
            "params": {
                "filter[videogame]": "valorant",
                "per_page": 50
            }
        },
        {
            "name": "General Matches with videogame_id=20 (Valorant ID)",
            "url": f"{base_url}/matches",
            "params": {
                "filter[videogame_id]": "20",
                "per_page": 50
            }
        },
        {
            "name": "General Matches with videogame slug",
            "url": f"{base_url}/matches",
            "params": {
                "filter[videogame]": "valorant",
                "filter[status]": "not_started",
                "per_page": 50
            }
        },
        {
            "name": "Search by game name in URL",
            "url": f"{base_url}/matches",
            "params": {
                "search[videogame]": "valorant",
                "per_page": 50
            }
        }
    ]
    
    for i, endpoint in enumerate(test_endpoints):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}: {endpoint['name']}")
        print(f"{'='*80}")
        print(f"URL: {endpoint['url']}")
        print(f"Params: {json.dumps(endpoint['params'], indent=2)}")
        
        try:
            response = requests.get(
                endpoint['url'],
                headers=headers,
                params=endpoint['params'],
                timeout=30
            )
            
            print(f"\nStatus: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Get matches array
                if isinstance(data, list):
                    matches = data
                elif isinstance(data, dict) and 'data' in data:
                    matches = data['data']
                else:
                    matches = []
                
                print(f"Total matches found: {len(matches)}")
                
                # Count Valorant vs other games
                valorant_count = 0
                other_games = {}
                valorant_matches = []
                
                for match in matches:
                    videogame = match.get('videogame', {})
                    if isinstance(videogame, dict):
                        game_name = videogame.get('name', 'Unknown')
                        game_slug = videogame.get('slug', 'unknown')
                        
                        if 'valorant' in game_name.lower() or 'valorant' in game_slug.lower():
                            valorant_count += 1
                            valorant_matches.append(match)
                        else:
                            other_games[game_name] = other_games.get(game_name, 0) + 1
                
                print(f"🎮 Valorant matches: {valorant_count}")
                print(f"🎯 Other games: {dict(other_games)}")
                
                if valorant_count > 0:
                    print(f"\n✅ SUCCESS! Found {valorant_count} Valorant matches")
                    print(f"📋 First 3 Valorant matches:")
                    
                    for i, match in enumerate(valorant_matches[:3]):
                        print(f"\n--- Valorant Match {i+1} ---")
                        print(f"ID: {match.get('id')}")
                        print(f"Name: {match.get('name', 'N/A')}")
                        print(f"Status: {match.get('status', 'N/A')}")
                        print(f"Begin: {match.get('begin_at', 'N/A')}")
                        
                        # Extract teams
                        teams = []
                        if 'opponents' in match and match['opponents']:
                            try:
                                teams = [opp['opponent']['name'] for opp in match['opponents']]
                            except:
                                pass
                        
                        if teams:
                            print(f"Teams: {' vs '.join(teams)}")
                        else:
                            print(f"Teams: Could not extract")
                        
                        print(f"Game: {match.get('videogame', {}).get('name', 'N/A')}")
                else:
                    print(f"❌ No Valorant matches found")
                    if matches:
                        print(f"Sample match videogame: {matches[0].get('videogame', {})}")
                        
            elif response.status_code == 404:
                print("❌ Endpoint not found")
            elif response.status_code == 429:
                print("❌ Rate limited")
                print(f"Rate limit headers: {dict(response.headers)}")
            else:
                print(f"❌ Error {response.status_code}")
                print(f"Response: {response.text[:300]}")
            
            print(f"\n⏱️ Waiting 2 seconds...")
            time.sleep(2)
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("🏁 VALORANT-ONLY TESTING COMPLETE")
    print(f"{'='*80}")

def test_videogame_list(api_token):
    """Get list of available videogames to find correct Valorant ID"""
    
    base_url = "https://api.pandascore.co"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    
    print(f"\n{'='*80}")
    print("🎮 GETTING AVAILABLE VIDEOGAMES")
    print(f"{'='*80}")
    
    try:
        response = requests.get(f"{base_url}/videogames", headers=headers, timeout=30)
        
        if response.status_code == 200:
            games = response.json()
            print(f"Found {len(games)} videogames:")
            
            for game in games:
                game_id = game.get('id')
                game_name = game.get('name')
                game_slug = game.get('slug')
                print(f"  ID: {game_id}, Name: '{game_name}', Slug: '{game_slug}'")
                
                # Highlight Valorant
                if 'valorant' in game_name.lower() or 'valorant' in game_slug.lower():
                    print(f"    ⭐ THIS IS VALORANT! Use ID: {game_id}")
        else:
            print(f"Error getting videogames: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    api_token = "ZrEdZx53byJC1dqBJB3JJ9bUoAZFRllj3eBY2kuTkKnc4La963E"
    
    # First, get the list of videogames to find Valorant's correct ID
    test_videogame_list(api_token)
    
    # Then test different ways to get Valorant matches
    test_valorant_only_endpoints(api_token)

if __name__ == "__main__":
    main()