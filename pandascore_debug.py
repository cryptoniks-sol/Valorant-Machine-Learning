#!/usr/bin/env python3
"""
PandaScore API Debug Script - Isolate the exact issue
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone

class PandaScoreDebugger:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.pandascore.co"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
    
    def test_api_connection(self):
        """Test basic API connectivity and authentication"""
        print("=" * 60)
        print("🔍 TESTING PANDASCORE API CONNECTION")
        print("=" * 60)
        
        # Test basic endpoint
        test_url = f"{self.base_url}/leagues"
        print(f"Testing basic connectivity: {test_url}")
        
        try:
            response = requests.get(test_url, headers=self.headers, timeout=30)
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Connection: SUCCESS")
                print(f"Response type: {type(data)}")
                if isinstance(data, list):
                    print(f"Number of leagues: {len(data)}")
                elif isinstance(data, dict) and 'data' in data:
                    print(f"Number of leagues: {len(data['data'])}")
                return True
            elif response.status_code == 401:
                print("❌ API Connection: AUTHENTICATION FAILED")
                print("Check your API token")
                return False
            elif response.status_code == 429:
                print("❌ API Connection: RATE LIMITED")
                print("Wait and try again")
                return False
            else:
                print(f"❌ API Connection: HTTP {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ API Connection: TIMEOUT")
            return False
        except requests.exceptions.ConnectionError:
            print("❌ API Connection: CONNECTION ERROR")
            return False
        except Exception as e:
            print(f"❌ API Connection: ERROR - {e}")
            return False
    
    def test_valorant_endpoints(self):
        """Test all Valorant endpoints to see which ones work"""
        print("\n" + "=" * 60)
        print("🎮 TESTING VALORANT ENDPOINTS")
        print("=" * 60)
        
        endpoints = [
            "/valorant/matches/upcoming",
            "/valorant/matches",
            "/matches?videogame=valorant",
            "/matches",
            "/valorant/leagues",
            "/valorant/tournaments"
        ]
        
        working_endpoints = []
        
        for endpoint in endpoints:
            full_url = f"{self.base_url}{endpoint}"
            print(f"\nTesting: {endpoint}")
            
            try:
                params = {}
                if "matches" in endpoint and "upcoming" not in endpoint:
                    params = {
                        "filter[status]": "not_started",
                        "sort": "begin_at",
                        "per_page": 10
                    }
                elif "upcoming" in endpoint:
                    params = {
                        "sort": "begin_at",
                        "per_page": 10
                    }
                
                response = requests.get(full_url, headers=self.headers, params=params, timeout=30)
                print(f"  Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Determine data structure
                    if isinstance(data, list):
                        items = data
                        total_count = len(data)
                    elif isinstance(data, dict) and 'data' in data:
                        items = data['data']
                        total_count = len(items)
                    else:
                        items = []
                        total_count = 0
                    
                    print(f"  ✅ SUCCESS: {total_count} items")
                    
                    # For match endpoints, check for Valorant content
                    if "matches" in endpoint and items:
                        valorant_matches = 0
                        upcoming_matches = 0
                        now = datetime.now(timezone.utc)
                        future_time = now + timedelta(hours=72)
                        
                        for item in items[:5]:  # Check first 5 items
                            # Check if it's Valorant
                            videogame = item.get('videogame', {})
                            if isinstance(videogame, dict):
                                game_name = videogame.get('name', '').lower()
                                if 'valorant' in game_name:
                                    valorant_matches += 1
                            
                            # Check if it's upcoming
                            begin_at = item.get('begin_at')
                            if begin_at:
                                try:
                                    match_time = datetime.fromisoformat(begin_at.replace('Z', '+00:00'))
                                    if now <= match_time <= future_time:
                                        upcoming_matches += 1
                                except:
                                    pass
                        
                        print(f"  📊 Valorant matches: {valorant_matches}/5 checked")
                        print(f"  📊 Upcoming matches: {upcoming_matches}/5 checked")
                        
                        if valorant_matches > 0 or upcoming_matches > 0:
                            working_endpoints.append(endpoint)
                    
                    elif total_count > 0:
                        working_endpoints.append(endpoint)
                
                elif response.status_code == 404:
                    print(f"  ❌ NOT FOUND")
                elif response.status_code == 429:
                    print(f"  ⏱️ RATE LIMITED")
                else:
                    print(f"  ❌ ERROR: {response.status_code}")
                    print(f"     {response.text[:200]}")
                
                time.sleep(1.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ EXCEPTION: {e}")
        
        print(f"\n✅ Working endpoints: {working_endpoints}")
        return working_endpoints
    
    def detailed_match_analysis(self, endpoint="/valorant/matches"):
        """Analyze match data structure in detail"""
        print("\n" + "=" * 60)
        print("🔍 DETAILED MATCH ANALYSIS")
        print("=" * 60)
        
        full_url = f"{self.base_url}{endpoint}"
        params = {
            "sort": "begin_at",
            "per_page": 20
        }
        
        try:
            response = requests.get(full_url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Failed to get data: {response.status_code}")
                return
            
            data = response.json()
            
            # Get matches array
            if isinstance(data, list):
                matches = data
            elif isinstance(data, dict) and 'data' in data:
                matches = data['data']
            else:
                print("❌ Unexpected data format")
                return
            
            print(f"📊 Total matches received: {len(matches)}")
            
            if not matches:
                print("❌ No matches found")
                return
            
            # Analyze first few matches
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=72)
            
            valorant_count = 0
            upcoming_count = 0
            valid_teams_count = 0
            
            print(f"\n📋 Analyzing first 10 matches:")
            print("-" * 40)
            
            for i, match in enumerate(matches[:10]):
                match_id = match.get('id', 'no_id')
                status = match.get('status', 'no_status')
                begin_at = match.get('begin_at', 'no_time')
                
                print(f"\nMatch {i+1}: ID={match_id}")
                print(f"  Status: {status}")
                print(f"  Begin: {begin_at}")
                
                # Check videogame
                videogame = match.get('videogame', {})
                if isinstance(videogame, dict):
                    game_name = videogame.get('name', 'unknown')
                    print(f"  Game: {game_name}")
                    if 'valorant' in game_name.lower():
                        valorant_count += 1
                        print(f"    ✅ Valorant match")
                    else:
                        print(f"    ❌ Not Valorant")
                
                # Check if upcoming
                if begin_at and begin_at != 'no_time':
                    try:
                        match_time = datetime.fromisoformat(begin_at.replace('Z', '+00:00'))
                        if now <= match_time <= future_time:
                            upcoming_count += 1
                            print(f"    ✅ Upcoming")
                        else:
                            time_diff = (match_time - now).total_seconds() / 3600
                            print(f"    ❌ Not upcoming ({time_diff:.1f}h from now)")
                    except Exception as e:
                        print(f"    ❌ Bad time format: {e}")
                
                # Check team data
                team1, team2 = self._extract_teams_debug(match)
                if team1 and team2:
                    valid_teams_count += 1
                    print(f"    ✅ Teams: {team1} vs {team2}")
                else:
                    print(f"    ❌ Missing/invalid team data")
                    self._debug_team_structure(match)
            
            print(f"\n📊 SUMMARY:")
            print(f"  Valorant matches: {valorant_count}/10")
            print(f"  Upcoming matches: {upcoming_count}/10")
            print(f"  Valid team data: {valid_teams_count}/10")
            
            # If we have valid upcoming Valorant matches, that's good
            if valorant_count > 0 and upcoming_count > 0 and valid_teams_count > 0:
                print(f"✅ Found valid data - issue may be in mapping logic")
            else:
                print(f"❌ Data quality issues detected")
                
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_teams_debug(self, match):
        """Extract team names with debug info"""
        # Method 1: opponents
        if 'opponents' in match and match['opponents'] and len(match['opponents']) >= 2:
            try:
                team1 = match['opponents'][0]['opponent']['name']
                team2 = match['opponents'][1]['opponent']['name']
                if team1 and team2 and team1 != team2:
                    return team1, team2
            except (KeyError, TypeError, IndexError):
                pass
        
        # Method 2: teams
        if 'teams' in match and len(match['teams']) >= 2:
            try:
                team1 = match['teams'][0]['name']
                team2 = match['teams'][1]['name']
                if team1 and team2 and team1 != team2:
                    return team1, team2
            except (KeyError, TypeError, IndexError):
                pass
        
        return None, None
    
    def _debug_team_structure(self, match):
        """Debug what team data actually exists"""
        print(f"      🔍 Available keys: {list(match.keys())}")
        
        if 'opponents' in match:
            opponents = match['opponents']
            print(f"      🔍 Opponents type: {type(opponents)}, length: {len(opponents) if opponents else 0}")
            if opponents and len(opponents) > 0:
                print(f"      🔍 First opponent: {opponents[0]}")
        
        if 'teams' in match:
            teams = match['teams']
            print(f"      🔍 Teams type: {type(teams)}, length: {len(teams) if teams else 0}")
            if teams and len(teams) > 0:
                print(f"      🔍 First team: {teams[0]}")
    
    def test_local_api_connection(self, local_url="http://localhost:5000/api/v1"):
        """Test local API connection"""
        print("\n" + "=" * 60)
        print("🔗 TESTING LOCAL API CONNECTION")
        print("=" * 60)
        
        try:
            response = requests.get(f"{local_url}/matches", timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('data', [])
                print(f"✅ Local API working: {len(matches)} matches")
                
                # Show some team names
                if matches:
                    print("Sample teams:")
                    for match in matches[:5]:
                        if 'teams' in match and len(match['teams']) >= 2:
                            team1 = match['teams'][0]['name']
                            team2 = match['teams'][1]['name']
                            print(f"  {team1} vs {team2}")
                
                return True
            else:
                print(f"❌ Local API error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Local API connection failed: {e}")
            return False

def main():
    # Use the API token from your bot
    api_token = "ZrEdZx53byJC1dqBJB3JJ9bUoAZFRllj3eBY2kuTkKnc4La963E"
    
    debugger = PandaScoreDebugger(api_token)
    
    print("🚀 PandaScore API Debugging Tool")
    print(f"🕐 Current time: {datetime.now(timezone.utc).isoformat()}")
    print(f"🌍 Timezone: {os.environ.get('TZ', 'System default')}")
    
    # Test sequence
    if debugger.test_api_connection():
        working_endpoints = debugger.test_valorant_endpoints()
        
        if working_endpoints:
            # Use the best endpoint for detailed analysis
            best_endpoint = working_endpoints[0]
            debugger.detailed_match_analysis(best_endpoint)
        else:
            print("❌ No working Valorant endpoints found")
    
    # Test local API
    debugger.test_local_api_connection()
    
    print("\n" + "=" * 60)
    print("🏁 DEBUGGING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()