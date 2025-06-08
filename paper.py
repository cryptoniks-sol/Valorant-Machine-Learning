#!/usr/bin/env python3
"""
Valorant Paper Trading Bot - Enhanced Version with PandaScore + Local API Integration
FIXED: Proper trade update functionality
"""

import os
import sys
import json
import time
import logging
import requests
import schedule
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

# Import your prediction pipeline
try:
    from train import (
        predict_with_consistent_ordering_enhanced,
        load_backtesting_models,
        analyze_betting_edge_for_backtesting,
        select_optimal_bets_conservative,
        BettingConstants,
        simulate_odds
    )
    print("✅ Successfully imported prediction pipeline")
except ImportError as e:
    print(f"❌ Error importing prediction pipeline: {e}")
    print("Make sure train.py is in the same directory")
    sys.exit(1)

@dataclass
class PaperTrade:
    """Represents a single paper trade"""
    trade_id: str
    timestamp: str
    team1: str
    team2: str
    bet_type: str
    bet_amount: float
    odds: float
    predicted_prob: float
    edge: float
    confidence: float
    match_id: str
    match_start_time: str
    status: str = "pending"  # pending, won, lost, cancelled
    actual_result: Optional[str] = None
    profit_loss: float = 0.0
    closing_odds: Optional[float] = None
    clv: float = 0.0
    mapping_confidence: Optional[Dict] = None
    # NEW: Store team mapping for better match updates
    pandascore_team1: Optional[str] = None
    pandascore_team2: Optional[str] = None
    local_team1: Optional[str] = None
    local_team2: Optional[str] = None

@dataclass
class PaperTradingState:
    """Current state of paper trading"""
    starting_bankroll: float
    current_bankroll: float
    total_trades: int
    winning_trades: int
    total_wagered: float
    total_profit: float
    max_drawdown: float
    max_bankroll: float
    trades: List[PaperTrade]
    last_update: str

class LocalAPI:
    """Local API client for getting match data and team mappings"""
    
    def __init__(self, base_url: str = "http://localhost:5000/api/v1", data_dir: Path = None):
        self.base_url = base_url
        self.data_dir = data_dir or Path("paper_trading_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Test connection and cache local matches
        self.local_matches = self._fetch_local_matches()
        
    def _fetch_local_matches(self):
        """Fetch and cache matches from local API"""
        try:
            response = requests.get(f"{self.base_url}/matches", timeout=10)
            if response.status_code == 200:
                data = response.json()
                matches = data.get('data', [])
                print(f"✅ Connected to local API - found {len(matches)} matches")
                return matches
            else:
                print(f"⚠️ Local API returned status {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Could not connect to local API: {e}")
            return []
    
    def get_local_team_names(self) -> set:
        """Get all unique team names from local API"""
        teams = set()
        for match in self.local_matches:
            if 'teams' in match and len(match['teams']) >= 2:
                teams.add(match['teams'][0]['name'])
                teams.add(match['teams'][1]['name'])
        return teams
    
    def find_match_by_teams_and_time(self, pandascore_team1: str, pandascore_team2: str, 
                                   pandascore_time: str) -> Optional[Dict]:
        """Find matching local match by team names and approximate time"""
        try:
            # Parse PandaScore time
            ps_time = datetime.fromisoformat(pandascore_time.replace('Z', '+00:00'))
            
            best_match = None
            best_score = 0
            
            for local_match in self.local_matches:
                if 'teams' not in local_match or len(local_match['teams']) < 2:
                    continue
                
                local_team1 = local_match['teams'][0]['name']
                local_team2 = local_match['teams'][1]['name']
                
                # Skip if one team is TBD
                if 'TBD' in [local_team1, local_team2]:
                    continue
                
                # Calculate team name similarity
                team_score = self._calculate_team_similarity(
                    pandascore_team1, pandascore_team2, 
                    local_team1, local_team2
                )
                
                if team_score > 0.3:  # Minimum similarity threshold
                    # Check time similarity (within 6 hours)
                    try:
                        local_time = datetime.fromisoformat(local_match['utc'].replace('Z', '+00:00'))
                        time_diff = abs((ps_time - local_time).total_seconds())
                        
                        if time_diff <= 6 * 3600:  # Within 6 hours
                            # Combine team and time scores
                            time_score = max(0, 1 - (time_diff / (6 * 3600)))
                            total_score = team_score * 0.7 + time_score * 0.3
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_match = {
                                    'local_match': local_match,
                                    'mapping': {
                                        'pandascore_team1': pandascore_team1,
                                        'pandascore_team2': pandascore_team2,
                                        'local_team1': local_team1,
                                        'local_team2': local_team2
                                    },
                                    'confidence': total_score
                                }
                    except:
                        continue
            
            return best_match
            
        except Exception as e:
            self.logger.error(f"Error finding local match: {e}")
            return None
    
    def _calculate_team_similarity(self, ps_team1: str, ps_team2: str, 
                                 local_team1: str, local_team2: str) -> float:
        """Calculate similarity between PandaScore and local team names"""
        
        # Try both orderings (team1->team1, team2->team2) and (team1->team2, team2->team1)
        score1 = (
            self._team_name_similarity(ps_team1, local_team1) + 
            self._team_name_similarity(ps_team2, local_team2)
        ) / 2
        
        score2 = (
            self._team_name_similarity(ps_team1, local_team2) + 
            self._team_name_similarity(ps_team2, local_team1)
        ) / 2
        
        return max(score1, score2)
    
    def _team_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two team names"""
        if not name1 or not name2:
            return 0
        
        name1_clean = self._normalize_team_name(name1)
        name2_clean = self._normalize_team_name(name2)
        
        # Exact match
        if name1_clean == name2_clean:
            return 1.0
        
        # Substring match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return 0.8
        
        # Word overlap
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            return overlap * 0.6
        
        # Fuzzy match
        try:
            import difflib
            return difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio() * 0.5
        except:
            return 0
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for comparison"""
        name = name.lower().strip()
        
        # Remove common words
        remove_words = ['esports', 'gaming', 'team', 'club', 'gg', 'gc', 'academy']
        for word in remove_words:
            name = name.replace(f' {word}', '').replace(f'{word} ', '')
        
        # Handle specific mappings
        mappings = {
            'karmine corp': 'karmine corp',
            'fnatic': 'fnc',
            'sentinels': 'sen',
            'cloud9': 'c9',
            'team liquid': 'tl',
        }
        
        for original, replacement in mappings.items():
            if original in name:
                name = name.replace(original, replacement)
        
        return name.strip()

class EnhancedPandaScoreAPI:
    def __init__(self, api_token: str, data_dir: Path = None, local_api: LocalAPI = None):
        self.api_token = api_token
        self.base_url = "https://api.pandascore.co"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        self.request_delay = 1.0
        self.data_dir = data_dir or Path("paper_trading_data")
        self.data_dir.mkdir(exist_ok=True)
        self.local_api = local_api
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load saved mappings
        self.saved_team_mappings = self.load_saved_team_mappings()

    def get_upcoming_matches_with_local_mapping(self, hours_ahead: int = 72) -> List[Dict]:
        """Get PandaScore matches and map them to local API teams"""
        
        # Get PandaScore matches
        pandascore_matches = self.get_upcoming_matches(hours_ahead)
        
        if not pandascore_matches or not self.local_api:
            return []
        
        mapped_matches = []
        
        print(f"🔗 Mapping {len(pandascore_matches)} PandaScore matches to local API...")
        
        for ps_match in pandascore_matches:
            try:
                # Get PandaScore team names
                ps_team1, ps_team2 = self.get_team_names_from_match(ps_match)
                if not ps_team1 or not ps_team2:
                    continue
                
                # Find corresponding local match
                local_match_data = self.local_api.find_match_by_teams_and_time(
                    ps_team1, ps_team2, ps_match.get('begin_at', '')
                )
                
                if local_match_data and local_match_data['confidence'] > 0.5:
                    mapping = local_match_data['mapping']
                    
                    print(f"✅ Mapped: {ps_team1} vs {ps_team2} -> {mapping['local_team1']} vs {mapping['local_team2']} (conf: {local_match_data['confidence']:.2f})")
                    
                    # Create enhanced match data
                    enhanced_match = {
                        'pandascore_match': ps_match,
                        'local_match': local_match_data['local_match'],
                        'team_mapping': mapping,
                        'mapping_confidence': local_match_data['confidence'],
                        'pandascore_team1': ps_team1,
                        'pandascore_team2': ps_team2,
                        'local_team1': mapping['local_team1'],
                        'local_team2': mapping['local_team2'],
                        'match_id': ps_match.get('id'),
                        'begin_at': ps_match.get('begin_at')
                    }
                    
                    mapped_matches.append(enhanced_match)
                    
                    # Save this mapping for future use
                    self.save_team_mapping(ps_team1, mapping['local_team1'])
                    self.save_team_mapping(ps_team2, mapping['local_team2'])
                    
                else:
                    print(f"❌ No local match found for: {ps_team1} vs {ps_team2}")
                    
            except Exception as e:
                print(f"❌ Error mapping match: {e}")
                continue
        
        print(f"🎯 Successfully mapped {len(mapped_matches)} matches")
        return mapped_matches
    
    def save_team_mapping(self, pandascore_name: str, local_name: str):
        """Save a team mapping for future use"""
        if pandascore_name not in self.saved_team_mappings:
            self.saved_team_mappings[pandascore_name] = local_name
            
            # Save to file
            try:
                mappings_file = self.data_dir / "pandascore_to_local_mappings.json"
                with open(mappings_file, 'w') as f:
                    json.dump(self.saved_team_mappings, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error saving team mapping: {e}")

    def load_saved_team_mappings(self):
        """Load previously saved team mappings"""
        try:
            mappings_file = self.data_dir / "pandascore_to_local_mappings.json"
            if mappings_file.exists():
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
                self.logger.info(f"📚 Loaded {len(mappings)} saved PandaScore->Local mappings")
                return mappings
            return {}
        except Exception as e:
            self.logger.error(f"Error loading saved team mappings: {e}")
            return {}

    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limited request with error handling"""
        try:
            time.sleep(self.request_delay)
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"⏱️ Rate limited, waiting 30 seconds...")
                time.sleep(30)
                return self._make_request(url, params)
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout for {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
        
    def get_upcoming_matches(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming Valorant matches with improved filtering"""
        try:
            now = datetime.now(timezone.utc)
            end_time = now + timedelta(hours=hours_ahead)
            
            print(f"🔍 Searching for matches between {now.isoformat()} and {end_time.isoformat()}")
            
            # Try different endpoints
            endpoints_to_try = [
                f"{self.base_url}/valorant/matches/upcoming",
                f"{self.base_url}/valorant/matches",
                f"{self.base_url}/matches"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    print(f"🌐 Trying endpoint: {endpoint}")
                    
                    if "upcoming" in endpoint:
                        params = {
                            "sort": "begin_at",
                            "per_page": 50
                        }
                    else:
                        params = {
                            "filter[status]": "not_started",
                            "sort": "begin_at", 
                            "per_page": 50
                        }
                    
                    data = self._make_request(endpoint, params)
                    if not data:
                        continue
                        
                    print(f"📊 Response received")
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        matches = data
                    elif isinstance(data, dict) and 'data' in data:
                        matches = data['data']
                    else:
                        matches = []
                    
                    # Filter for upcoming matches in time window
                    upcoming = []
                    for match in matches:
                        if self._is_upcoming_match(match, now, end_time):
                            upcoming.append(match)
                    
                    print(f"📡 Found {len(upcoming)} upcoming matches in next {hours_ahead} hours")
                    return self._filter_quality_matches(upcoming)
                        
                except Exception as e:
                    print(f"❌ Error with endpoint {endpoint}: {e}")
                    continue
            
            print("❌ All endpoints failed")
            return []
            
        except Exception as e:
            print(f"❌ Error fetching upcoming matches: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _is_upcoming_match(self, match: Dict, start_time: datetime, end_time: datetime) -> bool:
        """Check if match is in the upcoming time window"""
        try:
            begin_at = match.get('begin_at')
            if not begin_at:
                return False
                
            # Parse the begin_at time
            match_time = datetime.fromisoformat(begin_at.replace('Z', '+00:00'))
            return start_time <= match_time <= end_time
            
        except Exception as e:
            print(f"❌ Error parsing match time: {e}")
            return False
    
    def _filter_quality_matches(self, matches: List[Dict]) -> List[Dict]:
        """Filter matches for quality and completeness"""
        quality_matches = []
        
        for match in matches:
            try:
                # Skip canceled/postponed matches
                status = match.get('status', '').lower()
                if status in ['canceled', 'cancelled', 'postponed']:
                    print(f"⚠️ Skipping {status} match: {match.get('id')}")
                    continue
                
                # Check if match has proper team data
                team1, team2 = self.get_team_names_from_match(match)
                if not team1 or not team2 or team1 == team2:
                    print(f"⚠️ Skipping match with incomplete team data: {match.get('id')}")
                    continue
                
                # Check if match has reasonable start time
                begin_at = match.get('begin_at')
                if not begin_at:
                    print(f"⚠️ Skipping match without start time: {match.get('id')}")
                    continue
                
                # Check if it's a Valorant match
                videogame = match.get('videogame', {})
                if isinstance(videogame, dict) and videogame.get('name', '').lower() != 'valorant':
                    print(f"⚠️ Skipping non-Valorant match: {match.get('id')}")
                    continue
                
                quality_matches.append(match)
                
            except Exception as e:
                print(f"❌ Error filtering match {match.get('id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Quality filtered: {len(quality_matches)} matches")
        return quality_matches
    
    def get_team_names_from_match(self, match: Dict) -> Tuple[str, str]:
        """Enhanced team name extraction with multiple fallbacks"""
        try:
            # Primary method: opponents structure
            if 'opponents' in match and match['opponents'] and len(match['opponents']) >= 2:
                try:
                    team1 = match['opponents'][0]['opponent']['name']
                    team2 = match['opponents'][1]['opponent']['name']
                    if team1 and team2 and team1 != team2:
                        return team1, team2
                except (KeyError, TypeError, IndexError):
                    pass
            
            # Fallback 1: teams array
            if 'teams' in match and len(match['teams']) >= 2:
                try:
                    team1 = match['teams'][0]['name']
                    team2 = match['teams'][1]['name']
                    if team1 and team2 and team1 != team2:
                        return team1, team2
                except (KeyError, TypeError, IndexError):
                    pass
            
            # Fallback 2: participants
            if 'participants' in match and len(match['participants']) >= 2:
                try:
                    team1 = match['participants'][0]['name']
                    team2 = match['participants'][1]['name']
                    if team1 and team2 and team1 != team2:
                        return team1, team2
                except (KeyError, TypeError, IndexError):
                    pass
            
            # Debug logging for failed extractions
            print(f"❌ No team names found in match {match.get('id', 'unknown')}")
            available_keys = list(match.keys())
            print(f"Available keys: {available_keys[:10]}...")
            
            return None, None
            
        except Exception as e:
            print(f"❌ Error extracting team names: {e}")
            return None, None
    
    def get_finished_matches(self, hours_back: int = 6) -> List[Dict]:
        """Get recently finished matches with improved error handling"""
        try:
            now = datetime.now(timezone.utc)
            start_time = now - timedelta(hours=hours_back)
            
            print(f"🔍 Searching for finished matches between {start_time.isoformat()} and {now.isoformat()}")
            
            endpoints_to_try = [
                f"{self.base_url}/valorant/matches",
                f"{self.base_url}/matches"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    params = {
                        "filter[status]": "finished",
                        "sort": "-end_at",
                        "per_page": 100
                    }
                    
                    data = self._make_request(endpoint, params)
                    if not data:
                        continue
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        all_matches = data
                    elif isinstance(data, dict) and 'data' in data:
                        all_matches = data['data']
                    else:
                        all_matches = []
                    
                    # Filter for finished matches in time window
                    finished_matches = []
                    for match in all_matches:
                        if (match.get('status') == 'finished' and 
                            self._is_finished_match(match, start_time, now)):
                            finished_matches.append(match)
                    
                    print(f"✅ Found {len(finished_matches)} finished matches in last {hours_back} hours")
                    return finished_matches
                        
                except Exception as e:
                    print(f"❌ Error with endpoint {endpoint}: {e}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"❌ Error fetching finished matches: {e}")
            return []
    
    def _is_finished_match(self, match: Dict, start_time: datetime, end_time: datetime) -> bool:
        """Check if match finished in the time window"""
        try:
            if match.get('status') != 'finished':
                return False
                
            end_at = match.get('end_at')
            if not end_at:
                # If no end_at, check begin_at + some reasonable match duration
                begin_at = match.get('begin_at')
                if begin_at:
                    match_start = datetime.fromisoformat(begin_at.replace('Z', '+00:00'))
                    # Assume matches can last up to 3 hours
                    estimated_end = match_start + timedelta(hours=3)
                    return start_time <= estimated_end <= end_time
                return False
                
            match_end = datetime.fromisoformat(end_at.replace('Z', '+00:00'))
            return start_time <= match_end <= end_time
            
        except Exception as e:
            return False

    def get_match_by_id(self, match_id: str) -> Optional[Dict]:
        """Get a specific match by its ID"""
        try:
            url = f"{self.base_url}/matches/{match_id}"
            return self._make_request(url)
        except Exception as e:
            print(f"❌ Error fetching match {match_id}: {e}")
            return None

class EnhancedPaperTradingBot:
    """Enhanced paper trading bot with PandaScore + Local API integration"""
    
    def __init__(self, starting_bankroll: float = 500.0, data_dir: str = "paper_trading_data", 
                 local_api_url: str = "http://localhost:5000/api/v1"):
        self.starting_bankroll = starting_bankroll
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize Local API first
        print("🔗 Connecting to local API...")
        self.local_api = LocalAPI(base_url=local_api_url, data_dir=self.data_dir)
        
        # Initialize PandaScore API with local API integration
        print("🌐 Connecting to PandaScore API...")
        self.api = EnhancedPandaScoreAPI(
            "ZrEdZx53byJC1dqBJB3JJ9bUoAZFRllj3eBY2kuTkKnc4La963E",
            data_dir=self.data_dir,
            local_api=self.local_api
        )
        
        # Load or initialize state
        self.state_file = self.data_dir / "trading_state.json"
        self.state = self.load_state()
        
        # Load prediction models
        print("🤖 Loading prediction models...")
        try:
            self.ensemble_models, self.selected_features = load_backtesting_models()
            if not self.ensemble_models or not self.selected_features:
                raise RuntimeError("Models loaded but empty")
            print("✅ Prediction models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load prediction models: {e}")
            raise RuntimeError("Failed to load prediction models. Run training first.")
        
        # Setup enhanced logging with Unicode support
        log_file = self.data_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Clear any existing handlers to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter (without emojis for file, with emojis for console)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        # Processed matches to avoid duplicates
        self.processed_matches = set()
        
        # Statistics
        self.session_stats = {
            'matches_processed': 0,
            'predictions_made': 0,
            'trades_placed': 0,
            'trades_updated': 0,
            'errors': 0,
            'mappings_found': 0
        }
        
        # Add bankruptcy check
        self.bankruptcy_check = {'max_bet_size': self.state.current_bankroll * 0.05}
        
    def load_state(self) -> PaperTradingState:
        """Load trading state from file or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Convert trades back to PaperTrade objects
                trades = []
                for trade_data in data['trades']:
                    # Handle new fields that might not exist in old saves
                    if 'mapping_confidence' not in trade_data:
                        trade_data['mapping_confidence'] = None
                    
                    # Handle new team name fields for better matching
                    if 'pandascore_team1' not in trade_data:
                        trade_data['pandascore_team1'] = trade_data.get('team1')
                    if 'pandascore_team2' not in trade_data:
                        trade_data['pandascore_team2'] = trade_data.get('team2')
                    if 'local_team1' not in trade_data:
                        trade_data['local_team1'] = None
                    if 'local_team2' not in trade_data:
                        trade_data['local_team2'] = None
                        
                    trades.append(PaperTrade(**trade_data))
                
                state = PaperTradingState(
                    starting_bankroll=data['starting_bankroll'],
                    current_bankroll=data['current_bankroll'],
                    total_trades=data['total_trades'],
                    winning_trades=data['winning_trades'],
                    total_wagered=data['total_wagered'],
                    total_profit=data['total_profit'],
                    max_drawdown=data['max_drawdown'],
                    max_bankroll=data['max_bankroll'],
                    trades=trades,
                    last_update=data['last_update']
                )
                
                print(f"📊 Loaded existing state: ${state.current_bankroll:.2f} bankroll, {len(trades)} trades")
                return state
                
            except Exception as e:
                print(f"⚠️ Error loading state, creating new: {e}")
        
        # Create new state
        return PaperTradingState(
            starting_bankroll=self.starting_bankroll,
            current_bankroll=self.starting_bankroll,
            total_trades=0,
            winning_trades=0,
            total_wagered=0.0,
            total_profit=0.0,
            max_drawdown=0.0,
            max_bankroll=self.starting_bankroll,
            trades=[],
            last_update=datetime.now().isoformat()
        )
    
    def save_state(self):
        """Save current trading state to file"""
        try:
            # Convert to serializable format
            data = asdict(self.state)
            data['trades'] = [asdict(trade) for trade in self.state.trades]
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self._log_safe('error', f"Error saving state: {e}")
    
    def _log_safe(self, level, message):
        """Safe logging that removes emojis for file logging but keeps them for console"""
        # Remove emojis for file logging to avoid encoding issues
        import re
        clean_message = re.sub(r'[^\x00-\x7F]+', '', message)  # Remove non-ASCII characters
        
        # Log clean message to file
        if level == 'info':
            self.logger.info(clean_message)
        elif level == 'warning':
            self.logger.warning(clean_message)
        elif level == 'error':
            self.logger.error(clean_message)
        elif level == 'debug':
            self.logger.debug(clean_message)
        
        # Print original message with emojis to console
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {message}")

    def make_prediction_with_local_names(self, local_team1: str, local_team2: str) -> Optional[Dict]:
        """Make prediction using local API team names"""
        try:
            self._log_safe('info', f"🎯 Making prediction for: {local_team1} vs {local_team2}")
            
            # Make prediction using local team names
            prediction_results = predict_with_consistent_ordering_enhanced(
                local_team1, local_team2,
                self.ensemble_models,
                self.selected_features,
                use_cache=True
            )
            
            if 'error' in prediction_results:
                self._log_safe('warning', f"Prediction error: {prediction_results['error']}")
                return None
            
            self._log_safe('info', f"✅ Prediction complete: {local_team1} {prediction_results['win_probability']:.2%} vs {local_team2}")
            self.session_stats['predictions_made'] += 1
            
            return prediction_results
            
        except Exception as e:
            self._log_safe('error', f"Error making prediction: {e}")
            self.session_stats['errors'] += 1
            return None
    
    def simulate_market_odds(self, win_probability: float, match_format: str = 'bo3') -> Dict:
        """Simulate realistic betting odds with error handling"""
        try:
            odds_data = simulate_odds(
                win_probability, 
                vig=0.045,
                market_efficiency=0.92,
                match_format=match_format
            )
            
            self._log_safe('debug', f"Simulated odds: {odds_data}")
            return odds_data
            
        except Exception as e:
            self._log_safe('error', f"Error simulating odds: {e}")
            return {}
    
    def _detect_match_format(self, match: Dict) -> str:
        """Detect if match is BO3, BO5, etc."""
        try:
            num_games = match.get('number_of_games', 3)
            if num_games >= 5:
                return 'bo5'
            elif num_games >= 3:
                return 'bo3'
            else:
                return 'bo1'
        except Exception:
            return 'bo3'
    
    def place_paper_trade(self, team1: str, team2: str, bet_type: str, 
                         bet_amount: float, odds: float, predicted_prob: float,
                         edge: float, confidence: float, match_id: str, 
                         match_start_time: str, mapping_confidence: Dict = None,
                         local_team1: str = None, local_team2: str = None) -> str:
        """Place a paper trade with enhanced data"""
        
        # Check if we have enough bankroll
        if bet_amount > self.state.current_bankroll:
            self._log_safe('warning', f"Insufficient bankroll for {bet_type}: ${bet_amount:.2f} > ${self.state.current_bankroll:.2f}")
            return None
        
        trade_id = f"PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.state.trades)}"
        
        trade = PaperTrade(
            trade_id=trade_id,
            timestamp=datetime.now().isoformat(),
            team1=team1,
            team2=team2,
            bet_type=bet_type,
            bet_amount=bet_amount,
            odds=odds,
            predicted_prob=predicted_prob,
            edge=edge,
            confidence=confidence,
            match_id=match_id,
            match_start_time=match_start_time,
            mapping_confidence=mapping_confidence,
            pandascore_team1=team1,  # Store PandaScore team names
            pandascore_team2=team2,
            local_team1=local_team1,  # Store local team names for prediction reference
            local_team2=local_team2
        )
        
        # Update bankroll (reserve the bet amount)
        self.state.current_bankroll -= bet_amount
        self.state.total_wagered += bet_amount
        self.state.total_trades += 1
        self.state.trades.append(trade)
        
        self._log_safe('info', f"📝 Placed paper trade: {bet_type} ${bet_amount:.2f} @ {odds:.2f} ({edge:.2%} edge)")
        self.save_state()
        self.session_stats['trades_placed'] += 1
        
        return trade_id
    
    def process_upcoming_matches(self):
        """Process upcoming matches using PandaScore + Local API integration"""
        self._log_safe('info', "🔍 Scanning for upcoming matches with PandaScore + Local API integration...")
        
        # Get mapped matches (PandaScore matches mapped to local API teams)
        mapped_matches = self.api.get_upcoming_matches_with_local_mapping(hours_ahead=72)
        
        if not mapped_matches:
            self._log_safe('info', "❌ No mapped matches found")
            return 0
        
        trades_placed = 0
        
        for enhanced_match in mapped_matches:
            try:
                match_id = str(enhanced_match['match_id'])
                if match_id in self.processed_matches:
                    continue
                
                # Extract team names (use local API names for prediction)
                local_team1 = enhanced_match['local_team1']
                local_team2 = enhanced_match['local_team2']
                pandascore_team1 = enhanced_match['pandascore_team1']
                pandascore_team2 = enhanced_match['pandascore_team2']
                
                self.session_stats['matches_processed'] += 1
                self.session_stats['mappings_found'] += 1
                
                self._log_safe('info', f"🎮 Processing: {pandascore_team1} vs {pandascore_team2}")
                self._log_safe('info', f"🔗 Mapped to: {local_team1} vs {local_team2} (conf: {enhanced_match['mapping_confidence']:.2f})")
                
                # Make prediction using LOCAL API team names
                prediction = self.make_prediction_with_local_names(local_team1, local_team2)
                if not prediction:
                    self.processed_matches.add(match_id)
                    continue
                
                # Get PandaScore match for odds simulation
                pandascore_match = enhanced_match['pandascore_match']
                match_format = self._detect_match_format(pandascore_match)
                
                # Simulate market odds based on prediction
                odds_data = self.simulate_market_odds(prediction['win_probability'], match_format)
                
                if not odds_data:
                    self.processed_matches.add(match_id)
                    continue
                
                # Analyze betting edge
                betting_analysis = analyze_betting_edge_for_backtesting(
                    prediction['win_probability'], 
                    1 - prediction['win_probability'], 
                    odds_data,
                    prediction['confidence'], 
                    self.state.current_bankroll
                )
                
                # Select optimal bets
                optimal_bets = select_optimal_bets_conservative(
                    betting_analysis, 
                    pandascore_team1,  # Use PandaScore names for display
                    pandascore_team2, 
                    self.state.current_bankroll,
                    max_bets=2,
                    max_total_risk=0.10
                )
                
                if optimal_bets:
                    self._log_safe('info', f"💰 Found {len(optimal_bets)} betting opportunities")
                    for bet_type, analysis in optimal_bets.items():
                        trade_id = self.place_paper_trade(
                            team1=pandascore_team1,  # Use PandaScore names for trade records
                            team2=pandascore_team2,
                            bet_type=bet_type,
                            bet_amount=analysis['bet_amount'],
                            odds=analysis['odds'],
                            predicted_prob=analysis['probability'],
                            edge=analysis['edge'],
                            confidence=prediction['confidence'],
                            match_id=match_id,
                            match_start_time=enhanced_match.get('begin_at', ''),
                            mapping_confidence={
                                'pandascore_to_local': enhanced_match['team_mapping'],
                                'mapping_confidence': enhanced_match['mapping_confidence'],
                                'prediction_teams': f"{local_team1} vs {local_team2}"
                            },
                            local_team1=local_team1,  # Store local team names
                            local_team2=local_team2
                        )
                        if trade_id:
                            trades_placed += 1
                else:
                    self._log_safe('info', f"✅ No qualifying bets for {pandascore_team1} vs {pandascore_team2}")
                
                self.processed_matches.add(match_id)
                
            except Exception as e:
                self._log_safe('error', f"Error processing enhanced match: {e}")
                self.session_stats['errors'] += 1
                continue
        
        self._log_safe('info', f"📊 Scan complete: {trades_placed} new paper trades placed from {len(mapped_matches)} mapped matches")
        return trades_placed
    
    def update_finished_trades(self):
        """Update finished trades based on match results - ENHANCED VERSION"""
        self._log_safe('info', "🔄 Updating finished trades...")
        
        # Get pending trades
        pending_trades = [
            trade for trade in self.state.trades 
            if trade.status == "pending"
        ]
        
        if not pending_trades:
            self._log_safe('info', "No pending trades to update")
            return 0
        
        updates = 0
        
        # Check each pending trade individually
        for trade in pending_trades:
            try:
                self._log_safe('info', f"🔍 Checking trade {trade.trade_id} for match {trade.match_id}")
                
                # Get the specific match by ID
                match_data = self.api.get_match_by_id(trade.match_id)
                
                if not match_data:
                    # Try to find in recent finished matches
                    finished_matches = self.api.get_finished_matches(hours_back=72)  # Extended search
                    match_data = next(
                        (m for m in finished_matches if str(m.get('id')) == trade.match_id), 
                        None
                    )
                
                if not match_data:
                    # Check if match start time has passed significantly (might be cancelled)
                    try:
                        match_start = datetime.fromisoformat(trade.match_start_time.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        
                        # If match was supposed to start more than 6 hours ago and we can't find it
                        if (now - match_start).total_seconds() > 6 * 3600:
                            self._log_safe('warning', f"Match {trade.match_id} appears to be cancelled or missing")
                            trade.status = "cancelled"
                            # Return the bet amount to bankroll
                            self.state.current_bankroll += trade.bet_amount
                            updates += 1
                            continue
                    except:
                        pass
                    
                    self._log_safe('debug', f"No match data found for {trade.match_id}")
                    continue
                
                # Check if match is finished
                if match_data.get('status') != 'finished':
                    self._log_safe('debug', f"Match {trade.match_id} not finished yet (status: {match_data.get('status')})")
                    continue
                
                # Extract match result
                result = self.extract_match_result(match_data)
                if not result:
                    self._log_safe('warning', f"Could not extract result for match {trade.match_id}")
                    continue
                
                self._log_safe('info', f"🏆 Found result for {trade.match_id}: {result['team1']} vs {result['team2']} - {result['winner']} won {result['score']}")
                
                # Evaluate trade outcome using both PandaScore and local team names
                won = self.evaluate_trade_outcome(trade, result)
                
                if won:
                    profit = trade.bet_amount * (trade.odds - 1)
                    trade.status = "won"
                    trade.profit_loss = profit
                    self.state.current_bankroll += trade.bet_amount + profit
                    self.state.winning_trades += 1
                    self.state.total_profit += profit
                    
                    self._log_safe('info', f"✅ Trade WON: {trade.bet_type} +${profit:.2f}")
                else:
                    trade.status = "lost"
                    trade.profit_loss = -trade.bet_amount
                    self.state.total_profit -= trade.bet_amount
                    
                    self._log_safe('info', f"❌ Trade LOST: {trade.bet_type} -${trade.bet_amount:.2f}")
                
                trade.actual_result = f"{result['winner']} won {result['score']}"
                updates += 1
                self.session_stats['trades_updated'] += 1
                
            except Exception as e:
                self._log_safe('error', f"Error updating trade {trade.trade_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Update statistics
        self.update_statistics()
        self.save_state()
        
        self._log_safe('info', f"📈 Updated {updates} trades")
        return updates
    
    def extract_match_result(self, match: Dict) -> Optional[Dict]:
        """Extract match result from finished match - ENHANCED VERSION"""
        try:
            if match.get('status') != 'finished':
                return None
            
            # Get team names using the same method as when processing matches
            team1, team2 = self.api.get_team_names_from_match(match)
            if not team1 or not team2:
                print(f"❌ Could not extract team names from finished match {match.get('id')}")
                return None
            
            # Method 1: Try results array
            results = match.get('results', [])
            if results and len(results) >= 2:
                try:
                    team1_score = results[0].get('score', 0)
                    team2_score = results[1].get('score', 0)
                    
                    if team1_score != team2_score:  # Valid result
                        winner = team1 if team1_score > team2_score else team2
                        return {
                            'team1': team1,
                            'team2': team2,
                            'team1_score': team1_score,
                            'team2_score': team2_score,
                            'winner': winner,
                            'score': f"{team1_score}-{team2_score}"
                        }
                except (KeyError, TypeError):
                    pass
            
            # Method 2: Try games array
            games = match.get('games', [])
            if games:
                try:
                    team1_wins = 0
                    team2_wins = 0
                    
                    for game in games:
                        winner_info = game.get('winner', {})
                        if isinstance(winner_info, dict):
                            winner_name = winner_info.get('name', '')
                            if winner_name == team1:
                                team1_wins += 1
                            elif winner_name == team2:
                                team2_wins += 1
                    
                    if team1_wins != team2_wins:  # Valid result
                        winner = team1 if team1_wins > team2_wins else team2
                        return {
                            'team1': team1,
                            'team2': team2,
                            'team1_score': team1_wins,
                            'team2_score': team2_wins,
                            'winner': winner,
                            'score': f"{team1_wins}-{team2_wins}"
                        }
                except (KeyError, TypeError):
                    pass
            
            # Method 3: Try winner field directly
            winner_info = match.get('winner', {})
            if isinstance(winner_info, dict) and 'name' in winner_info:
                winner_name = winner_info['name']
                if winner_name in [team1, team2]:
                    # Assume 2-1 or 2-0 score if we don't have exact scores
                    return {
                        'team1': team1,
                        'team2': team2,
                        'team1_score': 2 if winner_name == team1 else 1,
                        'team2_score': 1 if winner_name == team1 else 2,
                        'winner': winner_name,
                        'score': "2-1" if winner_name == team1 else "1-2"
                    }
            
            print(f"❌ Could not extract match result for {match.get('id')} - no valid score data")
            print(f"Available keys: {list(match.keys())}")
            if 'results' in match:
                print(f"Results: {match['results']}")
            if 'games' in match:
                print(f"Games count: {len(match.get('games', []))}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error extracting match result: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_trade_outcome(self, trade: PaperTrade, result: Dict) -> bool:
        """Evaluate if a trade won based on the match result - ENHANCED VERSION"""
        try:
            bet_type = trade.bet_type
            team1_score = result['team1_score']
            team2_score = result['team2_score']
            winner_name = result['winner']
            
            # Get trade team names (could be PandaScore or local names)
            trade_team1 = trade.team1  # This should be PandaScore team name
            trade_team2 = trade.team2
            
            # Map result teams to trade teams using similarity matching
            result_team1 = result['team1']
            result_team2 = result['team2']
            
            # Try exact match first
            team1_matches_result_team1 = self._teams_match(trade_team1, result_team1)
            team1_matches_result_team2 = self._teams_match(trade_team1, result_team2)
            team2_matches_result_team1 = self._teams_match(trade_team2, result_team1)
            team2_matches_result_team2 = self._teams_match(trade_team2, result_team2)
            
            # Determine the correct mapping
            if team1_matches_result_team1 and team2_matches_result_team2:
                # trade_team1 -> result_team1, trade_team2 -> result_team2
                trade_team1_winner = (winner_name == result_team1)
                trade_team2_winner = (winner_name == result_team2)
            elif team1_matches_result_team2 and team2_matches_result_team1:
                # trade_team1 -> result_team2, trade_team2 -> result_team1
                trade_team1_winner = (winner_name == result_team2)
                trade_team2_winner = (winner_name == result_team1)
                # Swap scores for consistency
                team1_score, team2_score = team2_score, team1_score
            else:
                self._log_safe('warning', f"Could not map trade teams to result teams: {trade_team1}/{trade_team2} vs {result_team1}/{result_team2}")
                return False
            
            # Evaluate bet types
            if bet_type == 'team1_ml':
                return trade_team1_winner
            elif bet_type == 'team2_ml':
                return trade_team2_winner
            
            # Spread bets
            elif bet_type == 'team1_plus_1_5':
                # Team1 covers +1.5 spread if they win OR lose by 1
                return trade_team1_winner or (team1_score + 1.5 > team2_score)
            elif bet_type == 'team1_minus_1_5':
                # Team1 covers -1.5 spread if they win by 2+
                return trade_team1_winner and (team1_score - 1.5 > team2_score)
            elif bet_type == 'team2_plus_1_5':
                return trade_team2_winner or (team2_score + 1.5 > team1_score)
            elif bet_type == 'team2_minus_1_5':
                return trade_team2_winner and (team2_score - 1.5 > team1_score)
            
            # Total bets
            elif bet_type == 'over_2_5_maps':
                total_maps = team1_score + team2_score
                return total_maps > 2.5
            elif bet_type == 'under_2_5_maps':
                total_maps = team1_score + team2_score
                return total_maps < 2.5
            
            else:
                self._log_safe('warning', f"Unknown bet type: {bet_type}")
                return False
                
        except Exception as e:
            self._log_safe('error', f"Error evaluating trade outcome: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _teams_match(self, team1: str, team2: str) -> bool:
        """Check if two team names refer to the same team"""
        if not team1 or not team2:
            return False
        
        # Exact match
        if team1 == team2:
            return True
        
        # Case-insensitive match
        if team1.lower() == team2.lower():
            return True
        
        # Normalize and compare
        team1_norm = self._normalize_team_name_for_matching(team1)
        team2_norm = self._normalize_team_name_for_matching(team2)
        
        if team1_norm == team2_norm:
            return True
        
        # Substring match
        if team1_norm in team2_norm or team2_norm in team1_norm:
            return True
        
        return False
    
    def _normalize_team_name_for_matching(self, name: str) -> str:
        """Normalize team name for matching purposes"""
        name = name.lower().strip()
        
        # Remove common suffixes/prefixes
        for word in ['esports', 'gaming', 'team', 'club', 'gg', 'gc', 'academy', 'valkyries']:
            name = name.replace(f' {word}', '').replace(f'{word} ', '')
        
        # Handle common abbreviations
        abbreviations = {
            'generation g': 'gen.g',
            'geng': 'gen.g',
            'gen g': 'gen.g',
            'sentinels': 'sen',
            'cloud9': 'c9',
            'team liquid': 'tl',
            'fnatic': 'fnc'
        }
        
        for full, abbrev in abbreviations.items():
            if full in name:
                name = name.replace(full, abbrev)
        
        return name.strip()
    
    def update_statistics(self):
        """Update trading statistics"""
        try:
            # Calculate max drawdown
            running_balance = self.starting_bankroll
            peak = self.starting_bankroll
            max_dd = 0
            
            completed_trades = [t for t in self.state.trades if t.status in ['won', 'lost']]
            for trade in completed_trades:
                running_balance += trade.profit_loss
                if running_balance > peak:
                    peak = running_balance
                drawdown = (peak - running_balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            self.state.max_drawdown = max_dd
            self.state.max_bankroll = max(self.state.max_bankroll, self.state.current_bankroll)
            self.state.last_update = datetime.now().isoformat()
            
        except Exception as e:
            self._log_safe('error', f"Error updating statistics: {e}")
    
    def print_enhanced_status(self):
        """Print enhanced trading status"""
        roi = (self.state.current_bankroll - self.starting_bankroll) / self.starting_bankroll
        
        completed_trades = len([t for t in self.state.trades if t.status in ['won', 'lost']])
        pending_trades = len([t for t in self.state.trades if t.status == 'pending'])
        cancelled_trades = len([t for t in self.state.trades if t.status == 'cancelled'])
        win_rate = self.state.winning_trades / max(1, completed_trades)
        
        print(f"\n{'='*70}")
        print(f"📊 ENHANCED PAPER TRADING STATUS")
        print(f"{'='*70}")
        print(f"💰 Bankroll: ${self.state.current_bankroll:.2f} (started with ${self.starting_bankroll:.2f})")
        print(f"📈 P&L: ${self.state.current_bankroll - self.starting_bankroll:.2f} ({roi:.2%} ROI)")
        print(f"🎯 Trades: {completed_trades} completed, {pending_trades} pending, {cancelled_trades} cancelled")
        print(f"🏆 Win Rate: {self.state.winning_trades}/{completed_trades} ({win_rate:.2%})")
        print(f"💸 Total Wagered: ${self.state.total_wagered:.2f}")
        print(f"📉 Max Drawdown: {self.state.max_drawdown:.2%}")
        
        # Session statistics
        print(f"\n📋 Session Statistics:")
        print(f"  🔍 Matches Processed: {self.session_stats['matches_processed']}")
        print(f"  🔗 Team Mappings Found: {self.session_stats['mappings_found']}")
        print(f"  🎯 Predictions Made: {self.session_stats['predictions_made']}")
        print(f"  📝 Trades Placed: {self.session_stats['trades_placed']}")
        print(f"  📈 Trades Updated: {self.session_stats['trades_updated']}")
        print(f"  ❌ Errors: {self.session_stats['errors']}")
        
        # Show recent trades
        if self.state.trades:
            print(f"\n📝 Recent Trades:")
            recent_trades = sorted(self.state.trades, key=lambda x: x.timestamp, reverse=True)[:5]
            for trade in recent_trades:
                status_emoji = "✅" if trade.status == "won" else "❌" if trade.status == "lost" else "⏳" if trade.status == "pending" else "🚫"
                profit_str = f"${trade.profit_loss:+.2f}" if trade.status in ['won', 'lost'] else "N/A"
                print(f"  {status_emoji} {trade.team1} vs {trade.team2} | {trade.bet_type} | ${trade.bet_amount:.2f} @ {trade.odds:.2f} | {profit_str}")
        
        print(f"\n🕐 Last Update: {self.state.last_update}")
        print(f"{'='*70}\n")
    
    def run_once(self):
        """Run a single iteration of the trading bot"""
        try:
            print(f"🤖 Enhanced Paper Trading Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Reset session stats
            self.session_stats = {key: 0 for key in self.session_stats}
            
            # Update finished trades FIRST (this is important!)
            updates = self.update_finished_trades()
            
            # Then process new matches
            new_trades = self.process_upcoming_matches()
            
            # Print status
            self.print_enhanced_status()
            
            if new_trades > 0 or updates > 0:
                self._log_safe('info', f"Activity: {new_trades} new trades, {updates} updates")
            else:
                self._log_safe('info', "No new activity")
                
        except Exception as e:
            self._log_safe('error', f"Error in run_once: {e}")
            import traceback
            traceback.print_exc()
            self.session_stats['errors'] += 1
    
    def run_continuous(self, check_interval_minutes: int = 30):
        """Run the trading bot continuously"""
        print(f"🚀 Starting enhanced continuous paper trading (checking every {check_interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        # Schedule the bot to run
        schedule.every(check_interval_minutes).minutes.do(self.run_once)
        
        # Run once immediately
        self.run_once()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced paper trading bot...")
            print("Final status:")
            self.print_enhanced_status()

def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Valorant Paper Trading Bot with PandaScore + Local API")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Starting bankroll (default: $500)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in minutes (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--data-dir", type=str, default="paper_trading_data", help="Data directory")
    parser.add_argument("--local-api", type=str, default="http://localhost:5000/api/v1", help="Local API URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--update-trades", action="store_true", help="Force update all pending trades")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize bot with local API integration
        bot = EnhancedPaperTradingBot(
            starting_bankroll=args.bankroll, 
            data_dir=args.data_dir,
            local_api_url=args.local_api
        )
        
        if args.status:
            bot.print_enhanced_status()
        elif args.update_trades:
            print("🔄 Forcing update of all pending trades...")
            updates = bot.update_finished_trades()
            print(f"✅ Updated {updates} trades")
            bot.print_enhanced_status()
        elif args.once:
            bot.run_once()
        elif args.continuous:
            bot.run_continuous(check_interval_minutes=args.interval)
        else:
            print("Enhanced Valorant Paper Trading Bot with PandaScore + Local API Integration")
            print("Usage:")
            print("  --once           Run once and exit")
            print("  --continuous     Run continuously")
            print("  --status         Show status")
            print("  --update-trades  Force update all pending trades")
            print("  --bankroll 500   Set starting bankroll")
            print("  --interval 30    Set check interval (minutes)")
            print("  --local-api URL  Set local API URL (default: http://localhost:5000/api/v1)")
            print("\nExample:")
            print("  python paper.py --continuous --interval 300 --local-api http://localhost:5000/api/v1")
            print("  python paper.py --update-trades  # Force check for finished matches")
            print("\nThe bot will:")
            print("  1. Get upcoming matches from PandaScore")
            print("  2. Map team names to your local API teams")
            print("  3. Use local team names for predictions")
            print("  4. Use PandaScore odds for paper trading")
            print("  5. Automatically update trades when matches finish")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())