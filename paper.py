#!/usr/bin/env python3
"""
Valorant Paper Trading Bot - Enhanced Version with Optimized Team Mapping
Complete implementation with PandaScore + Local API Integration
"""

import os
import sys
import json
import time
import logging
import requests
import schedule
import re
import difflib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

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
class TeamMapping:
    pandascore_name: str
    local_name: str
    confidence: float
    mapping_type: str  # exact, abbrev, fuzzy, manual, etc.
    last_used: str
    usage_count: int = 0

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
    # Store team mapping for better match updates
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

class EnhancedTeamMapper:
    """Optimized team name mapping with multiple strategies"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("paper_trading_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Core mapping data
        self.verified_mappings: Dict[str, TeamMapping] = {}
        self.rejection_cache: Set[Tuple[str, str]] = set()
        
        # Performance tracking
        self.mapping_stats = {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'manual_mappings': 0,
            'cache_hits': 0,
            'rejections': 0,
            'org_pattern_matches': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load existing data
        self.load_mappings()
        self.init_team_knowledge_base()
        self.load_rejection_cache()
    
    def init_team_knowledge_base(self):
        """Initialize comprehensive team knowledge base"""
        
        # Known team organizations and their various names
        self.team_orgs = {
            # Tier 1 International Teams
            'sentinels': ['sentinels', 'sen', 'sentinels esports', 'sen valorant'],
            'fnatic': ['fnatic', 'fnc', 'fnatic valorant', 'fnatic team'],
            'cloud9': ['cloud9', 'c9', 'cloud9 blue', 'c9 blue', 'cloud 9'],
            'team_liquid': ['team liquid', 'tl', 'liquid', 'team liquid valorant', 'liquid valorant'],
            'geng': ['gen.g', 'geng', 'generation g', 'gen g', 'generation gaming'],
            'drx': ['drx', 'dragon x', 'drx korea', 'drx valorant'],
            'paper_rex': ['paper rex', 'prx', 'paper rex singapore', 'prx valorant'],
            'loud': ['loud', 'loud esports', 'loud brazil', 'loud valorant'],
            'nrg': ['nrg esports', 'nrg', 'optic gaming', 'optic'],
            'acend': ['acend', 'acend team', 'acend esports'],
            'gambit': ['gambit esports', 'gambit', 'm3 champions', 'gambit valorant'],
            
            # EMEA Teams
            'karmine_corp': ['karmine corp', 'kc', 'karmine corp gc', 'karmine', 'karmine corp valorant'],
            'vitality': ['team vitality', 'vitality', 'vit', 'vitality valorant'],
            'g2': ['g2 esports', 'g2', 'g2 valorant', 'g2 team'],
            'guild': ['guild esports', 'guild', 'guild valorant'],
            'fpx': ['funplus phoenix', 'fpx', 'fpx valorant'],
            'navi': ['natus vincere', 'navi', 'na\'vi', 'navi valorant'],
            'fut': ['fut esports', 'fut', 'futbolist'],
            'liquid': ['team liquid', 'liquid', 'tl'],
            'giants': ['giants gaming', 'giants', 'giants valorant'],
            'koi': ['koi', 'koi valorant'],
            'heretics': ['team heretics', 'heretics', 'th'],
            'bbl': ['bbl esports', 'bbl', 'bigboss layf'],
            
            # Americas Teams
            'eg': ['evil geniuses', 'eg', 'evil geniuses valorant'],
            '100t': ['100 thieves', '100t', 'hundred thieves'],
            'c9': ['cloud9', 'c9', 'cloud9 valorant'],
            'sen': ['sentinels', 'sen', 'sentinels valorant'],
            'nrg': ['nrg esports', 'nrg', 'nrg valorant'],
            'furia': ['furia', 'furia esports', 'furia valorant'],
            'mibr': ['mibr', 'made in brazil', 'mibr valorant'],
            'lev': ['leviatan', 'lev', 'leviatan valorant'],
            'kru': ['kru esports', 'kru', 'kru valorant'],
            '2g': ['2game esports', '2g', '2game'],
            
            # APAC Teams
            'prx': ['paper rex', 'prx', 'paper rex valorant'],
            'drx': ['drx', 'dragon x', 'drx valorant'],
            'geng': ['gen.g', 'geng', 'generation g'],
            'rrq': ['rex regum qeon', 'rrq', 'rrq valorant'],
            'ts': ['team secret', 'ts', 'secret valorant'],
            'talon': ['talon esports', 'talon', 'talon valorant'],
            'zeta': ['zeta division', 'zeta', 'zeta valorant'],
            'dfm': ['detonation focusme', 'dfm', 'detonation'],
            
            # China Teams
            'edg': ['edward gaming', 'edg', 'edg valorant'],
            'blg': ['bilibili gaming', 'blg', 'bili bili gaming'],
            'fpx_china': ['fpx', 'funplus phoenix china', 'fpx china'],
            'te': ['trace esports', 'te', 'trace'],
            'wolves': ['wolves esports', 'wolves', 'wolves valorant'],
            
            # Game Changers Teams
            'sen_gc': ['sentinels gc', 'sen gc', 'sentinels game changers'],
            'c9_white': ['cloud9 white', 'c9 white', 'c9w', 'cloud9 gc'],
            'shopify': ['shopify rebellion', 'shopify rebellion gc', 'sr gc', 'shopify gc'],
            'g2_gozen': ['g2 gozen', 'gozen', 'g2 gc'],
            'liquid_brazil_gc': ['liquid brazil gc', 'liquid gc', 'tl gc'],
            'giantx_gc': ['giantx gc', 'giants gc', 'gx gc'],
            'karmine_corp_gc': ['karmine corp gc', 'kc gc', 'karmine gc'],
            'blvkhvnd': ['blvkhvnd', 'blvk hvnd', 'black hand'],
            
            # Academy Teams
            'furia_academy': ['furia academy', 'furia aca', 'furia fem'],
            'mibr_academy': ['mibr academy', 'mibr aca'],
            'tl_academy': ['team liquid academy', 'tl academy', 'liquid academy'],
            'c9_academy': ['cloud9 academy', 'c9 academy'],
            '2game_academy': ['2game academy', '2g academy'],
            
            # Regional/Challengers Teams
            'eternal_fire': ['eternal fire', 'ef', 'eternal fire valorant'],
            'galatasaray': ['galatasaray esports', 'galatasaray', 'gs'],
            'supermassive': ['papara supermassive', 'supermassive', 'sup'],
            'team_nvus': ['team nvus', 'nvus', 'nvus esports'],
            'revenant_xspark': ['revenant xspark', 'rxs', 'revenant'],
            'reckoning_esports': ['reckoning esports', 'rge', 'reckoning'],
            'team_raad': ['team ra\'ad', 'team raad', 'raad'],
            'baam_esports': ['baam esports', 'baam'],
            'gamax_esports': ['gamax esports', 'gamax'],
            'one_more_esports': ['one more esports', 'one more'],
            'twisted_minds': ['twisted minds', 'tm'],
            'fraggerz': ['fraggerz', 'fraggerz esports'],
            'the_ultimates': ['the ultimates', 'ultimates'],
            'rvn': ['rvn', 'rvn esports'],
            'yung': ['yung', 'yung esports'],
            'nobles': ['nobles', 'nobles esports'],
            'team_gb': ['team gb', 'gb'],
            'rafha_esports': ['rafha esports', 'rafha'],
            'gng_esports': ['gng esports', 'gng'],
            '3bl_esports': ['3bl esports', '3bl'],
            'villanarc': ['villianarc', 'villian arc', 'villanarc'],
            'alqadsiah': ['alqadsiah esports', 'alqadsiah'],
            'red_canids': ['red canids', 'red', 'canids'],
            'agropesca_jacare': ['agropesca jacare', 'jacare', 'agropesca'],
            'sagaz': ['sagaz', 'sagaz esports'],
            'los_grandes': ['los grandes', 'grandes'],
            'tbk_esports': ['tbk esports', 'tbk'],
            'corinthians': ['corinthians esports', 'corinthians'],
            'peek': ['peek', 'peek esports'],
            'elevate': ['elevate', 'elevate esports'],
            'stellae_gaming': ['stellae gaming', 'stellae'],
            'diretoria': ['diretoria', 'diretoria esports'],
            'f4tality': ['f4tality', 'fatality']
        }
        
        # Common org suffixes/prefixes to normalize
        self.org_tokens = {
            'suffixes': ['esports', 'gaming', 'team', 'club', 'gg', 'gc', 'academy', 
                        'valorant', 'white', 'blue', 'red', 'black', 'fem', 'aca'],
            'prefixes': ['team', 'clan'],
            'game_changers': ['gc', 'game changers', 'changers']
        }
        
        # Regional indicators
        self.regional_indicators = {
            'korea': ['kr', 'korea', 'korean'],
            'japan': ['jp', 'japan', 'japanese'],
            'na': ['na', 'north america', 'usa', 'us', 'canada'],
            'eu': ['eu', 'europe', 'european'],
            'brazil': ['br', 'brazil', 'brazilian'],
            'latam': ['latam', 'latin america', 'latin'],
            'apac': ['apac', 'asia pacific', 'asia'],
            'mena': ['mena', 'middle east'],
            'china': ['cn', 'china', 'chinese']
        }
        
        # Common abbreviation expansions
        self.abbreviations = {
            'gen g': 'generation g',
            'gen.g': 'generation g', 
            'tl': 'team liquid',
            'fnc': 'fnatic',
            'sen': 'sentinels',
            'c9': 'cloud9',
            'prx': 'paper rex',
            'drx': 'dragon x',
            'kc': 'karmine corp',
            'vit': 'vitality',
            'fpx': 'funplus phoenix',
            'edg': 'edward gaming',
            'eg': 'evil geniuses',
            '100t': '100 thieves',
            'nrg': 'nrg esports',
            'lev': 'leviatan',
            'kru': 'kru esports',
            'rrq': 'rex regum qeon',
            'ts': 'team secret',
            'dfm': 'detonation focusme',
            'blg': 'bilibili gaming',
            'te': 'trace esports'
        }
    
    def load_mappings(self):
        """Load existing mappings with enhanced data structure"""
        try:
            mappings_file = self.data_dir / "enhanced_team_mappings.json"
            if mappings_file.exists():
                with open(mappings_file, 'r') as f:
                    data = json.load(f)
                
                for ps_name, mapping_data in data.items():
                    if isinstance(mapping_data, str):
                        # Convert old format
                        self.verified_mappings[ps_name] = TeamMapping(
                            pandascore_name=ps_name,
                            local_name=mapping_data,
                            confidence=1.0,
                            mapping_type="legacy",
                            last_used="unknown",
                            usage_count=1
                        )
                    else:
                        # New format
                        self.verified_mappings[ps_name] = TeamMapping(**mapping_data)
                
                self.logger.info(f"Loaded {len(self.verified_mappings)} verified team mappings")
        
        except Exception as e:
            self.logger.error(f"Error loading mappings: {e}")
    
    def load_rejection_cache(self):
        """Load rejection cache from file"""
        try:
            rejections_file = self.data_dir / "mapping_rejections.json"
            if rejections_file.exists():
                with open(rejections_file, 'r') as f:
                    rejections_list = json.load(f)
                    self.rejection_cache = set(tuple(pair) for pair in rejections_list)
                self.logger.info(f"Loaded {len(self.rejection_cache)} mapping rejections")
        except Exception as e:
            self.logger.error(f"Error loading rejections: {e}")
    
    def normalize_team_name(self, name: str, aggressive: bool = False) -> str:
        """Enhanced team name normalization with multiple levels"""
        if not name:
            return ""
        
        original = name
        name = name.lower().strip()
        
        # Remove special characters and normalize spacing
        name = re.sub(r'[^\w\s-]', '', name)
        name = ' '.join(name.split())  # Normalize whitespace
        
        # Handle common abbreviation patterns
        name = self.expand_abbreviations(name)
        
        if aggressive:
            # More aggressive normalization for fuzzy matching
            
            # Remove org tokens
            for suffix in self.org_tokens['suffixes']:
                # Remove as whole word
                name = re.sub(rf'\b{re.escape(suffix)}\b', '', name).strip()
            
            for prefix in self.org_tokens['prefixes']:
                # Remove as whole word at start
                name = re.sub(rf'^\b{re.escape(prefix)}\b\s*', '', name).strip()
            
            # Remove regional indicators if they don't help distinguish
            for region, indicators in self.regional_indicators.items():
                for indicator in indicators:
                    name = re.sub(rf'\b{re.escape(indicator)}\b', '', name).strip()
            
            # Handle numbers and special cases
            name = re.sub(r'\b(v2|2\.0|ii)\b', '2', name)
            name = re.sub(r'\bthe\b', '', name)
        
        return ' '.join(name.split())  # Final whitespace cleanup
    
    def expand_abbreviations(self, name: str) -> str:
        """Expand known abbreviations"""
        for abbrev, full in self.abbreviations.items():
            if name == abbrev or name.startswith(f"{abbrev} ") or name.endswith(f" {abbrev}"):
                name = name.replace(abbrev, full)
        
        return name
    
    def calculate_similarity_score(self, name1: str, name2: str) -> Tuple[float, str]:
        """Enhanced similarity calculation with detailed reasoning"""
        
        if not name1 or not name2:
            return 0.0, "empty_input"
        
        # Exact match
        if name1 == name2:
            self.mapping_stats['exact_matches'] += 1
            return 1.0, "exact_match"
        
        # Case-insensitive exact match
        if name1.lower() == name2.lower():
            self.mapping_stats['exact_matches'] += 1
            return 0.95, "case_insensitive_exact"
        
        # Normalize both names
        norm1 = self.normalize_team_name(name1, aggressive=False)
        norm2 = self.normalize_team_name(name2, aggressive=False)
        
        if norm1 == norm2:
            return 0.9, "normalized_exact"
        
        # Check org patterns
        org_score = self.check_org_patterns(name1, name2)
        if org_score > 0.8:
            self.mapping_stats['org_pattern_matches'] += 1
            return org_score, "org_pattern_match"
        
        # Aggressive normalization for fuzzy matching
        agg_norm1 = self.normalize_team_name(name1, aggressive=True)
        agg_norm2 = self.normalize_team_name(name2, aggressive=True)
        
        if agg_norm1 == agg_norm2:
            return 0.8, "aggressive_normalized"
        
        # Substring matching (longer substring = higher score)
        if agg_norm1 in agg_norm2 or agg_norm2 in agg_norm1:
            longer = max(agg_norm1, agg_norm2, key=len)
            shorter = min(agg_norm1, agg_norm2, key=len)
            if len(shorter) >= 3:  # Minimum length for meaningful substring
                return 0.6 + (len(shorter) / len(longer)) * 0.2, "substring_match"
        
        # Word overlap analysis
        words1 = set(agg_norm1.split())
        words2 = set(agg_norm2.split())
        
        if words1 and words2:
            intersection = words1 & words2
            union = words1 | words2
            
            if intersection:
                overlap_ratio = len(intersection) / len(union)
                # Bonus for meaningful words (length > 2)
                meaningful_overlap = sum(1 for word in intersection if len(word) > 2)
                meaningful_total = sum(1 for word in union if len(word) > 2)
                
                if meaningful_total > 0:
                    weighted_ratio = meaningful_overlap / meaningful_total
                    final_score = (overlap_ratio + weighted_ratio) / 2
                    
                    if final_score > 0.5:
                        return min(0.7, final_score), "word_overlap"
        
        # Fuzzy string matching as last resort
        similarity = difflib.SequenceMatcher(None, agg_norm1, agg_norm2).ratio()
        if similarity > 0.6:
            self.mapping_stats['fuzzy_matches'] += 1
            return similarity * 0.6, "fuzzy_match"  # Scale down fuzzy matches
        
        return 0.0, "no_match"
    
    def check_org_patterns(self, name1: str, name2: str) -> float:
        """Check if names belong to same organization using knowledge base"""
        norm1 = self.normalize_team_name(name1, aggressive=True)
        norm2 = self.normalize_team_name(name2, aggressive=True)
        
        for org_key, aliases in self.team_orgs.items():
            norm_aliases = [self.normalize_team_name(alias, aggressive=True) for alias in aliases]
            
            match1 = any(norm1 == alias or norm1 in alias or alias in norm1 for alias in norm_aliases)
            match2 = any(norm2 == alias or norm2 in alias or alias in norm2 for alias in norm_aliases)
            
            if match1 and match2:
                return 0.85  # High confidence for org pattern match
        
        return 0.0
    
    def find_best_match(self, pandascore_name: str, local_teams: List[str], 
                       time_similarity: float = 0.0) -> Optional[Dict]:
        """Find best matching local team with enhanced scoring"""
        
        # Check cache first
        cache_key = pandascore_name.lower()
        if cache_key in self.verified_mappings:
            mapping = self.verified_mappings[cache_key]
            if mapping.local_name in local_teams:
                mapping.usage_count += 1
                mapping.last_used = datetime.now().isoformat()
                self.mapping_stats['cache_hits'] += 1
                return {
                    'local_name': mapping.local_name,
                    'confidence': mapping.confidence,
                    'mapping_type': 'cached',
                    'reasoning': f"Previously verified mapping (used {mapping.usage_count} times)"
                }
        
        # Filter out rejected mappings and TBD teams
        available_teams = []
        for local_team in local_teams:
            if 'tbd' in local_team.lower():
                continue
            if (pandascore_name.lower(), local_team.lower()) in self.rejection_cache:
                continue
            available_teams.append(local_team)
        
        if not available_teams:
            return None
        
        best_match = None
        best_score = 0.0
        
        for local_team in available_teams:
            score, reasoning = self.calculate_similarity_score(pandascore_name, local_team)
            
            # Apply time bonus if provided
            if time_similarity > 0:
                score = score * 0.8 + time_similarity * 0.2
                reasoning += f" + time_bonus({time_similarity:.2f})"
            
            if score > best_score:
                best_score = score
                best_match = {
                    'local_name': local_team,
                    'confidence': score,
                    'mapping_type': reasoning,
                    'reasoning': f"Best similarity match: {reasoning}"
                }
        
        # Only return matches above threshold
        if best_match and best_score >= 0.6:
            return best_match
        
        return None
    
    def save_verified_mapping(self, pandascore_name: str, local_name: str, 
                            confidence: float, mapping_type: str):
        """Save a verified mapping for future use"""
        
        mapping = TeamMapping(
            pandascore_name=pandascore_name,
            local_name=local_name,
            confidence=confidence,
            mapping_type=mapping_type,
            last_used=datetime.now().isoformat(),
            usage_count=1
        )
        
        self.verified_mappings[pandascore_name.lower()] = mapping
        self.save_mappings_to_file()
    
    def save_mappings_to_file(self):
        """Save verified mappings to file"""
        try:
            mappings_file = self.data_dir / "enhanced_team_mappings.json"
            data = {}
            
            for key, mapping in self.verified_mappings.items():
                data[key] = {
                    'pandascore_name': mapping.pandascore_name,
                    'local_name': mapping.local_name,
                    'confidence': mapping.confidence,
                    'mapping_type': mapping.mapping_type,
                    'last_used': mapping.last_used,
                    'usage_count': mapping.usage_count
                }
            
            with open(mappings_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving mappings: {e}")
    
    def save_rejection_cache(self):
        """Save rejection cache to file"""
        try:
            rejections_file = self.data_dir / "mapping_rejections.json"
            rejections_list = list(self.rejection_cache)
            
            with open(rejections_file, 'w') as f:
                json.dump(rejections_list, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving rejections: {e}")
    
    def print_mapping_stats(self):
        """Print mapping statistics and performance metrics"""
        print(f"\n{'='*60}")
        print(f"📊 TEAM MAPPING STATISTICS")
        print(f"{'='*60}")
        print(f"🎯 Verified Mappings: {len(self.verified_mappings)}")
        print(f"🚫 Rejection Cache: {len(self.rejection_cache)}")
        print(f"📈 Performance Stats:")
        for stat, count in self.mapping_stats.items():
            print(f"  {stat.replace('_', ' ').title()}: {count}")
        
        # Show most used mappings
        if self.verified_mappings:
            print(f"\n🏆 Most Used Mappings:")
            sorted_mappings = sorted(
                self.verified_mappings.values(), 
                key=lambda x: x.usage_count, 
                reverse=True
            )[:5]
            
            for mapping in sorted_mappings:
                print(f"  {mapping.pandascore_name} -> {mapping.local_name} ({mapping.usage_count} uses)")

class LocalAPI:
    """Local API client for getting match data and team mappings"""
    
    def __init__(self, base_url: str = "http://localhost:5000/api/v1", data_dir: Path = None):
        self.base_url = base_url
        self.data_dir = data_dir or Path("paper_trading_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize enhanced team mapper
        self.team_mapper = EnhancedTeamMapper(data_dir)
        
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
        """Find matching local match using enhanced team mapping"""
        try:
            # Parse PandaScore time
            ps_time = datetime.fromisoformat(pandascore_time.replace('Z', '+00:00'))
            
            # Get all local teams for mapping
            local_teams = list(self.get_local_team_names())
            
            # Find best matches for both teams
            team1_match = self.team_mapper.find_best_match(pandascore_team1, local_teams)
            team2_match = self.team_mapper.find_best_match(pandascore_team2, local_teams)
            
            if not team1_match or not team2_match:
                return None
            
            # Find local match with these teams
            for local_match in self.local_matches:
                if 'teams' not in local_match or len(local_match['teams']) < 2:
                    continue
                
                local_team1_name = local_match['teams'][0]['name']
                local_team2_name = local_match['teams'][1]['name']
                
                # Check if this match uses our mapped teams (either order)
                if ((local_team1_name == team1_match['local_name'] and 
                     local_team2_name == team2_match['local_name']) or
                    (local_team1_name == team2_match['local_name'] and 
                     local_team2_name == team1_match['local_name'])):
                    
                    # Verify time similarity
                    try:
                        local_time = datetime.fromisoformat(local_match['utc'].replace('Z', '+00:00'))
                        time_diff = abs((ps_time - local_time).total_seconds())
                        
                        if time_diff <= 6 * 3600:  # Within 6 hours
                            # Calculate combined confidence
                            team_confidence = (team1_match['confidence'] + team2_match['confidence']) / 2
                            time_confidence = max(0, 1 - (time_diff / (6 * 3600)))
                            overall_confidence = team_confidence * 0.8 + time_confidence * 0.2
                            
                            # Save verified mappings
                            self.team_mapper.save_verified_mapping(
                                pandascore_team1, team1_match['local_name'], 
                                team1_match['confidence'], team1_match['mapping_type']
                            )
                            self.team_mapper.save_verified_mapping(
                                pandascore_team2, team2_match['local_name'], 
                                team2_match['confidence'], team2_match['mapping_type']
                            )
                            
                            return {
                                'local_match': local_match,
                                'mapping': {
                                    'pandascore_team1': pandascore_team1,
                                    'pandascore_team2': pandascore_team2,
                                    'local_team1': team1_match['local_name'],
                                    'local_team2': team2_match['local_name']
                                },
                                'confidence': overall_confidence,
                                'mapping_details': {
                                    'team1_mapping': team1_match,
                                    'team2_mapping': team2_match,
                                    'time_confidence': time_confidence
                                }
                            }
                    except:
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding local match: {e}")
            return None

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
                    
                else:
                    print(f"❌ No local match found for: {ps_team1} vs {ps_team2}")
                    
            except Exception as e:
                print(f"❌ Error mapping match: {e}")
                continue
        
        print(f"🎯 Successfully mapped {len(mapped_matches)} matches")
        return mapped_matches

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
    """Enhanced paper trading bot with optimized team mapping"""
    
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
        
        # Create formatter
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
        self._log_safe('info', "🔍 Scanning for upcoming matches with optimized team mapping...")
        
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
        
        # Use the enhanced team mapper for consistency
        score, reasoning = self.local_api.team_mapper.calculate_similarity_score(team1, team2)
        return score >= 0.8  # High confidence threshold for match evaluation
    
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
        """Print enhanced trading status with team mapping stats"""
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
        
        # Show team mapping statistics
        if self.local_api and self.local_api.team_mapper:
            print(f"\n🔗 Team Mapping Performance:")
            stats = self.local_api.team_mapper.mapping_stats
            for stat, count in stats.items():
                print(f"  {stat.replace('_', ' ').title()}: {count}")
        
        # Show recent trades
        if self.state.trades:
            print(f"\n📝 Recent Trades:")
            recent_trades = sorted(self.state.trades, key=lambda x: x.timestamp, reverse=True)[:5]
            for trade in recent_trades:
                status_emoji = "✅" if trade.status == "won" else "❌" if trade.status == "lost" else "⏳" if trade.status == "pending" else "🚫"
                profit_str = f"${trade.profit_loss:+.2f}" if trade.status in ['won', 'lost'] else "N/A"
                
                # Show mapping info if available
                mapping_info = ""
                if trade.mapping_confidence and 'mapping_confidence' in trade.mapping_confidence:
                    conf = trade.mapping_confidence['mapping_confidence']
                    mapping_info = f" (map: {conf:.2f})"
                
                print(f"  {status_emoji} {trade.team1} vs {trade.team2}{mapping_info} | {trade.bet_type} | ${trade.bet_amount:.2f} @ {trade.odds:.2f} | {profit_str}")
        
        print(f"\n🕐 Last Update: {self.state.last_update}")
        print(f"{'='*70}\n")
        
        # Print team mapping stats
        if self.local_api and self.local_api.team_mapper:
            self.local_api.team_mapper.print_mapping_stats()
    
    def run_once(self):
        """Run a single iteration of the trading bot"""
        try:
            print(f"🤖 Enhanced Paper Trading Bot with Optimized Team Mapping - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
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
        print(f"🚀 Starting enhanced continuous paper trading with optimized team mapping (checking every {check_interval_minutes} minutes)")
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
    
    parser = argparse.ArgumentParser(description="Enhanced Valorant Paper Trading Bot with Optimized Team Mapping")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Starting bankroll (default: $500)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in minutes (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--data-dir", type=str, default="paper_trading_data", help="Data directory")
    parser.add_argument("--local-api", type=str, default="http://localhost:5000/api/v1", help="Local API URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--update-trades", action="store_true", help="Force update all pending trades")
    parser.add_argument("--mapping-stats", action="store_true", help="Show team mapping statistics")
    parser.add_argument("--add-mapping", nargs=2, metavar=('PANDASCORE_NAME', 'LOCAL_NAME'), 
                       help="Manually add a team mapping")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize bot with optimized team mapping
        bot = EnhancedPaperTradingBot(
            starting_bankroll=args.bankroll, 
            data_dir=args.data_dir,
            local_api_url=args.local_api
        )
        
        if args.add_mapping:
            pandascore_name, local_name = args.add_mapping
            bot.local_api.team_mapper.save_verified_mapping(pandascore_name, local_name, 1.0, "manual")
            print(f"✅ Added manual mapping: {pandascore_name} -> {local_name}")
            
        elif args.mapping_stats:
            bot.local_api.team_mapper.print_mapping_stats()
            
        elif args.status:
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
            print("Enhanced Valorant Paper Trading Bot with Optimized Team Mapping")
            print("Usage:")
            print("  --once                     Run once and exit")
            print("  --continuous               Run continuously")
            print("  --status                   Show status")
            print("  --update-trades            Force update all pending trades")
            print("  --mapping-stats            Show team mapping statistics")
            print("  --add-mapping PS_NAME LOCAL_NAME  Add manual team mapping")
            print("  --bankroll 500             Set starting bankroll")
            print("  --interval 30              Set check interval (minutes)")
            print("  --local-api URL            Set local API URL")
            print("\nExamples:")
            print("  python paper.py --continuous --interval 300")
            print("  python paper.py --update-trades")
            print("  python paper.py --add-mapping 'Sentinels' 'SEN'")
            print("  python paper.py --mapping-stats")
            print("\nNew Features:")
            print("  ✅ Optimized team name mapping with knowledge base")
            print("  ✅ Verified mapping cache with usage tracking")
            print("  ✅ Rejection cache to avoid bad suggestions")
            print("  ✅ Enhanced similarity scoring (exact > org > fuzzy)")
            print("  ✅ Comprehensive team organization database")
            print("  ✅ Performance statistics and monitoring")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())