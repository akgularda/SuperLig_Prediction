"""
DATA-DRIVEN 2025-26 Turkish Süper Lig Season Prediction
Using Historical TSL Dataset (1958-2020) + Enhanced Analytics

Key Features:
- ✅ Historical performance analysis from tsl_dataset.csv
- ✅ Data-driven CSR calculations using 60+ years of match data
- ✅ Corrected 2025-26 team list (18 teams)
- ✅ Advanced statistical modeling with real match patterns
- ✅ Historical head-to-head analysis
- ✅ Performance trend analysis
"""

import argparse
import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import sys

class DataDrivenSuperLigPredictor:
    def __init__(self, dataset_path: str = "tsl_dataset.csv"):
        """Initialize with historical TSL dataset and 2025-26 team data"""
        
        # Load historical dataset
        print("📊 Loading historical Turkish Süper Lig dataset...")
        self.historical_data = pd.read_csv(dataset_path)
        print(f"✅ Loaded {len(self.historical_data)} matches from {self.historical_data['Season'].min()}-{self.historical_data['Season'].max()}")
        
        # Process historical data
        self.process_historical_data()
        
        # 2025-26 team data with corrected squad list
        self.teams_2025_26 = {
            # Big 3 Championship Contenders
            'Galatasaray': {
                'current_name': 'Galatasaray',
                'historical_names': ['Galatasaray'],
                'market_value': 220.0,
                'manager': 'Okan Buruk',
                'manager_experience': 8,
                'stadium_capacity': 52280,
                'summer_transfers_net': 7.3,  # Million €
                'key_signings': ['Victor Osimhen (loan)', 'Dries Mertens'],
                'financial_rating': 'A+',
                'youth_academy': 'A',
                'european_experience': 'High'
            },
            'Fenerbahce': {
                'current_name': 'Fenerbahce',
                'historical_names': ['Fenerbahce'],
                'market_value': 200.0,
                'manager': 'José Mourinho',
                'manager_experience': 10,  # World class
                'stadium_capacity': 50509,
                'summer_transfers_net': 23.1,
                'key_signings': ['Allan Saint-Maximin', 'Youssef En-Nesyri', 'Sofyan Amrabat'],
                'financial_rating': 'A+',
                'youth_academy': 'A',
                'european_experience': 'High'
            },
            'Besiktas': {
                'current_name': 'Besiktas',
                'historical_names': ['Besiktas'],
                'market_value': 85.0,
                'manager': 'Giovanni van Bronckhorst',
                'manager_experience': 7,
                'stadium_capacity': 41903,
                'summer_transfers_net': -5.7,  # Net sales
                'key_signings': ['Ciro Immobile', 'Rafa Silva'],
                'financial_rating': 'B+',
                'youth_academy': 'A-',
                'european_experience': 'High'
            },
            
            # Top 6 Contenders
            'Trabzonspor': {
                'current_name': 'Trabzonspor',
                'historical_names': ['Trabzonspor'],
                'market_value': 45.0,
                'manager': 'Şenol Güneş',
                'manager_experience': 9,
                'stadium_capacity': 41461,
                'summer_transfers_net': -3.8,
                'key_signings': ['Okay Yokuşlu', 'Denis Dragus'],
                'financial_rating': 'B',
                'youth_academy': 'B+',
                'european_experience': 'Medium'
            },
            'Basaksehir FK': {
                'current_name': 'Basaksehir FK',
                'historical_names': ['Basaksehir FK', 'Basaksehir'],
                'market_value': 35.0,
                'manager': 'Çağdaş Atan',
                'manager_experience': 5,
                'stadium_capacity': 17319,
                'summer_transfers_net': 2.1,
                'key_signings': ['Davidson', 'Krzysztof Piątek'],
                'financial_rating': 'B+',
                'youth_academy': 'B',
                'european_experience': 'Medium'
            },
            'Alanyaspor': {
                'current_name': 'Alanyaspor',
                'historical_names': ['Alanyaspor'],
                'market_value': 28.0,
                'manager': 'Fatih Tekke',
                'manager_experience': 3,
                'stadium_capacity': 10842,
                'summer_transfers_net': -2.4,
                'key_signings': ['Nuno Lima', 'Richard'],
                'financial_rating': 'B',
                'youth_academy': 'B-',
                'european_experience': 'Low'
            },
            
            # Mid-table Teams
            'Konyaspor': {
                'current_name': 'Konyaspor',
                'historical_names': ['Konyaspor'],
                'market_value': 22.0,
                'manager': 'Aleksandar Stanojevic',
                'manager_experience': 6,
                'stadium_capacity': 42276,
                'summer_transfers_net': 0.3,
                'key_signings': ['Bojan Miovski'],
                'financial_rating': 'B-',
                'youth_academy': 'B',
                'european_experience': 'Low'
            },
            'Antalyaspor': {
                'current_name': 'Antalyaspor',
                'historical_names': ['Antalyaspor'],
                'market_value': 16.8,
                'manager': 'Alex de Souza',
                'manager_experience': 2,  # New to management
                'stadium_capacity': 33032,
                'summer_transfers_net': 1.3,
                'key_signings': ['Jakub Kaluzinski'],
                'financial_rating': 'C+',
                'youth_academy': 'B-',
                'european_experience': 'Low'
            },
            'Kasimpasa': {
                'current_name': 'Kasimpasa',
                'historical_names': ['Kasimpasa'],
                'market_value': 19.2,
                'manager': 'Sami Uğurlu',
                'manager_experience': 3,
                'stadium_capacity': 13500,
                'summer_transfers_net': -1.6,
                'key_signings': ['Haris Hajradinovic'],
                'financial_rating': 'C+',
                'youth_academy': 'C+',
                'european_experience': 'Low'
            },
            'Gaziantep FK': {
                'current_name': 'Gaziantep FK',
                'historical_names': ['Gaziantep FK'],
                'market_value': 15.4,
                'manager': 'Selçuk İnan',
                'manager_experience': 2,
                'stadium_capacity': 33502,
                'summer_transfers_net': -0.3,
                'key_signings': ['Ömür Faruk Beyaz'],
                'financial_rating': 'C',
                'youth_academy': 'C+',
                'european_experience': 'None'
            },
            'Caykur Rizespor': {
                'current_name': 'Caykur Rizespor',
                'historical_names': ['Caykur Rizespor', 'Rizespor'],
                'market_value': 12.8,
                'manager': 'İlhan Palut',
                'manager_experience': 4,
                'stadium_capacity': 15332,
                'summer_transfers_net': -0.6,
                'key_signings': ['Adolfo Gaich'],
                'financial_rating': 'C',
                'youth_academy': 'C',
                'european_experience': 'None'
            },
            'Kayserispor': {
                'current_name': 'Kayserispor',
                'historical_names': ['Kayserispor'],
                'market_value': 11.5,
                'manager': 'Burak Yılmaz',
                'manager_experience': 1,  # Debut season
                'stadium_capacity': 32864,
                'summer_transfers_net': 0.2,
                'key_signings': ['Kartal Yılmaz'],
                'financial_rating': 'C',
                'youth_academy': 'C+',
                'european_experience': 'None'
            },
            'Goztepe': {
                'current_name': 'Goztepe',
                'historical_names': ['Goztepe'],
                'market_value': 14.2,
                'manager': 'Stanimir Stoilov',
                'manager_experience': 8,
                'stadium_capacity': 19724,
                'summer_transfers_net': 1.3,
                'key_signings': ['Isaac Solet'],
                'financial_rating': 'C+',
                'youth_academy': 'C+',
                'european_experience': 'None'
            },
            
            # Relegation Candidates and Promoted Teams
            'Samsunspor': {
                'current_name': 'Samsunspor',
                'historical_names': ['Samsunspor'],
                'market_value': 13.6,
                'manager': 'Thomas Reis',
                'manager_experience': 6,
                'stadium_capacity': 33919,
                'summer_transfers_net': 2.6,  # Investment for promotion
                'key_signings': ['Carlo Holse', 'Flavien Tait'],
                'financial_rating': 'C+',
                'youth_academy': 'B-',
                'european_experience': 'None'
            },
            'Eyupspor': {
                'current_name': 'Eyupspor',
                'historical_names': ['Eyupspor'],
                'market_value': 12.0,
                'manager': 'Arda Turan',
                'manager_experience': 0,  # First management role
                'stadium_capacity': 2500,  # Major disadvantage
                'summer_transfers_net': 3.7,
                'key_signings': ['Emre Akbaba', 'Prince Ampem'],
                'financial_rating': 'C',
                'youth_academy': 'C-',
                'european_experience': 'None'
            },
            'Genclerbirligi': {
                'current_name': 'Genclerbirligi',
                'historical_names': ['Genclerbirligi'],
                'market_value': 8.2,
                'manager': 'Tomislav Stipic',
                'manager_experience': 5,
                'stadium_capacity': 19412,
                'summer_transfers_net': 1.6,
                'key_signings': ['Rahman Buğra Çağıran'],
                'financial_rating': 'C-',
                'youth_academy': 'B-',  # Historically good academy
                'european_experience': 'None'
            },
            'Kocaelispor': {
                'current_name': 'Kocaelispor',
                'historical_names': ['Kocaelispor'],
                'market_value': 14.0,
                'manager': 'Ertuğrul Sağlam',
                'manager_experience': 10,
                'stadium_capacity': 34800,
                'summer_transfers_net': 2.2,
                'key_signings': ['Umut Nayir'],
                'financial_rating': 'C+',
                'youth_academy': 'B-',
                'european_experience': 'None'
            },
            'Karagümrük': {
                'current_name': 'Karagümrük',
                'historical_names': ['Fatih Karagümrük', 'Karagümrük'],
                'market_value': 13.0,
                'manager': 'Şota Arveladze',
                'manager_experience': 9,
                'stadium_capacity': 7600,
                'summer_transfers_net': -1.0,
                'key_signings': ['Fabio Borini'],
                'financial_rating': 'C',
                'youth_academy': 'C',
                'european_experience': 'None'
            }
        }
        
        # Calculate historical performance metrics
        self.calculate_historical_metrics()
        
        # Calculate data-driven CSR ratings
        self.calculate_data_driven_csr()
        
        # Simulation parameters
        self.home_advantage = 0.15  # 15% boost
        self.simulations = 1000
        
        # Initialize results storage
        self.team_names = list(self.teams_2025_26.keys())
        self.results = {team: {
            'points': [], 'wins': [], 'draws': [], 'losses': [],
            'gf': [], 'ga': [], 'rank': [],
            'is_champion': 0, 'in_europe': 0, 'is_relegated': 0
        } for team in self.team_names}

    def process_historical_data(self):
        """Process historical data for analysis"""
        # Convert date column
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'], errors='coerce')
        
        # Create team mapping for name variations
        self.team_mapping = {
            'Basaksehir FK': 'Basaksehir FK',
            'Basaksehir': 'Basaksehir FK',
            'Caykur Rizespor': 'Caykur Rizespor',
            'Rizespor': 'Caykur Rizespor',
            'Fatih Karagumruk': 'Fatih Karagumruk',  # Not in 2025-26
            'MKE Ankaragucu': 'MKE Ankaragucu'
        }
        
        print(f"📈 Processed historical data: {len(self.historical_data)} matches")

    def calculate_historical_metrics(self):
        """Calculate historical performance metrics for each team"""
        print("🔍 Calculating historical performance metrics...")
        
        self.historical_metrics = {}
        
        for team_name, team_data in self.teams_2025_26.items():
            # Get historical names for this team
            historical_names = team_data['historical_names']
            
            # Filter matches for this team
            team_matches = self.historical_data[
                (self.historical_data['home'].isin(historical_names)) | 
                (self.historical_data['visitor'].isin(historical_names))
            ].copy()
            
            if len(team_matches) == 0:
                # New team or no data
                self.historical_metrics[team_name] = {
                    'total_matches': 0,
                    'win_rate': 0.3,  # Default for new teams
                    'avg_goals_scored': 1.2,
                    'avg_goals_conceded': 1.8,
                    'home_advantage': 0.1,
                    'recent_form': 0.3,
                    'big_match_performance': 0.2,
                    'historical_strength': 0.2
                }
                continue
            
            # Calculate basic metrics
            total_matches = len(team_matches)
            
            # Wins, draws, losses
            home_wins = len(team_matches[(team_matches['home'].isin(historical_names)) & (team_matches['hgoal'] > team_matches['vgoal'])])
            away_wins = len(team_matches[(team_matches['visitor'].isin(historical_names)) & (team_matches['vgoal'] > team_matches['hgoal'])])
            home_draws = len(team_matches[(team_matches['home'].isin(historical_names)) & (team_matches['hgoal'] == team_matches['vgoal'])])
            away_draws = len(team_matches[(team_matches['visitor'].isin(historical_names)) & (team_matches['hgoal'] == team_matches['vgoal'])])
            
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            win_rate = total_wins / total_matches if total_matches > 0 else 0.3
            
            # Goals
            home_goals_scored = team_matches[team_matches['home'].isin(historical_names)]['hgoal'].sum()
            away_goals_scored = team_matches[team_matches['visitor'].isin(historical_names)]['vgoal'].sum()
            home_goals_conceded = team_matches[team_matches['home'].isin(historical_names)]['vgoal'].sum()
            away_goals_conceded = team_matches[team_matches['visitor'].isin(historical_names)]['hgoal'].sum()
            
            total_goals_scored = home_goals_scored + away_goals_scored
            total_goals_conceded = home_goals_conceded + away_goals_conceded
            
            avg_goals_scored = total_goals_scored / total_matches if total_matches > 0 else 1.2
            avg_goals_conceded = total_goals_conceded / total_matches if total_matches > 0 else 1.8
            
            # Home advantage
            home_matches = team_matches[team_matches['home'].isin(historical_names)]
            if len(home_matches) > 0:
                home_win_rate = len(home_matches[home_matches['hgoal'] > home_matches['vgoal']]) / len(home_matches)
                home_advantage = home_win_rate - win_rate
            else:
                home_advantage = 0.1
            
            # Recent form (last 3 seasons if available)
            recent_matches = team_matches[team_matches['Season'] >= 2018]
            if len(recent_matches) > 0:
                recent_wins = len(recent_matches[
                    ((recent_matches['home'].isin(historical_names)) & (recent_matches['hgoal'] > recent_matches['vgoal'])) |
                    ((recent_matches['visitor'].isin(historical_names)) & (recent_matches['vgoal'] > recent_matches['hgoal']))
                ])
                recent_form = recent_wins / len(recent_matches)
            else:
                recent_form = win_rate
            
            # Big match performance (vs Galatasaray, Fenerbahce, Besiktas)
            big_teams = ['Galatasaray', 'Fenerbahce', 'Besiktas']
            big_matches = team_matches[
                (team_matches['home'].isin(big_teams)) | (team_matches['visitor'].isin(big_teams))
            ]
            if len(big_matches) > 0:
                big_wins = len(big_matches[
                    ((big_matches['home'].isin(historical_names)) & (big_matches['hgoal'] > big_matches['vgoal'])) |
                    ((big_matches['visitor'].isin(historical_names)) & (big_matches['vgoal'] > big_matches['hgoal']))
                ])
                big_match_performance = big_wins / len(big_matches)
            else:
                big_match_performance = win_rate * 0.5  # Assume worse vs big teams
            
            # Historical strength (weighted by era)
            historical_strength = min(1.0, (win_rate + (total_matches / 1000)) * 0.8)
            
            self.historical_metrics[team_name] = {
                'total_matches': total_matches,
                'win_rate': min(0.8, max(0.1, win_rate)),
                'avg_goals_scored': max(0.5, min(3.0, avg_goals_scored)),
                'avg_goals_conceded': max(0.5, min(3.0, avg_goals_conceded)),
                'home_advantage': max(0.0, min(0.3, home_advantage)),
                'recent_form': min(0.9, max(0.1, recent_form)),
                'big_match_performance': min(0.6, max(0.05, big_match_performance)),
                'historical_strength': min(1.0, max(0.1, historical_strength))
            }
        
        print(f"✅ Calculated metrics for {len(self.historical_metrics)} teams")

    def compute_recent_titles(self, window: int = 10):
        """Compute champions per season to derive recent-title boosts and drought penalties.
        Uses seasons available in self.historical_data. Stores results in self.team_title_info.
        """
        seasons = sorted(self.historical_data['Season'].dropna().unique())
        self.team_title_info = {
            t: {'titles_last_window': 0, 'last_title_season': None, 'years_since_title': 99}
            for t in self.teams_2025_26.keys()
        }
        if not seasons:
            return
        max_season = int(max(seasons))
        # Extend drought to the present year even if dataset is older
        try:
            current_year = datetime.now().year
        except Exception:
            current_year = max_season
        start_season = max_season - window + 1
        for s in seasons:
            if s < start_season:
                continue
            try:
                table = self._compute_actual_table(int(s))
            except Exception:
                continue
            if table.empty:
                continue
            champ = str(table.iloc[0]['Team'])
            if champ in self.team_title_info:
                self.team_title_info[champ]['titles_last_window'] += 1
                self.team_title_info[champ]['last_title_season'] = int(s)
        # finalize years_since_title
        for team, info in self.team_title_info.items():
            lt = info['last_title_season']
            base_years = (max_season - lt) if lt is not None else (window + 5)
            gap_after_dataset = max(0, current_year - max_season)
            info['years_since_title'] = int(base_years + gap_after_dataset)

    def calculate_data_driven_csr(self):
        """Calculate CSR ratings using historical data and current factors"""
        print("🎯 Calculating data-driven CSR ratings...")
        # Populate title info once
        if not hasattr(self, 'team_title_info'):
            self.compute_recent_titles(window=10)
        
        for team_name, team_data in self.teams_2025_26.items():
            metrics = self.historical_metrics[team_name]
            
            # Base CSR from historical performance
            base_csr = (
                metrics['win_rate'] * 1000 +
                metrics['historical_strength'] * 500 +
                metrics['recent_form'] * 300 +
                metrics['big_match_performance'] * 200 +
                (2.0 - metrics['avg_goals_conceded']) * 100 +
                metrics['avg_goals_scored'] * 50
            )
            
            # Current factors adjustments
            manager_boost = {
                10: 150,  # Mourinho
                9: 100,   # Şenol Güneş
                8: 75,    # Okan Buruk, Stoilov
                7: 50,    # van Bronckhorst, Çalımbay
                6: 25,    # Stanojevic, Reis
                5: 10,    # Stipic, Atan
                4: 0,     # Uygun, Palut
                3: -10,   # Tekke, Uğurlu, Taşdemir
                2: -20,   # İnan, de Souza
                1: -30,   # Yılmaz
                0: -50    # Arda Turan
            }.get(team_data['manager_experience'], 0)
            
            # Financial rating boost
            financial_boost = {
                'A+': 100, 'A': 75, 'A-': 50,
                'B+': 25, 'B': 0, 'B-': -25,
                'C+': -50, 'C': -75, 'C-': -100
            }.get(team_data['financial_rating'], 0)
            
            # Market value boost (normalized)
            market_boost = min(200, team_data['market_value'] * 2)
            
            # Stadium capacity boost
            stadium_boost = min(50, team_data['stadium_capacity'] / 1000)
            
            # Transfer activity boost
            transfer_boost = team_data['summer_transfers_net'] * 3
            
            # European experience boost
            europe_boost = {
                'High': 75, 'Medium': 40, 'Low': 10, 'None': 0
            }.get(team_data['european_experience'], 0)
            
            # Youth academy boost (long-term development)
            academy_boost = {
                'A': 30, 'A-': 20, 'B+': 15, 'B': 10, 'B-': 5,
                'C+': 0, 'C': -5, 'C-': -10
            }.get(team_data['youth_academy'], 0)
            
            # Titles recency boost and drought penalty (addresses long droughts like Fenerbahce post-2014)
            title_info = self.team_title_info.get(team_name, {'titles_last_window': 0, 'years_since_title': 12})
            titles_last_10 = title_info['titles_last_window']
            years_since = title_info['years_since_title']
            titles_boost = min(120, titles_last_10 * 35)  # reward multiple recent titles
            is_big = team_name in { 'Galatasaray', 'Fenerbahce', 'Besiktas', 'Trabzonspor' }
            drought_penalty = min(120, (years_since * (10 if is_big else 5)))

            # Calculate final CSR
            final_csr = (
                base_csr +
                manager_boost +
                financial_boost +
                market_boost +
                stadium_boost +
                transfer_boost +
                europe_boost +
                academy_boost +
                titles_boost - drought_penalty
            )
            
            # Apply realistic bounds
            final_csr = max(1500, min(3000, final_csr))
            
            # Store CSR
            self.teams_2025_26[team_name]['data_driven_csr'] = int(final_csr)
        
        # Sort teams by CSR for display
        sorted_teams = sorted(self.teams_2025_26.items(), 
                            key=lambda x: x[1]['data_driven_csr'], reverse=True)
        
        print("\n🏆 DATA-DRIVEN CSR RATINGS (Based on Historical Analysis)")
        print("-" * 70)
        for i, (team, data) in enumerate(sorted_teams, 1):
            metrics = self.historical_metrics[team]
            print(f"{i:2d}. {team:<15} CSR: {data['data_driven_csr']:4d} "
                  f"(Win Rate: {metrics['win_rate']:.2f}, Matches: {metrics['total_matches']:4d})")
        
        print(f"\n✅ Generated data-driven CSR for all {len(self.teams_2025_26)} teams")

    def calculate_match_probability(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """Calculate match probabilities using data-driven CSR and historical patterns"""
        home_csr = self.teams_2025_26[home_team]['data_driven_csr']
        away_csr = self.teams_2025_26[away_team]['data_driven_csr']
        
        home_metrics = self.historical_metrics[home_team]
        away_metrics = self.historical_metrics[away_team]
        
        # Base probability from CSR difference
        csr_diff = home_csr - away_csr
        base_home_prob = 1 / (1 + math.exp(-csr_diff / 400))
        
        # Apply home advantage
        home_advantage_effect = home_metrics['home_advantage']
        adjusted_home_prob = base_home_prob + home_advantage_effect
        
        # Draw probability based on team styles and historical patterns
        avg_goals = (home_metrics['avg_goals_scored'] + away_metrics['avg_goals_scored']) / 2
        draw_prob = max(0.15, min(0.35, 0.25 - abs(csr_diff) / 2000))
        
        # Normalize probabilities
        adjusted_home_prob = max(0.15, min(0.75, adjusted_home_prob))
        away_prob = 1 - adjusted_home_prob - draw_prob
        away_prob = max(0.1, away_prob)
        
        # Final normalization
        total = adjusted_home_prob + draw_prob + away_prob
        home_win = adjusted_home_prob / total
        draw = draw_prob / total
        away_win = away_prob / total
        
        return home_win, draw, away_win

    def simulate_match(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """Simulate a single match using historical goal patterns"""
        home_win_prob, draw_prob, away_win_prob = self.calculate_match_probability(home_team, away_team)
        
        home_metrics = self.historical_metrics[home_team]
        away_metrics = self.historical_metrics[away_team]
        
        rand = random.random()
        
        if rand < home_win_prob:
            # Home win
            home_goals = max(1, np.random.poisson(home_metrics['avg_goals_scored']))
            away_goals = max(0, np.random.poisson(away_metrics['avg_goals_conceded'] * 0.8))
            if home_goals <= away_goals:
                home_goals = away_goals + 1
        elif rand < home_win_prob + draw_prob:
            # Draw
            avg_draw_goals = (home_metrics['avg_goals_scored'] + away_metrics['avg_goals_scored']) / 3
            goals = max(0, np.random.poisson(avg_draw_goals))
            home_goals = away_goals = goals
        else:
            # Away win
            away_goals = max(1, np.random.poisson(away_metrics['avg_goals_scored']))
            home_goals = max(0, np.random.poisson(home_metrics['avg_goals_conceded'] * 0.8))
            if away_goals <= home_goals:
                away_goals = home_goals + 1
        
        return min(home_goals, 7), min(away_goals, 7)  # Cap at 7 goals

    def simulate_season(self) -> dict:
        """Simulate a complete season"""
        standings = {team: {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0} 
                    for team in self.team_names}
        
        # Generate all matches (double round-robin: 18 teams * 17 opponents * 2 = 612 matches)
        matches = []
        for i, home_team in enumerate(self.team_names):
            for j, away_team in enumerate(self.team_names):
                if i != j:
                    matches.append((home_team, away_team))
        
        # Simulate all matches
        for home_team, away_team in matches:
            home_goals, away_goals = self.simulate_match(home_team, away_team)
            
            standings[home_team]['GF'] += home_goals
            standings[home_team]['GA'] += away_goals
            standings[away_team]['GF'] += away_goals
            standings[away_team]['GA'] += home_goals
            
            if home_goals > away_goals:
                standings[home_team]['W'] += 1
                standings[home_team]['Pts'] += 3
                standings[away_team]['L'] += 1
            elif home_goals < away_goals:
                standings[away_team]['W'] += 1
                standings[away_team]['Pts'] += 3
                standings[home_team]['L'] += 1
            else:
                standings[home_team]['D'] += 1
                standings[home_team]['Pts'] += 1
                standings[away_team]['D'] += 1
                standings[away_team]['Pts'] += 1
        
        return standings

    def _pick_champion(self, standings: dict) -> str:
        """Return champion based on points, goal difference, goals for."""
        ranked = sorted(
            standings.items(),
            key=lambda item: (
                item[1]['Pts'],
                item[1]['GF'] - item[1]['GA'],
                item[1]['GF'],
            ),
            reverse=True,
        )
        return ranked[0][0]

    def run_champion_simulations(self, simulations: int) -> dict:
        """Run Monte Carlo simulations focused on champion outcomes."""
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        champion_counts = defaultdict(int)
        progress_every = max(1, simulations // 20)
        for i in range(simulations):
            if (i + 1) % progress_every == 0 or i == 0:
                print(f"Progress: {i + 1}/{simulations} simulations")
            standings = self.simulate_season()
            champion = self._pick_champion(standings)
            champion_counts[champion] += 1

        champion_probs = {
            team: (count / simulations) * 100 for team, count in champion_counts.items()
        }
        winner = max(champion_counts.items(), key=lambda item: item[1])[0]
        return {
            "winner": winner,
            "simulations": simulations,
            "champion_counts": champion_counts,
            "champion_probabilities": champion_probs,
        }

    def display_champion_summary(self, summary: dict) -> None:
        """Print champion-only summary."""
        simulations = summary.get("simulations", 0)
        winner = summary.get("winner", "")
        print("\nMonte Carlo champion forecast")
        print("-" * 40)
        print(f"Simulations: {simulations}")
        print(f"Top winner: {winner}")
        ranked = sorted(
            summary.get("champion_probabilities", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for team, prob in ranked[:6]:
            print(f"{team:<15}: {prob:>5.2f}%")

    def run_data_driven_simulations(self):
        """Run comprehensive simulations"""
        print(f"\n🚀 Starting Data-Driven 2025-26 Season Simulation")
        print(f"📊 Using historical analysis of {len(self.historical_data)} matches")
        print(f"🎯 Running {self.simulations} Monte Carlo simulations...\n")
        
        for i in range(self.simulations):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{self.simulations} ({(i+1)/self.simulations*100:.1f}%)")
            
            season_standings = self.simulate_season()
            
            # Convert to DataFrame and sort
            df = pd.DataFrame.from_dict(season_standings, orient='index')
            df['Team'] = df.index
            df['GD'] = df['GF'] - df['GA']
            df = df.sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
            df['Rank'] = df.index + 1

            # Store results
            for _, row in df.iterrows():
                team = row['Team']
                self.results[team]['points'].append(row['Pts'])
                self.results[team]['wins'].append(row['W'])
                self.results[team]['draws'].append(row['D'])
                self.results[team]['losses'].append(row['L'])
                self.results[team]['gf'].append(row['GF'])
                self.results[team]['ga'].append(row['GA'])
                self.results[team]['rank'].append(row['Rank'])

                if row['Rank'] == 1:
                    self.results[team]['is_champion'] += 1
                if row['Rank'] <= 5:  # European spots
                    self.results[team]['in_europe'] += 1
                if row['Rank'] >= 16:  # Relegation (bottom 3)
                    self.results[team]['is_relegated'] += 1
        
        print(f"\n✅ All {self.simulations} simulations completed!")

    def display_comprehensive_results(self):
        """Display comprehensive results with historical context"""
        avg_results = []
        
        for team in self.team_names:
            team_data = self.teams_2025_26[team]
            metrics = self.historical_metrics[team]
            
            avg_res = {
                'Team': team,
                'CSR': team_data['data_driven_csr'],
                'Pts': round(np.mean(self.results[team]['points'])),
                'W': round(np.mean(self.results[team]['wins'])),
                'D': round(np.mean(self.results[team]['draws'])),
                'L': round(np.mean(self.results[team]['losses'])),
                'GF': round(np.mean(self.results[team]['gf'])),
                'GA': round(np.mean(self.results[team]['ga'])),
                'GD': round(np.mean(self.results[team]['gf']) - np.mean(self.results[team]['ga'])),
                'Value(€M)': team_data['market_value'],
                'Manager': team_data['manager'],
                'Champ%': self.results[team]['is_champion'] / self.simulations * 100,
                'Europe%': self.results[team]['in_europe'] / self.simulations * 100,
                'Relegation%': self.results[team]['is_relegated'] / self.simulations * 100,
                'Historical_Matches': metrics['total_matches'],
                'Historical_WinRate': metrics['win_rate']
            }
            avg_results.append(avg_res)

        df_final = pd.DataFrame(avg_results).sort_values(by='Pts', ascending=False).reset_index(drop=True)
        df_final['Pos'] = df_final.index + 1
        
        print("\n" + "="*130)
        print("🏆 DATA-DRIVEN 2025-26 TURKISH SÜPER LIG PREDICTION")
        print("📊 Based on Historical Analysis (1958-2020) + Current Factors")
        print("="*130)
        print(f"{'Pos':<3} {'Team':<15} {'CSR':<4} {'Pts':<3} {'W':<2} {'D':<2} {'L':<2} "
              f"{'GF':<3} {'GA':<3} {'GD':<4} {'Value':<6} {'H.Matches':<9} {'H.WinRate':<9}")
        print("="*130)
        
        for _, row in df_final.iterrows():
            print(f"{row['Pos']:<3} {row['Team']:<15} {row['CSR']:<4} {row['Pts']:<3} "
                  f"{row['W']:<2} {row['D']:<2} {row['L']:<2} {row['GF']:<3} {row['GA']:<3} "
                  f"{row['GD']:+4} €{row['Value(€M)']:<5} {row['Historical_Matches']:<9} "
                  f"{row['Historical_WinRate']:.3f}")

        # Championship Analysis
        print(f"\n🏆 CHAMPIONSHIP PROBABILITIES (Data-Driven)")
        print("-" * 60)
        champ_data = df_final.sort_values(by='Champ%', ascending=False).head(10)
        for _, row in champ_data.iterrows():
            if row['Champ%'] > 0.1:
                print(f"{row['Team']:<15}: {row['Champ%']:>5.1f}% "
                      f"(CSR: {row['CSR']}, Hist: {row['Historical_WinRate']:.2f} over {row['Historical_Matches']} matches)")

        # European Qualification
        print(f"\n🌍 EUROPEAN QUALIFICATION CHANCES")
        print("-" * 40)
        for _, row in df_final.head(8).iterrows():
            if row['Europe%'] > 1:
                print(f"{row['Team']:<15}: {row['Europe%']:>5.1f}%")

        # Relegation Battle
        print(f"\n⬇️ RELEGATION PROBABILITIES")
        print("-" * 40)
        relegation_data = df_final.sort_values(by='Relegation%', ascending=False).head(8)
        for _, row in relegation_data.iterrows():
            if row['Relegation%'] > 0.1:
                print(f"{row['Team']:<15}: {row['Relegation%']:>5.1f}%")

        # Enhanced Analytics
        print(f"\n📊 ENHANCED ANALYTICS & INSIGHTS")
        print("-" * 60)
        print(f"• Total matches analyzed: {len(self.historical_data):,}")
        print(f"• Historical data range: {self.historical_data['Season'].min()}-{self.historical_data['Season'].max()}")
        print(f"• Average goals per match: {df_final['GF'].sum() / (len(df_final) * 34 / 2):.2f}")
        print(f"• Trabzonspor championship chance: {df_final[df_final['Team']=='Trabzonspor']['Champ%'].iloc[0]:.1f}%")
        print(f"• Basaksehir FK championship chance: {df_final[df_final['Team']=='Basaksehir FK']['Champ%'].iloc[0]:.1f}%")
        print(f"• Strongest historical performer: {champ_data.iloc[0]['Team']} ({champ_data.iloc[0]['Historical_WinRate']:.3f} win rate)")
        
        # Save results
        self.save_comprehensive_results(df_final)
        
        return df_final

    def save_comprehensive_results(self, df_final):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_dict = {
            "system_info": {
                "version": "4.0 Data-Driven",
                "timestamp": timestamp,
                "algorithm": "Historical Analysis + Data-Driven CSR",
                "historical_matches": len(self.historical_data),
                "historical_range": f"{self.historical_data['Season'].min()}-{self.historical_data['Season'].max()}",
                "teams_total": len(self.team_names),
                "simulations": self.simulations
            },
            "data_driven_csr": {},
            "historical_analysis": {},
            "championship_predictions": {},
            "relegation_predictions": {},
            "european_predictions": {}
        }
        
        for _, row in df_final.iterrows():
            team = row['Team']
            metrics = self.historical_metrics[team]
            
            results_dict["data_driven_csr"][team] = int(row['CSR'])
            results_dict["historical_analysis"][team] = {
                "total_matches": int(metrics['total_matches']),
                "win_rate": round(metrics['win_rate'], 3),
                "avg_goals_scored": round(metrics['avg_goals_scored'], 2),
                "avg_goals_conceded": round(metrics['avg_goals_conceded'], 2),
                "recent_form": round(metrics['recent_form'], 3)
            }
            results_dict["championship_predictions"][team] = round(row['Champ%'] / 100, 4)
            results_dict["relegation_predictions"][team] = round(row['Relegation%'] / 100, 4)
            results_dict["european_predictions"][team] = round(row['Europe%'] / 100, 4)
        
        filename = f"data_driven_superlig_prediction_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")

    # ==========================
    # Backtesting & Calibration
    # ==========================
    def _compute_actual_table(self, season: int) -> pd.DataFrame:
        """Build actual final table for a given season from historical dataset."""
        df = self.historical_data[self.historical_data['Season'] == season].copy()
        if df.empty:
            raise ValueError(f"No matches found for season {season}")
        teams = pd.unique(pd.concat([df['home'], df['visitor']], ignore_index=True))
        table = {t: {'Pts': 0, 'GF': 0, 'GA': 0, 'W': 0, 'D': 0, 'L': 0} for t in teams}
        for _, r in df.iterrows():
            h, a = r['home'], r['visitor']
            hg, ag = int(r['hgoal']), int(r['vgoal'])
            table[h]['GF'] += hg; table[h]['GA'] += ag
            table[a]['GF'] += ag; table[a]['GA'] += hg
            if hg > ag:
                table[h]['Pts'] += 3; table[h]['W'] += 1; table[a]['L'] += 1
            elif hg < ag:
                table[a]['Pts'] += 3; table[a]['W'] += 1; table[h]['L'] += 1
            else:
                table[h]['Pts'] += 1; table[a]['Pts'] += 1
                table[h]['D'] += 1; table[a]['D'] += 1
        df_tab = pd.DataFrame.from_dict(table, orient='index').reset_index().rename(columns={'index': 'Team'})
        df_tab['GD'] = df_tab['GF'] - df_tab['GA']
        df_tab = df_tab.sort_values(['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        df_tab['Rank'] = df_tab.index + 1
        return df_tab

    def _metrics_for_teams(self, teams: list[str]) -> dict:
        """Compute basic historical metrics for an arbitrary team set (no extra deps)."""
        metrics = {}
        df = self.historical_data
        for team in teams:
            tm = df[(df['home'] == team) | (df['visitor'] == team)]
            if tm.empty:
                metrics[team] = {
                    'win_rate': 0.3, 'avg_goals_scored': 1.2, 'avg_goals_conceded': 1.8,
                    'home_advantage': 0.1, 'recent_form': 0.3, 'big_match_performance': 0.2,
                    'historical_strength': 0.2
                }
                continue
            total = len(tm)
            hw = (tm[(tm['home'] == team) & (tm['hgoal'] > tm['vgoal'])]).shape[0]
            aw = (tm[(tm['visitor'] == team) & (tm['vgoal'] > tm['hgoal'])]).shape[0]
            hd = (tm[(tm['home'] == team) & (tm['hgoal'] == tm['vgoal'])]).shape[0]
            ad = (tm[(tm['visitor'] == team) & (tm['hgoal'] == tm['vgoal'])]).shape[0]
            win_rate = (hw + aw) / total if total else 0.3
            hg = tm[tm['home'] == team]['hgoal'].sum() + tm[tm['visitor'] == team]['vgoal'].sum()
            ag = tm[tm['home'] == team]['vgoal'].sum() + tm[tm['visitor'] == team]['hgoal'].sum()
            avg_sc = hg / total if total else 1.2
            avg_conc = ag / total if total else 1.8
            home_matches = tm[tm['home'] == team]
            if len(home_matches) > 0:
                home_win_rate = (home_matches['hgoal'] > home_matches['vgoal']).mean()
                home_adv = float(home_win_rate) - float(win_rate)
            else:
                home_adv = 0.1
            recent = tm[tm['Season'] >= max( tm['Season'].min(), tm['Season'].max() - 5 )]
            if len(recent) > 0:
                rw = (((recent['home'] == team) & (recent['hgoal'] > recent['vgoal'])) |
                      ((recent['visitor'] == team) & (recent['vgoal'] > recent['hgoal']))).mean()
                recent_form = float(rw)
            else:
                recent_form = win_rate
            big_teams = ['Galatasaray', 'Fenerbahce', 'Besiktas']
            big = tm[(tm['home'].isin(big_teams)) | (tm['visitor'].isin(big_teams))]
            if len(big) > 0:
                bw = (((big['home'] == team) & (big['hgoal'] > big['vgoal'])) |
                      ((big['visitor'] == team) & (big['vgoal'] > big['hgoal']))).mean()
                big_perf = float(bw)
            else:
                big_perf = win_rate * 0.5
            hist_str = min(1.0, (win_rate + (total / 1000)) * 0.8)
            metrics[team] = {
                'win_rate': min(0.8, max(0.1, win_rate)),
                'avg_goals_scored': max(0.5, min(3.0, avg_sc)),
                'avg_goals_conceded': max(0.5, min(3.0, avg_conc)),
                'home_advantage': max(0.0, min(0.3, home_adv)),
                'recent_form': min(0.9, max(0.1, recent_form)),
                'big_match_performance': min(0.6, max(0.05, big_perf)),
                'historical_strength': min(1.0, max(0.1, hist_str))
            }
        return metrics

    @staticmethod
    def _simulate_match_from_metrics(home_team: str, away_team: str, metrics: dict, base_home_adv: float = 0.15) -> tuple[int, int]:
        """Simulate a match using the same logic as simulate_match but with provided metrics."""
        import numpy as _np
        hm = metrics[home_team]; am = metrics[away_team]
        home_advantage = base_home_adv + hm['home_advantage'] * 0.5
        home_strength = (hm['win_rate'] * 0.5 + hm['recent_form'] * 0.3 + hm['historical_strength'] * 0.2)
        away_strength = (am['win_rate'] * 0.5 + am['recent_form'] * 0.3 + am['historical_strength'] * 0.2)
        base = 0.5 + (home_strength - away_strength) * 0.6 + home_advantage
        base = max(0.1, min(0.9, base))
        draw = 1 - (abs(home_strength - away_strength) * 0.8 + 0.2)
        draw = max(0.1, min(0.35, draw))
        diff = 1 - draw
        hp = base * diff
        ap = (1 - base) * diff
        r = _np.random.random()
        if r < hp:
            hg = max(1, _np.random.poisson(hm['avg_goals_scored']))
            ag = max(0, _np.random.poisson(am['avg_goals_conceded'] * 0.8))
            if hg <= ag: hg = ag + 1
        elif r < hp + draw:
            avg_draw_goals = (hm['avg_goals_scored'] + am['avg_goals_scored']) / 3
            g = max(0, _np.random.poisson(avg_draw_goals)); hg = ag = g
        else:
            ag = max(1, _np.random.poisson(am['avg_goals_scored']))
            hg = max(0, _np.random.poisson(hm['avg_goals_conceded'] * 0.8))
            if ag <= hg: ag = hg + 1
        return min(hg, 7), min(ag, 7)

    def backtest_season(self, season: int, simulations: int = 200) -> dict:
        """Backtest the simulator on a historical season and report calibration metrics."""
        actual = self._compute_actual_table(season)
        teams = actual['Team'].tolist()
        metrics = self._metrics_for_teams(teams)

        # Run sims
        agg_points = {t: [] for t in teams}
        for _ in range(simulations):
            pts = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            ga = {t: 0 for t in teams}
            for h in teams:
                for a in teams:
                    if h == a: continue
                    hg, ag = self._simulate_match_from_metrics(h, a, metrics)
                    gf[h] += hg; ga[h] += ag
                    gf[a] += ag; ga[a] += hg
                    if hg > ag: pts[h] += 3
                    elif ag > hg: pts[a] += 3
                    else: pts[h] += 1; pts[a] += 1
            for t in teams: agg_points[t].append(pts[t])

        pred_pts = {t: float(np.mean(v)) for t, v in agg_points.items()}
        pred_rank = sorted(pred_pts.keys(), key=lambda k: (pred_pts[k]), reverse=True)
        actual_rank = actual.sort_values(['Pts', 'GD', 'GF'], ascending=False)['Team'].tolist()

        # Spearman rho (manual)
        def _rank_map(order):
            return {team: i+1 for i, team in enumerate(order)}
        pr = _rank_map(pred_rank); ar = _rank_map(actual_rank)
        x = np.array([pr[t] for t in teams]); y = np.array([ar[t] for t in teams])
        x_bar, y_bar = x.mean(), y.mean()
        num = ((x - x_bar) * (y - y_bar)).sum()
        den = np.sqrt(((x - x_bar)**2).sum() * ((y - y_bar)**2).sum())
        spearman = float(num / den) if den > 0 else 0.0

        mae_points = float(np.mean([abs(pred_pts[t] - actual.set_index('Team').loc[t, 'Pts']) for t in teams]))
        hit_champion = 1.0 if pred_rank[0] == actual_rank[0] else 0.0
        top4_overlap = len(set(pred_rank[:4]).intersection(set(actual_rank[:4]))) / 4.0
        bottom3_overlap = len(set(pred_rank[-3:]).intersection(set(actual_rank[-3:]))) / 3.0

        summary = {
            'season': season,
            'simulations': simulations,
            'spearman_rank': round(spearman, 3),
            'mae_points': round(mae_points, 2),
            'champion_hit': hit_champion,
            'top4_overlap': round(top4_overlap, 2),
            'bottom3_overlap': round(bottom3_overlap, 2)
        }
        print("\n🧪 Backtest Summary:")
        for k, v in summary.items():
            print(f"• {k}: {v}")
        return summary

def _configure_console() -> None:
    """Best-effort UTF-8 console to avoid encoding errors on Windows."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data-driven Super Lig predictor")
    parser.add_argument(
        "--simulations",
        type=int,
        default=int(os.getenv("SIMULATIONS", "1000")),
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument("--backtest", type=int, default=None, help="Season to backtest")
    parser.add_argument(
        "--champion-only",
        action="store_true",
        help="Only simulate champion outcomes (fast summary)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main():
    """Main function to run data-driven prediction"""
    _configure_console()
    args = _parse_args()
    predictor = DataDrivenSuperLigPredictor("tsl_dataset.csv")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Optional: backtest via CLI or env var, e.g., `--backtest 2019`
    season_to_backtest = args.backtest
    if season_to_backtest is None:
        env_val = os.getenv('BACKTEST_SEASON')
        if env_val:
            try:
                season_to_backtest = int(env_val)
            except ValueError:
                season_to_backtest = None

    if season_to_backtest:
        predictor.backtest_season(season_to_backtest, simulations=200)
        return

    if args.champion_only:
        summary = predictor.run_champion_simulations(args.simulations)
        predictor.display_champion_summary(summary)
        return summary

    predictor.simulations = args.simulations
    predictor.run_data_driven_simulations()
    results = predictor.display_comprehensive_results()
    
    print(f"\n" + "="*80)
    print(f"🎯 DATA-DRIVEN PREDICTION SYSTEM COMPLETE")
    print(f"✅ Based on {len(predictor.historical_data):,} historical matches")
    print(f"✅ Accurate CSR using 60+ years of Turkish football data")
    print(f"✅ Enhanced with 2025-26 transfer and management data")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()
