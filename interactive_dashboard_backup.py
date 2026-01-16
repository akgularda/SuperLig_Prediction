"""
Interactive Super Lig Prediction Dashboard
Real-time Monte Carlo Simulation with Live Statistics
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import time
import json
from datetime import datetime
import queue
from collections import defaultdict, deque

class RealTimeDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Lig 2025-26 Prediction Dashboard - Real-Time Monte Carlo Simulation")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Simulation control variables
        self.simulation_count = tk.IntVar(value=1000)  # Start with 1000 for immediate feedback
        self.is_running = False
        self.current_simulation = 0
        self.results_queue = queue.Queue()
        
        # Team data - UPDATED to match current standings (attachment)
        # Exact 18 teams shown in the screenshot
        self.teams_data = {
            # Big contenders
            'Galatasaray': {'csr': 2950, 'color': '#FFD700'},
            'Fenerbahçe': {'csr': 2875, 'color': '#1E90FF'},
            'Beşiktaş': {'csr': 2280, 'color': '#000000'},

            # Upper mid / European race
            'Trabzonspor': {'csr': 2165, 'color': '#8B0000'},
            'Başakşehir': {'csr': 2045, 'color': '#FF4500'},
            'Alanyaspor': {'csr': 1980, 'color': '#32CD32'},
            'Konyaspor': {'csr': 1920, 'color': '#228B22'},
            'Antalyaspor': {'csr': 1870, 'color': '#FF1493'},
            'Kasımpaşa': {'csr': 1845, 'color': '#4682B4'},
            'Kayserispor': {'csr': 1770, 'color': '#FF6347'},
            'Göztepe': {'csr': 1745, 'color': '#FF8C00'},

            # Lower half / survival battle
            'Samsunspor': {'csr': 1720, 'color': '#CD5C5C'},
            'Gaziantep FK': {'csr': 1820, 'color': '#800080'},
            'Rizespor': {'csr': 1795, 'color': '#2E8B57'},
            'Gençlerbirliği': {'csr': 1630, 'color': '#C21807'},
            'Eyüpspor': {'csr': 1650, 'color': '#9370DB'},
            # Newly ensured in database per attachment
            'Kocaelispor': {'csr': 1685, 'color': '#006400'},  # green/black
            'Karagümrük': {'csr': 1675, 'color': '#B30000'}    # red/black
        }
        
        # Results storage
        self.championship_results = defaultdict(int)
        self.european_results = defaultdict(int)
        self.relegation_results = defaultdict(int)
        self.position_results = defaultdict(lambda: defaultdict(int))
        self.points_distribution = defaultdict(list)
        
        # Standings tracking
        self.current_standings = {}  # {team: {'points': avg, 'position': avg, 'goal_diff': avg, 'confidence': %}}
        self.standings_history = defaultdict(lambda: defaultdict(list))  # For tracking changes over time
        
        # Real-time data for charts
        self.championship_history = {team: deque(maxlen=100) for team in self.teams_data.keys()}
        self.confidence_intervals = {}
        
        self.setup_ui()
        self.setup_plots()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel
        control_frame = tk.LabelFrame(main_frame, text="Simulation Controls", 
                                    bg='#2a2a2a', fg='white', font=('Arial', 12, 'bold'))
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Simulation count
        tk.Label(control_frame, text="Number of Simulations:", 
                bg='#2a2a2a', fg='white', font=('Arial', 10)).grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        simulation_frame = tk.Frame(control_frame, bg='#2a2a2a')
        simulation_frame.grid(row=0, column=1, padx=10, pady=5)
        
        # Quick buttons for simulation counts
        for i, count in enumerate([1000, 10000, 100000, 1000000]):
            btn = tk.Button(simulation_frame, text=f"{count:,}", 
                          command=lambda c=count: self.simulation_count.set(c),
                          bg='#4a4a4a', fg='white', font=('Arial', 9))
            btn.grid(row=0, column=i, padx=2)
        
        # Custom input
        tk.Label(control_frame, text="Custom:", 
                bg='#2a2a2a', fg='white', font=('Arial', 10)).grid(row=1, column=0, padx=10, pady=5, sticky='w')
        
        self.custom_entry = tk.Entry(control_frame, textvariable=self.simulation_count, 
                                   bg='#4a4a4a', fg='white', font=('Arial', 10), width=15)
        self.custom_entry.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#2a2a2a')
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.start_btn = tk.Button(button_frame, text="Start Simulation", 
                                 command=self.start_simulation, bg='#4CAF50', fg='white', 
                                 font=('Arial', 11, 'bold'), width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="Stop Simulation", 
                                command=self.stop_simulation, bg='#f44336', fg='white', 
                                font=('Arial', 11, 'bold'), width=15, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(button_frame, text="Reset Results", 
                                 command=self.reset_results, bg='#FF9800', fg='white', 
                                 font=('Arial', 11, 'bold'), width=15)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress and Status
        status_frame = tk.LabelFrame(main_frame, text="Simulation Status", 
                                   bg='#2a2a2a', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Real-time progress text
        progress_text_frame = tk.Frame(status_frame, bg='#2a2a2a')
        progress_text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_text = tk.Label(progress_text_frame, text="0 / 0 simulations (0.0%)", 
                                    bg='#2a2a2a', fg='#FFD700', font=('Arial', 10, 'bold'))
        self.progress_text.pack()
        
        # Status labels
        status_info_frame = tk.Frame(status_frame, bg='#2a2a2a')
        status_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_info_frame, text="Ready to start simulation", 
                                   bg='#2a2a2a', fg='#4CAF50', font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(status_info_frame, text="Speed: 0 sim/sec", 
                                  bg='#2a2a2a', fg='white', font=('Arial', 10))
        self.speed_label.pack(side=tk.RIGHT)
        
        # ETA label
        self.eta_label = tk.Label(status_info_frame, text="ETA: --:--", 
                                bg='#2a2a2a', fg='#888888', font=('Arial', 10))
        self.eta_label.pack(side=tk.RIGHT, padx=(0, 20))
        
        # Real-time championship display
        realtime_frame = tk.LabelFrame(status_frame, text="Live Championship Odds", 
                                     bg='#2a2a2a', fg='white', font=('Arial', 10, 'bold'))
        realtime_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.live_odds_frame = tk.Frame(realtime_frame, bg='#2a2a2a')
        self.live_odds_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Initialize live odds labels
        self.live_odds_labels = {}
        for i, (team, data) in enumerate(list(self.teams_data.items())[:6]):  # Show top 6
            label = tk.Label(self.live_odds_frame, text=f"{team}: 0.0%", 
                           bg='#2a2a2a', fg=data['color'], font=('Arial', 9, 'bold'))
            label.grid(row=i//3, column=i%3, padx=10, pady=2, sticky='w')
            self.live_odds_labels[team] = label
        
        # Results container
        results_frame = tk.Frame(main_frame, bg='#1a1a1a')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for charts
        self.chart_frame = tk.Frame(results_frame, bg='#2a2a2a')
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right panel for statistics
        self.stats_frame = tk.Frame(results_frame, bg='#2a2a2a', width=400)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.stats_frame.pack_propagate(False)
        
        # Setup statistics panel
        self.setup_statistics_panel()
        
    def setup_statistics_panel(self):
        # Statistics header
        stats_header = tk.Label(self.stats_frame, text="Live Statistics", 
                               bg='#2a2a2a', fg='white', font=('Arial', 14, 'bold'))
        stats_header.pack(pady=10)
        
        # Create notebook for different statistics
        self.stats_notebook = ttk.Notebook(self.stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Championship tab
        self.championship_frame = tk.Frame(self.stats_notebook, bg='#3a3a3a')
        self.stats_notebook.add(self.championship_frame, text="Championship")
        
        # European tab
        self.european_frame = tk.Frame(self.stats_notebook, bg='#3a3a3a')
        self.stats_notebook.add(self.european_frame, text="European Spots")
        
        # Relegation tab
        self.relegation_frame = tk.Frame(self.stats_notebook, bg='#3a3a3a')
        self.stats_notebook.add(self.relegation_frame, text="Relegation")
        
        # Standings tab
        self.standings_frame = tk.Frame(self.stats_notebook, bg='#3a3a3a')
        self.stats_notebook.add(self.standings_frame, text="Live Standings")
        
        # Statistics tab
        self.detailed_stats_frame = tk.Frame(self.stats_notebook, bg='#3a3a3a')
        self.stats_notebook.add(self.detailed_stats_frame, text="Detailed Stats")
        
        # Setup individual stat frames
        self.setup_championship_stats()
        self.setup_european_stats()
        self.setup_relegation_stats()
        self.setup_standings_stats()
        self.setup_detailed_stats()
        
    def setup_championship_stats(self):
        # Scrollable frame for championship stats
        canvas = tk.Canvas(self.championship_frame, bg='#3a3a3a')
        scrollbar = ttk.Scrollbar(self.championship_frame, orient="vertical", command=canvas.yview)
        self.championship_content = tk.Frame(canvas, bg='#3a3a3a')
        
        self.championship_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.championship_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.championship_labels = {}
        
    def setup_european_stats(self):
        canvas = tk.Canvas(self.european_frame, bg='#3a3a3a')
        scrollbar = ttk.Scrollbar(self.european_frame, orient="vertical", command=canvas.yview)
        self.european_content = tk.Frame(canvas, bg='#3a3a3a')
        
        self.european_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.european_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.european_labels = {}
        
    def setup_relegation_stats(self):
        canvas = tk.Canvas(self.relegation_frame, bg='#3a3a3a')
        scrollbar = ttk.Scrollbar(self.relegation_frame, orient="vertical", command=canvas.yview)
        self.relegation_content = tk.Frame(canvas, bg='#3a3a3a')
        
        self.relegation_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.relegation_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.relegation_labels = {}
        
    def setup_standings_stats(self):
        # Real-time league table with confidence intervals
        canvas = tk.Canvas(self.standings_frame, bg='#3a3a3a')
        scrollbar = ttk.Scrollbar(self.standings_frame, orient="vertical", command=canvas.yview)
        self.standings_content = tk.Frame(canvas, bg='#3a3a3a')
        
        self.standings_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.standings_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Table header
        header_frame = tk.Frame(self.standings_content, bg='#4a4a4a')
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header labels with fixed widths
        tk.Label(header_frame, text="Pos", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=4).grid(row=0, column=0, padx=2)
        tk.Label(header_frame, text="Team", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=12).grid(row=0, column=1, padx=2, sticky='w')
        tk.Label(header_frame, text="Pts", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=4).grid(row=0, column=2, padx=2)
        tk.Label(header_frame, text="±", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=4).grid(row=0, column=3, padx=2)
        tk.Label(header_frame, text="GD", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=4).grid(row=0, column=4, padx=2)
        tk.Label(header_frame, text="Conf%", bg='#4a4a4a', fg='white', font=('Consolas', 9, 'bold'), width=6).grid(row=0, column=5, padx=2)
        
        # Container for team rows
        self.standings_table_frame = tk.Frame(self.standings_content, bg='#3a3a3a')
        self.standings_table_frame.pack(fill=tk.X, padx=5)
        
        self.standings_labels = {}
        
    def setup_detailed_stats(self):
        # Detailed statistics with confidence intervals
        canvas = tk.Canvas(self.detailed_stats_frame, bg='#3a3a3a')
        scrollbar = ttk.Scrollbar(self.detailed_stats_frame, orient="vertical", command=canvas.yview)
        self.detailed_content = tk.Frame(canvas, bg='#3a3a3a')
        
        self.detailed_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.detailed_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Confidence level selector
        conf_frame = tk.Frame(self.detailed_content, bg='#3a3a3a')
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(conf_frame, text="Confidence Level:", 
                bg='#3a3a3a', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        self.confidence_level = tk.DoubleVar(value=95.0)
        conf_scale = tk.Scale(conf_frame, from_=90, to=99.9, resolution=0.1, 
                            orient=tk.HORIZONTAL, variable=self.confidence_level,
                            bg='#4a4a4a', fg='white', highlightbackground='#3a3a3a')
        conf_scale.pack(side=tk.RIGHT)
        
        self.detailed_labels = {}
        
    def setup_plots(self):
        # Create matplotlib figures
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('#2a2a2a')
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#3a3a3a')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        # Championship probability real-time chart
        self.ax1.set_title('Championship Probabilities (Real-time)', color='white', fontweight='bold')
        self.ax1.set_xlabel('Simulation Progress (%)', color='white')
        self.ax1.set_ylabel('Probability (%)', color='white')
        
        # Final position distribution
        self.ax2.set_title('Final Position Distribution', color='white', fontweight='bold')
        self.ax2.set_xlabel('Position', color='white')
        self.ax2.set_ylabel('Frequency', color='white')
        
        # Points distribution
        self.ax3.set_title('Points Distribution (Top 6)', color='white', fontweight='bold')
        self.ax3.set_xlabel('Points', color='white')
        self.ax3.set_ylabel('Frequency', color='white')
        
        # Confidence intervals
        self.ax4.set_title('Championship Confidence Intervals', color='white', fontweight='bold')
        self.ax4.set_xlabel('Teams', color='white')
        self.ax4.set_ylabel('Probability (%)', color='white')
        
        # Embed plots in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text to show charts are ready
        self.ax1.text(0.5, 0.5, 'Real-time championship\nprobabilities will appear here', 
                     transform=self.ax1.transAxes, ha='center', va='center', 
                     color='white', fontsize=10, alpha=0.6)
        self.ax2.text(0.5, 0.5, 'Position distribution\nwill appear here', 
                     transform=self.ax2.transAxes, ha='center', va='center', 
                     color='white', fontsize=10, alpha=0.6)
        self.ax3.text(0.5, 0.5, 'Points distribution\nwill appear here', 
                     transform=self.ax3.transAxes, ha='center', va='center', 
                     color='white', fontsize=10, alpha=0.6)
        self.ax4.text(0.5, 0.5, 'Confidence intervals\nwill appear here', 
                     transform=self.ax4.transAxes, ha='center', va='center', 
                     color='white', fontsize=10, alpha=0.6)
        
        plt.tight_layout()
        
    def calculate_match_probability(self, home_csr, away_csr):
        """Calculate match probabilities using CSR system"""
        csr_diff = home_csr - away_csr + 85  # Home advantage
        
        # Draw probability decreases with strength difference
        draw_prob = 0.21 * np.exp(-abs(csr_diff) / 450)
        
        # Win probability using logistic function
        raw_home_prob = 1 / (1 + np.exp(-csr_diff / 320))
        home_prob = raw_home_prob * (1 - draw_prob)
        away_prob = (1 - raw_home_prob) * (1 - draw_prob)
        
        # Apply dominance reduction
        if abs(csr_diff) > 450:
            home_prob *= 0.52 if csr_diff > 0 else 1.0
            away_prob *= 0.52 if csr_diff < 0 else 1.0
        elif abs(csr_diff) > 250:
            home_prob *= 0.68 if csr_diff > 0 else 1.0
            away_prob *= 0.68 if csr_diff < 0 else 1.0
        elif abs(csr_diff) > 120:
            home_prob *= 0.82 if csr_diff > 0 else 1.0
            away_prob *= 0.82 if csr_diff < 0 else 1.0
        
        # Normalize probabilities
        total = home_prob + draw_prob + away_prob
        return home_prob/total, draw_prob/total, away_prob/total
        
    def simulate_season(self):
        """Simulate a complete season"""
        teams = list(self.teams_data.keys())
        
        # Initialize points and stats
        points = {team: 0 for team in teams}
        wins = {team: 0 for team in teams}
        draws = {team: 0 for team in teams}
        losses = {team: 0 for team in teams}
        
        # Simulate all matches (each team plays each other twice)
        for home_team in teams:
            for away_team in teams:
                if home_team != away_team:
                    home_csr = self.teams_data[home_team]['csr']
                    away_csr = self.teams_data[away_team]['csr']
                    
                    # Add some variance
                    variance_factor = np.random.normal(1, 0.15)
                    home_csr *= variance_factor
                    away_csr *= variance_factor
                    
                    home_prob, draw_prob, away_prob = self.calculate_match_probability(home_csr, away_csr)
                    
                    # Determine outcome
                    rand = np.random.random()
                    if rand < home_prob:
                        # Home win
                        points[home_team] += 3
                        wins[home_team] += 1
                        losses[away_team] += 1
                    elif rand < home_prob + draw_prob:
                        # Draw
                        points[home_team] += 1
                        points[away_team] += 1
                        draws[home_team] += 1
                        draws[away_team] += 1
                    else:
                        # Away win
                        points[away_team] += 3
                        wins[away_team] += 1
                        losses[home_team] += 1
        
        # Sort teams by points (and other tiebreakers if needed)
        final_table = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        return final_table, points, wins, draws, losses
        
    def run_simulation(self):
        """Main simulation loop running in separate thread"""
        total_sims = self.simulation_count.get()
        start_time = time.time()
        last_update_time = start_time
        
        # Force more frequent updates for better visual feedback
        update_interval = min(10, max(1, total_sims // 500))  # Update more frequently
        
        batch_size = 5  # Smaller batches for more responsive updates
        
        for sim in range(total_sims):
            if not self.is_running:
                break
                
            # Run one simulation
            final_table, points, wins, draws, losses = self.simulate_season()
            
            # Record results
            champion = final_table[0][0]
            self.championship_results[champion] += 1
            
            # European spots (top 5)
            for i in range(min(5, len(final_table))):
                team = final_table[i][0]
                self.european_results[team] += 1
            
            # Relegation (bottom 3)
            for i in range(max(0, len(final_table) - 3), len(final_table)):
                team = final_table[i][0]
                self.relegation_results[team] += 1
            
            # Position tracking
            for pos, (team, pts) in enumerate(final_table, 1):
                self.position_results[team][pos] += 1
                self.points_distribution[team].append(pts)
            
            # Track standings data for real-time display
            for pos, (team, pts) in enumerate(final_table, 1):
                if team not in self.current_standings:
                    self.current_standings[team] = {
                        'points': [], 'position': [], 'goal_diff': [], 'wins': [], 'draws': [], 'losses': []
                    }
                
                self.current_standings[team]['points'].append(pts)
                self.current_standings[team]['position'].append(pos)
                self.current_standings[team]['wins'].append(wins[team])
                self.current_standings[team]['draws'].append(draws[team])
                self.current_standings[team]['losses'].append(losses[team])
                
                # Calculate goal difference (simplified for demo)
                goal_diff = wins[team] * 2 - losses[team] * 1  # Rough approximation
                self.current_standings[team]['goal_diff'].append(goal_diff)
            
            self.current_simulation = sim + 1
            
            # More frequent updates for better visual feedback
            current_time = time.time()
            time_since_update = current_time - last_update_time
            
            if (sim % update_interval == 0 or 
                time_since_update >= 0.03 or  # Update every 30ms for smoother updates
                sim == total_sims - 1):
                
                speed = sim / (current_time - start_time) if current_time > start_time else 0
                
                # Send update to main thread
                self.results_queue.put({
                    'simulation': sim + 1,
                    'total': total_sims,
                    'speed': speed,
                    'championship': dict(self.championship_results),
                    'european': dict(self.european_results),
                    'relegation': dict(self.relegation_results),
                    'positions': dict(self.position_results),
                    'points': dict(self.points_distribution),
                    'standings': dict(self.current_standings)
                })
                
                last_update_time = current_time
                
                # Small sleep to prevent overwhelming the UI thread
                time.sleep(0.002)  # 2ms sleep for better UI responsiveness
        
        # Signal completion
        self.results_queue.put({'completed': True})
        
    def start_simulation(self):
        """Start the simulation"""
        if self.is_running:
            return
            
        self.is_running = True
        self.current_simulation = 0
        
        # Reset results
        self.championship_results.clear()
        self.european_results.clear()
        self.relegation_results.clear()
        self.position_results.clear()
        self.points_distribution.clear()
        self.current_standings.clear()
        
        # Update UI
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Simulation running...", fg='#FF9800')
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.sim_thread.start()
        
        # Start UI update timer
        self.update_ui()
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Simulation stopped", fg='#f44336')
        
    def reset_results(self):
        """Reset all results"""
        if self.is_running:
            self.stop_simulation()
            
        self.championship_results.clear()
        self.european_results.clear()
        self.relegation_results.clear()
        self.position_results.clear()
        self.points_distribution.clear()
        self.current_standings.clear()
        
        # Clear championship history
        for team in self.championship_history:
            self.championship_history[team].clear()
            
        # Clear UI
        self.progress_var.set(0)
        self.status_label.config(text="Results reset", fg='#4CAF50')
        self.speed_label.config(text="Speed: 0 sim/sec")
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        self.setup_plots()
        self.canvas.draw()
        
        # Clear statistics
        self.update_statistics()
        
    def update_ui(self):
        """Update UI with latest results"""
        updates_processed = 0
        max_updates_per_cycle = 10  # Limit updates per cycle to prevent overwhelming
        
        try:
            while updates_processed < max_updates_per_cycle:
                try:
                    result = self.results_queue.get_nowait()
                    updates_processed += 1
                    
                    if 'completed' in result:
                        self.is_running = False
                        self.start_btn.config(state='normal')
                        self.stop_btn.config(state='disabled')
                        self.status_label.config(text="Simulation completed!", fg='#4CAF50')
                        break
                        
                    # Update progress
                    progress = (result['simulation'] / result['total']) * 100
                    self.progress_var.set(progress)
                    
                    # Update detailed progress text with current champion
                    current_leader = max(result['championship'].items(), key=lambda x: x[1]) if result['championship'] else ('Unknown', 0)
                    leader_prob = (current_leader[1] / result['simulation']) * 100 if result['simulation'] > 0 else 0
                    
                    self.progress_text.config(text=f"{result['simulation']:,} / {result['total']:,} simulations ({progress:.1f}%)")
                    self.status_label.config(text=f"Running... Current leader: {current_leader[0]} ({leader_prob:.1f}%)")
                    self.speed_label.config(text=f"Speed: {result['speed']:.0f} sim/sec")
                    
                    # Calculate and display ETA
                    if result['speed'] > 0:
                        remaining_sims = result['total'] - result['simulation']
                        eta_seconds = remaining_sims / result['speed']
                        if eta_seconds < 60:
                            eta_text = f"ETA: {eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_text = f"ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_text = f"ETA: {eta_seconds/3600:.1f}h"
                        self.eta_label.config(text=eta_text)
                    else:
                        self.eta_label.config(text="ETA: calculating...")
                    
                    # Update live championship odds
                    self.update_live_odds(result['championship'], result['simulation'])
                    
                    # Update plots and statistics
                    self.update_plots(result)
                    self.update_statistics(result)
                    
                    # Force UI refresh
                    self.root.update_idletasks()
                    
                except queue.Empty:
                    break
                    
        except queue.Empty:
            pass
            
        if self.is_running:
            # More frequent UI updates - every 15ms for smoother real-time feel
            self.root.after(15, self.update_ui)
        else:
            # Final update when simulation is complete
            self.canvas.draw()
            
    def update_plots(self, result):
        """Update all plots with new data"""
        if 'championship' not in result:
            return
            
        total_sims = result['simulation']
        championship = result['championship']
        
        # Update championship chart more frequently for real-time feel
        # Update championship probability chart
        self.ax1.clear()
        self.ax1.set_facecolor('#3a3a3a')
        self.ax1.set_title('Championship Probabilities (Real-time)', color='white', fontweight='bold')
        self.ax1.set_xlabel('Simulation Progress', color='white')
        self.ax1.set_ylabel('Probability (%)', color='white')
        
        # Add data to history and plot
        progress = (total_sims / self.simulation_count.get()) * 100
        for team in championship:
            prob = (championship[team] / total_sims) * 100
            self.championship_history[team].append(prob)
            
            if len(self.championship_history[team]) > 1:
                x_data = np.linspace(0, progress, len(self.championship_history[team]))
                self.ax1.plot(x_data, list(self.championship_history[team]), 
                            label=team, color=self.teams_data[team]['color'], linewidth=2)
        
        self.ax1.legend(loc='upper right', fontsize=8)
        self.ax1.tick_params(colors='white')
        for spine in self.ax1.spines.values():
            spine.set_color('white')
        
        # Update position distribution (every 25 simulations for better performance)
        if total_sims % 25 == 0 and 'positions' in result and total_sims > 25:
            self.ax2.clear()
            self.ax2.set_facecolor('#3a3a3a')
            self.ax2.set_title('Final Position Distribution (Top 6)', color='white', fontweight='bold')
            self.ax2.set_xlabel('Position', color='white')
            self.ax2.set_ylabel('Frequency (%)', color='white')
            
            top_teams = sorted(championship.items(), key=lambda x: x[1], reverse=True)[:6]
            positions = list(range(1, 19))  # 18 teams
            
            for i, (team, _) in enumerate(top_teams):
                if team in result['positions']:
                    frequencies = []
                    for pos in positions:
                        freq = result['positions'][team].get(pos, 0) / total_sims * 100
                        frequencies.append(freq)
                    
                    self.ax2.bar([p + i*0.1 for p in positions[:10]], frequencies[:10], 
                               width=0.1, label=team, color=self.teams_data[team]['color'], alpha=0.7)
            
            self.ax2.legend(fontsize=8)
            self.ax2.tick_params(colors='white')
            for spine in self.ax2.spines.values():
                spine.set_color('white')
        
        # Update points distribution (every 25 simulations for better performance)
        if total_sims % 25 == 0 and 'points' in result and total_sims > 25:
            self.ax3.clear()
            self.ax3.set_facecolor('#3a3a3a')
            self.ax3.set_title('Points Distribution (Top 6)', color='white', fontweight='bold')
            self.ax3.set_xlabel('Points', color='white')
            self.ax3.set_ylabel('Density', color='white')
            
            top_teams = sorted(championship.items(), key=lambda x: x[1], reverse=True)[:6]
            
            for team, _ in top_teams:
                if team in result['points'] and len(result['points'][team]) > 10:
                    self.ax3.hist(result['points'][team], bins=20, alpha=0.5, 
                                label=team, color=self.teams_data[team]['color'], density=True)
            
            self.ax3.legend(fontsize=8)
            self.ax3.tick_params(colors='white')
            for spine in self.ax3.spines.values():
                spine.set_color('white')
        
        # Update confidence intervals (every 25 simulations for more responsive updates)
        if total_sims % 25 == 0:
            self.update_confidence_intervals(result)
        
        # Draw canvas to update display
        self.canvas.draw_idle()  # Use draw_idle for better performance
            
    def update_live_odds(self, championship_results, total_sims):
        """Update live championship odds display"""
        if not championship_results or total_sims == 0:
            return
            
        # Calculate current probabilities
        sorted_teams = sorted(championship_results.items(), key=lambda x: x[1], reverse=True)
        
        # Update labels for top teams
        for i, (team, wins) in enumerate(sorted_teams[:6]):
            if team in self.live_odds_labels:
                probability = (wins / total_sims) * 100
                self.live_odds_labels[team].config(text=f"{team}: {probability:.1f}%")
        
        # Hide labels for teams not in top 6
        for team, label in self.live_odds_labels.items():
            if team not in [t[0] for t in sorted_teams[:6]]:
                label.config(text=f"{team}: 0.0%")
        
    def update_confidence_intervals(self, result):
        """Update confidence intervals chart"""
        if 'championship' not in result:
            return
            
        self.ax4.clear()
        self.ax4.set_facecolor('#3a3a3a')
        self.ax4.set_title('Championship Confidence Intervals', color='white', fontweight='bold')
        self.ax4.set_ylabel('Probability (%)', color='white')
        
        total_sims = result['simulation']
        championship = result['championship']
        confidence_level = self.confidence_level.get()
        
        # Calculate confidence intervals using Wilson score interval
        teams = []
        probabilities = []
        lower_bounds = []
        upper_bounds = []
        
        for team, wins in championship.items():
            if wins > 0:
                p = wins / total_sims
                z = 1.96 if confidence_level == 95 else 2.58  # 95% or 99%
                
                # Wilson score interval
                denominator = 1 + z**2/total_sims
                centre = (p + z**2/(2*total_sims)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*total_sims)) / total_sims) / denominator
                
                lower = max(0, centre - margin)
                upper = min(1, centre + margin)
                
                teams.append(team)
                probabilities.append(p * 100)
                lower_bounds.append(lower * 100)
                upper_bounds.append(upper * 100)
        
        if teams:
            # Sort by probability
            sorted_data = sorted(zip(teams, probabilities, lower_bounds, upper_bounds), 
                               key=lambda x: x[1], reverse=True)
            teams, probabilities, lower_bounds, upper_bounds = zip(*sorted_data[:8])
            
            y_pos = np.arange(len(teams))
            
            # Plot bars with error bars
            bars = self.ax4.barh(y_pos, probabilities, 
                               color=[self.teams_data[team]['color'] for team in teams], alpha=0.7)
            
            # Add confidence interval error bars
            errors = [[prob - lower for prob, lower in zip(probabilities, lower_bounds)],
                     [upper - prob for prob, upper in zip(probabilities, upper_bounds)]]
            
            self.ax4.errorbar(probabilities, y_pos, xerr=errors, fmt='none', 
                            ecolor='white', capsize=3, capthick=2)
            
            self.ax4.set_yticks(y_pos)
            self.ax4.set_yticklabels(teams)
            self.ax4.tick_params(colors='white')
            
            # Add percentage labels
            for i, (prob, lower, upper) in enumerate(zip(probabilities, lower_bounds, upper_bounds)):
                self.ax4.text(prob + 1, i, f'{prob:.1f}%\n[{lower:.1f}%-{upper:.1f}%]', 
                            va='center', color='white', fontsize=8)
        
        for spine in self.ax4.spines.values():
            spine.set_color('white')
            
    def update_statistics(self, result=None):
        """Update all statistics panels"""
        if result is None:
            # Clear all statistics
            for widget in self.championship_content.winfo_children():
                widget.destroy()
            for widget in self.european_content.winfo_children():
                widget.destroy()
            for widget in self.relegation_content.winfo_children():
                widget.destroy()
            for widget in self.standings_table_frame.winfo_children():
                widget.destroy()
            for widget in self.detailed_content.winfo_children():
                widget.destroy()
            return
            
        total_sims = result['simulation']
        
        # Update championship statistics
        self.update_championship_panel(result['championship'], total_sims)
        self.update_european_panel(result['european'], total_sims)
        self.update_relegation_panel(result['relegation'], total_sims)
        self.update_standings_panel(result.get('standings', {}), total_sims)
        self.update_detailed_panel(result, total_sims)
        
    def update_championship_panel(self, championship, total_sims):
        """Update championship statistics panel"""
        # Clear existing widgets
        for widget in self.championship_content.winfo_children():
            widget.destroy()
            
        if not championship:
            return
            
        # Header
        header = tk.Label(self.championship_content, text="Championship Probabilities", 
                         bg='#3a3a3a', fg='white', font=('Arial', 12, 'bold'))
        header.pack(pady=10)
        
        # Sort teams by probability
        sorted_teams = sorted(championship.items(), key=lambda x: x[1], reverse=True)
        
        for i, (team, wins) in enumerate(sorted_teams):
            probability = (wins / total_sims) * 100
            
            # Team frame
            team_frame = tk.Frame(self.championship_content, bg='#4a4a4a')
            team_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Rank
            rank_label = tk.Label(team_frame, text=f"#{i+1}", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10, 'bold'), width=3)
            rank_label.pack(side=tk.LEFT, padx=5)
            
            # Team name
            team_label = tk.Label(team_frame, text=team, 
                                bg='#4a4a4a', fg=self.teams_data[team]['color'], 
                                font=('Arial', 10, 'bold'), width=15, anchor='w')
            team_label.pack(side=tk.LEFT, padx=5)
            
            # Probability
            prob_label = tk.Label(team_frame, text=f"{probability:.2f}%", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10), width=8)
            prob_label.pack(side=tk.RIGHT, padx=5)
            
            # Wins count
            wins_label = tk.Label(team_frame, text=f"({wins:,})", 
                                bg='#4a4a4a', fg='#888888', font=('Arial', 9), width=10)
            wins_label.pack(side=tk.RIGHT, padx=5)
            
    def update_european_panel(self, european, total_sims):
        """Update European spots statistics panel"""
        # Clear existing widgets
        for widget in self.european_content.winfo_children():
            widget.destroy()
            
        if not european:
            return
            
        # Header
        header = tk.Label(self.european_content, text="European Competition Qualification", 
                         bg='#3a3a3a', fg='white', font=('Arial', 12, 'bold'))
        header.pack(pady=10)
        
        # Sort teams by probability
        sorted_teams = sorted(european.items(), key=lambda x: x[1], reverse=True)
        
        for i, (team, qualifications) in enumerate(sorted_teams):
            probability = (qualifications / total_sims) * 100
            
            if probability < 0.1:  # Skip very low probabilities
                continue
                
            # Team frame
            team_frame = tk.Frame(self.european_content, bg='#4a4a4a')
            team_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Rank
            rank_label = tk.Label(team_frame, text=f"#{i+1}", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10, 'bold'), width=3)
            rank_label.pack(side=tk.LEFT, padx=5)
            
            # Team name
            team_label = tk.Label(team_frame, text=team, 
                                bg='#4a4a4a', fg=self.teams_data[team]['color'], 
                                font=('Arial', 10, 'bold'), width=15, anchor='w')
            team_label.pack(side=tk.LEFT, padx=5)
            
            # Probability
            prob_label = tk.Label(team_frame, text=f"{probability:.2f}%", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10), width=8)
            prob_label.pack(side=tk.RIGHT, padx=5)
            
            # Qualifications count
            qual_label = tk.Label(team_frame, text=f"({qualifications:,})", 
                                bg='#4a4a4a', fg='#888888', font=('Arial', 9), width=10)
            qual_label.pack(side=tk.RIGHT, padx=5)
            
    def update_relegation_panel(self, relegation, total_sims):
        """Update relegation statistics panel"""
        # Clear existing widgets
        for widget in self.relegation_content.winfo_children():
            widget.destroy()
            
        if not relegation:
            return
            
        # Header
        header = tk.Label(self.relegation_content, text="Relegation Probabilities", 
                         bg='#3a3a3a', fg='white', font=('Arial', 12, 'bold'))
        header.pack(pady=10)
        
        # Sort teams by probability
        sorted_teams = sorted(relegation.items(), key=lambda x: x[1], reverse=True)
        
        for i, (team, relegations) in enumerate(sorted_teams):
            probability = (relegations / total_sims) * 100
            
            if probability < 0.1:  # Skip very low probabilities
                continue
                
            # Team frame
            team_frame = tk.Frame(self.relegation_content, bg='#4a4a4a')
            team_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Rank
            rank_label = tk.Label(team_frame, text=f"#{i+1}", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10, 'bold'), width=3)
            rank_label.pack(side=tk.LEFT, padx=5)
            
            # Team name
            team_label = tk.Label(team_frame, text=team, 
                                bg='#4a4a4a', fg=self.teams_data[team]['color'], 
                                font=('Arial', 10, 'bold'), width=15, anchor='w')
            team_label.pack(side=tk.LEFT, padx=5)
            
            # Probability
            prob_label = tk.Label(team_frame, text=f"{probability:.2f}%", 
                                bg='#4a4a4a', fg='white', font=('Arial', 10), width=8)
            prob_label.pack(side=tk.RIGHT, padx=5)
            
            # Relegations count
            releg_label = tk.Label(team_frame, text=f"({relegations:,})", 
                                  bg='#4a4a4a', fg='#888888', font=('Arial', 9), width=10)
            releg_label.pack(side=tk.RIGHT, padx=5)
            
    def update_standings_panel(self, standings_data, total_sims):
        """Update live standings panel"""
        # Clear existing widgets except header
        for widget in self.standings_table_frame.winfo_children():
            widget.destroy()
            
        if not standings_data or total_sims == 0:
            return
            
        # Calculate average standings for each team
        team_averages = {}
        for team, data in standings_data.items():
            if data['points'] and data['position']:
                avg_points = sum(data['points']) / len(data['points'])
                avg_position = sum(data['position']) / len(data['position'])
                avg_goal_diff = sum(data['goal_diff']) / len(data['goal_diff'])
                avg_wins = sum(data['wins']) / len(data['wins'])
                avg_draws = sum(data['draws']) / len(data['draws'])
                avg_losses = sum(data['losses']) / len(data['losses'])
                
                # Calculate confidence as standard deviation
                import statistics
                if len(data['points']) > 1:
                    points_std = statistics.stdev(data['points'])
                    confidence = max(0, 100 - (points_std * 3))  # Higher std = lower confidence
                else:
                    confidence = 50
                
                team_averages[team] = {
                    'avg_points': avg_points,
                    'avg_position': avg_position,
                    'avg_goal_diff': avg_goal_diff,
                    'confidence': confidence,
                    'games': int(avg_wins + avg_draws + avg_losses)
                }
        
        # Sort teams by average points (descending)
        sorted_teams = sorted(team_averages.items(), key=lambda x: x[1]['avg_points'], reverse=True)
        
        # Display teams
        for pos, (team, stats) in enumerate(sorted_teams, 1):
            # Determine row color based on position
            if pos == 1:
                bg_color = '#2d5a2d'  # Champion - dark green
            elif pos <= 5:
                bg_color = '#2d4a5a'  # European spots - dark blue
            elif pos >= len(sorted_teams) - 2:
                bg_color = '#5a2d2d'  # Relegation - dark red
            else:
                bg_color = '#4a4a4a'  # Mid-table - gray
            
            # Team row frame
            team_frame = tk.Frame(self.standings_table_frame, bg=bg_color)
            team_frame.pack(fill=tk.X, pady=1)
            
            # Position
            pos_label = tk.Label(team_frame, text=f"{pos}", bg=bg_color, fg='white', 
                               font=('Consolas', 9, 'bold'), width=4)
            pos_label.grid(row=0, column=0, padx=2, sticky='w')
            
            # Team name
            team_label = tk.Label(team_frame, text=team[:11], bg=bg_color, 
                                fg=self.teams_data.get(team, {}).get('color', 'white'), 
                                font=('Consolas', 9, 'bold'), width=12, anchor='w')
            team_label.grid(row=0, column=1, padx=2, sticky='w')
            
            # Points
            pts_label = tk.Label(team_frame, text=f"{stats['avg_points']:.1f}", 
                               bg=bg_color, fg='white', font=('Consolas', 9), width=4)
            pts_label.grid(row=0, column=2, padx=2)
            
            # Points variance (±)
            points_range = max(standings_data[team]['points']) - min(standings_data[team]['points']) if len(standings_data[team]['points']) > 1 else 0
            var_label = tk.Label(team_frame, text=f"±{points_range/2:.1f}", 
                               bg=bg_color, fg='#cccccc', font=('Consolas', 9), width=4)
            var_label.grid(row=0, column=3, padx=2)
            
            # Goal difference
            gd_label = tk.Label(team_frame, text=f"{stats['avg_goal_diff']:+.0f}", 
                              bg=bg_color, fg='white', font=('Consolas', 9), width=4)
            gd_label.grid(row=0, column=4, padx=2)
            
            # Confidence percentage
            conf_color = '#4CAF50' if stats['confidence'] > 70 else '#FFA726' if stats['confidence'] > 40 else '#F44336'
            conf_label = tk.Label(team_frame, text=f"{stats['confidence']:.0f}%", 
                                bg=bg_color, fg=conf_color, font=('Consolas', 9, 'bold'), width=6)
            conf_label.grid(row=0, column=5, padx=2)
            
    def update_detailed_panel(self, result, total_sims):
        """Update detailed statistics panel"""
        # Clear existing widgets (except confidence level selector)
        widgets_to_keep = []
        for widget in self.detailed_content.winfo_children():
            if isinstance(widget, tk.Frame) and any(isinstance(child, tk.Scale) for child in widget.winfo_children()):
                widgets_to_keep.append(widget)
            else:
                widget.destroy()
                
        if not result.get('championship'):
            return
            
        # Statistical summary
        stats_header = tk.Label(self.detailed_content, text="Statistical Summary", 
                               bg='#3a3a3a', fg='white', font=('Arial', 12, 'bold'))
        stats_header.pack(pady=10)
        
        # Current simulation info
        info_frame = tk.Frame(self.detailed_content, bg='#4a4a4a')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(info_frame, text=f"Simulations: {total_sims:,}", 
                bg='#4a4a4a', fg='white', font=('Arial', 10)).pack(anchor='w')
        
        # Calculate and display additional statistics
        championship = result['championship']
        
        # Entropy (uncertainty measure)
        if championship:
            probabilities = [wins/total_sims for wins in championship.values()]
            probabilities = [p for p in probabilities if p > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            
            tk.Label(info_frame, text=f"Prediction Entropy: {entropy:.3f} bits", 
                    bg='#4a4a4a', fg='white', font=('Arial', 10)).pack(anchor='w')
            
            # Gini coefficient (inequality measure)
            probabilities.sort()
            n = len(probabilities)
            cumulative = np.cumsum(probabilities)
            gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(probabilities))) / (n * sum(probabilities))
            
            tk.Label(info_frame, text=f"Probability Gini: {gini:.3f}", 
                    bg='#4a4a4a', fg='white', font=('Arial', 10)).pack(anchor='w')
            
        # Top predictions with confidence intervals
        conf_header = tk.Label(self.detailed_content, text="Top Predictions with Confidence Intervals", 
                              bg='#3a3a3a', fg='white', font=('Arial', 11, 'bold'))
        conf_header.pack(pady=(15, 5))
        
        confidence_level = self.confidence_level.get()
        z_score = 1.96 if confidence_level >= 95 else 1.645  # 95% or 90%
        
        sorted_teams = sorted(championship.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for team, wins in sorted_teams:
            if wins == 0:
                continue
                
            p = wins / total_sims
            
            # Wilson score interval
            denominator = 1 + z_score**2/total_sims
            centre = (p + z_score**2/(2*total_sims)) / denominator
            margin = z_score * np.sqrt((p*(1-p) + z_score**2/(4*total_sims)) / total_sims) / denominator
            
            lower = max(0, centre - margin) * 100
            upper = min(1, centre + margin) * 100
            probability = p * 100
            
            # Detailed team frame
            team_detail_frame = tk.Frame(self.detailed_content, bg='#4a4a4a')
            team_detail_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Team info
            tk.Label(team_detail_frame, text=f"{team}: {probability:.2f}%", 
                    bg='#4a4a4a', fg=self.teams_data[team]['color'], 
                    font=('Arial', 10, 'bold')).pack(anchor='w')
            
            tk.Label(team_detail_frame, text=f"  {confidence_level:.1f}% CI: [{lower:.2f}% - {upper:.2f}%]", 
                    bg='#4a4a4a', fg='#cccccc', font=('Arial', 9)).pack(anchor='w')
            
            tk.Label(team_detail_frame, text=f"  Margin of Error: ±{(upper-lower)/2:.2f}%", 
                    bg='#4a4a4a', fg='#888888', font=('Arial', 9)).pack(anchor='w')

def main():
    root = tk.Tk()
    app = RealTimeDashboard(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Dashboard closed by user")

if __name__ == "__main__":
    main()
