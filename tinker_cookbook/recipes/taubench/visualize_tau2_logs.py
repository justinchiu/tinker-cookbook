#!/usr/bin/env python3
"""
Streamlit app to visualize tau2 RL training logs.

Usage:
    streamlit run visualize_tau2_logs.py -- /path/to/tau2.log
"""

import streamlit as st
import pandas as pd
import re
import sys
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import argparse

st.set_page_config(page_title="Tau2 Log Analyzer", layout="wide")

def parse_log_line(line):
    """Parse a single log line."""
    # Pattern: 2025-11-20 11:07:47.734 | INFO | tau2.gym.gym_agent:_log:626 - [[scenario]] message
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| (\w+)\s+\| ([^-]+) - \[\[([^\]]+)\]\] (.*)'
    match = re.match(pattern, line)
    if match:
        timestamp_str, level, location, scenario, message = match.groups()
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        return {
            'timestamp': timestamp,
            'level': level.strip(),
            'location': location.strip(),
            'scenario': scenario,
            'message': message
        }
    return None

def load_tau2_log(filepath):
    """Load and parse tau2 log file."""
    episodes = defaultdict(list)
    all_events = []

    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                all_events.append(parsed)
                episodes[parsed['scenario']].append(parsed)

    return all_events, episodes

def extract_test_train_split(log_dir):
    """Extract test/train task IDs from logs.log if available."""
    test_tasks = []
    train_tasks = []

    logs_file = log_dir.replace('tau2.log', 'logs.log')
    if os.path.exists(logs_file):
        with open(logs_file, 'r') as f:
            in_dataset_section = False
            for line in f:
                if 'TAU2 DATASET SPLIT' in line:
                    in_dataset_section = True
                elif in_dataset_section:
                    if 'TEST:' in line:
                        task_id = line.split('TEST:')[1].strip()
                        test_tasks.append(task_id)
                    elif 'TRAIN:' in line:
                        task_id = line.split('TRAIN:')[1].strip()
                        train_tasks.append(task_id)
                    elif '=====' in line and in_dataset_section:
                        break

    return test_tasks, train_tasks

def analyze_episodes(episodes):
    """Analyze episode data."""
    stats = []

    for scenario, events in episodes.items():
        # Find start and end events
        starts = [e for e in events if 'Starting orchestrator' in e['message']]
        completions = [e for e in events if 'Simulation done:' in e['message']]

        for completion in completions:
            success = 'True' in completion['message']
            stats.append({
                'scenario': scenario,
                'completed_at': completion['timestamp'],
                'success': success,
                'num_events': len(events)
            })

        # Track incomplete episodes
        if len(starts) > len(completions):
            for i in range(len(completions), len(starts)):
                stats.append({
                    'scenario': scenario,
                    'completed_at': None,
                    'success': None,  # Didn't complete
                    'num_events': len(events)
                })

    return pd.DataFrame(stats)

def main():
    st.title("🤖 Tau2 RL Training Log Analyzer")

    # File selection
    log_file = st.sidebar.text_input(
        "Log file path",
        value="/tmp/tinker-examples/tau2-rl/20251120-110707/tau2.log"
    )

    if not log_file:
        st.warning("Please enter a log file path")
        return

    try:
        with st.spinner("Loading log file..."):
            all_events, episodes = load_tau2_log(log_file)
            df_episodes = analyze_episodes(episodes)
            test_tasks, train_tasks = extract_test_train_split(log_file)
    except FileNotFoundError:
        st.error(f"File not found: {log_file}")
        return
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # Summary stats
    st.header("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_scenarios = len(episodes)
        st.metric("Unique Scenarios", total_scenarios)

    with col2:
        total_episodes = len(df_episodes)
        st.metric("Total Episode Attempts", total_episodes)

    with col3:
        completed = df_episodes['success'].notna().sum()
        st.metric("Completed Episodes", f"{completed} ({completed/total_episodes*100:.1f}%)")

    with col4:
        successful = df_episodes['success'].sum()
        if completed > 0:
            st.metric("Success Rate", f"{successful}/{completed} ({successful/completed*100:.1f}%)")
        else:
            st.metric("Success Rate", "N/A")

    # Test/Train Split Display
    if test_tasks or train_tasks:
        st.header("🔍 Test/Train Task Split")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Test Tasks ({len(test_tasks)})")
            if test_tasks:
                for task in test_tasks:
                    st.write(f"• {task}")
            else:
                st.info("No test tasks found in logs")

        with col2:
            st.subheader(f"Train Tasks ({len(train_tasks)})")
            if train_tasks:
                for task in train_tasks[:5]:
                    st.write(f"• {task}")
                if len(train_tasks) > 5:
                    st.write(f"... and {len(train_tasks) - 5} more")
            else:
                st.info("No train tasks found in logs")

    # Task-specific Analysis
    st.header("🎯 Task-Specific Analysis")

    # Create task dropdown
    all_task_ids = list(episodes.keys())
    selected_task = st.selectbox(
        "Select a task ID to analyze",
        options=['All'] + sorted(all_task_ids),
        key='task_selector'
    )

    if selected_task != 'All':
        # Mark if it's test or train
        task_type = "Unknown"
        if selected_task in test_tasks:
            task_type = "TEST"
        elif selected_task in train_tasks:
            task_type = "TRAIN"

        st.write(f"**Task Type:** {task_type}")

        # Get all events for this task
        task_events = episodes[selected_task]

        # Count attempts
        start_events = [e for e in task_events if 'Starting orchestrator' in e['message']]
        completion_events = [e for e in task_events if 'Simulation done:' in e['message']]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Attempts", len(start_events))
        with col2:
            st.metric("Completions", len(completion_events))
        with col3:
            success_count = sum(1 for e in completion_events if 'True' in e['message'])
            st.metric("Successes", success_count)

        # Show timeline for this task
        task_timeline = []
        for i, start in enumerate(start_events):
            end_time = None
            success = None
            if i < len(completion_events):
                end_time = completion_events[i]['timestamp']
                success = 'True' in completion_events[i]['message']

            task_timeline.append({
                'attempt': f"Attempt {i+1}",
                'start': start['timestamp'],
                'end': end_time if end_time else start['timestamp'],
                'success': 'Success' if success else ('Failed' if success is False else 'Incomplete'),
                'duration_seconds': (end_time - start['timestamp']).total_seconds() if end_time else None
            })

        df_task_timeline = pd.DataFrame(task_timeline)

        if not df_task_timeline.empty:
            # Show attempts over time
            fig = px.timeline(
                df_task_timeline.head(50),  # Limit to 50 for readability
                x_start="start",
                x_end="end",
                y="attempt",
                color="success",
                title=f"Timeline of Attempts for {selected_task[:50]}... (First 50)",
                color_discrete_map={'Success': 'green', 'Failed': 'red', 'Incomplete': 'orange'},
                height=max(300, min(800, len(df_task_timeline) * 20))
            )
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)

            # Duration statistics for this task
            if df_task_timeline['duration_seconds'].notna().any():
                st.subheader("Duration Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_dur = df_task_timeline['duration_seconds'].mean()
                    st.metric("Avg Duration", f"{avg_dur:.1f}s" if not pd.isna(avg_dur) else "N/A")
                with col2:
                    med_dur = df_task_timeline['duration_seconds'].median()
                    st.metric("Median Duration", f"{med_dur:.1f}s" if not pd.isna(med_dur) else "N/A")
                with col3:
                    max_dur = df_task_timeline['duration_seconds'].max()
                    st.metric("Max Duration", f"{max_dur:.1f}s" if not pd.isna(max_dur) else "N/A")

    # Timeline visualization
    st.header("📈 Episode Timeline")

    # Create timeline data
    timeline_data = []
    for scenario, events in episodes.items():
        starts = [e for e in events if 'Starting orchestrator' in e['message']]
        ends = [e for e in events if 'Simulation done:' in e['message']]

        for i, start in enumerate(starts):
            end_time = None
            success = None
            if i < len(ends):
                end_time = ends[i]['timestamp']
                success = 'True' in ends[i]['message']

            timeline_data.append({
                'scenario': scenario[:50],  # Truncate long names
                'start': start['timestamp'],
                'end': end_time if end_time else start['timestamp'],
                'success': 'Success' if success else ('Failed' if success is False else 'Incomplete'),
                'duration_seconds': (end_time - start['timestamp']).total_seconds() if end_time else None
            })

    df_timeline = pd.DataFrame(timeline_data)

    if not df_timeline.empty:
        # Gantt chart for episode execution
        fig = px.timeline(
            df_timeline.head(100),  # Limit to first 100 for readability
            x_start="start",
            x_end="end",
            y="scenario",
            color="success",
            title="Episode Execution Timeline (First 100)",
            color_discrete_map={'Success': 'green', 'Failed': 'red', 'Incomplete': 'orange'},
            height=max(400, len(df_timeline['scenario'].unique()) * 20)
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    # Success rate by scenario
    st.header("🎯 Performance by Scenario")

    scenario_stats = df_episodes.groupby('scenario').agg({
        'success': ['count', 'sum', lambda x: x.notna().sum()],
        'num_events': 'mean'
    }).round(2)
    scenario_stats.columns = ['Total Attempts', 'Successes', 'Completions', 'Avg Events']
    scenario_stats['Success Rate'] = (scenario_stats['Successes'] / scenario_stats['Completions']).fillna(0).round(3)
    scenario_stats = scenario_stats.sort_values('Success Rate', ascending=False)

    # Split into successful and failed scenarios
    successful_scenarios = scenario_stats[scenario_stats['Success Rate'] > 0]
    failed_scenarios = scenario_stats[scenario_stats['Success Rate'] == 0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Successful Scenarios")
        if not successful_scenarios.empty:
            st.dataframe(successful_scenarios, height=300)
        else:
            st.info("No successful scenarios")

    with col2:
        st.subheader("❌ Failed Scenarios")
        if not failed_scenarios.empty:
            st.dataframe(failed_scenarios.head(20), height=300)
        else:
            st.info("No completely failed scenarios")

    # Distribution plots
    st.header("📊 Distributions")

    col1, col2 = st.columns(2)

    with col1:
        # Attempts per scenario
        attempts_per_scenario = scenario_stats['Total Attempts'].value_counts().sort_index()
        fig = px.bar(
            x=attempts_per_scenario.index,
            y=attempts_per_scenario.values,
            title="Distribution of Attempts per Scenario",
            labels={'x': 'Number of Attempts', 'y': 'Number of Scenarios'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Success rate distribution
        success_rates = scenario_stats[scenario_stats['Completions'] > 0]['Success Rate']
        fig = px.histogram(
            success_rates,
            nbins=20,
            title="Success Rate Distribution",
            labels={'value': 'Success Rate', 'count': 'Number of Scenarios'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Episode duration analysis
    if df_timeline['duration_seconds'].notna().any():
        st.header("⏱️ Episode Duration Analysis")

        completed_episodes = df_timeline[df_timeline['duration_seconds'].notna()]

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_duration = completed_episodes['duration_seconds'].mean()
            st.metric("Average Duration", f"{avg_duration:.1f}s")

        with col2:
            median_duration = completed_episodes['duration_seconds'].median()
            st.metric("Median Duration", f"{median_duration:.1f}s")

        with col3:
            max_duration = completed_episodes['duration_seconds'].max()
            st.metric("Max Duration", f"{max_duration:.1f}s")

        # Duration by success status
        fig = px.box(
            completed_episodes,
            x='success',
            y='duration_seconds',
            title="Episode Duration by Outcome",
            labels={'duration_seconds': 'Duration (seconds)', 'success': 'Outcome'},
            color='success',
            color_discrete_map={'Success': 'green', 'Failed': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw log viewer
    st.header("📜 Raw Log Viewer")

    scenario_filter = st.selectbox(
        "Filter by scenario",
        options=['All'] + list(episodes.keys()),
        key='scenario_filter'
    )

    message_filter = st.text_input("Filter by message content", key='message_filter')

    # Filter events
    filtered_events = all_events
    if scenario_filter != 'All':
        filtered_events = [e for e in filtered_events if e['scenario'] == scenario_filter]
    if message_filter:
        filtered_events = [e for e in filtered_events if message_filter.lower() in e['message'].lower()]

    # Show filtered events
    st.write(f"Showing {min(100, len(filtered_events))} of {len(filtered_events)} events")

    df_events = pd.DataFrame(filtered_events[:100])
    if not df_events.empty:
        st.dataframe(
            df_events[['timestamp', 'scenario', 'message']],
            use_container_width=True,
            height=400
        )

if __name__ == "__main__":
    main()