import streamlit as st
import pandas as pd
import os
import random
import pulp
from collections import defaultdict

###############################################################################
# 1. InterviewScheduler Class (from your script, simplified for demonstration)
###############################################################################

class InterviewScheduler:
    def __init__(self, time_slots, interviewers, capacities, initial_prices=None):
        self.time_slots = time_slots
        self.interviewers = interviewers
        self.capacities = capacities
        
        # Initialize prices - start low if not specified
        self.prices = initial_prices or {slot: 1.0 for slot in time_slots}
        
        # Equal budgets by default
        self.budgets = {interviewer: 1000 for interviewer in interviewers}
        
        # Preferences/utilities for each interviewer (0-100 scale)
        self.utilities = {
            interviewer: {slot: 0 for slot in time_slots}
            for interviewer in interviewers
        }
        
        # Adjustments for consecutive slots (optional; not used in this demo)
        self.consecutive_adjustments = {interviewer: {} for interviewer in interviewers}
        
        # Unavailable slots per interviewer
        self.unavailable_slots = {interviewer: set() for interviewer in interviewers}
        
        # Current allocation
        self.allocation = {interviewer: set() for interviewer in interviewers}
        
        # For tracking convergence
        self.history = []
    
    def set_budget(self, interviewer, budget):
        if interviewer in self.budgets:
            self.budgets[interviewer] = budget
    
    def set_preferences(self, interviewer, preferences):
        if interviewer in self.utilities:
            for slot, utility in preferences.items():
                if slot in self.time_slots:
                    self.utilities[interviewer][slot] = max(0, min(100, utility))
    
    def set_unavailable(self, interviewer, slots):
        if interviewer in self.unavailable_slots:
            self.unavailable_slots[interviewer] = set(s for s in slots if s in self.time_slots)
            # Set utility to 0 for unavailable
            for slot in self.unavailable_slots[interviewer]:
                self.utilities[interviewer][slot] = 0
    
    def calculate_demand(self, interviewer):
        if all(self.utilities[interviewer][slot] == 0 for slot in self.time_slots):
            return set()
        
        # Mixed Integer Linear Program for knapsack
        prob = pulp.LpProblem(f"Demand_{interviewer}", pulp.LpMaximize)
        
        x = {
            slot: pulp.LpVariable(f"{interviewer}_{slot}", cat='Binary')
            for slot in self.time_slots
        }
        
        # Objective: sum of utilities
        utility_expr = pulp.lpSum(
            [self.utilities[interviewer][slot] * x[slot] for slot in self.time_slots]
        )
        
        # (Skipping consecutive slot adjustments for simplicity)
        
        prob += utility_expr
        
        # Budget constraint
        prob += pulp.lpSum(
            [self.prices[slot] * x[slot] for slot in self.time_slots]
        ) <= self.budgets[interviewer]
        
        # Unavailable slots
        for slot in self.unavailable_slots[interviewer]:
            prob += x[slot] == 0
        
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        selected = {slot for slot in self.time_slots if x[slot].varValue > 0.5}
        return selected
    
    def run_market_mechanism(self, max_iterations=10, price_step=0.5, convergence_threshold=0.1):
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            # Calculate demand
            demands = {i: self.calculate_demand(i) for i in self.interviewers}
            
            # Determine slot demands
            slot_demands = defaultdict(int)
            for i, slots in demands.items():
                for s in slots:
                    slot_demands[s] += 1
            
            # Adjust prices
            price_changes = {}
            for slot in self.time_slots:
                excess = slot_demands[slot] - self.capacities[slot]
                if excess != 0:
                    price_change = price_step * excess
                    new_price = max(0.1, self.prices[slot] + price_change)
                    price_changes[slot] = new_price - self.prices[slot]
                    self.prices[slot] = new_price
            
            if price_changes:
                max_change = max(abs(ch) for ch in price_changes.values())
                converged = (max_change < convergence_threshold)
            else:
                converged = True
            
            self.history.append({
                "iteration": iteration,
                "prices": self.prices.copy(),
                "demands": demands
            })
            iteration += 1
        
        # Final allocation
        final_demands = {i: self.calculate_demand(i) for i in self.interviewers}
        
        # If a slot is oversubscribed, randomly allocate
        allocation = {i: set() for i in self.interviewers}
        for slot in self.time_slots:
            interested = [i for i, slots in final_demands.items() if slot in slots]
            if len(interested) <= self.capacities[slot]:
                for i in interested:
                    allocation[i].add(slot)
            else:
                chosen = random.sample(interested, self.capacities[slot])
                for i in chosen:
                    allocation[i].add(slot)
        
        self.allocation = allocation
        return allocation, converged

    def get_allocation_summary(self):
        """Returns a dict summarizing the allocation."""
        summary = {}
        for i, slots in self.allocation.items():
            summary[i] = sorted(list(slots))
        return summary

###############################################################################
# 2. Global Config: Time Slots and Capacities
###############################################################################

TIME_SLOTS = [
    "9:00-9:30", "9:30-10:00", "10:00-10:30", 
    "10:30-11:00", "11:00-11:30", "11:30-12:00",
    "13:00-13:30", "13:30-14:00", "14:00-14:30",
    "14:30-15:00", "15:00-15:30", "15:30-16:00"
]

CAPACITIES = {slot: 2 for slot in TIME_SLOTS}  # Example: 2 parallel interviews per slot

# CSV file that stores interviewer data
CSV_FILE = "availability.csv"

###############################################################################
# 3. Utility Functions to Load/Save from CSV
###############################################################################

def load_availability():
    """Load CSV into a DataFrame with columns: 
       [interviewer, time_slot, utility, unavailable(bool)]
    """
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=["interviewer", "time_slot", "utility", "unavailable"])

def save_availability(df):
    df.to_csv(CSV_FILE, index=False)

def get_interviewers_from_csv():
    """Return a sorted list of unique interviewer names from CSV."""
    df = load_availability()
    return sorted(df["interviewer"].unique())

###############################################################################
# 4. Streamlit Pages
###############################################################################

def interviewer_page():
    """Page for each interviewer to submit or update their preferences."""
    st.header("Interviewer Page")
    st.write("Please enter your name and set your preferences for each time slot.")

    name = st.text_input("Your Name")
    if not name:
        st.stop()  # Don’t show anything else until they provide a name

    df = load_availability()

    # Filter existing records for this interviewer
    my_rows = df[df["interviewer"] == name]

    # If no rows exist yet, create them
    if my_rows.empty:
        for slot in TIME_SLOTS:
            df = df.append({
                "interviewer": name,
                "time_slot": slot,
                "utility": 50,          # default utility
                "unavailable": False    # default false
            }, ignore_index=True)
        save_availability(df)
        my_rows = df[df["interviewer"] == name]

    # Display a table-like interface for the user to edit (using Streamlit widgets)
    new_utilities = {}
    new_unavailable = {}

    st.write("**Adjust your utility (0-100) for each time slot, or mark as unavailable.**")
    for slot in TIME_SLOTS:
        row = my_rows[my_rows["time_slot"] == slot]
        current_utility = int(row["utility"].values[0])
        current_unavail = bool(row["unavailable"].values[0])

        col1, col2 = st.columns([3, 1])
        with col1:
            slider_val = st.slider(
                f"Utility for {slot}", 0, 100, current_utility, key=f"{slot}_slider"
            )
        with col2:
            check_val = st.checkbox(
                "Unavailable", value=current_unavail, key=f"{slot}_unavailable"
            )

        new_utilities[slot] = slider_val
        new_unavailable[slot] = check_val

    if st.button("Save Preferences"):
        # Update DataFrame rows with new values
        for slot in TIME_SLOTS:
            # Find the row in df
            idx = df[
                (df["interviewer"] == name) & 
                (df["time_slot"] == slot)
            ].index
            df.loc[idx, "utility"] = new_utilities[slot]
            df.loc[idx, "unavailable"] = new_unavailable[slot]
        
        save_availability(df)
        st.success("Preferences saved successfully!")


def admin_page():
    """Page where an admin can run the market-based scheduling algorithm."""
    st.header("Admin Page")
    st.write("Run the scheduling mechanism and view final assignments.")

    df = load_availability()
    if df.empty:
        st.warning("No availability data found. Please ask interviewers to submit preferences first.")
        return
    
    # Build list of interviewers
    interviewers = sorted(df["interviewer"].unique())

    if st.button("Run Scheduling Algorithm"):
        st.info("Running the market-based mechanism...")

        # 1. Instantiate the scheduler
        scheduler = InterviewScheduler(
            time_slots=TIME_SLOTS,
            interviewers=interviewers,
            capacities=CAPACITIES
        )

        # 2. For each interviewer, set preferences / unavailabilities
        for interviewer in interviewers:
            # Filter rows for this interviewer
            subset = df[df["interviewer"] == interviewer]
            
            # Build a dict: {slot: utility}
            preferences = {}
            unavailable_slots = []

            for _, row in subset.iterrows():
                slot = row["time_slot"]
                utility = row["utility"]
                is_unavail = bool(row["unavailable"])
                if is_unavail:
                    # Mark as unavailable
                    unavailable_slots.append(slot)
                    # Utility in the scheduler is forced to 0 anyway
                    preferences[slot] = 0
                else:
                    preferences[slot] = utility
            
            scheduler.set_preferences(interviewer, preferences)
            scheduler.set_unavailable(interviewer, unavailable_slots)

        # 3. Run the mechanism
        allocation, converged = scheduler.run_market_mechanism(
            max_iterations=20, 
            price_step=0.5, 
            convergence_threshold=0.1
        )

        st.success("Scheduling complete!")
        st.write(f"Converged: **{converged}**")

        # 4. Display final schedule
        summary = scheduler.get_allocation_summary()
        
        st.subheader("Final Assignments")
        for interviewer, slots in summary.items():
            st.write(f"**{interviewer}**: {', '.join(slots) if slots else 'No slots assigned'}")

        st.subheader("Final Prices")
        for slot in TIME_SLOTS:
            st.write(f"{slot}: {scheduler.prices[slot]:.2f}")

    else:
        st.write("Click the button above to run scheduling.")

###############################################################################
# 5. Streamlit Main: Multi-Page Approach via Sidebar
###############################################################################

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Interviewer Page", "Admin Page"])

    if page == "Interviewer Page":
        interviewer_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()
