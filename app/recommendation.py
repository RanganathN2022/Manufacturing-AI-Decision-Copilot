def generate_recommendation(
    machine_data,
    failure_probability,
    risk_level
):
    """
    Generate maintenance recommendations
    based on predicted failure risk and
    machine operating conditions.
    """

    actions = []
    focus_areas = []

    # ---------------------------------------------------------
    # Risk-based priority
    # ---------------------------------------------------------

    if risk_level == "Critical":

        priority = "Immediate"

        actions.append(
            "Schedule immediate maintenance inspection."
        )

    elif risk_level == "High":

        priority = "High"

        actions.append(
            "Schedule maintenance inspection soon."
        )

    elif risk_level == "Medium":

        priority = "Medium"

        actions.append(
            "Monitor machine condition and plan maintenance."
        )

    else:

        priority = "Low"

        actions.append(
            "Continue normal monitoring."
        )

    # ---------------------------------------------------------
    # Vibration
    # ---------------------------------------------------------

    if machine_data["Vibration_mm_s"] >= 5:

        focus_areas.append(
            "Inspect vibration and bearing condition."
        )

    # ---------------------------------------------------------
    # Temperature
    # ---------------------------------------------------------

    if machine_data["Temperature_C"] >= 75:

        focus_areas.append(
            "Inspect cooling system and thermal condition."
        )

    # ---------------------------------------------------------
    # Maintenance interval
    # ---------------------------------------------------------

    if machine_data["Last_Maintenance_Days"] >= 200:

        focus_areas.append(
            "Maintenance interval is overdue."
        )

    # ---------------------------------------------------------
    # Operating hours
    # ---------------------------------------------------------

    if machine_data["Operating_Hours"] >= 30000:

        focus_areas.append(
            "Inspect high-hour components for wear."
        )

    # ---------------------------------------------------------
    # Failure history
    # ---------------------------------------------------------

    if machine_data["Failure_Count"] >= 3:

        focus_areas.append(
            "Review previous failure history."
        )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    return {
        "priority": priority,
        "recommended_actions": actions,
        "focus_areas": focus_areas
    }