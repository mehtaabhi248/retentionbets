import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Retention experiment Forecasting",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Calculator"

# Configuration
@st.cache_data
def get_config():
    return {
        "d1_baseline": 0.19,
        "pc_baseline": 0.55,
        "default_n": 2000,
        "f_ios": 0.2,
        "f_android": 0.8,
        "rc0_baseline": 0.3263907744,
        "r_org0_baseline": 0.1840149259,
        "r_nc_baseline": 0.01,
        "p_msg_base_total": 0.007334785794503418,
        "delta_rc_ratio": 0.4,
    }

def create_header():
    """Header with baseline metrics."""
    config = get_config()
    
    st.title("Retention Experiment Forecasting")
    st.markdown("**Mathematical forecasting for retention experiments**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("D1 Baseline", f"{config['d1_baseline']:.1%}")
    with col2:
        st.metric("Onboarding completion", f"{config['pc_baseline']:.0%}")
    with col3:
        st.metric("Push opt in rate", "55%")
    with col4:
        st.metric("Daily Users", f"{config['default_n']:,}")

def create_sidebar():
    """Improved sidebar with button navigation."""
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Calculator button
        if st.button(
            "📊 Calculator", 
            key="nav_calc",
            help="Retention forecasting tool",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "Calculator" else "secondary"
        ):
            st.session_state.current_page = "Calculator"
            st.rerun()
        
        # About button
        if st.button(
            "📚 About", 
            key="nav_about",
            help="Mathematical framework",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "About" else "secondary"
        ):
            st.session_state.current_page = "About"
            st.rerun()
        
        return st.session_state.current_page

# Enhancement 3: Visual Channel Contribution
def visualize_channel_mix(channels):
    """Show channel contributions visually"""
    if channels:
        contrib_data = pd.DataFrame([
            {
                "Channel": f"{ch_id} ({ch_data['name']})", 
                "Contribution": ch_data['contribution'] * 1000,  # Scale for visibility
                "Reach": ch_data['R'],
                "Conversion": ch_data['C']
            }
            for ch_id, ch_data in channels.items()
        ])
        
        st.subheader("📊 Channel Contribution Mix")
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(contrib_data.set_index('Channel')['Contribution'])
            st.caption("Contribution (scaled 1000x for visibility)")
        
        with col2:
            # Show reach vs conversion scatter
            st.scatter_chart(contrib_data.set_index('Channel')[['Reach', 'Conversion']])
            st.caption("Reach vs Conversion Rate")

# Enhancement 4: Color-Coded Impact Indicators
def show_impact_gauge(delta_d1_total):
    """Visual impact indicator with color coding"""
    impact_pp = delta_d1_total * 100
    
    if impact_pp > 2.0:
        st.success(f"🎯 High Impact: {impact_pp:.2f}pp")
        return "high"
    elif impact_pp > 1.0:
        st.info(f"📈 Moderate Impact: {impact_pp:.2f}pp")
        return "moderate"
    elif impact_pp > 0.5:
        st.warning(f"⚡ Small Impact: {impact_pp:.2f}pp")
        return "small"
    elif impact_pp > 0:
        st.error(f"📉 Minimal Impact: {impact_pp:.2f}pp")
        return "minimal"
    else:
        st.error("❌ No Impact")
        return "none"

# Enhancement 5: Before/After Comparison Cards
def show_before_after(config, results, delta_pc=0):
    """Before/after comparison with visual cards"""
    st.subheader("📋 Before vs After Comparison")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("#### Current State")
        st.metric("D1 Retention", f"{config['d1_baseline']:.1%}")
        st.metric("Onboarding Rate", f"{config['pc_baseline']:.0%}")
        st.metric("Organic D1", f"{config['r_org0_baseline']:.1%}")
    
    with col2:
        st.markdown("#### →")
        st.markdown("")
        st.markdown("**Experiment**")
        st.markdown("**Impact**")
        
    with col3:
        st.markdown("#### After Experiment")
        st.metric("D1 Retention", f"{results['d1_new']:.1%}", 
                 delta=f"{results['delta_d1_total']*100:+.2f}pp")
        
        new_pc = config['pc_baseline'] + delta_pc if delta_pc > 0 else config['pc_baseline']
        pc_delta = f"{delta_pc*100:+.0f}pp" if delta_pc > 0 else "No change"
        st.metric("Onboarding Rate", f"{new_pc:.0%}", delta=pc_delta)
        
        st.metric("New Organic D1", f"{results['r_org1']:.1%}", 
                 delta=f"{results['delta_r_org']*100:+.3f}pp")

# Enhancement 6: Interactive Sliders with Live Preview
def show_live_formula_preview(delta_pc, delta_rc, config):
    """Show live formula calculation as user adjusts sliders"""
    if delta_pc > 0 or delta_rc > 0:
        st.subheader("🔄 Live Formula Preview")
        
        # Core calculation components
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completer_advantage = config['rc0_baseline'] - config['r_nc_baseline']
            st.metric("Completer Advantage", f"{completer_advantage:.3f}")
            st.caption("(rc - rnc)")
        
        with col2:
            st.metric("Completion Boost", f"{delta_pc:.3f}")
            st.caption("ΔPc")
        
        with col3:
            st.metric("Quality Boost", f"{delta_rc:.3f}")
            st.caption("Pc × Δrc")
        
        with col4:
            total_organic_impact = completer_advantage * delta_pc + config['pc_baseline'] * delta_rc
            st.metric("Total Organic Impact", f"{total_organic_impact:.6f}")
            st.caption("Δr_org")
        
        # Show the calculation breakdown
        with st.expander("📝 Calculation Breakdown"):
            st.code(f"""
Δr_org = (rc - rnc) × ΔPc + Pc × Δrc
Δr_org = {completer_advantage:.3f} × {delta_pc:.3f} + {config['pc_baseline']:.2f} × {delta_rc:.3f}
Δr_org = {completer_advantage * delta_pc:.6f} + {config['pc_baseline'] * delta_rc:.6f}
Δr_org = {total_organic_impact:.6f}
            """)

# Enhancement 7: Effort vs Impact Visualization
def show_effort_impact_matrix(results, effort, experiment_name):
    """Show experiment positioning on effort/impact matrix"""
    impact_score = results['delta_d1_total'] * 100  # Convert to pp
    
    st.subheader("🎯 Effort vs Impact Positioning")
    
    # Determine quadrant
    high_impact = impact_score > 1.0  # 1pp threshold
    high_effort = effort > 3
    
    # Create visual matrix
    col1, col2 = st.columns(2)
    
    with col1:
        if high_impact and not high_effort:
            st.success("🏆 QUICK WIN")
            st.write("High impact, low effort")
            priority = "Immediate priority"
        elif high_impact and high_effort:
            st.info("🏗️ MAJOR PROJECT")
            st.write("High impact, high effort")
            priority = "Strategic investment"
        elif not high_impact and not high_effort:
            st.warning("🔧 FILL-IN")
            st.write("Low impact, low effort")
            priority = "Nice to have"
        else:
            st.error("❌ AVOID")
            st.write("Low impact, high effort")
            priority = "Reconsider scope"
    
    with col2:
        # Show positioning metrics
        st.metric("Impact Score", f"{impact_score:.2f}pp")
        st.metric("Effort Level", f"{effort}/5")
        st.metric("Priority", priority)
        
        # ROI-style metric
        roi = impact_score / effort if effort > 0 else 0
        st.metric("Impact/Effort Ratio", f"{roi:.2f}")

def experiment_input():
    """Enhanced experiment setup input with live preview"""
    config = get_config()
    
    st.header("Experiment Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        experiment_name = st.text_input(
            "Experiment Name",
            placeholder="e.g., Rich Notifs with copy + sound + priority"
        )
        
        experiment_type = st.selectbox(
            "Type",
            ["Product Improvement", "Messaging Only", "Product + Messaging"]
        )
    
    with col2:
        effort = st.slider("Effort (1-5)", 1, 5, 3, 
                          help="1=Quick tweak, 5=Major engineering effort")
        
        n_users = st.number_input(
            "Daily Users",
            min_value=100,
            value=config["default_n"],
            step=500
        )
        
        platform = st.selectbox(
            "Platform",
            ["Both (1.0)", "Android (0.8)", "iOS (0.2)"]
        )
        
        if "Android" in platform:
            f_target = config["f_android"]
        elif "iOS" in platform:
            f_target = config["f_ios"]
        else:
            f_target = 1.0
        
        st.caption(f"f_target = {f_target}")
    
    return {
        "name": experiment_name,
        "type": experiment_type,
        "effort": effort,
        "n_users": n_users,
        "platform": platform,
        "f_target": f_target
    }

def product_parameters():
    """Enhanced product parameter input with live preview"""
    config = get_config()
    
    st.subheader("Product Changes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        delta_pc = st.slider(
            "Onboarding completion Improvement (pp)",
            min_value=0,
            max_value=15,
            value=0,
            step=1,
            help="Pc = Percentage of users who complete onboarding"
        ) / 100
        
        if delta_pc > 0:
            new_pc = config["pc_baseline"] + delta_pc
            st.caption(f"{config['pc_baseline']:.0%} → {new_pc:.0%}")
            
            # Live impact preview
            immediate_impact = (config["rc0_baseline"] - config["r_nc_baseline"]) * delta_pc
            st.info(f"💡 Expected organic lift: {immediate_impact*100:.3f}pp")
    
    with col2:
        if delta_pc > 0:
            expected_rc = delta_pc * config["delta_rc_ratio"]
            
            override_rc = st.checkbox("Override completor retention")
            
            if override_rc:
                delta_rc = st.number_input(
                    "Expected completor retention improvement (pp)",
                    min_value=0.0,
                    max_value=5.0,
                    value=expected_rc * 100,
                    step=0.1,
                    format="%.1f",
                    help="rc = Retention rate for users who complete onboarding"
                ) / 100
                st.caption("Manual override")
            else:
                delta_rc = expected_rc
                st.caption(f"Expected: {delta_rc*100:.1f}pp (based on completion improvement)")
            
            if delta_rc > 0:
                new_rc = config["rc0_baseline"] + delta_rc
                st.caption(f"{config['rc0_baseline']:.1%} → {new_rc:.1%}")
        else:
            delta_rc = 0.0
            override_rc = False
    
    # Show live formula preview for product changes
    if delta_pc > 0 or delta_rc > 0:
        show_live_formula_preview(delta_pc, delta_rc, config)
    
    return delta_pc, delta_rc, override_rc

def messaging_abc_channels():
    """Enhanced messaging channels with visual feedback"""
    st.subheader("Messaging Channels (A/B/C)")
    st.markdown("*Configure channels with opt-in rates and union calculation*")
    
    config = get_config()
    channels = {}
    
    # Channel A - Push Notifications
    with st.expander("Channel A (Push Notifications)", expanded=False):
        enable_a = st.checkbox("Enable Channel A", key="enable_channel_a")
        
        if enable_a:
            a_col1, a_col2, a_col3, a_col4 = st.columns(4)
            
            with a_col1:
                push_opt_in = st.number_input(
                    "Push Opt-in Rate (%)",
                    min_value=0.0,
                    max_value=95.0,
                    value=55.0,
                    step=1.0,
                    format="%.1f",
                    help="Percentage of completers who opt-in to push notifications",
                    key="push_opt_in"
                ) / 100
                
                R_A = config["pc_baseline"] * push_opt_in
                st.caption(f"R_A = {config['pc_baseline']:.0%} × {push_opt_in:.0%} = {R_A:.4f}")
            
            with a_col2:
                p_A = st.number_input(
                    "p_A (Lift per push)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.0116,
                    step=0.0001,
                    format="%.5f",
                    help="Incremental retention per push",
                    key="p_a_input"
                )
            
            with a_col3:
                k_A = st.number_input(
                    "k_A (Pushes on D1)",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of push notifications",
                    key="k_a_input"
                )
            
            with a_col4:
                baseline_opt_in = 0.55
                if push_opt_in != baseline_opt_in:
                    improvement = (push_opt_in - baseline_opt_in) * 100
                    st.metric("Opt-in Change", f"{improvement:+.0f}pp")
                else:
                    st.caption("Baseline opt-in rate")
            
            C_A = 1 - (1 - p_A) ** k_A if p_A > 0 else 0
            contribution_A = R_A * C_A
            
            # Color-coded feedback
            if contribution_A > 0.01:
                st.success(f"C_A = {C_A:.6f} | Contribution: {contribution_A:.6f}")
            elif contribution_A > 0.005:
                st.info(f"C_A = {C_A:.6f} | Contribution: {contribution_A:.6f}")
            else:
                st.warning(f"C_A = {C_A:.6f} | Contribution: {contribution_A:.6f}")
            
            channels['A'] = {
                'R': R_A, 'p': p_A, 'k': k_A, 'C': C_A, 
                'contribution': contribution_A, 'name': 'Push',
                'opt_in_rate': push_opt_in
            }
    
    # Channel B - Email (similar enhancement pattern)
    with st.expander("Channel B (Email)", expanded=False):
        enable_b = st.checkbox("Enable Channel B", key="enable_channel_b")
        
        if enable_b:
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            
            with b_col1:
                email_signup = st.number_input(
                    "Email Signup Rate (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=12.0,
                    step=1.0,
                    format="%.1f",
                    help="Percentage of users who sign up for email",
                    key="email_signup"
                ) / 100
                
                R_B = email_signup
                st.caption(f"R_B = {R_B:.4f}")
            
            with b_col2:
                p_B = st.number_input(
                    "p_B (Lift per email)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.003,
                    step=0.0001,
                    format="%.5f",
                    help="Incremental retention per email",
                    key="p_b_input"
                )
            
            with b_col3:
                k_B = st.number_input(
                    "k_B (Emails on D1)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of emails",
                    key="k_b_input"
                )
            
            with b_col4:
                baseline_signup = 0.12
                if email_signup != baseline_signup:
                    improvement = (email_signup - baseline_signup) * 100
                    st.metric("Signup Change", f"{improvement:+.0f}pp")
                else:
                    st.caption("Baseline signup rate")
            
            C_B = 1 - (1 - p_B) ** k_B if p_B > 0 else 0
            contribution_B = R_B * C_B
            
            # Color-coded feedback
            if contribution_B > 0.01:
                st.success(f"C_B = {C_B:.6f} | Contribution: {contribution_B:.6f}")
            elif contribution_B > 0.005:
                st.info(f"C_B = {C_B:.6f} | Contribution: {contribution_B:.6f}")
            else:
                st.warning(f"C_B = {C_B:.6f} | Contribution: {contribution_B:.6f}")
            
            channels['B'] = {
                'R': R_B, 'p': p_B, 'k': k_B, 'C': C_B,
                'contribution': contribution_B, 'name': 'Email',
                'signup_rate': email_signup
            }
    
    # Channel C - Experimental (similar pattern)
    with st.expander("Channel C (Experimental)", expanded=False):
        enable_c = st.checkbox("Enable Channel C", key="enable_channel_c")
        
        if enable_c:
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            
            with c_col1:
                channel_c_name = st.selectbox(
                    "Channel C Type",
                    ["Live Activity", "In-App Messages", "App Icon Changes", "Streak Prompts", "Other"],
                    key="channel_c_type"
                )
            
            with c_col2:
                R_C = st.number_input(
                    "R_C (Reach)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.001,
                    format="%.4f",
                    help="Fraction of users reached (experimental channels use direct reach)",
                    key="r_c_input"
                )
            
            with c_col3:
                p_C = st.number_input(
                    "p_C (Lift per touch)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.0232,
                    step=0.0001,
                    format="%.5f",
                    help="Incremental retention per touch",
                    key="p_c_input"
                )
            
            with c_col4:
                k_C = st.number_input(
                    "k_C (Touches on D1)",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Number of touches",
                    key="k_c_input"
                )
            
            C_C = 1 - (1 - p_C) ** k_C if p_C > 0 else 0
            contribution_C = R_C * C_C
            
            # Color-coded feedback
            if contribution_C > 0.01:
                st.success(f"C_C = {C_C:.6f} | Contribution: {contribution_C:.6f}")
            elif contribution_C > 0.005:
                st.info(f"C_C = {C_C:.6f} | Contribution: {contribution_C:.6f}")
            else:
                st.warning(f"C_C = {C_C:.6f} | Contribution: {contribution_C:.6f}")
            
            channels['C'] = {
                'R': R_C, 'p': p_C, 'k': k_C, 'C': C_C,
                'contribution': contribution_C, 'name': channel_c_name
            }
    
    # Union calculation with visualization
    if channels:
        st.subheader("Union Calculation")
        
        # Show individual contributions
        for ch_id, ch_data in channels.items():
            if ch_id == 'A' and 'opt_in_rate' in ch_data:
                st.write(f"• Channel {ch_id} ({ch_data['name']}): {ch_data['opt_in_rate']:.0%} opt-in → R × C = {ch_data['contribution']:.6f}")
            elif ch_id == 'B' and 'signup_rate' in ch_data:
                st.write(f"• Channel {ch_id} ({ch_data['name']}): {ch_data['signup_rate']:.0%} signup → R × C = {ch_data['contribution']:.6f}")
            else:
                st.write(f"• Channel {ch_id} ({ch_data['name']}): R × C = {ch_data['contribution']:.6f}")
        
        # Calculate P_new
        individual_contribs = [ch_data['contribution'] for ch_data in channels.values()]
        P_new = 1 - np.prod([1 - contrib for contrib in individual_contribs])
        
        st.markdown("**Union Formula:** P_new = 1 - ∏(1 - R_i × C_i)")
        st.success(f"**P_new (union of A/B/C): {P_new:.8f}**")
        
        # Add channel mix visualization
        visualize_channel_mix(channels)
        
        # Show detailed calculation
        with st.expander("Union Calculation Details"):
            calc_steps = []
            for i, contrib in enumerate(individual_contribs):
                calc_steps.append(f"(1 - {contrib:.6f}) = {1-contrib:.6f}")
            
            product = np.prod([1 - contrib for contrib in individual_contribs])
            
            st.code(f"""
Individual contributions: {individual_contribs}

Step by step:
{chr(10).join([f'Channel {list(channels.keys())[i]}: {step}' for i, step in enumerate(calc_steps)])}

Product: {product:.8f}
P_new = 1 - {product:.8f} = {P_new:.8f}
            """)
        
        return P_new, channels
    else:
        return 0, {}

def calculate_impact(experiment_details, delta_pc, delta_rc, p_new):
    """Calculate retention impact using exact formulas."""
    config = get_config()
    
    f_target = experiment_details["f_target"]
    n_users = experiment_details["n_users"]
    effort = experiment_details["effort"]
    
    pc_baseline = config["pc_baseline"]
    rc0_baseline = config["rc0_baseline"]
    r_org0_baseline = config["r_org0_baseline"]
    r_nc_baseline = config["r_nc_baseline"]
    p_msg_base = config["p_msg_base_total"]
    
    # Core calculation
    delta_r_org = (rc0_baseline - r_nc_baseline) * delta_pc + pc_baseline * delta_rc
    
    # Product effect
    delta_d1_product = f_target * (1 - p_msg_base) * delta_r_org
    
    # Messaging effect
    r_org1 = r_org0_baseline + delta_r_org
    delta_d1_messages = (1 - r_org1) * (1 - p_msg_base) * p_new
    
    # Total impact
    delta_d1_total = delta_d1_product + delta_d1_messages
    d1_new = config["d1_baseline"] + delta_d1_total
    score = (n_users * delta_d1_total) / effort if effort > 0 else 0
    
    return {
        "delta_d1_product": delta_d1_product,
        "delta_d1_messages": delta_d1_messages,
        "delta_d1_total": delta_d1_total,
        "d1_new": d1_new,
        "score": score,
        "delta_r_org": delta_r_org,
        "r_org1": r_org1
    }

def display_results(experiment_details, results, delta_pc=0):
    """Enhanced results display with visual enhancements"""
    if not experiment_details["name"]:
        st.warning("Please enter an experiment name")
        return
    
    st.header("Results")
    st.subheader(experiment_details["name"])
    
    # Show impact gauge first
    impact_level = show_impact_gauge(results['delta_d1_total'])
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("ΔD1 Product", f"{results['delta_d1_product']*100:.3f}pp")
    
    with col2:
        st.metric("ΔD1 Messages", f"{results['delta_d1_messages']*100:.3f}pp")
    
    with col3:
        st.metric("ΔD1 Total", f"{results['delta_d1_total']*100:.3f}pp")
    
    with col4:
        st.metric("D1 New", f"{results['d1_new']:.1%}")
    
    with col5:
        st.metric("Score", f"{results['score']:,.0f}")
    
    # Before/after comparison
    config = get_config()
    show_before_after(config, results, delta_pc)
    
    # Effort vs impact matrix
    show_effort_impact_matrix(results, experiment_details["effort"], experiment_details["name"])
    
    # Breakdown table
    if results["delta_d1_product"] > 0 or results["delta_d1_messages"] > 0:
        st.subheader("Impact Breakdown")
        
        breakdown_data = []
        
        if results["delta_d1_product"] > 0:
            breakdown_data.append({
                "Component": "Product Effect",
                "ΔD1 Value": f"{results['delta_d1_product']*100:.3f}pp"
            })
        
        if results["delta_d1_messages"] > 0:
            breakdown_data.append({
                "Component": "Messaging Effect", 
                "ΔD1 Value": f"{results['delta_d1_messages']*100:.3f}pp"
            })
        
        breakdown_data.append({
            "Component": "Total",
            "ΔD1 Value": f"{results['delta_d1_total']*100:.3f}pp"
        })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

def calculation_details(results, delta_pc, delta_rc):
    """Show calculation details in expander."""
    with st.expander("Calculation Details"):
        config = get_config()
        
        st.code(f"""
Core Formula:
Δr_org = (rc - rnc) × ΔPc + Pc × Δrc
Δr_org = ({config['rc0_baseline']:.3f} - {config['r_nc_baseline']:.2f}) × {delta_pc:.3f} + {config['pc_baseline']:.2f} × {delta_rc:.3f}
Δr_org = {results['delta_r_org']:.6f}

Product Effect:
ΔD1_product = f_target × (1 - P_msg_base) × Δr_org
ΔD1_product = {results['delta_d1_product']:.6f}

Total:
ΔD1_total = {results['delta_d1_total']:.6f}
Score = (N × ΔD1_total) / Effort = {results['score']:,.0f}
        """)

def calculator_page():
    """Enhanced calculator page with visual improvements"""
    experiment_details = experiment_input()
    
    delta_pc, delta_rc, override_rc = 0, 0, False
    p_new, channels = 0, {}
    
    if experiment_details["type"] in ["Product Improvement", "Product + Messaging"]:
        delta_pc, delta_rc, override_rc = product_parameters()
    
    if experiment_details["type"] in ["Messaging Only", "Product + Messaging"]:
        p_new, channels = messaging_abc_channels()
    
    if experiment_details["name"] and st.button("Calculate Impact", type="primary"):
        results = calculate_impact(experiment_details, delta_pc, delta_rc, p_new)
        
        st.markdown("---")
        display_results(experiment_details, results, delta_pc)
        
        calculation_details(results, delta_pc, delta_rc)

def about_page():
    """About page with mathematical framework."""
    st.header("Mathematical Framework")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retention Model")
        st.markdown("""
        **Constant-α Model:**
        ```
        R_t ≈ D1 × α^(t-1)
        α = (D7/D1)^(1/6)
        ```
        Models retention decay over time with constant continuation rate.
        """)
        
        st.subheader("Organic vs Messaging")
        st.markdown("""
        **D1 Composition:**
        ```
        D1 = r_org + (1−r_org) × P_msg,base
        ```
        **Solving for Organic:**
        ```
        r_org = (D1 − P_msg,base) / (1 − P_msg,base)
        ```
        """)
        
        st.subheader("Completers vs Non-completers")
        st.markdown("""
        **Organic Breakdown:**
        ```
        r_org = Pc × rc + (1−Pc) × rnc
        ```
        **Solving for rc:**
        ```
        rc = (r_org − (1−Pc) × rnc) / Pc
        ```
        """)
    
    with col2:
        st.subheader("Product Impact")
        st.markdown("""
        **Organic Shift:**
        ```
        Δr_org ≈ (rc − rnc) × ΔPc + Pc × Δrc
        ```
        **Product Effect:**
        ```
        ΔD1_product = (1 − P_msg,base) × Δr_org × f_target
        ```
        """)
        
        st.subheader("Messaging Impact")
        st.markdown("""
        **Per-channel:**
        ```
        Ci = 1 − (1 − pi)^ki
        ```
        **Union (prevents overlap):**
        ```
        P_any = 1 − ∏(1 − Ri × Ci)
        ```
        **Messaging Effect:**
        ```
        ΔD1_messages = (1−r_org1) × (1−P_msg,base) × P_new
        ```
        """)
        
        st.subheader("Total Impact")
        st.markdown("""
        **Combined:**
        ```
        ΔD1_total = ΔD1_product + ΔD1_messages
        ΔR = N × ΔD1
        Score = ΔR / Effort
        ```
        """)
    
    st.markdown("---")
    st.subheader("Variable Definitions")
    
    var_col1, var_col2 = st.columns(2)
    
    with var_col1:
        st.markdown("""
        **Core Variables:**
        - **D1**: Day-1 retention
        - **Pc**: Onboarding completion rate
        - **rc, rnc**: Organic D1 for completers/non-completers
        - **r_org**: Overall organic retention
        - **α**: Continuation rate from D1 onward
        """)
    
    with var_col2:
        st.markdown("""
        **Messaging Variables:**
        - **Ri**: Channel reach (A=Push, B=Email, C=Experimental)
        - **pi**: Per-touch lift for channel i
        - **ki**: Number of touches for channel i
        - **Ci**: Channel conversion rate
        - **P_new**: Union probability of new messaging
        """)

def main():
    create_header()
    
    # Get current page from sidebar
    current_page = create_sidebar()
    
    # Show appropriate page
    if current_page == "Calculator":
        calculator_page()
    else:
        about_page()
    
    # Footer with last updated timestamp
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()