"""
Global config: colours, layout, cache settings.
This centralises style so desktop + mobile stay consistent.
"""

# ============================================================
# COLOURS
# ============================================================

PRIMARY_COLOR = "#4E79A7"     # Main line / area color (blue)
ACCENT_COLOR = "#F28E2B"      # Secondary line (orange)
POSITIVE_COLOR = "#59A14F"    # Positive bars (green)
NEGATIVE_COLOR = "#E15759"    # Negative bars (red)
GOLD_COLOR = "#D4AF37"        # Gold line
SILVER_COLOR = "#C0C0C0"      # Silver line

# ============================================================
# PLOTLY LAYOUT (DESKTOP + MOBILE)
# ============================================================

BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,12,16,1)",
    hovermode="x",
    margin=dict(l=40, r=20, t=60, b=40),
    font=dict(
        family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        size=12,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
    ),
    hoverlabel=dict(
        namelength=-1,
        bgcolor="rgba(30,30,40,0.9)",
        bordercolor="rgba(0,0,0,0)",
        font_size=11,
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
    ),
)


# Default figure height for all charts (keeps things readable on mobile,
# without being gigantic on desktop).
DEFAULT_FIG_HEIGHT = 420

# ============================================================
# CACHE SETTINGS
# ============================================================

# Cache lifetime: 10 minutes
CACHE_TTL_SECONDS = 600

# Global “10y” history ceiling (in days)
MAX_HISTORY_DAYS = 3650
