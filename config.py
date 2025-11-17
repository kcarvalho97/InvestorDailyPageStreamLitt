"""
Shared configuration for the Markets Dashboard:
- Colors
- Plotly base layout
- Cache / history settings
"""

PRIMARY_COLOR = "#4E79A7"     # Main line / area color (blue)
ACCENT_COLOR = "#F28E2B"      # Second line (orange)
POSITIVE_COLOR = "#59A14F"    # Positive bars (green)
NEGATIVE_COLOR = "#E15759"    # Negative bars (red)
GOLD_COLOR = "#D4AF37"        # Gold line
SILVER_COLOR = "#C0C0C0"      # Silver line

BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
)

# Cache lifetime: 10 minutes
CACHE_TTL_SECONDS = 600

# Global “10y” history ceiling (in days)
MAX_HISTORY_DAYS = 3650

