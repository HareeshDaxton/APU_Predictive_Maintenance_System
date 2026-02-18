




























# _normalize_operating_conditions() 
# To make sure the model learns engine health, not differences caused by operating modes.
# We normalize operating conditions so the model focuses on degradation, not how the engine is being used.

# _add_cycle_normalization
# It converts raw cycle number into a percentage of engine life used.
# Standardize lifecycle progression across engines so the model learns degradation stage instead of raw time.

#Rolling standard deviation measures recent sensor instability, helping detect early signs of mechanical failure.