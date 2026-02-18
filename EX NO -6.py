# Bayesian Network Example: Rain → Sprinkler → WetGrass

# Given probabilities
P_Rain = 0.3                  # P(Rain=Yes)
P_NotRain = 0.7               # P(Rain=No)

P_Sprinkler_given_Rain = 0.1  # P(Sprinkler=Yes | Rain=Yes)
P_Sprinkler_given_NoRain = 0.6 # P(Sprinkler=Yes | Rain=No)

P_WetGrass_given_Rain_Sprinkler = 0.99   # P(WetGrass=Yes | Rain=Yes, Sprinkler=Yes)
P_WetGrass_given_Rain_NoSprinkler = 0.8  # P(WetGrass=Yes | Rain=Yes, Sprinkler=No)
P_WetGrass_given_NoRain_Sprinkler = 0.9  # P(WetGrass=Yes | Rain=No, Sprinkler=Yes)
P_WetGrass_given_NoRain_NoSprinkler = 0.0 # P(WetGrass=Yes | Rain=No, Sprinkler=No)

# Calculate total probability of WetGrass
P_WetGrass = (
    P_WetGrass_given_Rain_Sprinkler * P_Rain * P_Sprinkler_given_Rain +
    P_WetGrass_given_Rain_NoSprinkler * P_Rain * (1 - P_Sprinkler_given_Rain) +
    P_WetGrass_given_NoRain_Sprinkler * P_NotRain * P_Sprinkler_given_NoRain +
    P_WetGrass_given_NoRain_NoSprinkler * P_NotRain * (1 - P_Sprinkler_given_NoRain)
)

# Calculate P(Rain | WetGrass) using Bayes theorem
P_Rain_given_WetGrass = (
    (P_WetGrass_given_Rain_Sprinkler * P_Rain * P_Sprinkler_given_Rain +
     P_WetGrass_given_Rain_NoSprinkler * P_Rain * (1 - P_Sprinkler_given_Rain))
    / P_WetGrass
)

print("Probability of Rain given WetGrass =", round(P_Rain_given_WetGrass, 3))
