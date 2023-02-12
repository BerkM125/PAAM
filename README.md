# PAAM
Predicting &amp; Analyzing Athletic Movement

## UPDATES

February 11: Contour detection and ROIs have made the process much better, especially when many people are in one video, except there’s still a ways to go. Simply passing a frame to the model produces limited results, since multi-person analysis or 3D keypoint detection is not possible with this mode. Some successful (and unsuccessful) examples below:

## Goals / General Planning

Convert footage of people on a court into real coordinate space data using computer vision and pre-existing models of human skeletal structure.
Analyze compiled data from a lot of footage (of joint movements, racket movements, etc) to see if there are any patterns that exist in purely kinematic data that indicate different styles of playing, or even the styles of specific players.
Eventually, if such patterns exist, construct a SKILL INDEX of a player based on a certain amount of data around their movements. To make such an index, the program would need to observe a particular moment in the opponent’s game, and calculate exactly what the ideal move would be based on their move. Similar to a chess bot but with more physics.
There are a  LOT of applications that could come from this kind of technology / software. Simply analyzing games and seeing where one can improve, running match videos through the software to keep track of various statistics, or even becoming intelligent enough to identify strategies to combat certain kinds of players and playing styles on court.

