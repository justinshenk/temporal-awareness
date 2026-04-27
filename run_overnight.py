#!/usr/bin/env python3
"""
COMPREHENSIVE OVERNIGHT RUN — EMNLP-GRADE
============================================
Everything needed to make results bulletproof.

PHASE 1: GPT-J-6B
  1a. Large domain study (50 prompts × 7 conditions, K=1,3,5)
      - chain_of_thought, chain_of_thought_scrambled, chain_of_thought_nonmath
      - free_prose, structured_prose, code, poetry
  1b. Bootstrap CIs on all domain gaps (300 samples, 3 seeds)
  1c. Attention vs MLP decomposition on CoT, code, poetry (K=3)
  1d. Cross-domain probe transfer (binary: common vs rare next-token)

PHASE 2: Qwen-7B
  2a. 500-sig code staircase (fills in "---" entries)  
  2b. Large domain study (50 prompts × 7 conditions)
  2c. Attention vs MLP on CoT (K=3)

Saves intermediate results after each section.
Expected: ~10-12 hours on RTX 3090 (24GB).
"""

import json, os, sys, time, random, gc
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy import stats as scipy_stats

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PCA_DIM = 128
N_BOOTSTRAP = 300
PROBE_SEEDS = [42, 123, 456]
N_GEN = 60
K_VALUES = [1, 3, 5]
OUTDIR = "results/lookahead/final"
os.makedirs(OUTDIR, exist_ok=True)

def save_intermediate(data, name):
    """Save intermediate results so we don't lose progress on crash."""
    path = f"{OUTDIR}/overnight_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  [SAVED] {path}")

# ================================================================
# DOMAIN PROMPTS — 50 per condition
# ================================================================
DOMAINS = {
    "chain_of_thought": [
        "Question: What is 247 + 389? Let me solve this step by step.",
        "Problem: If a train travels at 60 mph for 3 hours, how far does it go? Solution:",
        "Calculate: 15% of 240. Step 1: Convert 15% to decimal:",
        "Question: How many seconds are in 3.5 hours? Let me think through this.",
        "If x + 7 = 15, what is x? Let me solve: First, subtract",
        "Problem: A rectangle has length 12 and width 8. Find the area.",
        "Calculate the average of 23, 45, 67, and 89. First, add them:",
        "Question: What is 1000 minus 347? Let me work through this:",
        "If 3 apples cost $2.25, how much does one apple cost? Solution:",
        "Problem: Convert 72 degrees Fahrenheit to Celsius. Formula: C =",
        "Question: What is 18 times 23? Step 1: Break it down.",
        "Problem: A car uses 8 gallons for 240 miles. What is the mpg? Solution:",
        "Calculate: What is 3/4 plus 2/3? First, find a common denominator:",
        "Question: If you save $50 per week, how much in 6 months? Let me calculate:",
        "Problem: Find the hypotenuse of a right triangle with legs 3 and 4.",
        "Question: What is 15 squared? Step 1: multiply 15 by 15.",
        "Problem: A pizza is cut into 8 slices. 3 people eat 2 slices each. How many remain?",
        "Calculate: 20% tip on a $85 bill. Step 1: find 10% first:",
        "Question: How many minutes in 3 days? Let me break this down:",
        "Problem: If a shirt costs $40 and is 25% off, what is the sale price?",
        "Calculate: the perimeter of a rectangle with length 15 and width 8.",
        "Question: What is 144 divided by 12? Let me solve:",
        "Problem: A train leaves at 2:15 PM and arrives at 5:45 PM. Trip duration?",
        "Calculate: compound interest on $1000 at 5% for 2 years. Formula:",
        "Question: What is the volume of a cube with side length 5? Step 1:",
        "Problem: If 4 workers finish in 6 hours, how long for 3 workers?",
        "Calculate: the area of a circle with radius 7. Using pi equals 3.14:",
        "Question: What is 2 to the power of 8? Let me compute step by step.",
        "Problem: A store sells notebooks for $3 each. How much for 15 notebooks?",
        "Calculate: the diagonal of a rectangle with sides 6 and 8.",
        "Question: If 1 inch equals 2.54 cm, how many cm in 12 inches?",
        "Problem: Probability of drawing red from 3 red, 5 blue, 2 green marbles?",
        "Calculate: 33% of 900. Step 1: convert percentage to decimal:",
        "Question: What is the sum of integers from 1 to 10? Let me add them:",
        "Problem: Two trains at 60 and 40 mph from 200 miles apart. When do they meet?",
        "Calculate: days between January 15 and March 10. Step by step:",
        "Question: What is 7 factorial? Let me compute: 7 times 6 times",
        "Problem: A recipe needs 2 cups flour for 12 cookies. How much for 30?",
        "Calculate: the slope between points (2,3) and (8,15). Rise over run:",
        "Question: A number increased by 40% becomes 280. What was the original?",
        "Problem: Find the mean, median, and mode of: 4, 7, 2, 9, 4, 8, 4.",
        "Calculate: simple interest on $5000 at 3% for 4 years. Formula:",
        "Question: What is the GCD of 48 and 36? Using Euclidean algorithm:",
        "Problem: A cone has radius 3 and height 4. Find its volume.",
        "Calculate: 3.5 hours in hours and minutes. Step 1:",
        "Question: How many ways can you arrange the letters in CAT?",
        "Problem: If f(x) = 2x + 3, what is f(7)? Substituting:",
        "Calculate: the distance between (1,1) and (4,5). Distance formula:",
        "Question: What is 0.125 as a fraction? Let me convert step by step.",
        "Problem: A tank fills at 5 liters per minute. How long to fill 200 liters?",
    ],
    "chain_of_thought_scrambled": [
        "Question: What is 247 + 389? Therefore the answer. Step 3: carry over.",
        "Problem: Train at 60 mph for 3 hours? The answer is: Step 2 multiply",
        "Calculate: 15% of 240. The result equals. Step 2: Now convert back:",
        "Question: How many seconds in 3.5 hours? The answer is. Step 1: multiply",
        "If x + 7 = 15, the answer is x equals. To verify: add back",
        "Problem: Rectangle length 12 width 8. Therefore area equals. Step 1:",
        "Calculate average of 23, 45, 67, 89. The average is. To compute: add",
        "Question: What is 1000 minus 347? Answer: the result. Step 2: subtract",
        "If 3 apples cost $2.25, the answer is. Step 3: so each costs",
        "Problem: Convert 72°F to Celsius. Answer equals. Step 1: subtract",
        "Question: 18 times 23? The product is. Step 2: add partial products.",
        "Problem: Car 8 gallons 240 miles. The mpg is. Divide:",
        "Calculate: 3/4 plus 2/3. The sum equals. Step 3: convert denominators.",
        "Question: Save $50 weekly for 6 months. The total. Step 2: weeks count.",
        "Problem: Hypotenuse legs 3 and 4. The answer is. Square each:",
        "Question: 15 squared equals. The result. Step 1: break into parts.",
        "Problem: Pizza 8 slices, 3 eat 2 each. Remaining is. Total eaten:",
        "Calculate: 20% tip on $85. The tip equals. Step 2: multiply by 2.",
        "Question: Minutes in 3 days. The total is. Step 2: hours per day.",
        "Problem: Shirt $40, 25% off. Sale price is. Step 2: subtract discount.",
        "Calculate: perimeter, length 15 width 8. The answer. Add all sides:",
        "Question: 144 divided by 12. The quotient is. Long division:",
        "Problem: Train 2:15 PM to 5:45 PM. Duration is. Subtract times:",
        "Calculate: compound interest $1000 at 5% 2 years. The amount. Apply:",
        "Question: Volume cube side 5. The answer is. Cube the side:",
        "Problem: 4 workers 6 hours, 3 workers how long. The time. Inverse:",
        "Calculate: circle area radius 7. The area is. Pi times r squared:",
        "Question: 2 to the 8th. The answer. Double repeatedly:",
        "Problem: Notebooks $3 each, 15 total. Total cost is. Multiply:",
        "Calculate: diagonal rectangle 6 by 8. The length. Sum of squares:",
        "Question: 12 inches in cm. The conversion. Multiply by 2.54:",
        "Problem: 3 red 5 blue 2 green marbles. P(red) is. Total marbles:",
        "Calculate: 33% of 900. The result. Multiply 0.33 times:",
        "Question: Sum 1 to 10. The total is. Use formula n(n+1)/2:",
        "Problem: Trains 60+40 mph, 200 miles apart. Meet time. Combined speed:",
        "Calculate: days Jan 15 to Mar 10. The count. January has 31:",
        "Question: 7 factorial. The product is. Multiply 7 times 6:",
        "Problem: 2 cups for 12 cookies, for 30. Amount is. Scale up:",
        "Calculate: slope (2,3) to (8,15). The slope. Differences: y then x:",
        "Question: Number plus 40% equals 280. Original is. Divide by 1.4:",
        "Problem: Mean median mode of 4,7,2,9,4,8,4. Results. Sort first:",
        "Calculate: simple interest $5000, 3%, 4 years. Interest is. PRT:",
        "Question: GCD 48 and 36. The answer. Divide larger by smaller:",
        "Problem: Cone radius 3 height 4. Volume is. 1/3 pi r²h:",
        "Calculate: 3.5 hours to hrs min. That is. 0.5 times 60:",
        "Question: Arrangements of CAT. The count. Three factorial:",
        "Problem: f(x)=2x+3, f(7). The value. Substitute 7:",
        "Calculate: distance (1,1) to (4,5). The distance. Square differences:",
        "Question: 0.125 as fraction. The fraction is. Multiply by 1000:",
        "Problem: Tank 5 L/min, 200 liters. Time is. Divide:",
    ],
    "chain_of_thought_nonmath": [
        "Question: Is a tomato a fruit or vegetable? Let me think step by step.",
        "Problem: What season comes after winter? Let me reason through this.",
        "Question: Which is heavier, a pound of feathers or a pound of rocks? Step 1:",
        "Problem: If today is Monday, what day was it 3 days ago? Let me count:",
        "Question: Can penguins fly? Let me consider the evidence step by step.",
        "Problem: Which continent is Brazil in? Let me think about geography.",
        "Question: Is water wet? Let me analyze this step by step.",
        "Problem: What color do you get mixing red and blue? Step 1: recall",
        "Question: Does the sun rise in the east or west? Let me reason:",
        "Problem: How many legs does a spider have? Let me think carefully.",
        "Question: Is a whale a fish or mammal? Let me consider step by step.",
        "Problem: Which is taller, a giraffe or an elephant? Let me compare:",
        "Question: Can sound travel through space? Step 1: Consider what space is.",
        "Problem: What language do they speak in Brazil? Let me think:",
        "Question: Is glass a solid or liquid? Let me reason through this.",
        "Problem: Which planet is closest to the sun? Step 1: recall the order:",
        "Question: Do plants need sunlight to grow? Let me think step by step.",
        "Problem: What is the capital of Japan? Let me recall step by step.",
        "Question: Can humans breathe underwater? Let me consider why. Step 1:",
        "Problem: Is the moon bigger than the earth? Let me compare sizes.",
        "Question: Do all birds fly? Let me think of examples step by step.",
        "Problem: What material is glass made from? Step 1: recall the process.",
        "Question: Is ice cream a solid or liquid? Let me analyze this.",
        "Problem: Which ocean is the largest? Let me think about geography.",
        "Question: Can cats see in complete darkness? Step 1: consider cat eyes.",
        "Problem: What happens when you mix oil and water? Let me reason:",
        "Question: Is a mushroom a plant? Let me think about biology.",
        "Problem: Which came first, the chicken or the egg? Let me reason:",
        "Question: Do fish drink water? Let me think about this step by step.",
        "Problem: Is the Great Wall visible from space? Step 1: consider the scale.",
        "Question: Can lightning strike the same place twice? Let me analyze:",
        "Problem: What causes rainbows? Step 1: think about light and water.",
        "Question: Do polar bears eat penguins? Let me think geographically.",
        "Problem: Is a coconut a fruit or nut? Let me classify step by step.",
        "Question: Can you fold paper more than 7 times? Step 1: think about it.",
        "Problem: Why is the sky blue? Let me reason through the physics.",
        "Question: Do goldfish really have 3 second memory? Let me consider evidence.",
        "Problem: Is Pluto still a planet? Step 1: recall the reclassification.",
        "Question: Can dogs see colors? Let me think about canine vision.",
        "Problem: Which metal is liquid at room temperature? Step 1: recall metals.",
        "Question: Is a hot dog a sandwich? Let me define terms step by step.",
        "Problem: Why do leaves change color in autumn? Let me reason:",
        "Question: Can you survive without sleep? Let me think about biology.",
        "Problem: Is zero even or odd? Step 1: recall the definition of even.",
        "Question: Do we use only 10% of our brain? Let me analyze this claim.",
        "Problem: What makes popcorn pop? Step 1: think about the kernel.",
        "Question: Is a square always a rectangle? Let me reason geometrically.",
        "Problem: Why do boats float? Step 1: consider density and buoyancy.",
        "Question: Can you hear sound in space? Let me think about this.",
        "Problem: Why is honey never spoiling? Step 1: consider its properties.",
    ],
    "free_prose": [
        "The old man sat by the river and watched the sun slowly",
        "She had always dreamed of traveling to distant lands where",
        "In the quiet moments before dawn the world seemed to hold",
        "He remembered the summer they spent together at the lake",
        "The city streets were empty except for a few stray cats",
        "Growing up in a small town meant everyone knew your",
        "The rain had been falling for three days straight and the",
        "After years of searching she finally found what she was",
        "The library was his favorite place in the world because",
        "When the music stopped everyone turned to look at the",
        "The village was nestled in a valley surrounded by mountains",
        "Every morning she would wake up before the alarm and",
        "The letter arrived on a Tuesday and changed everything about",
        "He had never been the kind of person who enjoyed",
        "The garden behind the house was overgrown with wildflowers",
        "They met at a coffee shop on the corner of",
        "The wind howled through the empty corridors of the old",
        "She picked up the phone and dialed the number she",
        "The children played in the yard while their parents watched",
        "It was the kind of day when nothing seemed to go",
        "The train pulled into the station just as the clock",
        "He opened the door to find a package sitting on",
        "The ocean stretched out before them endless and blue and",
        "She closed her eyes and tried to remember the last",
        "The market was crowded with vendors selling fruits and vegetables",
        "He walked along the beach collecting shells and thinking about",
        "The snow began to fall softly covering the ground in",
        "She sat at her desk staring at the blank page",
        "The forest was dense and dark but he kept walking",
        "They drove for hours along the winding mountain road until",
        "The sunset painted the sky in shades of orange and",
        "He found the old photograph tucked inside a book he",
        "The smell of fresh bread drifted from the bakery across",
        "She stood at the edge of the cliff looking down",
        "The cat curled up on the windowsill and watched the",
        "He sat in the waiting room flipping through old magazines",
        "The bridge swayed gently in the wind as they crossed",
        "She packed her bags and left without saying goodbye to",
        "The stars came out one by one as darkness fell",
        "He turned the corner and saw something he never expected",
        "The dog barked at the mailman every single morning without",
        "She opened the window and let the cool breeze fill",
        "The fire crackled in the hearth casting shadows on the",
        "He looked at his watch and realized he was late",
        "The flowers bloomed early that year surprising everyone in the",
        "She wrote the letter by hand because she wanted it",
        "The road ahead was long and straight disappearing into the",
        "He poured himself a cup of coffee and sat down",
        "The birds sang in the trees outside her bedroom window",
        "She smiled when she saw the familiar face in the",
    ],
    "structured_prose": [
        "Step 1: Preheat the oven to 350 degrees. Step 2:",
        "To assemble the furniture, first lay out all the pieces",
        "WARNING: Before operating this device, ensure that the power",
        "Instructions for use: Apply a thin layer of the solution",
        "Day 1 of the workout plan: Begin with a five minute",
        "Section 3.2: The applicant must submit all required documents",
        "Ingredients: 2 cups flour, 1 cup sugar, 3 eggs.",
        "Troubleshooting guide: If the device does not turn on check",
        "Meeting agenda: 1. Review of previous minutes. 2. Budget",
        "Installation guide: First, download the latest version from",
        "Step 1: Gather all materials needed for the project including",
        "Recipe: Chocolate chip cookies. Prep time: 15 minutes. Bake",
        "Safety guidelines: Always wear protective equipment when operating heavy",
        "Maintenance schedule: Check oil level every 3000 miles. Replace",
        "Lesson plan: Objective: Students will learn to identify the main",
        "Protocol 1: Begin by washing hands thoroughly with soap and",
        "Assembly instructions: Connect part A to part B using the",
        "Training module 1: Introduction to workplace safety. Section 1:",
        "Checklist: Before departure ensure all passengers have their boarding",
        "Operating procedure: Turn the machine on by pressing the green",
        "Phase 1: Planning. Identify the key stakeholders and establish",
        "User manual: To charge the device, connect the USB cable",
        "Quality control steps: Inspect the product for visible defects.",
        "First aid instructions: If the person is not breathing, call",
        "Setup wizard: Welcome to the installation process. Click next",
        "Lab procedure: Measure 50ml of the solution using a graduated",
        "Warranty terms: This product is covered for 12 months from",
        "Cooking directions: Bring 4 cups of water to a boil.",
        "Exercise routine: Monday: 30 minutes cardio. Tuesday: upper body",
        "Packing list for camping trip: tent, sleeping bag, flashlight,",
        "Daily schedule: 6:00 AM wake up. 6:30 AM breakfast.",
        "Emergency procedure: In case of fire, proceed to the nearest",
        "Configuration guide: Open the settings menu and navigate to",
        "Cleaning instructions: Wipe the surface with a damp cloth and",
        "Travel itinerary: Day 1: Arrive at the airport by 8",
        "Experiment setup: Place the beaker on the hot plate and",
        "Scoring rubric: Excellent (5 points): demonstrates thorough understanding of",
        "Budget breakdown: Housing: 30%. Food: 15%. Transportation: 10%.",
        "Return policy: Items may be returned within 30 days with",
        "Meeting minutes: Attendees: John, Sarah, Mike. Date: March 15.",
        "Study guide: Chapter 1 covers the fundamental concepts of",
        "Inventory list: Item 001: screwdriver set, quantity: 12, location:",
        "Filing instructions: Sort documents alphabetically by last name then",
        "Medication dosage: Adults: take 2 tablets every 6 hours with",
        "Project timeline: Week 1-2: Research phase. Week 3-4: Design",
        "Terms of service: By using this application you agree to",
        "Grading criteria: Participation: 10%. Homework: 20%. Midterm: 30%.",
        "Registration process: Step 1: Create an account with your email.",
        "Site preparation: Clear the area of debris and level the",
        "Booking confirmation: Your reservation at Hotel Grand is confirmed for",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "def sort_list(items):\n    for i in range(len(items)):\n        for",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def",
        "def binary_search(arr, target):\n    left, right = 0, len(arr)",
        "def reverse_string(s):\n    result = ''\n    for char in",
        "def count_words(text):\n    words = text.split()\n    return",
        "def merge_lists(a, b):\n    result = []\n    i, j = 0,",
        "def factorial(n):\n    if n == 0:\n        return 1\n    return",
        "def flatten(nested):\n    result = []\n    for item in nested:",
        "def max_element(arr):\n    best = arr[0]\n    for x in arr:",
        "def remove_duplicates(lst):\n    seen = set()\n    result = []",
        "def power(base, exp):\n    if exp == 0:\n        return 1",
        "def gcd(a, b):\n    while b != 0:\n        a, b = b,",
        "def matrix_multiply(A, B):\n    rows = len(A)\n    cols = len(",
        "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue",
        "def depth_first_search(graph, start):\n    visited = set()\n    stack =",
        "def insertion_sort(arr):\n    for i in range(1, len(arr)):",
        "def longest_common_prefix(strs):\n    if not strs:\n        return",
        "def two_sum(nums, target):\n    seen = {}\n    for i,",
        "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(',",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr",
        "def linked_list_reverse(head):\n    prev = None\n    current = head",
        "def tree_height(node):\n    if node is None:\n        return 0",
        "def palindrome_check(s):\n    s = s.lower().replace(' ', '')\n    return",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(",
        "def rotate_array(arr, k):\n    k = k % len(arr)\n    return",
        "def count_chars(text):\n    freq = {}\n    for ch in text:",
        "def find_median(arr):\n    arr.sort()\n    n = len(arr)\n    if",
        "def zip_lists(a, b):\n    result = []\n    for i in range(",
        "def string_to_int(s):\n    result = 0\n    for char in s:",
        "def matrix_transpose(matrix):\n    rows = len(matrix)\n    cols =",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot",
        "def bfs(graph, start):\n    visited = set()\n    queue = [start]",
        "def dijkstra(graph, source):\n    distances = {node: float('inf') for",
        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp",
        "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount",
        "def edit_distance(s1, s2):\n    m, n = len(s1), len(",
        "def topological_sort(graph):\n    visited = set()\n    result = []",
        "def lru_cache(capacity):\n    cache = {}\n    order = []\n    def",
        "def trie_insert(root, word):\n    node = root\n    for char in",
        "def heap_push(heap, val):\n    heap.append(val)\n    i = len(",
        "def union_find(n):\n    parent = list(range(n))\n    def find(",
        "def sliding_window_max(arr, k):\n    from collections import deque\n    dq",
        "def count_inversions(arr):\n    if len(arr) <= 1:\n        return arr",
        "def serialize_tree(root):\n    if root is None:\n        return '",
        "def permutations(arr):\n    if len(arr) <= 1:\n        return [arr",
        "def subsets(nums):\n    result = [[]]\n    for num in nums:",
        "def longest_increasing_subsequence(arr):\n    n = len(arr)\n    dp = [1]",
        "def min_path_sum(grid):\n    m, n = len(grid), len(grid",
    ],
    "poetry": [
        "Roses are red, violets are blue, sugar is sweet and",
        "Once upon a midnight dreary, while I pondered weak and",
        "Shall I compare thee to a summer day? Thou art more",
        "Two roads diverged in a yellow wood, and sorry I could",
        "The fog comes on little cat feet. It sits looking over",
        "I wandered lonely as a cloud that floats on high over",
        "Do not go gentle into that good night. Old age should",
        "Because I could not stop for Death, he kindly stopped for",
        "Twas brillig and the slithy toves did gyre and gimble in",
        "A thing of beauty is a joy forever. Its loveliness increases",
        "How do I love thee? Let me count the ways. I",
        "The road not taken is the one that leads to where",
        "In Xanadu did Kubla Khan a stately pleasure dome decree",
        "I think that I shall never see a poem lovely as",
        "The world is too much with us late and soon getting",
        "Tyger tyger burning bright in the forests of the night what",
        "O Captain my Captain our fearful trip is done the ship",
        "Hope is the thing with feathers that perches in the soul",
        "I hear America singing the varied carols I hear each one",
        "The love song of a wandering soul who seeks the distant",
        "Stop all the clocks cut off the telephone prevent the dog",
        "Whose woods these are I think I know his house is",
        "Let us go then you and I when the evening is",
        "I carry your heart with me I carry it in my",
        "The wind was a torrent of darkness among the gusty trees",
        "Season of mists and mellow fruitfulness close bosom friend of",
        "My heart leaps up when I behold a rainbow in the",
        "When I heard the learned astronomer when the proofs the figures",
        "Out of the night that covers me black as the pit",
        "Wild nights wild nights were I with thee wild nights should",
        "The apparition of these faces in the crowd petals on a",
        "Gather ye rosebuds while ye may old time is still a",
        "Come live with me and be my love and we will",
        "Death be not proud though some have called thee mighty and",
        "I met a traveler from an antique land who said two",
        "Still I rise out of the huts of history shame I",
        "The lake isle of Innisfree I will arise and go now",
        "Annabel Lee it was many and many a year ago in",
        "The raven quoth the raven nevermore upon the pallid bust of",
        "Fire and ice some say the world will end in fire",
        "If you can keep your head when all about you are",
        "The second coming turning and turning in the widening gyre the",
        "Ode on a Grecian urn thou still unravished bride of",
        "So much depends upon a red wheel barrow glazed with rain",
        "Not all those who wander are lost the old that is",
        "Invictus out of the night that covers me I am the",
        "The love that dare not speak its name in praise of",
        "Daffodils I wandered lonely as a cloud that floats on",
        "Ozymandias look on my works ye mighty and despair nothing",
        "The hollow men we are the hollow men we are the",
    ],
}

# Code staircase signatures (100 for Qwen-7B)
CODE_SIGS = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'),
    ('def greet(name):', 'str'), ('def farewell(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def to_lower(text):', 'str'), ('def capitalize(text):', 'str'), ('def strip_whitespace(text):', 'str'),
    ('def reverse_string(s):', 'str'), ('def repeat_string(s, n):', 'str'),
    ('def join_words(words):', 'str'), ('def first_word(text):', 'str'), ('def last_word(text):', 'str'),
    ('def remove_spaces(s):', 'str'), ('def replace_char(s, old, new):', 'str'),
    ('def first_name(full_name):', 'str'), ('def last_name(full_name):', 'str'),
    ('def format_date(year, month, day):', 'str'), ('def format_time(hours, minutes):', 'str'),
    ('def to_binary(n):', 'str'), ('def to_hex(n):', 'str'), ('def slug(text):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_odd(n):', 'bool'), ('def is_positive(x):', 'bool'),
    ('def is_negative(x):', 'bool'), ('def is_zero(x):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def is_palindrome(s):', 'bool'), ('def is_empty(s):', 'bool'), ('def is_sorted(items):', 'bool'),
    ('def contains(items, target):', 'bool'), ('def starts_with(text, prefix):', 'bool'),
    ('def ends_with(text, suffix):', 'bool'), ('def is_alpha(text):', 'bool'),
    ('def is_digit(text):', 'bool'), ('def is_upper(text):', 'bool'), ('def is_lower(text):', 'bool'),
    ('def has_duplicates(items):', 'bool'), ('def all_positive(numbers):', 'bool'),
    ('def any_negative(numbers):', 'bool'), ('def is_valid_email(text):', 'bool'),
    ('def get_evens(numbers):', 'list'), ('def get_odds(numbers):', 'list'),
    ('def filter_positive(numbers):', 'list'), ('def filter_negative(numbers):', 'list'),
    ('def unique(items):', 'list'), ('def flatten(nested):', 'list'),
    ('def sort_ascending(items):', 'list'), ('def sort_descending(items):', 'list'),
    ('def reverse_list(items):', 'list'), ('def split_words(text):', 'list'),
    ('def split_lines(text):', 'list'), ('def split_chars(text):', 'list'),
    ('def zip_lists(a, b):', 'list'), ('def merge_sorted(a, b):', 'list'),
    ('def remove_duplicates(items):', 'list'), ('def take(items, n):', 'list'),
    ('def drop(items, n):', 'list'), ('def chunk(items, size):', 'list'),
    ('def interleave(a, b):', 'list'), ('def get_keys(d):', 'list'),
    ('def average(numbers):', 'float'), ('def median(numbers):', 'float'),
    ('def variance(numbers):', 'float'), ('def std_dev(numbers):', 'float'),
    ('def to_celsius(f):', 'float'), ('def to_fahrenheit(c):', 'float'),
    ('def percentage(part, total):', 'float'), ('def ratio(a, b):', 'float'),
    ('def distance(x1, y1, x2, y2):', 'float'), ('def magnitude(x, y, z):', 'float'),
    ('def dot_product(a, b):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def circle_area(radius):', 'float'), ('def sphere_volume(radius):', 'float'),
    ('def triangle_area(base, height):', 'float'), ('def hypotenuse(a, b):', 'float'),
    ('def sigmoid(x):', 'float'), ('def relu(x):', 'float'),
    ('def log_base(x, base):', 'float'), ('def square_root(x):', 'float'),
]


def bootstrap_gap(X_probe, X_baseline, y, n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap CI on the gap between probe and baseline accuracy."""
    rng = np.random.RandomState(seed)
    gaps = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        oob = list(set(range(len(y))) - set(idx))
        if len(oob) < 5 or len(np.unique(y[idx])) < 2:
            continue
        try:
            clf_p = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
            clf_p.fit(X_probe[idx], y[idx])
            acc_p = clf_p.score(X_probe[oob], y[oob])
            clf_b = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
            clf_b.fit(X_baseline[idx], y[idx])
            acc_b = clf_b.score(X_baseline[oob], y[oob])
            gaps.append(acc_p - acc_b)
        except:
            continue
    if len(gaps) < 50:
        return 0.0, 0.0, 0.0, False
    mean_gap = np.mean(gaps)
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    significant = ci_lo > 0  # entire CI above zero
    return float(mean_gap), float(ci_lo), float(ci_hi), bool(significant)


def run_domain_study(model, model_name, domains, min_target=10):
    """Run full domain study with bootstrap CIs."""
    logger.info(f"\n  DOMAIN STUDY: {model_name}")
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    W_E = model.W_E.detach()
    
    results = {"model": model_name}
    
    for domain_name, prompts in domains.items():
        logger.info(f"\n  Domain: {domain_name} ({len(prompts)} prompts)")
        
        # Generate
        all_seqs = []
        for pi, prompt in enumerate(prompts):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
            if (pi + 1) % 10 == 0:
                logger.info(f"    Generated {pi+1}/{len(prompts)}")
        
        # 25 train / 25 test
        train_seqs = all_seqs[:25]
        test_seqs = all_seqs[25:]
        
        domain_results = {}
        for k in K_VALUES:
            test_tgts = []
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                for n in range(pl, len(ids) - k):
                    test_tgts.append(ids[n + k])
            
            tc = Counter(test_tgts)
            frequent = {t for t, c in tc.items() if c >= min_target}
            t2i = {t: i for i, t in enumerate(sorted(frequent))}
            n_cls = len(t2i)
            
            if n_cls < 3:
                logger.info(f"    K={k}: {n_cls} classes, skip")
                continue
            
            activations = {l: [] for l in layers}
            ctx_embs, labels = [], []
            
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    _, cache = model.run_with_cache(inp,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
                for n in range(pl, len(ids) - k):
                    tgt = ids[n + k]
                    if tgt not in t2i: continue
                    labels.append(t2i[tgt])
                    for l in layers:
                        activations[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                    ws = max(0, n - 4)
                    ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                    ctx_embs.append(ctx.mean(axis=0))
                del cache; torch.cuda.empty_cache()
            
            labels = np.array(labels)
            n_ex = len(labels)
            if n_ex < 30:
                logger.info(f"    K={k}: {n_ex} examples, skip")
                continue
            cc = Counter(labels)
            chance = max(cc.values()) / n_ex
            min_c = min(cc.values())
            n_splits = min(5, min_c)
            if n_splits < 2: continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Context baseline
            X_ctx_raw = np.stack(ctx_embs)
            scaler_ctx = StandardScaler()
            X_ctx_s = scaler_ctx.fit_transform(X_ctx_raw)
            X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_ctx_s)
            
            ctx_accs = []
            for seed in PROBE_SEEDS:
                cv_s = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                      X_ctx, labels, cv=cv_s, scoring="accuracy").mean()
                ctx_accs.append(acc)
            ctx_acc = np.mean(ctx_accs)
            
            # Best probe across layers (with multi-seed)
            best_p, best_l = 0, 0
            layer_results = {}
            best_X_probe = None
            for l in layers:
                X_raw = np.stack(activations[l])
                scaler_p = StandardScaler()
                X_s = scaler_p.fit_transform(X_raw)
                X_pca = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_s)
                
                p_accs = []
                for seed in PROBE_SEEDS:
                    cv_s = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                          X_pca, labels, cv=cv_s, scoring="accuracy").mean()
                    p_accs.append(acc)
                p_acc = np.mean(p_accs)
                layer_results[str(l)] = float(p_acc)
                if p_acc > best_p:
                    best_p, best_l = p_acc, l
                    best_X_probe = X_pca
            
            gap = best_p - ctx_acc
            
            # Bootstrap CI on the gap
            boot_gap, boot_lo, boot_hi, boot_sig = bootstrap_gap(
                best_X_probe, X_ctx, labels, N_BOOTSTRAP, 42)
            
            logger.info(f"    K={k}: {n_ex} ex, {n_cls} cls | chance={chance:.3f} "
                        f"ctx={ctx_acc:.3f} probe(L{best_l})={best_p:.3f} "
                        f"gap={gap:+.3f} [{boot_lo:+.3f},{boot_hi:+.3f}] "
                        f"{'SIG' if boot_sig else 'ns'}")
            
            domain_results[f"k{k}"] = {
                "n_examples": int(n_ex), "n_classes": int(n_cls),
                "chance": float(chance), "context": float(ctx_acc),
                "best_probe": float(best_p), "best_layer": int(best_l),
                "gap": float(gap),
                "gap_ci_lo": float(boot_lo), "gap_ci_hi": float(boot_hi),
                "gap_significant": bool(boot_sig),
                "per_layer": layer_results,
            }
        
        results[domain_name] = domain_results
    
    return results


def run_attn_mlp(model, model_name, domains_to_test, k=3):
    """Attention vs MLP decomposition on specified domains."""
    logger.info(f"\n  ATTN vs MLP DECOMPOSITION: {model_name}, K={k}")
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    results = {"model": model_name, "k": k}
    
    for domain_name, prompts in domains_to_test.items():
        logger.info(f"\n  Domain: {domain_name}")
        
        all_seqs = []
        for prompt in prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        
        test_seqs = all_seqs[25:]
        
        test_tgts = []
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            for n in range(pl, len(ids) - k):
                test_tgts.append(ids[n + k])
        tc = Counter(test_tgts)
        frequent = {t for t, c in tc.items() if c >= 8}
        t2i = {t: i for i, t in enumerate(sorted(frequent))}
        n_cls = len(t2i)
        if n_cls < 3:
            logger.info(f"    {n_cls} classes, skip")
            continue
        
        attn_acts = {l: [] for l in layers}
        mlp_acts = {l: [] for l in layers}
        resid_acts = {l: [] for l in layers}
        labels = []
        
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            inp = torch.tensor([ids], device="cuda")
            cache_names = []
            for l in layers:
                cache_names.extend([
                    f"blocks.{l}.hook_resid_post",
                    f"blocks.{l}.hook_attn_out",
                    f"blocks.{l}.hook_mlp_out"])
            with torch.no_grad():
                _, cache = model.run_with_cache(inp, names_filter=cache_names)
            for n in range(pl, len(ids) - k):
                tgt = ids[n + k]
                if tgt not in t2i: continue
                labels.append(t2i[tgt])
                for l in layers:
                    resid_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                    attn_acts[l].append(cache[f"blocks.{l}.hook_attn_out"][0, n, :].cpu().numpy())
                    mlp_acts[l].append(cache[f"blocks.{l}.hook_mlp_out"][0, n, :].cpu().numpy())
            del cache; torch.cuda.empty_cache()
        
        labels = np.array(labels)
        n_ex = len(labels)
        min_c = min(Counter(labels).values())
        n_splits = min(5, min_c)
        if n_splits < 2: continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        logger.info(f"    {n_ex} ex, {n_cls} cls")
        logger.info(f"    {'Layer':>6} {'Resid':>8} {'Attn':>8} {'MLP':>8} {'Winner':>8}")
        
        domain_decomp = {}
        for l in layers:
            accs = {}
            for name, acts in [("resid", resid_acts), ("attn", attn_acts), ("mlp", mlp_acts)]:
                X = np.stack(acts[l])
                X = StandardScaler().fit_transform(X)
                if X.shape[1] > PCA_DIM:
                    X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X)
                acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                      X, labels, cv=cv, scoring="accuracy").mean()
                accs[name] = float(acc)
            winner = "ATTN" if accs["attn"] > accs["mlp"] else "MLP"
            logger.info(f"    L{l:>4} {accs['resid']:>8.4f} {accs['attn']:>8.4f} "
                        f"{accs['mlp']:>8.4f} {winner:>8}")
            domain_decomp[str(l)] = accs
        
        results[domain_name] = domain_decomp
    
    return results


def run_code_staircase(model, model_name, sigs):
    """Full code staircase with name+params baseline."""
    logger.info(f"\n  CODE STAIRCASE: {model_name} ({len(sigs)} sigs)")
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    targets = sorted(set(r for _, r in sigs))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[r] for _, r in sigs])
    
    # Extract activations
    all_acts = {l: [] for l in layers}
    for si, (sig, ret) in enumerate(sigs):
        tokens = model.to_tokens(sig + "\n    ", prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
        for l in layers:
            all_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu().numpy())
        del cache; torch.cuda.empty_cache()
        if (si + 1) % 20 == 0:
            logger.info(f"    Extracted {si+1}/{len(sigs)}")
    
    cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())),
                         shuffle=True, random_state=42)
    
    # Name+params baseline
    np_texts = [sig.replace("def ", "").replace(":", "").replace("(", " ").replace(")", " ").replace(",", " ")
                for sig, _ in sigs]
    tfidf = TfidfVectorizer()
    X_np = tfidf.fit_transform(np_texts).toarray()
    np_acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                              X_np, labels, cv=cv, scoring="accuracy").mean()
    logger.info(f"  Name+Params: {np_acc:.4f}")
    
    # Probe each layer
    logger.info(f"  {'Layer':>6} {'Probe':>8} {'N+P':>8} {'Gap':>8}")
    results = {"model": model_name, "n_sigs": len(sigs), "name_params": float(np_acc)}
    layer_results = {}
    for l in layers:
        X = np.stack(all_acts[l])
        X = StandardScaler().fit_transform(X)
        if X.shape[1] > PCA_DIM:
            X = PCA(n_components=min(PCA_DIM, X.shape[0]-1), random_state=42).fit_transform(X)
        acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                              X, labels, cv=cv, scoring="accuracy").mean()
        gap = acc - np_acc
        logger.info(f"  L{l:>4} {acc:>8.4f} {np_acc:>8.4f} {gap:>+8.4f}")
        layer_results[str(l)] = {"probe": float(acc), "gap": float(gap)}
    
    results["layers"] = layer_results
    best = max(layer_results.items(), key=lambda x: x[1]["probe"])
    results["best_probe"] = float(best[1]["probe"])
    results["best_layer"] = best[0]
    results["best_gap"] = float(best[1]["gap"])
    return results


def run_cross_domain_transfer(model, model_name, domain_data):
    """Cross-domain probe transfer: train on one domain, test on another."""
    logger.info(f"\n  CROSS-DOMAIN TRANSFER: {model_name}")
    n_layers = model.cfg.n_layers
    mid_layer = n_layers // 2
    k = 3
    
    # Use pre-generated sequences from domain_data
    # Binary task: is next token in top-20 most common tokens overall?
    
    # First pass: find overall top-20 tokens across all domains
    all_tokens = []
    for domain_name, seqs in domain_data.items():
        for seq in seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            for n in range(pl, len(ids)):
                all_tokens.append(ids[n])
    
    top20 = set(t for t, _ in Counter(all_tokens).most_common(20))
    logger.info(f"  Top-20 tokens cover {sum(1 for t in all_tokens if t in top20)/len(all_tokens)*100:.1f}% of all tokens")
    
    # Extract binary labels and activations per domain
    domain_features = {}
    for domain_name, seqs in domain_data.items():
        test_seqs = seqs[25:]  # use test split
        acts, labels = [], []
        
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                _, cache = model.run_with_cache(inp,
                    names_filter=[f"blocks.{mid_layer}.hook_resid_post"])
            for n in range(pl, len(ids) - k):
                target = ids[n + k]
                acts.append(cache[f"blocks.{mid_layer}.hook_resid_post"][0, n, :].cpu().numpy())
                labels.append(1 if target in top20 else 0)
            del cache; torch.cuda.empty_cache()
        
        X = np.stack(acts)
        X = StandardScaler().fit_transform(X)
        if X.shape[1] > PCA_DIM:
            X = PCA(n_components=min(PCA_DIM, X.shape[0]-1), random_state=42).fit_transform(X)
        y = np.array(labels)
        domain_features[domain_name] = (X, y)
        logger.info(f"    {domain_name}: {len(y)} examples, {sum(y)} common / {len(y)-sum(y)} rare")
    
    # Cross-domain transfer matrix
    transfer_domains = ["chain_of_thought", "code", "free_prose", "poetry"]
    results = {"model": model_name, "layer": mid_layer, "k": k}
    
    logger.info(f"\n    {'Train\\Test':<20}" + "".join(f"{d[:8]:>10}" for d in transfer_domains))
    
    transfer_matrix = {}
    for train_dom in transfer_domains:
        if train_dom not in domain_features: continue
        X_train, y_train = domain_features[train_dom]
        row = {}
        row_str = f"    {train_dom[:18]:<20}"
        
        for test_dom in transfer_domains:
            if test_dom not in domain_features: continue
            X_test, y_test = domain_features[test_dom]
            
            if train_dom == test_dom:
                # Within-domain CV
                cv = StratifiedKFold(n_splits=min(5, min(Counter(y_train).values())),
                                     shuffle=True, random_state=42)
                acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                      X_train, y_train, cv=cv, scoring="accuracy").mean()
            else:
                # Cross-domain
                clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
            
            row[test_dom] = float(acc)
            row_str += f"{acc:>10.3f}"
        
        transfer_matrix[train_dom] = row
        logger.info(row_str)
    
    results["transfer_matrix"] = transfer_matrix
    return results


def main():
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE OVERNIGHT RUN — EMNLP-GRADE")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    all_results = {}
    
    # ============================================================
    # PHASE 1: GPT-J-6B
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: GPT-J-6B")
    logger.info("=" * 70)
    
    model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", device="cuda", dtype=torch.float16)
    model.eval()
    
    # 1a: Large domain study with bootstrap CIs
    logger.info("\n--- 1a: Large domain study (50 prompts, 7 conditions) ---")
    gptj_domains = run_domain_study(model, "EleutherAI/gpt-j-6b", DOMAINS)
    all_results["gptj_domains_50"] = gptj_domains
    save_intermediate(all_results, "phase1a_domains")
    
    # 1b: Attention vs MLP on CoT, code, poetry
    logger.info("\n--- 1b: Attention vs MLP decomposition ---")
    attn_mlp_domains = {k: v for k, v in DOMAINS.items() 
                        if k in ["chain_of_thought", "code", "poetry"]}
    gptj_attn_mlp = run_attn_mlp(model, "EleutherAI/gpt-j-6b", attn_mlp_domains)
    all_results["gptj_attn_mlp"] = gptj_attn_mlp
    save_intermediate(all_results, "phase1b_attnmlp")
    
    # 1c: Cross-domain transfer
    logger.info("\n--- 1c: Cross-domain probe transfer ---")
    # Generate sequences for transfer (reuse domain study data)
    domain_seqs = {}
    for domain_name, prompts in DOMAINS.items():
        if domain_name in ["chain_of_thought", "code", "free_prose", "poetry"]:
            seqs = []
            for prompt in prompts:
                tokens = model.to_tokens(prompt, prepend_bos=True)
                with torch.no_grad():
                    gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
                seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
            domain_seqs[domain_name] = seqs
    
    gptj_transfer = run_cross_domain_transfer(model, "EleutherAI/gpt-j-6b", domain_seqs)
    all_results["gptj_transfer"] = gptj_transfer
    save_intermediate(all_results, "phase1c_transfer")
    
    del model; torch.cuda.empty_cache(); gc.collect()
    
    # ============================================================
    # PHASE 2: Qwen-7B
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Qwen-7B")
    logger.info("=" * 70)
    
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B", device="cuda", dtype=torch.float16)
    model.eval()
    
    # 2a: Code staircase (100 sigs)
    logger.info("\n--- 2a: Code staircase (100 sigs) ---")
    qwen_code = run_code_staircase(model, "Qwen/Qwen2.5-7B", CODE_SIGS)
    all_results["qwen7b_code"] = qwen_code
    save_intermediate(all_results, "phase2a_code")
    
    # 2b: Domain study (50 prompts)
    logger.info("\n--- 2b: Domain study (50 prompts) ---")
    qwen_domains = run_domain_study(model, "Qwen/Qwen2.5-7B", DOMAINS)
    all_results["qwen7b_domains_50"] = qwen_domains
    save_intermediate(all_results, "phase2b_domains")
    
    # 2c: Attn vs MLP on CoT
    logger.info("\n--- 2c: Attention vs MLP on CoT ---")
    qwen_attn_mlp = run_attn_mlp(model, "Qwen/Qwen2.5-7B", 
                                  {"chain_of_thought": DOMAINS["chain_of_thought"]})
    all_results["qwen7b_attn_mlp"] = qwen_attn_mlp
    save_intermediate(all_results, "phase2c_attnmlp")
    
    del model; torch.cuda.empty_cache(); gc.collect()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info("OVERNIGHT RUN COMPLETE — SUMMARY")
    logger.info(f"{'='*70}")
    
    # Domain study summary
    for model_key, model_label in [("gptj_domains_50", "GPT-J-6B"), 
                                     ("qwen7b_domains_50", "Qwen-7B")]:
        ds = all_results.get(model_key, {})
        logger.info(f"\n  {model_label} DOMAIN STUDY (50 prompts):")
        logger.info(f"  {'Domain':<30} {'K=1':>8} {'K=3':>8} {'K=5':>8} {'K=3 CI':>16} {'Sig?':>5}")
        for dom in ["chain_of_thought", "chain_of_thought_scrambled", 
                     "chain_of_thought_nonmath", "free_prose", "structured_prose",
                     "code", "poetry"]:
            if dom not in ds: continue
            d = ds[dom]
            k1 = d.get("k1", {}).get("gap", None)
            k3 = d.get("k3", {}).get("gap", None)
            k5 = d.get("k5", {}).get("gap", None)
            ci = f"[{d.get('k3',{}).get('gap_ci_lo',0):+.3f},{d.get('k3',{}).get('gap_ci_hi',0):+.3f}]" if k3 else ""
            sig = "YES" if d.get("k3", {}).get("gap_significant", False) else "no"
            logger.info(f"  {dom:<30} {k1 if k1 is not None else '---':>8} "
                        f"{k3 if k3 is not None else '---':>8} "
                        f"{k5 if k5 is not None else '---':>8} {ci:>16} {sig:>5}")
    
    # Attn vs MLP
    for model_key, label in [("gptj_attn_mlp", "GPT-J-6B"), ("qwen7b_attn_mlp", "Qwen-7B")]:
        am = all_results.get(model_key, {})
        if "chain_of_thought" in am:
            logger.info(f"\n  {label} ATTN vs MLP (CoT, K=3):")
            for l, accs in am["chain_of_thought"].items():
                if isinstance(accs, dict) and "resid" in accs:
                    logger.info(f"    L{l}: resid={accs['resid']:.4f} attn={accs['attn']:.4f} mlp={accs['mlp']:.4f}")
    
    # Qwen code
    qc = all_results.get("qwen7b_code", {})
    if "best_probe" in qc:
        logger.info(f"\n  QWEN-7B CODE: probe={qc['best_probe']:.4f} N+P={qc['name_params']:.4f} "
                    f"gap={qc['best_gap']:+.4f}")
    
    # Transfer
    tr = all_results.get("gptj_transfer", {})
    if "transfer_matrix" in tr:
        logger.info(f"\n  CROSS-DOMAIN TRANSFER (GPT-J, K=3):")
        logger.info(f"  Diagonal = within-domain, off-diagonal = transfer")
    
    # Save final
    save_intermediate(all_results, "complete")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — OVERNIGHT BULLETPROOFING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
