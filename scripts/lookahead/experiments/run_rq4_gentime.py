#!/usr/bin/env python3
"""
RQ4 GENERATION-TIME COMMITMENT PROBING
========================================
The most direct test of planning: does the model build planning
representations DURING code generation, beyond what the signature gives?

Design:
- Start with function signature (e.g., "def sum_list(numbers):\n    ")
- Generate 20 tokens autoregressively
- At EACH generated token, extract activations from last position
- Probe: can we predict return type at each generation step?
- Compare to name+params baseline (FIXED at signature, doesn't change)

Key finding we're looking for:
- If probe accuracy at step K > name+params → model ADDS planning info during generation
- If probe accuracy stays ≤ name+params → model just propagates signature info

Also tracks:
- What tokens are generated (to identify when type-revealing tokens appear)
- Generated-text baseline (BoW of generated tokens so far)
- Commitment point: first step where probe exceeds threshold consistently

Models: GPT-2 XL, Pythia-2.8B, SantaCoder, CodeLlama-7B
500 examples (100 per type), 20 generation steps, 3 layers per model
"""

import json, os, sys, hashlib, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# ================================================================
# DATASET (same 500 from final run, abbreviated)
# ================================================================
DATASET_500 = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'), ('def gcd(a, b):', 'int'), ('def lcm(a, b):', 'int'),
    ('def abs_value(n):', 'int'), ('def sign(n):', 'int'), ('def clamp(val, lo, hi):', 'int'),
    ('def popcount(n):', 'int'), ('def manhattan_distance(x1, y1, x2, y2):', 'int'),
    ('def depth(tree):', 'int'), ('def height(node):', 'int'), ('def size(collection):', 'int'),
    ('def count_occurrences(items, target):', 'int'), ('def bit_length(n):', 'int'),
    ('def digit_sum(n):', 'int'), ('def count_zeros(numbers):', 'int'),
    ('def count_positives(numbers):', 'int'), ('def count_negatives(numbers):', 'int'),
    ('def max_depth(tree):', 'int'), ('def min_depth(tree):', 'int'),
    ('def num_leaves(tree):', 'int'), ('def num_nodes(tree):', 'int'),
    ('def edit_distance(s1, s2):', 'int'), ('def longest_common_prefix_len(s1, s2):', 'int'),
    ('def count_unique(items):', 'int'), ('def binary_search(arr, target):', 'int'),
    ('def partition_index(arr, pivot):', 'int'), ('def rank(items, target):', 'int'),
    ('def floor_div(a, b):', 'int'), ('def ceil_div(a, b):', 'int'),
    ('def count_consonants(text):', 'int'), ('def count_spaces(text):', 'int'),
    ('def count_digits(text):', 'int'), ('def count_upper(text):', 'int'),
    ('def count_lower(text):', 'int'), ('def count_special(text):', 'int'),
    ('def sum_digits(n):', 'int'), ('def reverse_int(n):', 'int'),
    ('def max_subarray_sum(arr):', 'int'), ('def min_subarray_sum(arr):', 'int'),
    ('def longest_streak(items):', 'int'), ('def shortest_path_len(graph, start, end):', 'int'),
    ('def degree(graph, node):', 'int'), ('def in_degree(graph, node):', 'int'),
    ('def out_degree(graph, node):', 'int'), ('def num_components(graph):', 'int'),
    ('def num_edges(graph):', 'int'), ('def diameter(graph):', 'int'),
    ('def page_count(items, page_size):', 'int'), ('def word_count(paragraph):', 'int'),
    ('def sentence_count(text):', 'int'), ('def paragraph_count(text):', 'int'),
    ('def line_number(text, target):', 'int'), ('def column_count(matrix):', 'int'),
    ('def row_count(matrix):', 'int'), ('def rank_matrix(matrix):', 'int'),
    ('def trace(matrix):', 'int'), ('def determinant_int(matrix):', 'int'),
    ('def hamming_weight(n):', 'int'), ('def leading_zeros(n):', 'int'),
    ('def trailing_zeros(n):', 'int'), ('def next_power_of_two(n):', 'int'),
    ('def prev_power_of_two(n):', 'int'), ('def count_bits(n):', 'int'),
    ('def xor_sum(a, b):', 'int'), ('def and_sum(a, b):', 'int'),
    ('def or_sum(a, b):', 'int'), ('def nand_sum(a, b):', 'int'),
    ('def count_primes(n):', 'int'), ('def nth_prime(n):', 'int'),
    ('def euler_totient(n):', 'int'), ('def mobius(n):', 'int'),
    ('def catalan(n):', 'int'), ('def bell_number(n):', 'int'),
    ('def stirling(n, k):', 'int'), ('def binomial(n, k):', 'int'),
    ('def tribonacci(n):', 'int'), ('def lucas(n):', 'int'),
    ('def ackermann(m, n):', 'int'), ('def collatz_steps(n):', 'int'),
    ('def digital_root(n):', 'int'), ('def perfect_number_count(n):', 'int'),
    ('def greet(name):', 'str'), ('def farewell(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def to_lower(text):', 'str'), ('def capitalize(text):', 'str'), ('def strip_whitespace(text):', 'str'),
    ('def reverse_string(s):', 'str'), ('def repeat_string(s, n):', 'str'),
    ('def join_words(words):', 'str'), ('def first_word(text):', 'str'), ('def last_word(text):', 'str'),
    ('def remove_spaces(s):', 'str'), ('def replace_char(s, old, new):', 'str'),
    ('def first_name(full_name):', 'str'), ('def last_name(full_name):', 'str'),
    ('def format_date(year, month, day):', 'str'), ('def format_time(hours, minutes):', 'str'),
    ('def to_binary(n):', 'str'), ('def to_hex(n):', 'str'), ('def to_roman(n):', 'str'),
    ('def slug(text):', 'str'), ('def title_case(text):', 'str'), ('def snake_case(text):', 'str'),
    ('def camel_case(text):', 'str'), ('def pad_left(s, width, char):', 'str'),
    ('def pad_right(s, width, char):', 'str'), ('def truncate(text, length):', 'str'),
    ('def extract_domain(email):', 'str'), ('def extract_extension(filename):', 'str'),
    ('def base64_encode(data):', 'str'), ('def base64_decode(data):', 'str'),
    ('def url_encode(text):', 'str'), ('def url_decode(text):', 'str'),
    ('def html_escape(text):', 'str'), ('def html_unescape(text):', 'str'),
    ('def caesar_cipher(text, shift):', 'str'), ('def rot13(text):', 'str'),
    ('def morse_encode(text):', 'str'), ('def morse_decode(code):', 'str'),
    ('def pig_latin(word):', 'str'), ('def reverse_words(text):', 'str'),
    ('def remove_vowels(text):', 'str'), ('def remove_consonants(text):', 'str'),
    ('def remove_punctuation(text):', 'str'), ('def remove_digits(text):', 'str'),
    ('def center_text(text, width):', 'str'), ('def left_justify(text, width):', 'str'),
    ('def right_justify(text, width):', 'str'), ('def wrap_text(text, width):', 'str'),
    ('def indent(text, spaces):', 'str'), ('def dedent(text):', 'str'),
    ('def quote(text):', 'str'), ('def unquote(text):', 'str'),
    ('def pluralize(word):', 'str'), ('def singularize(word):', 'str'),
    ('def ordinalize(n):', 'str'), ('def humanize(number):', 'str'),
    ('def kebab_case(text):', 'str'), ('def dot_case(text):', 'str'),
    ('def path_join(parts):', 'str'), ('def path_basename(path):', 'str'),
    ('def path_dirname(path):', 'str'), ('def path_stem(path):', 'str'),
    ('def get_initials(name):', 'str'), ('def abbreviate(text):', 'str'),
    ('def expand_tabs(text, size):', 'str'), ('def normalize_whitespace(text):', 'str'),
    ('def mask_email(email):', 'str'), ('def mask_phone(phone):', 'str'),
    ('def format_currency(amount, symbol):', 'str'), ('def format_percentage(value):', 'str'),
    ('def format_number(n, decimals):', 'str'), ('def format_bytes(size):', 'str'),
    ('def char_at(s, index):', 'str'), ('def substring(s, start, end):', 'str'),
    ('def encrypt(text, key):', 'str'), ('def decrypt(text, key):', 'str'),
    ('def compress(text):', 'str'), ('def decompress(data):', 'str'),
    ('def hash_string(text):', 'str'), ('def checksum(data):', 'str'),
    ('def generate_password(length):', 'str'), ('def generate_uuid():', 'str'),
    ('def timestamp_to_string(ts):', 'str'), ('def date_to_string(date):', 'str'),
    ('def json_to_string(obj):', 'str'), ('def xml_to_string(obj):', 'str'),
    ('def error_message(code):', 'str'), ('def status_text(code):', 'str'),
    ('def color_name(rgb):', 'str'), ('def hex_color(r, g, b):', 'str'),
    ('def render_template(template, data):', 'str'), ('def interpolate(template, values):', 'str'),
    ('def transliterate(text):', 'str'), ('def romanize(text):', 'str'),
    ('def sanitize(text):', 'str'), ('def escape_regex(text):', 'str'),
    ('def mime_type(filename):', 'str'), ('def file_extension(mime):', 'str'),
    ('def shorten_url(url):', 'str'), ('def expand_url(short):', 'str'),
    ('def to_ascii(text):', 'str'), ('def to_utf8(text):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_odd(n):', 'bool'), ('def is_positive(x):', 'bool'),
    ('def is_negative(x):', 'bool'), ('def is_zero(x):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def is_palindrome(s):', 'bool'), ('def is_empty(s):', 'bool'), ('def is_sorted(items):', 'bool'),
    ('def contains(items, target):', 'bool'), ('def starts_with(text, prefix):', 'bool'),
    ('def ends_with(text, suffix):', 'bool'), ('def is_alpha(text):', 'bool'),
    ('def is_digit(text):', 'bool'), ('def is_upper(text):', 'bool'), ('def is_lower(text):', 'bool'),
    ('def has_duplicates(items):', 'bool'), ('def all_positive(numbers):', 'bool'),
    ('def any_negative(numbers):', 'bool'), ('def is_valid_email(text):', 'bool'),
    ('def is_substring(s, sub):', 'bool'), ('def is_anagram(s1, s2):', 'bool'),
    ('def is_power_of_two(n):', 'bool'), ('def is_leap_year(year):', 'bool'),
    ('def is_vowel(char):', 'bool'), ('def is_consonant(char):', 'bool'),
    ('def file_exists(path):', 'bool'), ('def is_balanced(parens):', 'bool'),
    ('def is_symmetric(matrix):', 'bool'), ('def is_connected(graph):', 'bool'),
    ('def is_bipartite(graph):', 'bool'), ('def is_cyclic(graph):', 'bool'),
    ('def is_tree(graph):', 'bool'), ('def is_dag(graph):', 'bool'),
    ('def is_complete(graph):', 'bool'), ('def is_planar(graph):', 'bool'),
    ('def is_perfect(n):', 'bool'), ('def is_abundant(n):', 'bool'),
    ('def is_deficient(n):', 'bool'), ('def is_square(n):', 'bool'),
    ('def is_cube(n):', 'bool'), ('def is_fibonacci(n):', 'bool'),
    ('def is_armstrong(n):', 'bool'), ('def is_harshad(n):', 'bool'),
    ('def is_happy(n):', 'bool'), ('def is_narcissistic(n):', 'bool'),
    ('def is_valid_ip(text):', 'bool'), ('def is_valid_url(text):', 'bool'),
    ('def is_valid_phone(text):', 'bool'), ('def is_valid_date(text):', 'bool'),
    ('def is_valid_json(text):', 'bool'), ('def is_valid_xml(text):', 'bool'),
    ('def is_valid_hex(text):', 'bool'), ('def is_valid_binary(text):', 'bool'),
    ('def is_valid_base64(text):', 'bool'), ('def is_valid_uuid(text):', 'bool'),
    ('def is_ascii(text):', 'bool'), ('def is_utf8(text):', 'bool'),
    ('def is_printable(text):', 'bool'), ('def is_whitespace(text):', 'bool'),
    ('def is_title(text):', 'bool'), ('def is_capitalized(text):', 'bool'),
    ('def is_numeric(text):', 'bool'), ('def is_alphanumeric(text):', 'bool'),
    ('def is_blank(text):', 'bool'), ('def is_null(value):', 'bool'),
    ('def is_instance(obj, cls):', 'bool'), ('def is_iterable(obj):', 'bool'),
    ('def is_callable(obj):', 'bool'), ('def is_mutable(obj):', 'bool'),
    ('def is_hashable(obj):', 'bool'), ('def is_frozen(obj):', 'bool'),
    ('def is_subset(a, b):', 'bool'), ('def is_superset(a, b):', 'bool'),
    ('def is_disjoint(a, b):', 'bool'), ('def is_equal(a, b):', 'bool'),
    ('def is_greater(a, b):', 'bool'), ('def is_less(a, b):', 'bool'),
    ('def is_between(x, lo, hi):', 'bool'), ('def is_close(a, b, tol):', 'bool'),
    ('def is_divisible(a, b):', 'bool'), ('def is_coprime(a, b):', 'bool'),
    ('def is_monotonic(seq):', 'bool'), ('def is_increasing(seq):', 'bool'),
    ('def is_decreasing(seq):', 'bool'), ('def is_constant(seq):', 'bool'),
    ('def is_unique(items):', 'bool'), ('def is_permutation(a, b):', 'bool'),
    ('def is_rotation(s1, s2):', 'bool'), ('def is_subsequence(s, sub):', 'bool'),
    ('def is_prefix(s, pre):', 'bool'), ('def is_suffix(s, suf):', 'bool'),
    ('def is_match(text, pattern):', 'bool'), ('def is_valid_password(text):', 'bool'),
    ('def is_expired(date):', 'bool'), ('def is_weekend(date):', 'bool'),
    ('def is_holiday(date):', 'bool'), ('def is_business_day(date):', 'bool'),
    ('def has_key(d, key):', 'bool'), ('def has_value(d, value):', 'bool'),
    ('def has_attribute(obj, attr):', 'bool'), ('def has_method(obj, method):', 'bool'),
    ('def can_convert(value, target_type):', 'bool'), ('def supports(obj, feature):', 'bool'),
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
    ('def get_values(d):', 'list'), ('def range_list(start, stop):', 'list'),
    ('def neighbors(graph, node):', 'list'), ('def find_all(text, pattern):', 'list'),
    ('def permutations(items):', 'list'), ('def combinations(items, k):', 'list'),
    ('def topk(items, k):', 'list'), ('def bottomk(items, k):', 'list'),
    ('def sliding_window(items, size):', 'list'), ('def rotate(items, k):', 'list'),
    ('def filter_none(items):', 'list'), ('def filter_empty(items):', 'list'),
    ('def compact(items):', 'list'), ('def partition(items, predicate):', 'list'),
    ('def group_by(items, key):', 'list'), ('def sort_by(items, key):', 'list'),
    ('def map_values(items, func):', 'list'), ('def filter_by(items, predicate):', 'list'),
    ('def reject(items, predicate):', 'list'), ('def sample(items, n):', 'list'),
    ('def shuffle(items):', 'list'), ('def repeat_list(items, n):', 'list'),
    ('def enumerate_items(items):', 'list'), ('def pairwise(items):', 'list'),
    ('def cumsum(numbers):', 'list'), ('def cumprod(numbers):', 'list'),
    ('def diff(numbers):', 'list'), ('def running_mean(numbers, window):', 'list'),
    ('def histogram(items):', 'list'), ('def frequencies(items):', 'list'),
    ('def intersection(a, b):', 'list'), ('def union(a, b):', 'list'),
    ('def difference(a, b):', 'list'), ('def symmetric_difference(a, b):', 'list'),
    ('def cartesian_product(a, b):', 'list'), ('def power_set(items):', 'list'),
    ('def subsequences(items):', 'list'), ('def prefixes(items):', 'list'),
    ('def suffixes(items):', 'list'), ('def rotations(items):', 'list'),
    ('def transpose(matrix):', 'list'), ('def diagonal(matrix):', 'list'),
    ('def anti_diagonal(matrix):', 'list'), ('def row(matrix, i):', 'list'),
    ('def column(matrix, j):', 'list'), ('def flatten_matrix(matrix):', 'list'),
    ('def bfs(graph, start):', 'list'), ('def dfs(graph, start):', 'list'),
    ('def shortest_path(graph, start, end):', 'list'), ('def topological_sort(graph):', 'list'),
    ('def connected_components(graph):', 'list'), ('def cycle_detect(graph):', 'list'),
    ('def primes_up_to(n):', 'list'), ('def factors(n):', 'list'),
    ('def divisors(n):', 'list'), ('def prime_factors(n):', 'list'),
    ('def fibonacci_sequence(n):', 'list'), ('def pascal_row(n):', 'list'),
    ('def collatz_sequence(n):', 'list'), ('def digits(n):', 'list'),
    ('def chars(text):', 'list'), ('def words(text):', 'list'),
    ('def lines(text):', 'list'), ('def tokens(text):', 'list'),
    ('def ngrams(text, n):', 'list'), ('def bigrams(text):', 'list'),
    ('def trigrams(text):', 'list'), ('def parse_csv_row(line):', 'list'),
    ('def parse_path(path):', 'list'), ('def parse_query(query):', 'list'),
    ('def find_indices(items, target):', 'list'), ('def where(items, condition):', 'list'),
    ('def select(items, indices):', 'list'), ('def exclude(items, indices):', 'list'),
    ('def head(items, n):', 'list'), ('def tail(items, n):', 'list'),
    ('def rest(items):', 'list'), ('def butlast(items):', 'list'),
    ('def deduplicate(items):', 'list'), ('def sorted_unique(items):', 'list'),
    ('def most_common(items, n):', 'list'), ('def least_common(items, n):', 'list'),
    ('def average(numbers):', 'float'), ('def median(numbers):', 'float'),
    ('def variance(numbers):', 'float'), ('def std_dev(numbers):', 'float'),
    ('def to_celsius(f):', 'float'), ('def to_fahrenheit(c):', 'float'),
    ('def percentage(part, total):', 'float'), ('def ratio(a, b):', 'float'),
    ('def distance(x1, y1, x2, y2):', 'float'), ('def magnitude(x, y, z):', 'float'),
    ('def dot_product(a, b):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def circle_area(radius):', 'float'), ('def sphere_volume(radius):', 'float'),
    ('def triangle_area(base, height):', 'float'), ('def hypotenuse(a, b):', 'float'),
    ('def sigmoid(x):', 'float'), ('def relu(x):', 'float'), ('def tanh(x):', 'float'),
    ('def log_base(x, base):', 'float'), ('def square_root(x):', 'float'),
    ('def cube_root(x):', 'float'), ('def lerp(a, b, t):', 'float'),
    ('def normalize(value, min_val, max_val):', 'float'), ('def bmi(weight, height):', 'float'),
    ('def compound_interest(principal, rate, years):', 'float'),
    ('def moving_average(values, window):', 'float'), ('def entropy(probs):', 'float'),
    ('def rmse(predicted, actual):', 'float'), ('def correlation(x, y):', 'float'),
    ('def covariance(x, y):', 'float'), ('def skewness(values):', 'float'),
    ('def kurtosis(values):', 'float'), ('def geometric_mean(values):', 'float'),
    ('def harmonic_mean(values):', 'float'), ('def weighted_average(values, weights):', 'float'),
    ('def percentile(values, p):', 'float'), ('def interquartile_range(values):', 'float'),
    ('def mean_absolute_error(predicted, actual):', 'float'),
    ('def mean_squared_error(predicted, actual):', 'float'),
    ('def r_squared(predicted, actual):', 'float'),
    ('def precision(tp, fp):', 'float'), ('def recall(tp, fn):', 'float'),
    ('def f1_score(precision, recall):', 'float'), ('def accuracy(correct, total):', 'float'),
    ('def jaccard_similarity(a, b):', 'float'), ('def dice_coefficient(a, b):', 'float'),
    ('def euclidean_distance(a, b):', 'float'), ('def manhattan_distance_f(a, b):', 'float'),
    ('def chebyshev_distance(a, b):', 'float'), ('def minkowski_distance(a, b, p):', 'float'),
    ('def angle_between(v1, v2):', 'float'), ('def cross_product_magnitude(v1, v2):', 'float'),
    ('def polygon_area(vertices):', 'float'), ('def polygon_perimeter(vertices):', 'float'),
    ('def cylinder_volume(radius, height):', 'float'), ('def cone_volume(radius, height):', 'float'),
    ('def trapezoid_area(a, b, h):', 'float'), ('def ellipse_area(a, b):', 'float'),
    ('def arc_length(radius, angle):', 'float'), ('def sector_area(radius, angle):', 'float'),
    ('def simple_interest(principal, rate, time):', 'float'),
    ('def present_value(future, rate, periods):', 'float'),
    ('def future_value(present, rate, periods):', 'float'),
    ('def annuity_payment(pv, rate, periods):', 'float'),
    ('def loan_balance(principal, rate, payment, periods):', 'float'),
    ('def tax_amount(income, rate):', 'float'),
    ('def tip_amount(bill, percentage):', 'float'),
    ('def discount_price(original, discount):', 'float'),
    ('def markup_price(cost, markup):', 'float'),
    ('def profit_margin(revenue, cost):', 'float'),
    ('def return_on_investment(gain, cost):', 'float'),
    ('def speed(distance, time):', 'float'),
    ('def acceleration(velocity, time):', 'float'),
    ('def kinetic_energy(mass, velocity):', 'float'),
    ('def potential_energy(mass, height):', 'float'),
    ('def force(mass, acceleration_val):', 'float'),
    ('def pressure(force_val, area):', 'float'),
    ('def density(mass, volume):', 'float'),
    ('def wavelength(frequency, speed_val):', 'float'),
    ('def decibels(intensity, reference):', 'float'),
    ('def ph_value(concentration):', 'float'),
    ('def celsius_to_kelvin(c):', 'float'),
    ('def kelvin_to_celsius(k):', 'float'),
    ('def miles_to_km(miles):', 'float'),
    ('def km_to_miles(km):', 'float'),
    ('def pounds_to_kg(pounds):', 'float'),
    ('def kg_to_pounds(kg):', 'float'),
    ('def gallons_to_liters(gallons):', 'float'),
    ('def liters_to_gallons(liters):', 'float'),
    ('def inches_to_cm(inches):', 'float'),
    ('def cm_to_inches(cm):', 'float'),
    ('def radians_to_degrees(rad):', 'float'),
    ('def degrees_to_radians(deg):', 'float'),
    ('def noise_level(signal, noise):', 'float'),
    ('def signal_to_noise(signal, noise):', 'float'),
    ('def attenuation(input_power, output_power):', 'float'),
    ('def gain(output_power, input_power):', 'float'),
]


N_GEN_STEPS = 20  # generate 20 tokens
PCA_DIM = 128
N_BOOT = 300  # fewer bootstrap for speed (300 is fine for CIs)


def run_generation_probing(model_name, dtype):
    """Generate tokens and probe at each step."""
    logger.info("=" * 70)
    logger.info(f"MODEL: {model_name}")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    # 3 layers: early, mid, late
    layers = [n_layers // 4, n_layers // 2, n_layers - 1]
    logger.info(f"  {n_layers} layers, probing: {layers}")

    targets = sorted(set(ret for _, ret in DATASET_500))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[ret] for _, ret in DATASET_500])
    n_examples = len(DATASET_500)

    # Step 1: Generate tokens for all examples
    logger.info(f"  Generating {N_GEN_STEPS} tokens for {n_examples} examples...")
    all_generated_tokens = []  # [n_examples, N_GEN_STEPS] token ids
    all_generated_text = []    # [n_examples] full generated strings

    for i, (sig, ret) in enumerate(DATASET_500):
        prompt = sig + "\n    "
        tokens = model.to_tokens(prompt, prepend_bos=True)
        gen_toks = []
        with torch.no_grad():
            for step in range(N_GEN_STEPS):
                logits = model(tokens)
                next_tok = logits[0, -1, :].argmax().item()
                gen_toks.append(next_tok)
                tokens = torch.cat([tokens, torch.tensor([[next_tok]], device="cuda")], dim=1)

        all_generated_tokens.append(gen_toks)
        gen_text = model.to_string(torch.tensor(gen_toks))
        all_generated_text.append(gen_text)

        if i < 3:
            logger.info(f"    [{i}] {sig} → '{gen_text[:60]}'")
        if (i + 1) % 100 == 0:
            logger.info(f"    Generated {i+1}/{n_examples}")

    # Step 2: Extract activations at EACH generation step
    logger.info(f"\n  Extracting activations at each of {N_GEN_STEPS} generation steps...")

    # activations[step][layer] = np.array of shape [n_examples, d_model]
    step_activations = {step: {layer: [] for layer in layers} for step in range(N_GEN_STEPS)}

    for i, (sig, ret) in enumerate(DATASET_500):
        prompt = sig + "\n    "
        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            for step in range(N_GEN_STEPS):
                # Run with cache to get activations
                _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])

                for layer in layers:
                    # Get activation of LAST token position
                    act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                    step_activations[step][layer].append(act)

                # Add next generated token
                next_tok = all_generated_tokens[i][step]
                tokens = torch.cat([tokens, torch.tensor([[next_tok]], device="cuda")], dim=1)

                del cache
                torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            logger.info(f"    Extracted {i+1}/{n_examples}")

    # Convert to numpy
    for step in range(N_GEN_STEPS):
        for layer in layers:
            step_activations[step][layer] = np.stack(step_activations[step][layer])

    # Step 3: Name+params baseline (FIXED — computed once from signature activations)
    logger.info(f"\n  Computing name+params baseline (fixed at signature)...")
    prompt_tokens_list = [model.to_tokens(sig + "\n    ", prepend_bos=True) for sig, _ in DATASET_500]

    np_baselines = {}
    for layer in layers:
        np_feats = []
        for i in range(n_examples):
            tokens = prompt_tokens_list[i]
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_post"])
            # Mean pool all signature tokens
            acts = cache[f"blocks.{layer}.hook_resid_post"][0, :, :].cpu().numpy()
            np_feats.append(acts.mean(axis=0))
            del cache
        
        X_np = np.stack(np_feats)
        scaler_np = StandardScaler()
        X_np_s = scaler_np.fit_transform(X_np)
        if X_np_s.shape[1] > PCA_DIM:
            X_np_s = PCA(n_components=min(PCA_DIM, X_np_s.shape[0]-1), random_state=42).fit_transform(X_np_s)

        np_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            X_np_s, labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        np_baselines[layer] = np_scores.mean()
        logger.info(f"    L{layer} name+params baseline: {np_baselines[layer]:.3f}")

    # Step 4: Probe at each generation step
    logger.info(f"\n  === GENERATION-TIME COMMITMENT CURVES ===")
    results = {"model": model_name, "n_layers": n_layers, "layers": layers,
               "n_gen_steps": N_GEN_STEPS, "n_examples": n_examples}
    results["name_params_baselines"] = {str(l): float(np_baselines[l]) for l in layers}

    step_results = {}
    for layer in layers:
        logger.info(f"\n  Layer {layer}:")
        logger.info(f"  {'Step':>6} {'Probe':>8} {'CI_lo':>8} {'CI_hi':>8} {'N+P':>8} {'Gap':>8} {'Tokens':>40}")

        layer_results = []
        for step in range(N_GEN_STEPS):
            X = step_activations[step][layer]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > PCA_DIM:
                X_s = PCA(n_components=min(PCA_DIM, X_s.shape[0]-1), random_state=42).fit_transform(X_s)

            # CV accuracy
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_s, labels,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            probe_acc = scores.mean()

            # Bootstrap CI
            rng = np.random.RandomState(42)
            boot_accs = []
            for _ in range(N_BOOT):
                idx = rng.choice(len(X_s), len(X_s), replace=True)
                oob = list(set(range(len(X_s))) - set(idx))
                if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                    continue
                p = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                p.fit(X_s[idx], labels[idx])
                boot_accs.append(p.score(X_s[oob], labels[oob]))

            ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5]) if boot_accs else (0, 0)

            gap = probe_acc - np_baselines[layer]

            # Show common generated tokens at this step
            common_toks = Counter(all_generated_tokens[i][step] for i in range(n_examples))
            top3 = common_toks.most_common(3)
            tok_str = ", ".join(f"'{model.to_string(torch.tensor([t]))}'" for t, c in top3)

            logger.info(f"  {step:>6} {probe_acc:>8.3f} {ci_lo:>8.3f} {ci_hi:>8.3f} "
                        f"{np_baselines[layer]:>8.3f} {gap:>+8.3f} {tok_str:>40}")

            layer_results.append({
                "step": step, "probe": float(probe_acc),
                "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "name_params": float(np_baselines[layer]),
                "gap": float(gap),
            })

        step_results[str(layer)] = layer_results

    results["commitment_curves"] = step_results

    # Save example generations
    results["example_generations"] = [
        {"sig": sig, "ret": ret, "generated": all_generated_text[i][:200]}
        for i, (sig, ret) in enumerate(DATASET_500[:20])
    ]

    # Save
    safe_name = model_name.replace("/", "_")
    outfile = f"results/lookahead/final/gentime_{safe_name}.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n  >>> Saved to {outfile}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    logger.info("=" * 70)
    logger.info("GENERATION-TIME COMMITMENT PROBING")
    logger.info("500 examples × 20 generation steps × 3 layers")
    logger.info("=" * 70)

    models = [
        ("gpt2-xl", torch.float32),           # Negative control (0% behavioral)
        ("pythia-2.8b", torch.float16),        # Best general model
        ("bigcode/santacoder", torch.float16), # Best code model
        ("codellama/CodeLlama-7b-Python-hf", torch.float16),  # Largest model
    ]

    all_results = {}
    for model_name, dtype in models:
        try:
            all_results[model_name] = run_generation_probing(model_name, dtype)
        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # Save all
    outfile = "results/lookahead/final/gentime_all.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("DONE — GENERATION-TIME COMMITMENT PROBING")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
