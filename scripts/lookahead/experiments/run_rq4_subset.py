#!/usr/bin/env python3
"""
RQ4 SUBSET ANALYSIS
====================
Look for hidden positive signal:
1. Identify "ambiguous" functions where parameter names DON'T predict return type
2. Test if probe beats name+params on that subset
3. Confusion matrix: what does name+params get wrong that probe gets right?
4. Misleading names: def greet(numbers) — name says str, params say int
5. SantaCoder L12 deep dive (closest to beating name+params)
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
from src.lookahead.probing.activation_extraction import extract_activations_batch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# ================================================================
# DATASET - same 500 as final run
# ================================================================
DATASET_500 = [
    # INT (100)
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
    # STR (100)
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
    # BOOL (100)
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
    # LIST (100)
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
    # FLOAT (100)
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


def extract_param_names(sig):
    """Extract parameter names from function signature."""
    params_str = sig[sig.index("(")+1:sig.index(")")]
    if not params_str.strip():
        return []
    return [p.strip() for p in params_str.split(",")]


def classify_ambiguity(sig, ret_type):
    """
    Classify how 'ambiguous' a function is.
    Ambiguous = parameter names are shared across multiple return types.
    """
    params = extract_param_names(sig)
    name = sig.split("(")[0].replace("def ", "")

    # Strong name signals
    name_signals = {
        'bool': ['is_', 'has_', 'can_', 'should_', 'contains', 'starts_with', 'ends_with',
                 'supports', 'file_exists', 'is_valid'],
        'str': ['greet', 'farewell', 'to_upper', 'to_lower', 'capitalize', 'format_',
                'encode', 'decode', 'escape', 'slug', 'case', 'pad_', 'truncate',
                'reverse_string', 'join_', 'first_word', 'last_word', 'extract_',
                'generate_password', 'generate_uuid', 'render_', 'interpolate',
                'encrypt', 'decrypt', 'compress', 'decompress', 'hash_', 'checksum',
                'to_binary', 'to_hex', 'to_roman', 'to_ascii', 'to_utf8',
                'mask_', 'shorten_', 'expand_url', 'mime_type', 'file_extension',
                'color_name', 'hex_color', 'error_message', 'status_text',
                'path_basename', 'path_dirname', 'path_stem', 'path_join',
                'get_initials', 'abbreviate', 'sanitize', 'transliterate', 'romanize',
                'remove_vowels', 'remove_consonants', 'remove_punctuation', 'remove_digits',
                'pig_latin', 'morse_', 'rot13', 'caesar_cipher', 'quote', 'unquote',
                'pluralize', 'singularize', 'ordinalize', 'humanize',
                'char_at', 'substring', 'repeat_string', 'replace_char', 'remove_spaces',
                'first_name', 'last_name', 'normalize_whitespace', 'expand_tabs',
                'indent', 'dedent', 'center_text', 'left_justify', 'right_justify',
                'wrap_text', 'reverse_words', 'strip_whitespace',
                'base64_', 'url_encode', 'url_decode', 'html_escape', 'html_unescape',
                'escape_regex', 'format_currency', 'format_percentage', 'format_number',
                'format_bytes', 'timestamp_to_string', 'date_to_string', 'json_to_string',
                'xml_to_string', 'dot_case', 'kebab_case', 'snake_case', 'camel_case',
                'title_case'],
        'list': ['get_evens', 'get_odds', 'filter_', 'unique', 'flatten', 'sort_',
                 'reverse_list', 'split_', 'zip_', 'merge_', 'remove_duplicates',
                 'take', 'drop', 'chunk', 'interleave', 'get_keys', 'get_values',
                 'range_list', 'neighbors', 'find_all', 'permutations', 'combinations',
                 'topk', 'bottomk', 'sliding_window', 'rotate', 'compact', 'partition',
                 'group_by', 'sort_by', 'map_values', 'filter_by', 'reject', 'sample',
                 'shuffle', 'repeat_list', 'enumerate_items', 'pairwise',
                 'cumsum', 'cumprod', 'diff', 'running_mean', 'histogram', 'frequencies',
                 'intersection', 'union', 'difference', 'symmetric_difference',
                 'cartesian_product', 'power_set', 'subsequences', 'prefixes', 'suffixes',
                 'rotations', 'transpose', 'diagonal', 'anti_diagonal', 'row', 'column',
                 'flatten_matrix', 'bfs', 'dfs', 'shortest_path', 'topological_sort',
                 'connected_components', 'cycle_detect', 'primes_up_to', 'factors',
                 'divisors', 'prime_factors', 'fibonacci_sequence', 'pascal_row',
                 'collatz_sequence', 'digits', 'chars', 'words', 'lines', 'tokens',
                 'ngrams', 'bigrams', 'trigrams', 'parse_csv_row', 'parse_path',
                 'parse_query', 'find_indices', 'where', 'select', 'exclude',
                 'head', 'tail', 'rest', 'butlast', 'deduplicate', 'sorted_unique',
                 'most_common', 'least_common', 'filter_none', 'filter_empty'],
        'float': ['average', 'median', 'variance', 'std_dev', 'percentage', 'ratio',
                  'distance', 'magnitude', 'similarity', 'area', 'volume', 'hypotenuse',
                  'sigmoid', 'relu', 'tanh', 'log_', 'square_root', 'cube_root',
                  'lerp', 'normalize', 'bmi', 'interest', 'entropy', 'rmse',
                  'correlation', 'covariance', 'skewness', 'kurtosis', 'mean',
                  'percentile', 'error', 'precision', 'recall', 'f1_score', 'accuracy',
                  'jaccard', 'dice', 'euclidean', 'manhattan_distance_f', 'chebyshev',
                  'minkowski', 'angle_', 'cross_product', 'perimeter', 'arc_length',
                  'sector_area', 'present_value', 'future_value', 'annuity', 'loan_balance',
                  'tax_amount', 'tip_amount', 'discount_', 'markup_', 'profit_margin',
                  'return_on_investment', 'speed', 'acceleration', 'kinetic_energy',
                  'potential_energy', 'force', 'pressure', 'density', 'wavelength',
                  'decibels', 'ph_value', 'to_celsius', 'to_fahrenheit',
                  'celsius_to_kelvin', 'kelvin_to_celsius', 'miles_to_km', 'km_to_miles',
                  'pounds_to_kg', 'kg_to_pounds', 'gallons_to_liters', 'liters_to_gallons',
                  'inches_to_cm', 'cm_to_inches', 'radians_to_degrees', 'degrees_to_radians',
                  'noise_level', 'signal_to_noise', 'attenuation', 'gain',
                  'weighted_average', 'geometric_mean', 'harmonic_mean', 'interquartile_range',
                  'mean_absolute_error', 'mean_squared_error', 'r_squared',
                  'polygon_area', 'polygon_perimeter', 'cylinder_volume', 'cone_volume',
                  'trapezoid_area', 'ellipse_area', 'simple_interest',
                  'circle_area', 'sphere_volume', 'triangle_area',
                  'dot_product', 'cosine_similarity', 'moving_average', 'compound_interest'],
        'int': ['count_', 'num_', 'find_max', 'find_min', 'sum_', 'product',
                'factorial', 'fibonacci', 'string_length', 'index_of', 'hamming_',
                'gcd', 'lcm', 'abs_value', 'sign', 'clamp', 'popcount',
                'manhattan_distance', 'depth', 'height', 'size',
                'add', 'subtract', 'multiply', 'divide_int', 'modulo', 'power',
                'bit_length', 'digit_sum', 'max_depth', 'min_depth',
                'edit_distance', 'longest_common_prefix_len', 'binary_search',
                'partition_index', 'rank', 'floor_div', 'ceil_div',
                'reverse_int', 'max_subarray_sum', 'min_subarray_sum',
                'longest_streak', 'shortest_path_len', 'degree', 'in_degree',
                'out_degree', 'num_components', 'num_edges', 'diameter',
                'page_count', 'word_count', 'sentence_count', 'paragraph_count',
                'line_number', 'column_count', 'row_count', 'rank_matrix',
                'trace', 'determinant_int', 'hamming_weight', 'leading_zeros',
                'trailing_zeros', 'next_power_of_two', 'prev_power_of_two',
                'count_bits', 'xor_sum', 'and_sum', 'or_sum', 'nand_sum',
                'count_primes', 'nth_prime', 'euler_totient', 'mobius',
                'catalan', 'bell_number', 'stirling', 'binomial',
                'tribonacci', 'lucas', 'ackermann', 'collatz_steps',
                'digital_root', 'perfect_number_count'],
    }

    # Check if name strongly signals the type
    name_predicted = None
    for t, patterns in name_signals.items():
        for p in patterns:
            if name.startswith(p) or name == p or p in name:
                name_predicted = t
                break
        if name_predicted:
            break

    # Ambiguous params = params that appear across multiple return types
    # e.g., (a, b) appears in int, float, bool, list functions
    AMBIGUOUS_PARAMS = {'a', 'b', 'n', 'x', 's', 'items', 'values', 'obj',
                        'value', 'data', 'key', 'start', 'end', 'target'}
    param_set = set(params)
    all_ambiguous = param_set.issubset(AMBIGUOUS_PARAMS) if params else True

    if name_predicted == ret_type and not all_ambiguous:
        return "easy"  # name AND params both predict correctly
    elif name_predicted == ret_type and all_ambiguous:
        return "name_only"  # name predicts but params are ambiguous
    elif name_predicted != ret_type and not all_ambiguous:
        return "params_only"  # params predict but name doesn't/is wrong
    else:
        return "ambiguous"  # neither name nor params clearly predict


def main():
    logger.info("=" * 70)
    logger.info("SUBSET ANALYSIS")
    logger.info("=" * 70)

    # Classify all 500 examples
    categories = {"easy": [], "name_only": [], "params_only": [], "ambiguous": []}
    for idx, (sig, ret) in enumerate(DATASET_500):
        cat = classify_ambiguity(sig, ret)
        categories[cat].append((idx, sig, ret))

    logger.info("\n=== AMBIGUITY CLASSIFICATION ===")
    for cat, items in categories.items():
        types = Counter(ret for _, _, ret in items)
        logger.info(f"  {cat}: {len(items)} examples — {dict(types)}")
        if len(items) <= 20:
            for idx, sig, ret in items:
                logger.info(f"    [{idx}] {sig} → {ret}")

    # Now run probing on AMBIGUOUS subset only using SantaCoder (best model)
    ambiguous_indices = set(idx for idx, _, _ in categories["ambiguous"])
    name_only_indices = set(idx for idx, _, _ in categories["name_only"])

    if len(categories["ambiguous"]) < 10:
        logger.info("\nToo few ambiguous examples, expanding to include name_only...")
        test_indices = ambiguous_indices | name_only_indices
        test_name = "ambiguous+name_only"
    else:
        test_indices = ambiguous_indices
        test_name = "ambiguous"

    logger.info(f"\n=== PROBING ON {test_name.upper()} SUBSET ({len(test_indices)} examples) ===")

    # Load SantaCoder
    model = HookedTransformer.from_pretrained("bigcode/santacoder", device="cuda", dtype=torch.float16)
    model.eval()
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]))

    # Make full dataset + extract activations
    all_examples = []
    for idx, (sig, ret) in enumerate(DATASET_500):
        all_examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"d500_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False, "index": idx},
        ))
    targets = sorted(set(ret for _, ret in DATASET_500))
    t2i = {t: i for i, t in enumerate(targets)}
    all_labels = np.array([t2i[ret] for _, ret in DATASET_500])

    logger.info("  Extracting activations...")
    caches = extract_activations_batch(model, model.tokenizer, all_examples, layers=layers, device="cuda")

    # Subset indices
    subset_mask = np.array([i in test_indices for i in range(len(DATASET_500))])
    subset_labels = all_labels[subset_mask]
    subset_examples = [all_examples[i] for i in range(len(all_examples)) if i in test_indices]
    subset_caches = [caches[i] for i in range(len(caches)) if i in test_indices]

    logger.info(f"  Subset: {len(subset_labels)} examples, {len(set(subset_labels))} classes")
    if len(set(subset_labels)) < 3:
        logger.warning("  Not enough classes in subset for meaningful probing!")

    pca_dim = 128
    for layer in layers:
        # Full dataset probe
        min_seq = min(len(c.token_ids) for c in caches)
        best_pos = 5  # typical best
        X_full = np.stack([caches[i].activations[layer][min(best_pos, len(caches[i].token_ids)-1)]
                           for i in range(len(caches))])
        scaler = StandardScaler()
        X_full_s = scaler.fit_transform(X_full)
        if X_full_s.shape[1] > pca_dim:
            pca = PCA(n_components=min(pca_dim, X_full_s.shape[0]-1), random_state=42)
            X_full_s = pca.fit_transform(X_full_s)
        else:
            pca = None

        # Train on full, predict subset
        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_full_s, all_labels)
        full_acc = probe.score(X_full_s, all_labels)

        # Get subset predictions
        X_sub = np.stack([subset_caches[i].activations[layer][min(best_pos, len(subset_caches[i].token_ids)-1)]
                          for i in range(len(subset_caches))])
        X_sub_s = scaler.transform(X_sub)
        if pca is not None:
            X_sub_s = pca.transform(X_sub_s)

        probe_preds = probe.predict(X_sub_s)
        subset_probe_acc = np.mean(probe_preds == subset_labels)

        # Name+params baseline on subset (CV)
        name_params_feats = []
        for i, idx in enumerate(sorted(test_indices)):
            cache = caches[idx]
            max_pos = min(len(cache.token_ids), 10)
            acts = cache.activations[layer][:max_pos]
            name_params_feats.append(acts.mean(axis=0))
        X_np = np.stack(name_params_feats)
        scaler_np = StandardScaler()
        X_np_s = scaler_np.fit_transform(X_np)
        if X_np_s.shape[1] > pca_dim:
            X_np_s = PCA(n_components=min(pca_dim, X_np_s.shape[0]-1), random_state=42).fit_transform(X_np_s)

        if len(set(subset_labels)) >= 2 and len(subset_labels) >= 10:
            n_splits = min(5, min(Counter(subset_labels).values()))
            if n_splits >= 2:
                np_scores = cross_val_score(
                    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                    X_np_s, subset_labels,
                    cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                    scoring="accuracy",
                )
                np_acc = np_scores.mean()
            else:
                np_acc = -1
        else:
            np_acc = -1

        logger.info(f"  L{layer}: subset_probe={subset_probe_acc:.3f} subset_n+p={np_acc:.3f} "
                     f"gap={subset_probe_acc - np_acc:.3f} full_probe={full_acc:.3f}")

    # ================================================================
    # MISLEADING NAMES EXPERIMENT
    # ================================================================
    logger.info("\n=== MISLEADING NAMES ===")
    misleading = [
        ('def greet(numbers):', 'str'),       # name=str, params=int/list
        ('def count(text):', 'int'),           # name=int, params=str
        ('def get_items(n):', 'list'),         # name=list, params=int
        ('def is_valid(numbers):', 'bool'),    # name=bool, params=int/list
        ('def calculate(text):', 'float'),     # name=float, params=str
        ('def process(items):', 'str'),        # generic name, params=list
        ('def transform(n):', 'list'),         # generic name, params=int
        ('def compute(text, items):', 'int'),  # generic name, mixed params
        ('def validate(numbers, text):', 'bool'),  # name=bool, mixed params
        ('def analyze(a, b):', 'float'),       # generic everything
        ('def handle(data):', 'str'),          # generic
        ('def execute(values):', 'int'),       # generic name, params=float/list
        ('def run(s):', 'list'),               # generic name, params=str
        ('def check(n, s):', 'bool'),          # name=bool, mixed params
        ('def build(items, n):', 'str'),       # generic name, mixed params
    ]

    misleading_examples = []
    for idx, (sig, ret) in enumerate(misleading):
        misleading_examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"misleading_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False},
        ))

    m_caches = extract_activations_batch(model, model.tokenizer, misleading_examples, layers=layers, device="cuda")
    m_labels = np.array([t2i[ret] for _, ret in misleading])

    logger.info("  Training probe on full 500, testing on misleading names:")
    for layer in layers:
        # Reuse full probe from above
        X_full = np.stack([caches[i].activations[layer][min(5, len(caches[i].token_ids)-1)]
                           for i in range(len(caches))])
        scaler = StandardScaler()
        X_full_s = scaler.fit_transform(X_full)
        if X_full_s.shape[1] > pca_dim:
            pca = PCA(n_components=min(pca_dim, X_full_s.shape[0]-1), random_state=42)
            X_full_s = pca.fit_transform(X_full_s)
        else:
            pca = None

        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_full_s, all_labels)

        X_m = np.stack([m_caches[i].activations[layer][min(5, len(m_caches[i].token_ids)-1)]
                        for i in range(len(m_caches))])
        X_m_s = scaler.transform(X_m)
        if pca is not None:
            X_m_s = pca.transform(X_m_s)

        m_preds = probe.predict(X_m_s)
        m_acc = np.mean(m_preds == m_labels)

        # Show individual predictions
        wrong = []
        for i, (sig, ret) in enumerate(misleading):
            pred_type = targets[m_preds[i]]
            if pred_type != ret:
                wrong.append(f"{sig} → pred={pred_type}, true={ret}")

        logger.info(f"  L{layer}: misleading_acc={m_acc:.3f} ({sum(m_preds == m_labels)}/{len(m_labels)})")
        if wrong:
            for w in wrong[:5]:
                logger.info(f"    WRONG: {w}")

    # Save results
    results = {
        "categories": {k: len(v) for k, v in categories.items()},
        "ambiguous_examples": [(sig, ret) for _, sig, ret in categories["ambiguous"]],
        "name_only_examples": [(sig, ret) for _, sig, ret in categories["name_only"]],
    }
    outfile = "results/lookahead/final/subset_analysis.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {outfile}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
