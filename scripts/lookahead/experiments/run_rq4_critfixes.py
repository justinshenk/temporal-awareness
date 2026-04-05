#!/usr/bin/env python3
"""
RQ4 CRITICAL FIXES
===================
Fix 1: MEAN-POOLING CONFOUND
  Problem: probe reads 1 position, name+params mean-pools 10. Unfair.
  Fix: probe ALSO mean-pools. Compare probe-meanpool vs name+params-meanpool.
  If probe-meanpool still loses → finding holds.
  If probe-meanpool wins → previous result was confounded.

Fix 2: FIXED-POSITION GENERATION-TIME PROBING  
  Problem: probing last position during generation measures "what is the model 
  predicting now" not "does it maintain planning info."
  Fix: probe at the FIXED position of the `def` token throughout generation.
  If return type info persists at `def` even at step 19 → maintained commitment.
  If it decays → model truly forgets.

Models: SantaCoder (best code model), CodeLlama (strongest decay)
"""

import json, os, sys, hashlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType
from src.lookahead.probing.activation_extraction import extract_activations_batch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# Same 500 dataset (first 20 per type for brevity in comments, full list in code)
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

PCA_DIM = 128
N_GEN_STEPS = 20


def make_examples():
    examples = []
    for idx, (sig, ret) in enumerate(DATASET_500):
        examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"d500_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False},
        ))
    return examples


def run_fix1_meanpool(model_name, dtype):
    """FIX 1: Fair mean-pooling comparison."""
    logger.info("=" * 70)
    logger.info(f"FIX 1: MEAN-POOLING CONFOUND — {model_name}")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))

    examples = make_examples()
    targets = sorted(set(ret for _, ret in DATASET_500))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[ret] for _, ret in DATASET_500])

    caches = extract_activations_batch(model, model.tokenizer, examples, layers=layers, device="cuda")

    logger.info(f"\n  {'Layer':>6} {'Probe-Pos5':>12} {'Probe-MeanPool':>16} {'Name+Params-MP':>16} {'Name-Only':>12} "
                f"{'gap(PMP-NP)':>14} {'gap(PP5-NP)':>14}")

    results = {}
    for layer in layers:
        min_seq = min(len(c.token_ids) for c in caches)

        # 1. Probe at single position (pos 5) — original method
        pos = min(5, min_seq - 1)
        X_pos5 = np.stack([caches[i].activations[layer][pos] for i in range(len(caches))])
        scaler1 = StandardScaler()
        X_pos5_s = scaler1.fit_transform(X_pos5)
        if X_pos5_s.shape[1] > PCA_DIM:
            X_pos5_s = PCA(n_components=min(PCA_DIM, X_pos5_s.shape[0]-1), random_state=42).fit_transform(X_pos5_s)
        pos5_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            X_pos5_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy")
        probe_pos5 = pos5_scores.mean()

        # 2. Probe with MEAN POOLING (same as name+params) — FAIR comparison
        X_meanpool = np.stack([
            caches[i].activations[layer][:min(10, len(caches[i].token_ids))].mean(axis=0)
            for i in range(len(caches))
        ])
        scaler2 = StandardScaler()
        X_mp_s = scaler2.fit_transform(X_meanpool)
        if X_mp_s.shape[1] > PCA_DIM:
            X_mp_s = PCA(n_components=min(PCA_DIM, X_mp_s.shape[0]-1), random_state=42).fit_transform(X_mp_s)
        mp_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            X_mp_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy")
        probe_meanpool = mp_scores.mean()

        # 3. Name+params mean pool (positions 0-10, same as before)
        # This IS the same as probe meanpool since they see the same tokens
        # The question is: does the probe learn DIFFERENT features from the same input?
        # Actually name+params and probe-meanpool are IDENTICAL — same input, same model
        # The real comparison should be: do they learn different things?
        # But since same X → same model → same accuracy, they're the same.

        # 4. Name-only (position 2)
        X_name = np.stack([
            caches[i].activations[layer][min(2, len(caches[i].token_ids)-1)]
            for i in range(len(caches))
        ])
        scaler3 = StandardScaler()
        X_name_s = scaler3.fit_transform(X_name)
        if X_name_s.shape[1] > PCA_DIM:
            X_name_s = PCA(n_components=min(PCA_DIM, X_name_s.shape[0]-1), random_state=42).fit_transform(X_name_s)
        name_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            X_name_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy")
        name_only = name_scores.mean()

        gap_mp_vs_np = probe_meanpool - probe_meanpool  # 0 by definition since same input
        gap_p5_vs_mp = probe_pos5 - probe_meanpool

        logger.info(f"  L{layer:>4} {probe_pos5:>12.3f} {probe_meanpool:>16.3f} {probe_meanpool:>16.3f} "
                    f"{name_only:>12.3f} {gap_mp_vs_np:>+14.3f} {gap_p5_vs_mp:>+14.3f}")

        results[f"layer_{layer}"] = {
            "probe_pos5": float(probe_pos5),
            "probe_meanpool": float(probe_meanpool),
            "name_params_meanpool": float(probe_meanpool),  # same thing
            "name_only": float(name_only),
        }

    logger.info("\n  KEY INSIGHT: probe_meanpool IS name+params — they're the same input!")
    logger.info("  The original gap was: probe(pos5) vs name+params(meanpool)")
    logger.info("  This compares 1 token vs 10 tokens — different information, not different learning")
    logger.info("  Fair test: does pos5 beat pos2 (name-only)?")
    logger.info("  If yes → probe reads more than just the name (but could be params at other positions)")
    logger.info("  If no → probe only reads the name")

    del model, caches
    torch.cuda.empty_cache()
    return results


def run_fix2_fixed_position_gentime(model_name, dtype):
    """FIX 2: Probe at FIXED def-token position during generation."""
    logger.info("=" * 70)
    logger.info(f"FIX 2: FIXED-POSITION GENTIME — {model_name}")
    logger.info("  Probing at position of 'def' token (pos 0) throughout generation")
    logger.info("  If return-type info persists at def → maintained commitment")
    logger.info("  If it decays → model truly overwrites signature info")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    # Use 3 layers: early, mid, late
    layers = [n_layers // 4, n_layers // 2, n_layers - 1]

    targets = sorted(set(ret for _, ret in DATASET_500))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[ret] for _, ret in DATASET_500])
    n_examples = len(DATASET_500)

    # Find the position of 'def' token for each example
    # In most tokenizers: "def sum_list(numbers):\n    " → ['def', ' sum', '_list', '(', ...]
    # 'def' is typically at position 0 (after BOS) or position 1
    DEF_POS = 1  # position 1 (after BOS token at 0)

    # Generate tokens
    logger.info(f"  Generating {N_GEN_STEPS} tokens for {n_examples} examples...")
    all_generated_tokens = []
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
        if (i + 1) % 100 == 0:
            logger.info(f"    Generated {i+1}/{n_examples}")

    # Extract activations at DEF_POS and LAST_POS at each step
    logger.info(f"\n  Extracting activations at def position (pos {DEF_POS}) and last position...")

    def_activations = {step: {l: [] for l in layers} for step in range(N_GEN_STEPS)}
    last_activations = {step: {l: [] for l in layers} for step in range(N_GEN_STEPS)}

    for i, (sig, ret) in enumerate(DATASET_500):
        prompt = sig + "\n    "
        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            for step in range(N_GEN_STEPS):
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers]
                )
                for layer in layers:
                    resid = cache[f"blocks.{layer}.hook_resid_post"]
                    # Fixed position: def token
                    def_act = resid[0, DEF_POS, :].cpu().numpy()
                    def_activations[step][layer].append(def_act)
                    # Last position (for comparison)
                    last_act = resid[0, -1, :].cpu().numpy()
                    last_activations[step][layer].append(last_act)

                next_tok = all_generated_tokens[i][step]
                tokens = torch.cat([tokens, torch.tensor([[next_tok]], device="cuda")], dim=1)
                del cache
                torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            logger.info(f"    Extracted {i+1}/{n_examples}")

    # Convert to numpy
    for step in range(N_GEN_STEPS):
        for layer in layers:
            def_activations[step][layer] = np.stack(def_activations[step][layer])
            last_activations[step][layer] = np.stack(last_activations[step][layer])

    # Probe at each step
    logger.info(f"\n  === COMMITMENT CURVES: DEF-POSITION vs LAST-POSITION ===")

    results = {}
    for layer in layers:
        logger.info(f"\n  Layer {layer}:")
        logger.info(f"  {'Step':>6} {'Def-Pos':>10} {'Last-Pos':>10} {'Diff':>10}")

        layer_results = []
        for step in range(N_GEN_STEPS):
            # Probe at def position
            X_def = def_activations[step][layer]
            scaler_d = StandardScaler()
            X_def_s = scaler_d.fit_transform(X_def)
            if X_def_s.shape[1] > PCA_DIM:
                X_def_s = PCA(n_components=min(PCA_DIM, X_def_s.shape[0]-1),
                              random_state=42).fit_transform(X_def_s)
            def_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_def_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy")
            def_acc = def_scores.mean()

            # Probe at last position
            X_last = last_activations[step][layer]
            scaler_l = StandardScaler()
            X_last_s = scaler_l.fit_transform(X_last)
            if X_last_s.shape[1] > PCA_DIM:
                X_last_s = PCA(n_components=min(PCA_DIM, X_last_s.shape[0]-1),
                               random_state=42).fit_transform(X_last_s)
            last_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_last_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy")
            last_acc = last_scores.mean()

            logger.info(f"  {step:>6} {def_acc:>10.3f} {last_acc:>10.3f} {def_acc - last_acc:>+10.3f}")

            layer_results.append({
                "step": step,
                "def_pos_acc": float(def_acc),
                "last_pos_acc": float(last_acc),
                "diff": float(def_acc - last_acc),
            })

        results[str(layer)] = layer_results

    # Save
    safe_name = model_name.replace("/", "_")
    outfile = f"results/lookahead/final/fixes_{safe_name}.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump({"model": model_name, "fix1_meanpool": {}, "fix2_defpos": results},
                  f, indent=2, default=str)
    logger.info(f"\n  >>> Saved to {outfile}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    logger.info("=" * 70)
    logger.info("CRITICAL FIXES")
    logger.info("=" * 70)

    all_results = {}

    # FIX 1: Mean-pooling on 3 models
    for model_name, dtype in [
        ("bigcode/santacoder", torch.float16),
        ("codellama/CodeLlama-7b-Python-hf", torch.float16),
        ("pythia-2.8b", torch.float16),
    ]:
        try:
            all_results[f"fix1_{model_name}"] = run_fix1_meanpool(model_name, dtype)
        except Exception as e:
            logger.error(f"Fix1 {model_name} failed: {e}")
            import traceback; traceback.print_exc()

    # FIX 2: Fixed-position gentime on 2 models (those with strongest decay)
    for model_name, dtype in [
        ("bigcode/santacoder", torch.float16),
        ("codellama/CodeLlama-7b-Python-hf", torch.float16),
    ]:
        try:
            all_results[f"fix2_{model_name}"] = run_fix2_fixed_position_gentime(model_name, dtype)
        except Exception as e:
            logger.error(f"Fix2 {model_name} failed: {e}")
            import traceback; traceback.print_exc()

    # Save all
    outfile = "results/lookahead/final/critical_fixes_all.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("DONE — CRITICAL FIXES")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
