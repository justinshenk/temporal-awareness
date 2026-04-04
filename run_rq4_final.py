#!/usr/bin/env python3
"""
RQ4 FINAL COMPREHENSIVE RUN
============================
Addresses ALL reviewer concerns in one overnight run:

1. 500 examples (100 per type) — tighter CIs
2. Name+params baseline — rules out parameter name confound
3. FP32 steering for capable models — causal evidence
4. Cross-task transfer (H5) — code probe → rhyme test
5. Base vs instruction-tuned — Llama-3.2-1B vs Instruct
6. 1000 bootstrap, PCA(128), incremental saves

Models: GPT-2 (S/M/XL), Pythia (410M/1B/1.4B/2.8B),
        SantaCoder, CodeLlama-7B, Llama-3.2-1B, Llama-3.2-1B-Instruct
"""

import json, logging, os, sys, hashlib, time, random, traceback
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType
from src.lookahead.probing.activation_extraction import extract_activations_batch
from src.lookahead.probing.behavioral_validation import run_behavioral_validation, compute_behavioral_summary
from src.lookahead.datasets.rhyme import generate_rhyme_dataset

RESULTS_DIR = "results/lookahead/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 500 EXAMPLES (100 per type)
# ================================================================
DATASET_500 = [
    # INT (100 examples)
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
    # STR (100 examples)
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
    # BOOL (100 examples)
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
    # LIST (100 examples)
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
    # FLOAT (100 examples)
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


def make_examples(dataset=None):
    if dataset is None:
        dataset = DATASET_500
    examples = []
    for idx, (sig, ret) in enumerate(dataset):
        examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"d500_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False},
        ))
    return examples


def make_nonsense_examples(examples):
    random.seed(42)
    names = ["xyzq", "blorpf", "qwmx", "fnrd", "ghtk", "zmvp", "krtl", "dwqn",
             "pxvs", "jhlm", "vbnq", "wrtx", "ycfg", "nmzk", "tplr", "sdhj",
             "bxcv", "lfgn", "mwqz", "rkpt", "hdsx", "jvnq", "cwfl", "nbtm",
             "gzpr", "xkdw", "fqms", "ylrv", "thcn", "pdwk"]
    out = []
    for ex in examples:
        sig = ex.metadata["signature"]
        params = sig[sig.index("("):]
        nn = random.choice(names)
        new_sig = f"def {nn}{params}"
        out.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=new_sig + "\n    ",
            target_value=ex.target_value, target_token_positions=[],
            example_id=hashlib.md5(f"nonsense_{nn}_{params}".encode()).hexdigest()[:12],
            metadata={"signature": new_sig, "original": sig,
                      "has_type_annotation": False, "return_type": ex.target_value,
                      "is_control": False},
        ))
    return out


def get_name_and_params_features(caches, examples, layer, pca_dim):
    """Extract features from function name token AND parameter tokens."""
    features = []
    for i, (ex, cache) in enumerate(zip(examples, caches)):
        sig = ex.metadata["signature"]
        # Tokens: [BOS, def, <space>, name, (, param1, comma, param2, ..., ), :]
        # Name is at position 2, params start at position 3
        n_tokens = len(cache.token_ids)
        # Get all tokens from position 2 (name) through the end of signature
        max_pos = min(n_tokens, 10)  # cap at 10 tokens
        acts = cache.activations[layer][:max_pos]  # [max_pos, d_model]
        features.append(acts.mean(axis=0))  # mean pool name + params
    X = np.stack(features)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if X_s.shape[1] > pca_dim:
        X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1, X_s.shape[1]),
                  random_state=42).fit_transform(X_s)
    return X_s


def run_probing_full(model, examples, labels, targets, layers, pca_dim, n_boot):
    """Full probing with name-only AND name+params baselines."""
    chance = 1.0 / len(targets)
    results = {}
    caches = extract_activations_batch(model, model.tokenizer, examples, layers=layers, device="cuda")

    for layer in layers:
        min_seq = min(len(c.token_ids) for c in caches)
        best_acc, best_pos = 0, 0

        for pos in range(min_seq):
            X = np.stack([caches[i].activations[layer][pos] for i in range(len(examples))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > pca_dim:
                X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1, X_s.shape[1]),
                          random_state=42).fit_transform(X_s)
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_pos = pos

        # Name-only baseline (position 2)
        name_acts = np.stack([
            caches[i].activations[layer][min(2, len(caches[i].token_ids) - 1)]
            for i in range(len(examples))
        ])
        scaler_n = StandardScaler()
        name_s = scaler_n.fit_transform(name_acts)
        if name_s.shape[1] > pca_dim:
            name_s = PCA(n_components=min(pca_dim, name_s.shape[0]-1, name_s.shape[1]),
                         random_state=42).fit_transform(name_s)
        name_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            name_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        best_name = name_scores.mean()

        # Name+Params baseline (mean pool all signature tokens)
        name_params_s = get_name_and_params_features(caches, examples, layer, pca_dim)
        name_params_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            name_params_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        best_name_params = name_params_scores.mean()

        # BoW baseline
        bow_dim = min(max(max(c.token_ids) for c in caches) + 1, 50257)
        X_bow = np.zeros((len(examples), bow_dim), dtype=np.float32)
        for row in range(len(examples)):
            for t in range(min(best_pos + 1, len(caches[row].token_ids))):
                tid = caches[row].token_ids[t]
                if tid < bow_dim:
                    X_bow[row, tid] = 1.0
        nz = X_bow.sum(axis=0) > 0
        X_bow_f = X_bow[:, nz]
        if X_bow_f.shape[1] > 0:
            scaler_b = StandardScaler()
            X_bow_s = scaler_b.fit_transform(X_bow_f)
            bow_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_bow_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_bow = bow_scores.mean()
        else:
            best_bow = chance

        # Bootstrap CI + probe vs name test + probe vs name+params test
        X_best = np.stack([caches[i].activations[layer][best_pos] for i in range(len(examples))])
        scaler_bp = StandardScaler()
        X_best_s = scaler_bp.fit_transform(X_best)
        if X_best_s.shape[1] > pca_dim:
            X_best_s = PCA(n_components=min(pca_dim, X_best_s.shape[0]-1, X_best_s.shape[1]),
                           random_state=42).fit_transform(X_best_s)

        rng = np.random.RandomState(42)
        boot_accs = []
        probe_wins_name = 0
        probe_wins_nameparams = 0
        valid_boots = 0
        for _ in range(n_boot):
            idx = rng.choice(len(X_best_s), len(X_best_s), replace=True)
            oob = list(set(range(len(X_best_s))) - set(idx))
            if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                continue
            valid_boots += 1
            p1 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p1.fit(X_best_s[idx], labels[idx])
            a1 = p1.score(X_best_s[oob], labels[oob])
            boot_accs.append(a1)

            # vs name-only
            p2 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p2.fit(name_s[idx], labels[idx])
            a2 = p2.score(name_s[oob], labels[oob])
            if a1 > a2:
                probe_wins_name += 1

            # vs name+params
            p3 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p3.fit(name_params_s[idx], labels[idx])
            a3 = p3.score(name_params_s[oob], labels[oob])
            if a1 > a3:
                probe_wins_nameparams += 1

        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5]) if boot_accs else (0, 0)
        p_vs_name = 1.0 - (probe_wins_name / valid_boots) if valid_boots else 1.0
        p_vs_nameparams = 1.0 - (probe_wins_nameparams / valid_boots) if valid_boots else 1.0

        logger.info(
            f"  L{layer}: probe={best_acc:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
            f"BoW={best_bow:.3f} name={best_name:.3f} name+params={best_name_params:.3f} "
            f"gap_name={best_acc - best_name:.3f} gap_np={best_acc - best_name_params:.3f} "
            f"p_name={p_vs_name:.4f} p_np={p_vs_nameparams:.4f} pos={best_pos}"
        )

        results[f"layer_{layer}"] = {
            "probe": float(best_acc), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "bow": float(best_bow), "name_only": float(best_name),
            "name_params": float(best_name_params),
            "gap_name": float(best_acc - best_name),
            "gap_name_params": float(best_acc - best_name_params),
            "p_vs_name": float(p_vs_name),
            "p_vs_name_params": float(p_vs_nameparams),
            "best_pos": int(best_pos), "valid_boots": valid_boots,
        }

    return results, caches


def run_fp32_steering(model_name, examples, caches, layers, targets):
    """Reload model in fp32 and run steering."""
    logger.info(f"  Loading {model_name} in fp32 for steering...")
    try:
        model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float32)
        model.eval()
    except Exception as e:
        logger.error(f"  Cannot load {model_name} in fp32: {e}")
        return []

    t2i = {t: i for i, t in enumerate(targets)}
    type_examples = {}
    for e, c in zip(examples, caches):
        type_examples.setdefault(e.target_value, []).append((e, c))

    results = []
    for alpha in [1.0, 3.0, 5.0, 10.0]:
        if "int" not in type_examples or "str" not in type_examples:
            continue
        exs_int = type_examples["int"][:5]
        exs_str = type_examples["str"][:5]

        steer_layers = [layers[0], layers[len(layers)//2], layers[-1]]
        for layer in steer_layers:
            mean_int = np.mean([c.activations[layer].mean(axis=0) for _, c in exs_int], axis=0)
            mean_str = np.mean([c.activations[layer].mean(axis=0) for _, c in exs_str], axis=0)
            sv = torch.tensor((mean_str - mean_int) * alpha, dtype=torch.float32, device="cuda")
            hook_name = f"blocks.{layer}.hook_resid_post"

            for ex, cache in exs_int[:2]:
                tokens = model.to_tokens(ex.prompt, prepend_bos=True)
                with torch.no_grad():
                    orig = model(tokens)
                op = torch.softmax(orig[0, -1, :], dim=-1)

                def make_hook(s):
                    def h(value, hook):
                        value[0, :, :] += s
                        return value
                    return h

                with torch.no_grad():
                    steered = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(sv))])
                sp = torch.softmax(steered[0, -1, :], dim=-1)
                kl = torch.sum(op * (torch.log(op + 1e-10) - torch.log(sp + 1e-10))).item()

                ot5 = [model.to_string(torch.tensor([t])) for t in torch.topk(op, 5).indices.tolist()]
                st5 = [model.to_string(torch.tensor([t])) for t in torch.topk(sp, 5).indices.tolist()]

                logger.info(f"    alpha={alpha:.0f} L{layer}: KL={kl:.3f} orig={ot5} steered={st5}")
                results.append({"alpha": alpha, "layer": layer, "kl": kl,
                                "sig": ex.metadata["signature"],
                                "orig_top5": ot5, "steered_top5": st5})

    del model
    torch.cuda.empty_cache()
    return results


def run_cross_task_transfer(model, code_caches, code_examples, code_labels, targets, layers, pca_dim):
    """Train probe on code, test on rhyme (H5)."""
    logger.info("  === CROSS-TASK TRANSFER (H5) ===")
    rhyme_ex = generate_rhyme_dataset(n_per_rhyme_set=5, include_controls=False)
    rhyme_caches = extract_activations_batch(model, model.tokenizer, rhyme_ex, layers=layers, device="cuda")

    results = {}
    for layer in layers:
        min_seq_code = min(len(c.token_ids) for c in code_caches)
        min_seq_rhyme = min(len(c.token_ids) for c in rhyme_caches)

        # Train on code (best position from code probing)
        best_pos = min(5, min_seq_code - 1)
        X_train = np.stack([code_caches[i].activations[layer][best_pos] for i in range(len(code_examples))])
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        if X_train_s.shape[1] > pca_dim:
            pca = PCA(n_components=min(pca_dim, X_train_s.shape[0]-1), random_state=42)
            X_train_s = pca.fit_transform(X_train_s)
        else:
            pca = None

        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_train_s, code_labels)
        train_acc = probe.score(X_train_s, code_labels)

        # Test on rhyme (predict what? — just check if probe predictions are non-random)
        rhyme_pos = min(best_pos, min_seq_rhyme - 1)
        X_test = np.stack([rhyme_caches[i].activations[layer][rhyme_pos] for i in range(len(rhyme_ex))])
        X_test_s = scaler.transform(X_test)
        if pca is not None:
            X_test_s = pca.transform(X_test_s)

        rhyme_preds = probe.predict(X_test_s)
        pred_dist = {t: int(np.sum(rhyme_preds == i)) for i, t in enumerate(targets)}

        logger.info(f"  L{layer}: train_acc={train_acc:.3f} rhyme_pred_dist={pred_dist}")
        results[f"layer_{layer}"] = {
            "train_acc": float(train_acc),
            "rhyme_pred_distribution": pred_dist,
        }

    return results


def run_model_full(model_name, dtype, pca_dim=128, n_boot=1000, do_steering_fp32=False):
    """Run complete suite for one model."""
    logger.info("=" * 70)
    logger.info(f"MODEL: {model_name} (500 examples, pca={pca_dim}, boot={n_boot})")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]))
    logger.info(f"  {n_layers} layers, d_model={d_model}, probing layers: {layers}")

    examples = make_examples()
    targets = sorted(set(e.target_value for e in examples))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[e.target_value] for e in examples])

    result = {"model": model_name, "n_layers": n_layers, "d_model": d_model,
              "n_examples": len(examples), "pca_dim": pca_dim, "n_boot": n_boot}

    # Behavioral
    logger.info("\n  === BEHAVIORAL (500 examples) ===")
    beh = run_behavioral_validation(model, examples, max_new_tokens=50)
    beh_sum = compute_behavioral_summary(beh)
    for t, s in beh_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%}")
    result["behavioral"] = beh_sum
    type_correct = {}
    for r in beh:
        type_correct.setdefault(r.target_value, []).append(r.task_success)
    for t in sorted(type_correct):
        logger.info(f"    {t}: {np.mean(type_correct[t]):.1%}")
    result["behavioral_per_type"] = {t: float(np.mean(v)) for t, v in type_correct.items()}

    # Probing with name+params baseline
    logger.info(f"\n  === PROBING (500 ex, name+params baseline, boot={n_boot}) ===")
    probe_results, caches = run_probing_full(model, examples, labels, targets, layers, pca_dim, n_boot)
    result["probing"] = probe_results

    # Nonsense names
    logger.info("\n  === NONSENSE ===")
    nonsense = make_nonsense_examples(examples)
    n_labels = np.array([t2i[e.target_value] for e in nonsense])
    n_caches = extract_activations_batch(model, model.tokenizer, nonsense, layers=layers, device="cuda")
    nonsense_results = {}
    for layer in layers:
        min_seq = min(len(c.token_ids) for c in n_caches)
        best_acc = 0
        for pos in range(min_seq):
            X = np.stack([n_caches[i].activations[layer][pos] for i in range(len(nonsense))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > pca_dim:
                X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1, X_s.shape[1]),
                          random_state=42).fit_transform(X_s)
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_s, n_labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_acc = max(best_acc, scores.mean())
        logger.info(f"  Nonsense L{layer}: {best_acc:.3f} (chance=0.200)")
        nonsense_results[f"nonsense_L{layer}"] = {"probe": float(best_acc)}
    result.update(nonsense_results)
    del n_caches

    # Cross-task transfer (H5)
    logger.info("\n  === CROSS-TASK TRANSFER (H5) ===")
    try:
        transfer_results = run_cross_task_transfer(model, caches, examples, labels, targets, layers, pca_dim)
        result["cross_task_transfer"] = transfer_results
    except Exception as e:
        logger.error(f"  Cross-task transfer failed: {e}")
        result["cross_task_transfer"] = {"error": str(e)}

    # Rhyme behavioral
    logger.info("\n  === RHYME ===")
    rhyme_ex = generate_rhyme_dataset(n_per_rhyme_set=3, include_controls=False)
    beh_r = run_behavioral_validation(model, rhyme_ex, max_new_tokens=50)
    beh_r_sum = compute_behavioral_summary(beh_r)
    for t, s in beh_r_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%}")
    result["rhyme_behavioral"] = beh_r_sum

    # FP32 steering
    if do_steering_fp32:
        logger.info("\n  === FP32 STEERING ===")
        del model
        torch.cuda.empty_cache()
        steering = run_fp32_steering(model_name, examples, caches, layers, targets)
        result["steering_fp32"] = steering
    else:
        del model

    del caches
    torch.cuda.empty_cache()

    # Save incrementally
    safe_name = model_name.replace("/", "_")
    outfile = f"{RESULTS_DIR}/{safe_name}_final.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  >>> Saved to {outfile}")

    return result


# ================================================================
# MAIN
# ================================================================
def main():
    logger.info("=" * 70)
    logger.info("RQ4 FINAL COMPREHENSIVE RUN")
    logger.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Dataset: 500 examples (100 per type)")
    logger.info("=" * 70)

    all_results = {}

    # Define all models
    models = [
        # GPT-2 family
        ("gpt2", torch.float32, True),
        ("gpt2-medium", torch.float32, True),
        ("gpt2-xl", torch.float32, True),
        # Pythia scaling
        ("pythia-410m", torch.float32, True),
        ("pythia-1b", torch.float16, True),
        ("pythia-1.4b", torch.float16, True),
        ("pythia-2.8b", torch.float16, True),  # FP32 steering
        # Code models
        ("bigcode/santacoder", torch.float16, True),  # FP32 steering
        ("codellama/CodeLlama-7b-Python-hf", torch.float16, False),
        # Base vs instruction-tuned
        ("meta-llama/Llama-3.2-1B", torch.float16, True),
        ("meta-llama/Llama-3.2-1B-Instruct", torch.float16, True),
    ]

    for model_name, dtype, do_steering in models:
        try:
            all_results[model_name] = run_model_full(
                model_name, dtype, pca_dim=128, n_boot=1000,
                do_steering_fp32=do_steering,
            )
        except Exception as e:
            logger.error(f"MODEL {model_name} FAILED: {e}")
            logger.error(traceback.format_exc())
            all_results[model_name] = {"error": str(e)}

    # Save everything
    outfile = f"{RESULTS_DIR}/all_final_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Tar
    os.system(f"tar czf /workspace/rq4_final_results.tar.gz {RESULTS_DIR}/")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    for name, res in all_results.items():
        if isinstance(res, dict) and "probing" in res:
            beh = res.get("behavioral", {})
            beh_acc = "N/A"
            for t, s in beh.items():
                beh_acc = s.get("task_accuracy", "N/A")
                break
            logger.info(f"\n{name}: behavioral={beh_acc}")
            for lk, lv in res["probing"].items():
                logger.info(
                    f"  {lk}: probe={lv['probe']:.3f} [{lv['ci_lo']:.3f},{lv['ci_hi']:.3f}] "
                    f"name={lv['name_only']:.3f} n+p={lv['name_params']:.3f} "
                    f"gap_name={lv['gap_name']:.3f} gap_np={lv['gap_name_params']:.3f} "
                    f"p_name={lv['p_vs_name']:.4f} p_np={lv['p_vs_name_params']:.4f}"
                )

    logger.info(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results: {outfile}")


if __name__ == "__main__":
    main()
