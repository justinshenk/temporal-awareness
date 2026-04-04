#!/usr/bin/env python3
"""
RQ4 SUBSET ANALYSIS v2 — FAIR AND COMPREHENSIVE
=================================================
Finding 1 FIX: Train probe on EASY examples, test on HARD examples
Finding 2 FIX: 50 misleading examples, test on 3 models

This makes both findings bulletproof for reviewers.
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
# DATASETS
# ================================================================

# Same 500 from final run (abbreviated — just signatures + types)
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

# 50 misleading functions — name suggests one type, params suggest another
MISLEADING_50 = [
    # Name suggests STR, params suggest INT/LIST
    ('def greet(numbers):', 'str'), ('def farewell(items):', 'str'),
    ('def format_output(n):', 'str'), ('def describe(numbers):', 'str'),
    ('def label(items, n):', 'str'), ('def title(n):', 'str'),
    ('def caption(numbers):', 'str'), ('def header(n, items):', 'str'),
    ('def message(n):', 'str'), ('def response(items):', 'str'),
    # Name suggests INT, params suggest STR
    ('def count(text):', 'int'), ('def measure(s):', 'int'),
    ('def evaluate(text, s):', 'int'), ('def score(text):', 'int'),
    ('def grade(s):', 'int'), ('def tally(text):', 'int'),
    ('def total(text, s):', 'int'), ('def quantity(text):', 'int'),
    ('def amount(s):', 'int'), ('def level(text):', 'int'),
    # Name suggests LIST, params suggest INT/STR
    ('def collect(n):', 'list'), ('def gather(text):', 'list'),
    ('def fetch(n, text):', 'list'), ('def retrieve(n):', 'list'),
    ('def extract(n):', 'list'), ('def obtain(text):', 'list'),
    ('def assemble(n):', 'list'), ('def compile_items(text):', 'list'),
    ('def harvest(n):', 'list'), ('def accumulate(text):', 'list'),
    # Name suggests BOOL, params suggest INT/LIST
    ('def validate(numbers):', 'bool'), ('def verify(items, numbers):', 'bool'),
    ('def confirm(numbers):', 'bool'), ('def test(items):', 'bool'),
    ('def inspect(numbers):', 'bool'), ('def audit(items):', 'bool'),
    ('def review(numbers):', 'bool'), ('def examine(items):', 'bool'),
    ('def assess(numbers):', 'bool'), ('def probe_fn(items):', 'bool'),
    # Name suggests FLOAT, params suggest STR/LIST
    ('def calculate(text):', 'float'), ('def compute(text, items):', 'float'),
    ('def estimate(text):', 'float'), ('def approximate(items):', 'float'),
    ('def interpolate_val(text):', 'float'), ('def extrapolate(items):', 'float'),
    ('def project(text):', 'float'), ('def forecast(items):', 'float'),
    ('def predict(text):', 'float'), ('def model(items):', 'float'),
]


def classify_example(sig, ret):
    """Classify based on whether params are type-predictive."""
    params_str = sig[sig.index("(")+1:sig.index(")")]
    params = [p.strip() for p in params_str.split(",") if p.strip()]
    
    # Type-predictive parameter names
    TYPE_SIGNALS = {
        'int': {'n', 'numbers', 'arr', 'matrix', 'graph', 'tree', 'node'},
        'str': {'text', 's', 'name', 'full_name', 'email', 'filename', 'url',
                'path', 'word', 'code', 'template', 'phone', 'mime',
                'short', 'data', 'line', 'query'},
        'bool': {'text', 'pattern', 'prefix', 'suffix', 'sub', 'date',
                 'parens', 'seq', 'cls', 'attr', 'method', 'feature'},
        'list': {'items', 'nested', 'words', 'text', 'numbers', 'predicate',
                 'func', 'key', 'indices', 'condition', 'graph', 'matrix'},
        'float': {'numbers', 'values', 'radius', 'base', 'height', 'x',
                  'mass', 'velocity', 'rate', 'probs', 'weights', 'predicted',
                  'actual', 'vertices', 'a', 'b'},
    }
    
    # Count how many types the params could predict
    predicted_types = set()
    for p in params:
        for t, signals in TYPE_SIGNALS.items():
            if p in signals:
                predicted_types.add(t)
    
    if len(predicted_types) == 0:
        return "no_param_signal"
    elif len(predicted_types) == 1 and ret in predicted_types:
        return "easy"
    elif len(predicted_types) == 1 and ret not in predicted_types:
        return "misleading_params"
    else:
        return "ambiguous_params"


def make_examples(dataset):
    examples = []
    for idx, (sig, ret) in enumerate(dataset):
        examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"v2_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False, "index": idx},
        ))
    return examples


def run_on_model(model_name, dtype, pca_dim=128):
    """Run complete subset + misleading analysis on one model."""
    logger.info("=" * 70)
    logger.info(f"MODEL: {model_name}")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]))

    targets = sorted(set(ret for _, ret in DATASET_500))
    t2i = {t: i for i, t in enumerate(targets)}

    # Classify examples
    easy_idx, hard_idx = [], []
    for i, (sig, ret) in enumerate(DATASET_500):
        cat = classify_example(sig, ret)
        if cat == "easy":
            easy_idx.append(i)
        else:
            hard_idx.append(i)

    logger.info(f"  Easy: {len(easy_idx)}, Hard: {len(hard_idx)}")
    hard_types = Counter(DATASET_500[i][1] for i in hard_idx)
    logger.info(f"  Hard type distribution: {dict(hard_types)}")

    # Check we have all 5 types in both sets
    easy_types = set(DATASET_500[i][1] for i in easy_idx)
    hard_types_set = set(DATASET_500[i][1] for i in hard_idx)
    if len(hard_types_set) < 5:
        logger.warning(f"  Hard set missing types: {targets - hard_types_set}")
        logger.warning(f"  Cannot do 5-class probing on hard set. Adding examples...")
        # Add a few from each missing type
        for t in targets:
            if t not in hard_types_set:
                for i in easy_idx[:]:
                    if DATASET_500[i][1] == t:
                        hard_idx.append(i)
                        easy_idx.remove(i)
                        hard_types_set.add(t)
                        break

    all_examples = make_examples(DATASET_500)
    all_labels = np.array([t2i[ret] for _, ret in DATASET_500])

    # Extract activations
    logger.info("  Extracting activations for 500 examples...")
    caches = extract_activations_batch(model, model.tokenizer, all_examples, layers=layers, device="cuda")

    # ================================================================
    # FINDING 1: FAIR train-on-easy, test-on-hard
    # ================================================================
    logger.info(f"\n  === FINDING 1: TRAIN ON EASY ({len(easy_idx)}), TEST ON HARD ({len(hard_idx)}) ===")

    for layer in layers:
        # Train probe on EASY examples only
        X_easy = np.stack([caches[i].activations[layer][min(5, len(caches[i].token_ids)-1)]
                           for i in easy_idx])
        y_easy = all_labels[easy_idx]
        scaler = StandardScaler()
        X_easy_s = scaler.fit_transform(X_easy)
        if X_easy_s.shape[1] > pca_dim:
            pca = PCA(n_components=min(pca_dim, X_easy_s.shape[0]-1), random_state=42)
            X_easy_s = pca.fit_transform(X_easy_s)
        else:
            pca = None

        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_easy_s, y_easy)
        train_acc = probe.score(X_easy_s, y_easy)

        # Test on HARD examples
        X_hard = np.stack([caches[i].activations[layer][min(5, len(caches[i].token_ids)-1)]
                           for i in hard_idx])
        y_hard = all_labels[hard_idx]
        X_hard_s = scaler.transform(X_hard)
        if pca is not None:
            X_hard_s = pca.transform(X_hard_s)
        probe_test_acc = probe.score(X_hard_s, y_hard)

        # Name+params baseline on HARD (CV within hard set)
        np_feats_hard = []
        for i in hard_idx:
            cache = caches[i]
            max_pos = min(len(cache.token_ids), 10)
            acts = cache.activations[layer][:max_pos]
            np_feats_hard.append(acts.mean(axis=0))
        X_np_hard = np.stack(np_feats_hard)
        scaler_np = StandardScaler()
        X_np_hard_s = scaler_np.fit_transform(X_np_hard)
        if X_np_hard_s.shape[1] > pca_dim:
            X_np_hard_s = PCA(n_components=min(pca_dim, X_np_hard_s.shape[0]-1),
                              random_state=42).fit_transform(X_np_hard_s)

        min_class = min(Counter(y_hard).values())
        n_splits = min(5, min_class)
        if n_splits >= 2:
            np_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_np_hard_s, y_hard,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            np_acc = np_scores.mean()
        else:
            np_acc = -1

        # Name-only baseline on HARD (CV within hard set)
        name_feats_hard = []
        for i in hard_idx:
            cache = caches[i]
            name_feats_hard.append(cache.activations[layer][min(2, len(cache.token_ids)-1)])
        X_name_hard = np.stack(name_feats_hard)
        scaler_name = StandardScaler()
        X_name_hard_s = scaler_name.fit_transform(X_name_hard)
        if X_name_hard_s.shape[1] > pca_dim:
            X_name_hard_s = PCA(n_components=min(pca_dim, X_name_hard_s.shape[0]-1),
                                random_state=42).fit_transform(X_name_hard_s)

        if n_splits >= 2:
            name_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_name_hard_s, y_hard,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            name_acc = name_scores.mean()
        else:
            name_acc = -1

        gap_vs_np = probe_test_acc - np_acc
        gap_vs_name = probe_test_acc - name_acc

        logger.info(
            f"  L{layer}: train={train_acc:.3f} | HARD: probe={probe_test_acc:.3f} "
            f"name={name_acc:.3f} n+p={np_acc:.3f} | "
            f"gap_name={gap_vs_name:+.3f} gap_np={gap_vs_np:+.3f}"
        )

    # ================================================================
    # FINDING 2: 50 MISLEADING NAMES
    # ================================================================
    logger.info(f"\n  === FINDING 2: 50 MISLEADING NAMES ===")

    misleading_examples = make_examples(MISLEADING_50)
    m_caches = extract_activations_batch(model, model.tokenizer, misleading_examples, layers=layers, device="cuda")
    m_labels = np.array([t2i[ret] for _, ret in MISLEADING_50])

    for layer in layers:
        # Train probe on ALL 500 normal examples
        X_all = np.stack([caches[i].activations[layer][min(5, len(caches[i].token_ids)-1)]
                          for i in range(len(caches))])
        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        if X_all_s.shape[1] > pca_dim:
            pca = PCA(n_components=min(pca_dim, X_all_s.shape[0]-1), random_state=42)
            X_all_s = pca.fit_transform(X_all_s)
        else:
            pca = None

        probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        probe.fit(X_all_s, all_labels)

        # Test on misleading
        X_m = np.stack([m_caches[i].activations[layer][min(5, len(m_caches[i].token_ids)-1)]
                        for i in range(len(m_caches))])
        X_m_s = scaler.transform(X_m)
        if pca is not None:
            X_m_s = pca.transform(X_m_s)

        m_preds = probe.predict(X_m_s)
        m_acc = np.mean(m_preds == m_labels)

        # Analyze: does probe follow name or params?
        follows_name = 0
        follows_params = 0
        follows_neither = 0
        for i, (sig, true_type) in enumerate(MISLEADING_50):
            pred = targets[m_preds[i]]
            name = sig.split("(")[0].replace("def ", "")
            params_str = sig[sig.index("(")+1:sig.index(")")]

            # What would name predict?
            name_pred = None
            if any(w in name for w in ['greet', 'farewell', 'format', 'describe', 'label',
                                        'title', 'caption', 'header', 'message', 'response']):
                name_pred = 'str'
            elif any(w in name for w in ['count', 'measure', 'score', 'grade', 'tally',
                                          'total', 'quantity', 'amount', 'level', 'evaluate']):
                name_pred = 'int'
            elif any(w in name for w in ['collect', 'gather', 'fetch', 'retrieve', 'extract',
                                          'obtain', 'assemble', 'compile', 'harvest', 'accumulate']):
                name_pred = 'list'
            elif any(w in name for w in ['validate', 'verify', 'confirm', 'test', 'inspect',
                                          'audit', 'review', 'examine', 'assess', 'probe_fn']):
                name_pred = 'bool'
            elif any(w in name for w in ['calculate', 'compute', 'estimate', 'approximate',
                                          'interpolate', 'extrapolate', 'project', 'forecast',
                                          'predict', 'model']):
                name_pred = 'float'

            # What would params predict?
            param_pred = None
            if any(p in params_str for p in ['numbers', 'items']):
                param_pred = 'list'  # or int
            elif any(p in params_str for p in ['text', 's']):
                param_pred = 'str'
            elif 'n' in [p.strip() for p in params_str.split(',')]:
                param_pred = 'int'

            if pred == name_pred:
                follows_name += 1
            elif pred == param_pred:
                follows_params += 1
            else:
                follows_neither += 1

        logger.info(
            f"  L{layer}: misleading_acc={m_acc:.3f} ({sum(m_preds == m_labels)}/50) "
            f"follows_name={follows_name} follows_params={follows_params} follows_neither={follows_neither}"
        )

    del model, caches, m_caches
    torch.cuda.empty_cache()


def main():
    logger.info("=" * 70)
    logger.info("SUBSET ANALYSIS v2 — FAIR AND COMPREHENSIVE")
    logger.info("=" * 70)

    # Test on 3 models: SantaCoder (best), Pythia-2.8B (best general), GPT-2 XL (baseline)
    models = [
        ("bigcode/santacoder", torch.float16),
        ("pythia-2.8b", torch.float16),
        ("gpt2-xl", torch.float32),
    ]

    for model_name, dtype in models:
        try:
            run_on_model(model_name, dtype)
        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
