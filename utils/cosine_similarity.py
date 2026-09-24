#!/usr/bin/env python
# coding: utf-8

import math

def calculate_cosine_similarity_two_vectors(dot_product, vector_1_norm, vector_2_norm):
    ''' Find the cosine similarity 2 vectors '''
    
    cosine_similarity = dot_product/(vector_1_norm * vector_2_norm)

    return cosine_similarity

def calculate_angle_from_cosine_similarity(cosine_similarity):
    ''' Convert cosine similarity into absolute angle (radians) '''

    # clip values just outside [-1, 1] from floating point error, which make math.acos raise.
    # A missing value (NaN) stays NaN
    if cosine_similarity > 1:
        cosine_similarity = 1.0
    elif cosine_similarity < -1:
        cosine_similarity = -1.0

    theta = math.acos(cosine_similarity)

    return theta


