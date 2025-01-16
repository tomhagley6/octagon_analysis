#!/usr/bin/env python
# coding: utf-8

import math

def calculate_cosine_similarity_two_vectors(dot_product, vector_1_norm, vector_2_norm):
    ''' Find the cosine similarity 2 vectors '''
    
    cosine_similarity = dot_product/(vector_1_norm * vector_2_norm)

    return cosine_similarity

def calculate_angle_from_cosine_similarity(cosine_similarity):
    ''' Convert cosine similarity into absolute angle (radians) '''

    theta = math.acos(cosine_similarity)

    return theta


