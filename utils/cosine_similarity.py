#!/usr/bin/env python
# coding: utf-8

def calculate_cosine_similarity_two_vectors(dot_product, vector_1_norm, vector_2_norm):
    ''' Find the cosine similarity 2 vectors'''
    
    cosine_similarity = dot_product/(vector_1_norm * vector_2_norm)

    return cosine_similarity