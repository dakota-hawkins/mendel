import pytest

import mendel
import numpy as np



class TestIndividual(object):

    def setup(self):
        key = {'gene1': 2,
               'gene2': 'Aa',
               'gene3': 10,
               'gene4': 'GG'}
        self.individual = mendel.Individual(key)

    def test_chromosome_keys(self):
        """Test access to gene values."""
        truth_array = [self.individual.chromosome['gene1'] == 2,
                       self.individual.chromosome['gene2'] == 'Aa',
                       self.individual.chromosome['gene3'] == 10,
                       self.individual.chromosome['gene4'] == 'GG']
        assert(all(truth_array))

    def test_individual_equivalence(self):
        """Test equivalence between individuals."""
        key = {'gene1': 2,
               'gene2': 'Aa',
               'gene3': 10,
               'gene4': 'GG'}
        other = mendel.Individual(key)
        assert self.individual == other

    def test_individual_copy(self):
        """Test copy's ability to generate equivalence."""
        other = self.individual.copy()
        assert self.individual == other
    
    def test_genetic_cross(self):
        """Test cross between individuals."""
        # set seed for expected breakpoint at i = 0
        np.random.seed(0)
        key = {'gene1': 7,
               'gene2': 'AA',
               'gene3': 5,
               'gene4': 'gg'}
        other = mendel.Individual(key)
        child1, child2 = self.individual.cross(other)
        expected_child1 = mendel.Individual({'gene1': 2, 'gene2': 'AA',
                                             'gene3': 5, 'gene4': 'gg'})
        expected_child2 = mendel.Individual({'gene1': 7, 'gene2': 'Aa',
                                             'gene3': 10, 'gene4': 'GG'})
        assert all([child1 == expected_child1, child2 == expected_child2])

    def test_nonnested_chromosome_parse(self):
        """Test non-nested chromosomes."""
        key = {'gene1': 2,
               'gene2': 'Aa',
               'gene3': 10,
               'gene4': 'GG'}
        assert key == self.individual.parse_chromosome()

    def test_nested_chromosome_parse(self):
        """Test unpacking of chromosome to expected dictionary."""
        expected_key = {'gene1': {'exon1': 'aa',
                                  'exon2': 'bb'},
                        'gene2': 'cc',
                        'gene3': {'exon1': {'transcript1': 'GG',
                                            'transcript2': 'CC'},
                                  'exon2': 'Aa'},
                        'gene4': {'exon1': {'transcript1': 'GG',
                                            'transcript2': 'CC'}}}
        input_key = {'gene1.exon1': 'aa',
                     'gene1.exon2': 'bb',
                     'gene2': 'cc',
                     'gene3.exon1.transcript1': 'GG',
                     'gene3.exon1.transcript2': 'CC',
                     'gene3.exon2': 'Aa',
                     'gene4.exon1.transcript1': 'GG',
                     'gene4.exon1.transcript2': 'CC'}
        individual = mendel.Individual(input_key)
        assert expected_key == individual.parse_chromosome()

    def test_set_membership(self):
        """Test proper set membership."""
        first_copy = self.individual.copy()
        second_copy = self.individual.copy()
        key = {'gene1': 3,
               'gene2': 'Aa',
               'gene3': 10,
               'gene4': 'GG'}
        second_individual = mendel.Individual(key)
        key_2 = {'genex': 2,
                 'gene2': 'Aa',
                 'gene3': 10,
                 'gene4': 'GG'}
        third_individual = mendel.Individual(key_2)
        indi_set = set([self.individual, first_copy, second_copy,
                        second_individual, third_individual])
        assert len(indi_set) == 3

class TestGeneticAlgorithm:

    def setup(self):
        parameter_space = {'gene1': {'exon1': 'aa',
                                     'exon2': 'bb'},
                           'gene2': 'cc',
                           'gene3': {'exon1': {'transcript1': 'GG',
                                               'transcript2': 'CC'},
                                     'exon2': 'Aa'},
                           'gene4': {'exon1': {'transcript1': 'GG',
                                               'transcript2': 'CC'}}}
        self.ga = mendel.GeneticAlgorithm(parameter_space)