import pytest
import mendel



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
        key = {'gene1': 7,
               'gene2': 'AA',
               'gene3': 5,
               'gene4': 'gg'}
        other = mendel.Individual(key)
        child1, child2 = self.individual.cross(other, break_point=1)
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

class TestGeneticAlgorithm(object):

    def setup(self):
        self.a_s = ['aa', 'aA', 'Aa', 'AA']
        self.b_s = ['bb', 'bB', 'Bb', 'BB']
        self.c_s = ['cc', 'cC', 'Cc', 'CC']
        self.g_s = ['gg', 'gG', 'Gg', 'GG']
        input_space = {'gene1': {'exon1': self.a_s, 'exon2': self.b_s},
                        'gene2': self.c_s,
                        'gene3': {'exon1': {'transcript1': self.g_s,
                                            'transcript2': self.c_s},
                                  'exon2': self.a_s},
                        'gene4': {'exon1': {'transcript1': self.g_s,
                                            'transcript2': self.c_s}}}
        self.ga = mendel.GeneticAlgorithm(input_space)

    def test_genomic_space(self):

        expected_space = {'gene1.exon1': self.a_s,
                          'gene1.exon2': self.b_s,
                          'gene2': self.c_s,
                          'gene3.exon1.transcript1': self.g_s,
                          'gene3.exon1.transcript2': self.c_s,
                          'gene3.exon2': self.a_s,
                          'gene4.exon1.transcript1': self.g_s,
                          'gene4.exon1.transcript2': self.c_s}
        self.ga.genomic_space == expected_space

    def test_random_ind(self):
        expected_space = {'gene1.exon1': self.a_s,
                          'gene1.exon2': self.b_s,
                          'gene2': self.c_s,
                          'gene3.exon1.transcript1': self.g_s,
                          'gene3.exon1.transcript2': self.c_s,
                          'gene3.exon2': self.a_s,
                          'gene4.exon1.transcript1': self.g_s,
                          'gene4.exon1.transcript2': self.c_s}
        random = self.ga.random_individual()
        assert all([random.chromosome[x] in expected_space[x]\
                    for x in random.chromosome])

    def test_best_performer_single_best(self):
        self.ga.generations = 1
        # fitness mixin returns 1 for all fitness scores
        score_fitness = mendel.FitnessMixin()
        self.ga.breed(score_fitness)
        self.ga.population[0].fitness = 2
        top = self.ga.best_performer()
        assert self.ga.population[0] == top[0]

    def test_best_performer_multiple_best(self):
        self.ga.generations = 1
        # fitness mixin returns 1 for all fitness scores
        score_fitness = mendel.FitnessMixin()
        self.ga.breed(score_fitness)
        self.ga.population[0].fitness = 2
        idx = 1
        flag = False
        while idx < self.ga.n - 1 and not flag:
            if self.ga.population[0] != self.ga.population[idx]:
                flag = True
            idx += 1
        self.ga.population[idx].fitness = 2
        top = self.ga.best_performer()
        assert self.ga.population[0] in top and self.ga.population[idx] in top

    def test_mutate(self):
        indi = self.ga.random_individual()
        old_indi = indi.copy()
        mutated = self.ga.mutate(indi)
        assert old_indi != mutated
    
                    