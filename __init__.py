import collections

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import numpy as np
from scipy import stats, special

from tqdm import tqdm


class Individual(object):
    """
    Class for holding paramter values for individuals in a genetic algorithm.
    
    Attributes
    ------
    chromosome : dict
    fitness : float
    
    Methods
    -------
    copy : create a copy of an Individual.
    cross : cross to individual to create children.
    parse_chromosome : return paramter names and values as key words.
    """


    def __init__(self, chromosome=None):
        """
        Class for holding paramter values for individuals in a genetic algorithm.
        
        Parameters
        ----------
        chromosome : dict, optional
            A dictionary containing parameter names with associated values. The
            default is None, and an empty chromosome will be instantiated.
        """

        self.chromosome = chromosome
        self.fitness = None
        # create a hash key by concatenating all key-values pairs as a single
        # string, ensures chromosome and has equality
        self.__hashkey = ''
        for key, value in self.chromosome.items():
            self.__hashkey += str(key) + str(value)

    def __repr__(self):
        """
        String representation of an individual.
        
        Returns
        -------
        str
            String containing all gene/allele values and current
            fitness score.
        """
        msg = "Individual with the following genotype:\n"
        for key, item in self.chromosome.items():
            msg += "  {}: {}\n".format(key, item)
        msg += "  Fitness: {}\n".format(self.fitness)
        return msg

    def __eq__(self, other):
        """Individuals are equal if chromsomes are genetically identical."""
        if isinstance(other, self.__class__):
            return self.chromosome == other.chromosome
        return False
    
    def __hash__(self):
        """Hash an individual by string contactenation of their chromosome."""
        return hash(self.__hashkey)

    def copy(self):
        """
        Create a copy of the the current individual.
        
        Returns
        -------
        Individual
            New Individual with the same chromosome and fitness values. 
        """
        new = Individual(self.chromosome.copy())
        new.fitness = self.fitness
        return new
    
    def cross(self, partner, break_point=None):
        """
        Generate children from two parent Individuals.
        
        Parameters
        ----------
        partner : Individual
            Individual to cross with.
        break_point : int, optional
            Which key, when ordered as a list, to set as the break point for a
            genetic cross. Value should be between 1 and `#genes - 1`. The
            default is None, and the point will be randomly selected.
        
        Raises
        ------
        ValueError
            If `partner` is not an Individual object.
        ValueError
            If `partner` and `self` do not have the same genes.
        ValueError
            If `break_point` is not between 1 and `len(self.chromosome) - 1`.
        ValueError
            If passed `break_point` is neither None nor an integer.
        
        Returns
        -------
        Individual
            Mutated individual.
        """

        if not isinstance(partner, self.__class__):
            raise ValueError("`partner` must be an Individual.")
        # ensure individuals are comparable
        if self.chromosome.keys() != partner.chromosome.keys():
            raise ValueError("Incomparable indivuals. Genes between" +
                             "individuals do not match.")
        # randomly choose breakpoint for genetic cross
        genes = list(self.chromosome.keys())
        if isinstance(break_point, int):
            if not 1 <= break_point < len(genes) - 1:
                raise ValueError('`break_point` must fall between 1 and ' +
                                 '{}. Got {}.'.format(len(genes) - 1,
                                                      break_point))
        elif break_point is None:
            break_point = np.random.choice(range(1, len(genes) - 1))
        else:
            raise ValueError('Expected integer or None Type for `break_point.' +
                             ' Got {}.'.format(type(break_point)))

        # create chromosome for offsprings
        child1 = {x: None for x in genes}
        child2 = child1.copy()

        # cross genes between parents and children
        for x in genes[0:break_point]:
            child1[x] = self.chromosome[x]
            child2[x] = partner.chromosome[x]
        for y in genes[break_point:]:
            child1[y] = partner.chromosome[y]
            child2[y] = self.chromosome[y]

        return Individual(child1), Individual(child2)

    def parse_chromosome(self, sep='.'):
        """
        Unpack a chromosome
        
        Parameters
        ----------
        sep : str, optional
            String value separating nested keys. The default is '.', which
            assumes child keys are separated from parent keys by '.'.
        
        Returns
        -------
        dict
            Dictionary of unpacked keys and values.
        """

        parameter_kwargs = {}
        for key in self.chromosome.keys():
            if sep in key:
                kwargs = Individual.__parse_chr_key(key.split(sep),
                                                    self.chromosome[key])
                parameter_kwargs = Individual.__merge_dict(parameter_kwargs,
                                                           kwargs)
            else:
                parameter_kwargs[key] = self.chromosome[key]

        return parameter_kwargs

    @staticmethod
    def __parse_chr_key(keys, value):
        """Parse embedded keys."""
        if len(keys) > 1:
            new_value = {keys[-1]: value}
            return Individual.__parse_chr_key(keys[:-1], new_value)
        else:
            return {keys[0]: value}
    
    @staticmethod
    def __merge_dict(d1, d2):
        """Merge dictionaries together."""
        new_dict = {}
        for key in d1.keys():
            if key in d2.keys():
                if isinstance(d1[key], collections.MutableMapping) and\
                isinstance(d2[key], collections.MutableMapping):
                    if d1[key].keys() == d2[key].keys():
                        new_dict[key] = Individual.__merge_dict(d1[key],
                                                                d2[key])
                    else:
                        new_dict[key] = {**d1[key], **d2[key]}
            else:
                new_dict[key] = d1[key]
        for key in d2.keys():
            if key not in d1.keys():
                new_dict[key] = d2[key]
        return new_dict


class FitnessMixin(object):
    """
    Mixin class for calculating fitness of an individual.

    Fitness scores are problem dependent, thus abstracting out a single-class
    that covers all needs seems unlikely. However, some assumptions about
    fitness evaluators are necessary to ensure the `GeneticAlgorithm`
    class works properly. Therefore, all user-created fitness evaluators should
    extend this class, while implementing their own `score` function.

    The function `individual_to_kwargs` can be revised to return key-word
    arguments as desired.
    
    Functions
    ---------

    score : (Individual)
        Evaluate the fitness of a provided individual. Should be re-implemented
        in children classes.
    
    individual_to_kwargs : (Individual)
        Unflatten the chromosome dictionary of an individual. Can be
        re-implemented to parse a uniquely structured chromosome if necessary.
    """

        
    def score(self, individual):
        """
        Skeleton method to evaluate the fitness of a given individual
        
        Parameters
        ----------
        individual : Individual
            Individual to evaluate.
        
        Returns
        -------
        float
            Fitness score.
        """
        individual.fitness = 1

        return 1

    @staticmethod
    def individual_to_kwargs(individual):
        """
        Unpack an individual to parameter values.

        Unpack an individual to parameter/kwarg values. Can be extended to
        properly unpack problem-specific kwargs. Can often to be called to feed
        parameters to `score()`.
        
        Parameters
        ----------
        individual : Individual
            Individual whose parameters should be unpacked.
        
        Returns
        -------
        dict
            Dictionary of parameter keys and values.
        """

        param_dict = individual.parse_chromosome()
        return param_dict
    

class GeneticAlgorithm(object):
    """
    Attributes
    ----------
    genomic_space : dict
        Dictionary where each key represents a gene. Values are lists
        of possible values (alleles), each gene can exhibit. Dictionary
        is essentially a flattened version of `parameter_space`. 

        Example
            {'x.range': [0, 1, 2],
                'x.domain': [3, 4, 5],
                'y': [-1, -2, -3]}

    population : list
        List of current individuals.
    pop_size : int
        Total population size. The default is 100.
    generations : int
        Total number of generation to spawn. The default is 500.
    mutation_rate : float
        Probability to induce random mutation after a cross. The default is
        0.03, which will randomly mutate a single gene in a child in 3% of
        the crosses.
    elite_rate : float
        The percentage of individuals with the highest fitness scores to
        retain at the end of a generation. Top performers will move onto the
        next generation, without having their parameters changed. Default is
        0.1, and 10% of individuals with the highest fitness scores 
        will be kept.
    drift_rate : float
        The percentage of random individual to introduce for each
        generation. The default is 0.1, and the 10% least fit individuals
        will be replaced by new, random individuals. This helps move the
        parameters to a maximum, while also guarding against local maximas.
    verbose : boolean
        Whether to print progress as generations proceed.
    estimate_converge : boolean
        Whether to estimate convergence for possible early stoppage.
        Convergence is estimated by estimating the probability the next
        generation will lead to an increase in average fitness among elite
        individuals. Convergence is met when this probability is below a
        given threshold. 
    p_threshold : float, optional
        Minimum probability threshold for estimating convergence.
    gen_min : int, optional
        Minimum generations before convergence can be reached.
    zero_point : float, optional
        The percentile of number of generations that should act as a center
        when calculating success/failure weights. The probability of increasing
        fitness is modelled as a beta distribution as a conjugate prior to a
        binomial distribution. Because the probability of increasing fitness
        is dynamic, weighting successes/failures from the oldest generations
        equally with successes/failures from newest generations is improper.
        Therefore, success/failures are weighted using a sigmoid function with
        their distances centered at a "zero point". Given 100 generation and
        `zero_point=0.66`, this "zero point" will occurr at the 66th generation.
        Letting :math:`i` be an index for generations, the weight of that
        generation is calculated using the following formula:
        .. math::
            w_i = \frac{1}{1 + \exp{i - 66}}
        This weight is then added to the appropiate :math:`\alpha` or
        :math:`\beta` sum depending on whether fitness was increased or
        maintained at generation :math:`i`. Increasing the value towards 1, will
        weight newer measurement more heavily. The opposite is true for
        decreasing the value toward 0.

    Methods
    -------
        breed : perform the genetic algorithm search function.
        mutate : mutate a gene in a given individual.
        random_individual : return an individual with a random genotype.
    """

    def __init__(self, parameter_space, pop_size=100, generations=500, 
                 mutation_rate=0.03, elite_rate=0.1, drift_rate=0.1,
                 verbose=True, estimate_convergence=True, p_threshold=0.02,
                 gen_min=50, zero_point=0.66):
        """
        A class to optimize hyperparameters using a genetic algorithm.
        
        Parameters
        ----------
        parameter_space : dict
            A possibly nested dictionary containing parameter values to be
            evaluated. Keys pointing to non-dictionary entries should point to
            lists containing all possible test values. All keys should be
            strings with no "."s, as these are used when flattening input.

            Example:
                {'x': {'range': [0, 1, 2],
                       'domain': [3, 4, 5]},
                 'y': [-1, -2, -3]}
        pop_size : int, optional
            Total population size. The default is 100.
        generations : int, optional
            Total number of generation to spawn. The default is 500.
        mutation_rate : float, optional
            Probability to induce random mutation after a cross. The default is
            0.03, which will randomly mutate a single gene in a child in 3% of
            the crosses.
        elite_rate : float, optional
            The percentage of individuals with the highest fitness scores to
            retain at the end of a generation. Top performers will move onto the
            next generation, without having their parameters changed. Default is
            0.1, and 10% of individuals with the highest fitness scores 
            will be kept.
        drift_rate : float, optional
            The percentage of random individual to introduce for each
            generation. The default is 0.1, and the 10% least fit individuals
            will be replaced by new, random individuals. This helps move the
            parameters to a maximum, while also guarding against local maximas.
        verbose : boolean, optional
            Whether to print progress as generations proceed. Default is yes,
            and a progess bar will be displayed. If `estimate_convergence` is
            also True, a plot of convergence will also be displayed.
        estimate_converge : boolean, optional
            Whether to estimate convergence for possible early stoppage.
            Convergence is estimated by estimating the probability the next
            generation will lead to an increase in average fitness among elite
            individuals. Convergence is met when this probability is below a
            given threshold. Default is True. 
        p_threshold : float, optional
            Minimum probability threshold for estimating convergence. Default is
            0.02, and convergence will be assumed when the probability of an
            increase in fitness in the next generation is below 0.02.
        gen_min : int, optional
            Minimum generations before convergence can be reached. Default is
            50, and the model will not be assumed to be at convergence (even if 
            `p(increase)` < `p_threshold`) until the set number of generations
            has passed.
        zero_point : float, optional
            The percentile of number of generations that should act as a center
            when calculating success/failure weights. Default is 0.66. Increase
            toward 1 to increase the model's sensitivity, decrease to increase
            robustness.
        """

        self.__set_genomic_space(parameter_space)
        self.__initial_population(pop_size)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.estimate_convergence = estimate_convergence
        self.p_threshold = p_threshold
        self.gen_min = gen_min
        self.zero_point = zero_point
        self.elite_rate = elite_rate
        self.drift_rate = drift_rate
        self.verbose = verbose

    @staticmethod
    def __evaluate_key(key, sep):
        if not isinstance(key, str):
            raise ValueError("Keys must be strings for easy concatentation.")
        if sep in key:
            raise ValueError("Cannot have `{}` in dictionary keys.".format(sep))

    @staticmethod
    def flatten(d, parent_key='', sep='.'):
        """
        Flatten a dictionary containing nested dictionaries.
        
        Parameters
        ----------
        d : dict
            Dictionary to flatten
        parent_key : str, optional
            Key in parent dictionary pointing to `d`. The default is '', which
            assumes `d` is the highest level nested dictionary.
        sep : str, optional
            String value to separate child and parent keys. The default is '.',
            which will place a '.' between each key. All parent and child keys
            will be assessed to ensure they do not contain a `sep` character;
            therefore, `sep` should be set to a delimiter not present in current
            keys.
        
        Returns
        -------
        dict
            Flattened dictionary with parent and child keys separted by `sep`.

        References
        ----------

        Taken shamelessly from here:
            https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
        """

        items = []
        for k, v in d.items():
            GeneticAlgorithm.__evaluate_key(k, sep)
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(GeneticAlgorithm.flatten(v, new_key,
                                                      sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __set_genomic_space(self, parameter_space):
        """Set genomic space to a flattened dictionary."""
        self.genomic_space = self.flatten(parameter_space, sep='.')

    def __initial_population(self, n=1000):
        """Randomally initialize a population of set size."""
        if n <= 1:
            raise ValueError('Population size must exceed 1.')
        self.n = n
        self.population = [None]*self.n
        for i in range(self.n):
            self.population[i] = self.random_individual()
    
    @staticmethod
    def __score2prob(scores):
        """
        Convert fitness scores to probabilities using the softmax function.
        
        Parameters
        ----------
        scores : numpy.array
            Fitness scores for each individual in the current population
        
        Raises
        ------
        ValueError
            Raised if `scores` is not castable to numpy.array. 
        
        Returns
        -------
        numpy.array
            Probability to select each individual within the population 
        """

        try:
            scores = np.array(scores)
        except:
            raise ValueError("`scores` must be castable to numpy array.")
        ex_scores = np.exp(scores - np.max(scores))
        return ex_scores / ex_scores.sum()

    def random_individual(self):
        """
        Create a random individual.

        Returns
        -------
        Individual
            Individual with randomally selected values for
            `Individual.chromosome`. 
        """
        chromosome = {}
        for key, value in self.genomic_space.items():
            if isinstance(value, (np.ndarray, list, set)):
                chromosome[key] = np.random.choice(self.genomic_space[key])
            elif isinstance(value, (str, int, float)):
                chromosome[key] = value
            else:
                raise ValueError("Unexpected type: {}".format(type(value)))
        return Individual(chromosome)

    def best_performer(self):
        """
        Find indidivual with the highest fitness.

        Returns the individual with the highest fitness scores. If multiple
        individuals, with unique genotypes, share the same fitness score,
        all unique individuals are returned. 
        
        Returns
        -------
        list, Individual
            List of individuals with the highest fitness scores. 
        """

        scores = [x.fitness for x in self.population if x.fitness is not None]
        if len(scores) == 0:
            raise ValueError("Population fitness has not been assessed")
        best = np.max(scores)
        best_performers = [x for x in self.population if x.fitness == best]
        return list(set(best_performers))
    
    def update_posterior(self, increases, no_increases, n_generations):
        zero_point = np.max((1, n_generations * self.zero_point))
        alpha = weight_results(np.array(increases), zero_point)
        beta = weight_results(np.array(no_increases), zero_point)
        with open('alpha-beta.log', 'a') as f:
            f.write('alpha: {}, beta: {}\n'.format(alpha, beta))
        # self.p_increase_ = stats.beta(a=alpha, b=beta)
        self.p_increase_.a = alpha
        self.p_increase_.b = beta
        with open('a-b.log', 'a') as f:
            f.write('a: {}, b: {}\n'.format(self.p_increase_.a, self.p_increase_.b))
        self.p_of_increase_.append(beta_binom(1, 1, alpha, beta))

    def __initialize_diagnostic_plot(self):
        figure = plt.figure()
        gs = GridSpec(2, 2, figure=figure)
        ax1 = figure.add_subplot(gs[1, :])
        ax2 = figure.add_subplot(gs[0, 0])
        ax3 = figure.add_subplot(gs[0, 1])
        ax1.set_ylabel('Average Fitness')
        ax2.set_ylabel('$p($increase$)$')
        ax1.set_xlabel('Generation')
        ax2.set_xlabel('Generation')
        ax3.set_ylabel('$p($increase$)$')
        ax3.set_xlabel('x')
        
        norm = colors.Normalize(vmin=0, vmax=self.generations)
        scalar_map = plt.cm.ScalarMappable(norm=norm)
        scalar_map._A = []
        plt.colorbar(scalar_map, ax=ax3)
        return figure, scalar_map

    def __update_diagnostic_plot(self, figure, scalar_map, generation):
        axes = figure.axes
        axes[0].plot(range(generation), self.fitness_avgs_)
        axes[1].plot(range(generation), self.p_of_increase_)
        plt.suptitle('{} iterations '.format(generation)
                     + r'($\alpha =$ {:.2f}, $\beta =$ {:.2f})'.format(
                          self.p_increase_.a, self.p_increase_.b))
        axes[2].plot(self.x_, self.pdf_, color=scalar_map.to_rgba(generation),
                 alpha=0.25)
        plt.pause(0.01)


    def breed(self, fitness_function, n_min=None, p_threshold=None,
              zero_point=None):
        """
        Breed individuals in a population to optimize parameter values.
        
        Parameters
        ----------
        fitness_function : FitnessMixin
            An extension the FitnessMixin class, that should re-implement the
            `score` function to evaluate fitness on a problem-specific basis.

        Returns
        -------
            None
        """
        n_elite = int(self.elite_rate * self.n)
        n_rand = int(self.drift_rate * self.n)
        n_pairs = int((self.n - n_elite - n_rand) / 2)
        if self.estimate_convergence:
            # model probability of successfully increasing the fitness over each
            # iteration as a beta (conjugate prior of binomial)
            self.p_increase_ = stats.beta(a=1, b=1)
            self.fitness_avgs_ = [0]
            # set first generation to both increase and decrease as uninformed
            # "prior"
            increases = [1] # track generations where fitness increases occurr
            no_increases = [1] # track gens where fitness does not increase
            # track p(increase) to estimate convergence
            self.p_of_increase_ = [beta_binom(1, 1, 1, 1)]
            if self.verbose:
                self.x_ = np.arange(0, 1, 0.001)
                self.pdf_ = stats.beta.pdf(self.x_, self.p_increase_.a,
                                           self.p_increase_.b)
                fig, sm = self.__initialize_diagnostic_plot()
        # iterate through all generations, print progress bar if verbose
        iterator = range(self.generations)
        if self.verbose:
            iterator = tqdm(iterator)
            
        for i in iterator:
            new_population = []
            # calculate fitness for current population
            scores = np.array([0]*self.n)
            for j, each in enumerate(self.population):
                scores[j] = fitness_function.score(each)

            # order fitness scores in decreasing order
            ranked = np.argsort(-1 * scores)
            if self.estimate_convergence:
                current_fitness = np.mean(scores[ranked][:n_elite])
                if current_fitness > self.fitness_avgs_[i]:
                    increases.append(i + 1)
                else:
                    no_increases.append(i + 1)
                self.fitness_avgs_.append(current_fitness)
                self.update_posterior(increases, no_increases, i + 1)
                if self.verbose:
                    self.pdf_ = self.p_increase_.pdf(self.x_)
                    self.__update_diagnostic_plot(fig, sm, i + 2)
                # probability of next generation below threshold
                # -> likely converged, stop iteration 
                print(self.p_increase_.a, self.p_increase_.b)
                if self.p_of_increase_[-1] < self.p_threshold and\
                i + 1 > self.gen_min:
                    if self.verbose:
                        iterator.close()
                    break
            # pass best performers to the next generation
            new_population += [self.population[j] for j in ranked[:n_elite]]
            # add genetic drift to population via random samples
            new_population += [self.random_individual() for j in range(n_rand)]
            # remove poorest performers from selection
            selection = self.population[:self.n - n_rand]
            # convert fitness scores to probabilities for pairing selection
            p_select = self.__score2prob([x.fitness for x in selection])
            # select pairs
            pairs = np.random.choice(selection, (n_pairs, 2),
                                     replace=True, p=p_select)
            # create children by crossing parents, add to new generation
            for parent1, parent2 in pairs:
                child1, child2 = parent1.cross(parent2)
                r1, r2 = np.random.random(2)
                if r1 <= self.mutation_rate:
                    child1 = self.mutate(child1)
                if r2 <= self.mutation_rate:
                    child2 = self.mutate(child2)
                new_population += [child1, child2]
            self.population = new_population

        # score final population
        for each in self.population:
            fitness_function.score(each)

    def mutate(self, individual):
        """
        Mutate an individual.

        Mutates a single allele value in an individual's chromosome to a new
        value for the same gene.
        
        Parameters
        ----------
        individual : Individual
            Individual to mutate.
        
        Returns
        -------
        Individual
            Mutated individual. Individual is directly mutated within the
            function as well, so returning the object is just formality.
        """

        # choose random 'gene' to change
        mutable = [k for k, v in self.genomic_space.items()\
                   if isinstance(v, (np.ndarray, list, set)) and len(v) > 1]
        key = np.random.choice(mutable)
        # mutate current gene value to a new value. 
        value = np.random.choice([x for x in self.genomic_space[key]\
                                  if individual.chromosome[key] != x])
        individual.chromosome[key] = value
        return individual

def beta_binom(n, k, a, b):
    gammaln = special.gammaln
    out = gammaln(n + 1) + gammaln(k + a) + gammaln(n - k + b) + gammaln(a + b)\
        - (gammaln(k + 1) + gammaln(n - k + 1) + gammaln(a) + gammaln(b)\
           + gammaln(n + a + b))
    return np.exp(out)
        
def weight_results(hits, zero_point):
    weights = 1 / (1 + np.exp(-(hits - zero_point)))
    return weights.sum()