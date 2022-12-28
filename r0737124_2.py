import time
from typing import Tuple
import Reporter
import numpy as np
from numba import jit

@jit(nopython=True)
def distanceIndividual(numberOfCities: int, distanceMatrix: np.array, individual: np.array) -> float:
    distance = 0
    for i in range(numberOfCities - 1):
        distance += distanceMatrix[individual[i], individual[i+1]]
    distance += distanceMatrix[individual[numberOfCities-1], individual[0]]

    return distance

@jit(nopython=True)
def naive2Opt(solution: np.array, numberOfCities: int, distanceMatrix: np.array) -> np.array:
    """
    Naive 2-opt implementation: for each pair of nodes (the node and the one following it), check if the new solution is better.
    solution: 1D numpy array in the cycle notation containing the solution to optimize
    numberOfCities: number of cities in the problem
    distanceMatrix: 2D numpy array containing the distance matrix
    returns: 1D numpy array in the cycle notation containing the optimized solution
    """
    for point1 in range(numberOfCities):
        p1 = solution[point1 - 1]
        p2 = solution[point1]
        p3 = solution[(point1 + 1) % numberOfCities]
        p4 = solution[(point1 + 2) % numberOfCities]

        distance = distanceMatrix[p1][p2] + distanceMatrix[p2][p3] + distanceMatrix[p3][p4]
        newDistance = distanceMatrix[p1][p3] + distanceMatrix[p3][p2] + distanceMatrix[p2][p4]

        # If the new solution is better, swap the arcs
        if newDistance < distance:
            solution[point1] = p3
            solution[(point1 + 1) % numberOfCities] = p2
    
    return solution

@jit(nopython=True)
def advanced2Opt(solution: np.array, numberOfCities: int, distanceMatrix: np.array) -> np.array:
    for point1 in range(numberOfCities):
        p1 = solution[0]
        p2 = solution[point1 - 1]
        p3 = solution[point1]
        p6 = solution[-1]

        for point2 in range(point1 + 1, numberOfCities):
            p4 = solution[point2 - 1]
            p5 = solution[point2]

            distance = distanceMatrix[p2][p3] + distanceMatrix[p4][p5] + distanceMatrix[p6][p1]
            newDistance = distanceMatrix[p2][p5] + distanceMatrix[p6][p3] + distanceMatrix[p4][p1]

            # If the new solution is better, swap the arcs
            if newDistance < distance:
                a = solution[:point1]
                b = solution[point1:point2]
                c = solution[point2:]
                solution = np.concatenate((a, c, b))

    return solution

def randomGreedy(numberOfCities: int, rcl: float, distanceMatrix: np.array) -> np.array:
        individual = np.zeros(numberOfCities, dtype=np.int)
        individual[0] = np.random.randint(0, numberOfCities)
        for i in range(1, numberOfCities, 1):
            candidateList = buildCandidateList(numberOfCities, rcl, distanceMatrix, individual[:i])
            individual[i] = np.random.choice(candidateList, 1)[0]
        return individual

@jit(nopython=True)
def buildCandidateList(numberOfCities: int, rcl: float, distanceMatrix: np.array, indices: np.array) -> np.array:
    last = indices[-1]
    
    notUsed = []
    for i in range(numberOfCities):
        if i not in indices[:-1]:
            notUsed.append(i)
    
    notUsed.remove(last)
    
    candidates = []
    minimalDistance = np.inf
    for i in range(len(notUsed)):
        if distanceMatrix[last][notUsed[i]] < minimalDistance:
            minimalDistance = distanceMatrix[last][notUsed[i]]

    allowed_distance = (1 + rcl) * minimalDistance
    for x in notUsed:
        if distanceMatrix[last][x] <= allowed_distance:
            candidates.append(x)
    return np.array(candidates)

class r0737124:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.distanceMatrix = None
        self.numberOfCities = 0
        self.bestSolution = None
        self.bestObjective = np.inf

        # OPTIONS
        self.populationSize: int = 25
        self.numberOffspring: int = 50
        self.kTournamentSize: int = 5
        self.rcl: float = 0.1
        self.mutationRate: float = 0.05


    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        self.distanceMatrix = distanceMatrix

        # Get the number of cities
        self.numberOfCities = distanceMatrix.shape[0]

        # Initialize the population
        population = self.initializePopulation()

        generation = 1
        yourConvergenceTestsHere = True
        while( yourConvergenceTestsHere ):
            # Recombine the population
            offspring = self.recombination(population)

            # Mutate the offspring
            population = self.mutation(population, offspring)
            
            # Run local optimizer
            population = self.localOptimization(population)

            
            population, distances = self.elimination(population)
            population, distances = self.removeDuplicates(population, distances)
            population, distances = self.elitism(population, distances)

            meanObjective = np.mean(distances)
            bestObjective = np.min(distances)
            bestSolution = population[np.argmin(distances)]
            generation += 1
            
            print(f"Generation {generation}: best objective {bestObjective}, mean objective {meanObjective}")
            
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0

            # Shift the best solution to start at city 0
            bestSolution = np.roll(bestSolution, -np.argmin(bestSolution))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0
    
    def initializePopulation(self) -> np.array: #TODO check if self.populationSize is not > self.numberOfCities
        population = np.zeros((self.populationSize, self.numberOfCities), dtype=np.int)
        
        # Select random cities to start with
        randomStartCities = np.random.choice(np.arange(self.numberOfCities), self.populationSize, replace=False)

        # Create a greedy solution for each city
        for index, startCity in enumerate(randomStartCities):
            population[index] = self.createGreedySolution(startCity)
        
        return population

    
    def createGreedySolution(self, startCity: int) -> np.array:
        solution = np.zeros(self.numberOfCities, dtype=np.int)
        solution[0] = startCity

        # Create a list of all cities
        cities = np.arange(self.numberOfCities)
        cities = np.delete(cities, startCity)

        # Create a greedy solution
        for i in range(1, self.numberOfCities):
            # Get the city with the shortest distance to the previous city
            city = np.argmin(self.distanceMatrix[solution[i-1], cities])
            solution[i] = cities[city]

            # Remove the city from the list of cities
            cities = np.delete(cities, city)

        return solution

    def recombination(self, population: np.array) -> np.array:
        offspring = np.empty([self.numberOffspring, self.numberOfCities], dtype=np.int)

        # Select parents
        #parents = self.kTournamentSelection(population, self.numberOffspring)
        parents = self.kTournamentSelection(population, self.numberOffspring)

        # Recombine parents
        for index in range(0, self.numberOffspring, 2):
            offspring[index], offspring[index+1] = self.orderCrossover(parents[index], parents[index+1])
        
        return offspring


    def kTournamentSelection(self, population: np.array, amountOfParents: int) -> np.array:
        selectedParents = np.empty((amountOfParents, self.numberOfCities), dtype=np.int)

        for index in range(amountOfParents):
            # Select k random individuals
            randomIndividuals = population[np.random.choice(self.populationSize, self.kTournamentSize, replace=True)]

            # Select the individual with the best fitness
            distances = self.distancePopulation(randomIndividuals)
            indexBestIndividual = np.argsort(distances)

            # Add the best individual to the list of selected parents
            selectedParents[index] = randomIndividuals[indexBestIndividual[0]]
        
        return selectedParents

    def rouletteSelection(self, population: np.array, amountOfParents: int) -> np.array:
        selectedParents = np.empty((amountOfParents, self.numberOfCities), dtype=np.int)

        for index in range(amountOfParents):
            # Calculate the fitness of each individual
            distances = self.distancePopulation(population)
            fitness = 1 / distances

            # Select a random individual
            randomIndividual = np.random.choice(self.populationSize, 1, replace=True, p=fitness/np.sum(fitness))[0]

            # Add the individual to the list of selected parents
            selectedParents[index] = population[randomIndividual]
        
        return selectedParents
            
    def distancePopulation(self, population: np.array) -> np.array:
        #return np.apply_along_axis(self.numberOfCities, self.distanceMatrix, distanceIndividual, 1, population)
        distances = np.zeros(population.shape[0])
        for index, individual in enumerate(population):
            distances[index] = distanceIndividual(self.numberOfCities, self.distanceMatrix, individual)
        return distances

    def orderCrossover(self, parent1: np.array, parent2: np.array) -> Tuple[np.array, np.array]:
        # Select two random points
        randomPoints = np.random.choice(np.arange(self.numberOfCities), 2, replace=False)
        start = min(randomPoints)
        end = max(randomPoints)

        # Create the offspring
        offspring1 = np.zeros(self.numberOfCities, dtype=np.int)
        offspring2 = np.zeros(self.numberOfCities, dtype=np.int)

        # Copy the selected part from the parents to the offspring
        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]

        # Fill the rest of the offspring with the genes of the other parent
        usedInOffspring1 = set(parent1[start:end])
        usedInOffspring2 = set(parent2[start:end])

        sequenceParent1 = []
        sequenceParent2 = []

        index = end
        start = True
        while index != end or start:
            start = False

            if parent1[index] not in usedInOffspring2:
                sequenceParent1.append(parent1[index])
            if parent2[index] not in usedInOffspring1:
                sequenceParent2.append(parent2[index])
            index = (index + 1) % self.numberOfCities
        
        while len(sequenceParent1) != 0:
            offspring1[index] = sequenceParent2.pop(0)
            offspring2[index] = sequenceParent1.pop(0)
            index = (index + 1) % self.numberOfCities

        return offspring1, offspring2

    def mutation(self, population: np.array, offspring: np.array) -> np.array:
        newPopulation = np.vstack((population, offspring))

        for index in range(newPopulation.shape[0]):
            if np.random.rand() < self.mutationRate:
                newPopulation[index] = self.inverseMutation(newPopulation[index])
        
        return newPopulation
    
    def inverseMutation(self, individual: np.array) -> np.array:
        # Select two random points
        randomPoints = np.random.choice(np.arange(self.numberOfCities), 2, replace=False)
        start = min(randomPoints)
        end = max(randomPoints)

        # Inverse the order of the cities between the two points
        individual[start:end] = individual[start:end][::-1]

        return individual

    def elimination(self, population: np.array) -> Tuple[np.array, np.array]:
        distances = self.distancePopulation(population)
        sortedIndices = np.argsort(distances)

        return population[sortedIndices[:self.populationSize]], distances[sortedIndices[:self.populationSize]]
    
    def removeDuplicates(self, population: np.array, distances: np.array) -> Tuple[np.array, np.array]:
        newPopulation = np.empty((self.populationSize, self.numberOfCities), dtype=np.int)
        newPopulation[0] = population[0]

        for index in range(self.populationSize - 1, 0, -1):
            if distances[index] == distances[index-1]:
                newPopulation[index] = randomGreedy(self.numberOfCities, self.rcl, self.distanceMatrix)
            else:
                newPopulation[index] = population[index]

        newDistances = self.distancePopulation(newPopulation)

        return newPopulation, newDistances

    def localOptimization(self, population: np.array) -> np.array:
        for i in range(population.shape[0]):
            population[i] = advanced2Opt(population[i], self.numberOfCities, self.distanceMatrix)
        return population

    def elitism(self, population: np.array, distances: np.array) -> Tuple[np.array, np.array]:
        # Find the best individual's distance
        bestDistance = np.min(distances)

        # If the best distance so far is better than the best distance in the current population, replace the worst individual with the best individual so far
        if self.bestObjective < bestDistance:
            population[np.argmax(distances)] = self.bestSolution
            distances[np.argmax(distances)] = self.bestObjective

        return population, distances

    