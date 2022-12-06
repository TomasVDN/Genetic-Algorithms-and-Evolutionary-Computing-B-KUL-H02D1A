import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0737124:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        self.numberOfCities = 0
        self.distanceMatrix: np.array = None

        self.nearestNeighbourList = None

        self.populationSize = 100

        self.population: np.array = None
        self.populationFitness: np.array = None

        self.kTournamentSelectionSize = 5
        self.kTournamentEliminationSize = 5
        self.mutationChance = 0.05

        self.children: np.array = None

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.		
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        self.numberOfCities = self.distanceMatrix.shape[0]
        print(f'Number of cities: {self.numberOfCities}')

        print("Starting initialization...")
        self.initialize()

        self.calculatePopulationFitness()
        print("Initialization done.")

        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1,2,3,4,5])

        generation = 1

        # Your code here.
        yourConvergenceTestsHere = True
        while( yourConvergenceTestsHere ):
            
            # Create population size children, and append them to the population.
            print("Creating children...")
            self.children = np.empty((0, self.numberOfCities), dtype=int)
            while len(self.children) < self.populationSize:
                # Select two parents using tournament selection.
                parent1 = self.kTournamentSelection(self.kTournamentSelectionSize)
                parent2 = self.kTournamentSelection(self.kTournamentSelectionSize)
                while parent2 == parent1:
                    parent2 = self.kTournamentSelection(self.kTournamentSelectionSize)

                # Crossover the parents to create the children.
                child1, child2 = self.orderCrossover(parent1, parent2)

                # Calculate the fitness of the children.
                child1Fitness = self.calculateFitness(child1)
                child2Fitness = self.calculateFitness(child2)

                # Add the children to the population if their fitness is not infinite.
                if not np.isinf(child1Fitness):
                    self.children = np.append(self.children, [child1], axis=0)
                if not np.isinf(child2Fitness):
                    self.children = np.append(self.children, [child2], axis=0)
            print("Children created.")

            # Add the children to the population.
            self.population = np.append(self.population, self.children, axis=0)

            # Remove the duplicates from the population.
            self.population = np.unique(self.population, axis=0)

            # Calculate the fitness of the population.
            self.calculatePopulationFitness()

            # Execute elimination step.
            print("Executing elimination step...")
            self.eliminationStep()
            print("Elimination step executed.")

            # Calculate the fitness of the population.
            self.normalize()
            self.calculatePopulationFitness()
            
            # Calculate the mean and best objective function value of the population.
            meanObjective = np.mean(self.populationFitness)
            bestObjective = np.min(self.populationFitness)
            bestSolution = self.population[np.argmin(self.populationFitness)]

            generation += 1
            print(f'Generation {generation} completed. (mean: {meanObjective}, best: {bestObjective}')

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0

    def initialize(self):
        '''
        Initialize the population. For this, create 30% greedy individuals, and 70% random individuals (non infinite).
        '''
        self.calculateNearestNeighbourList()
        amountGreedy = int(self.populationSize * 0.3)

        # Keep the population in a numpy array, where each row is a candidate solution.
        self.population = np.empty((self.populationSize, self.numberOfCities), dtype=int)

        # Generate the greedy individuals, assuming a candidate could be None.
        for i in range(amountGreedy):
            candidate = None
            while candidate is None:
                candidate = self.generateGreedyCandidate(np.random.randint(0, self.numberOfCities))
            self.population[i] = candidate
        
        # Generate the random individuals, assuming a candidate could be None.
        for i in range(amountGreedy, self.populationSize):
            candidate = None
            while candidate is None:
                candidate = self.generateRandomCandidate()
            self.population[i] = candidate



    def calculateNearestNeighbourList(self):
        """
        For each city in distanceMatrix, sort the list of indices of the other cities in function of 
        the distance relative to the current city. Remove all indices for which the distance is 0 of inf.
        """
        nearestNeighboursList = []
        for i in range(self.numberOfCities):
            nearestNeighboursList.append(np.argsort(self.distanceMatrix[i]))
            nearestNeighboursList[i] = np.delete(nearestNeighboursList[i], np.where(self.distanceMatrix[i][nearestNeighboursList[i]] == 0))
            nearestNeighboursList[i] = np.delete(nearestNeighboursList[i], np.where(np.isinf(self.distanceMatrix[i][nearestNeighboursList[i]])))

        self.nearestNeighbourList = nearestNeighboursList

    def generateGreedyCandidate(self, startCityIndex: int):
        """
        Starting from the given startCityIndex, add the nearest neighbor using self.nearestNeighbourList not already in the visited cities. Continue until all 
        cities are visited. If the city for which a neighbor must be found does not have any unvisited neighbors, return None.
        """
        visitedCities = np.array([startCityIndex])
        currentCity = startCityIndex
        while len(visitedCities) < self.numberOfCities:
            # Find the nearest neighbor not already visited.
            for neighbor in self.nearestNeighbourList[currentCity]:
                if neighbor not in visitedCities:
                    visitedCities = np.append(visitedCities, neighbor)
                    currentCity = neighbor
                    break
            else:
                return None

        return visitedCities

    def generateRandomCandidate(self):
        """
        Starting from a random city, add a random neighbor using self.nearestNeighbourList not already in the visited cities. Continue until all 
        cities are visited. If the city for which a neighbor must be found does not have any unvisited neighbors, return None.
        """
        startCityIndex = np.random.randint(0, self.numberOfCities)
        visitedCities = np.array([startCityIndex])
        currentCity = startCityIndex
        while len(visitedCities) < self.numberOfCities:
            # Make a numpy array with all the neighbors, then filter out those already visited.
            neighbors = self.nearestNeighbourList[currentCity]
            neighbors = np.delete(neighbors, np.where(np.isin(neighbors, visitedCities)))
            
            # If there are no neighbors left, return None.
            if len(neighbors) == 0:
                return None
            
            # Choose a random neighbor and add it to the visited cities.
            neighbor = np.random.choice(neighbors)
            visitedCities = np.append(visitedCities, neighbor)
            currentCity = neighbor

        return visitedCities
    
    def calculateFitness(self, solution: np.array):
        """
        Calculate the fitness of the given solution.
        """
        fitness = 0
        for i in range(len(solution) - 1):
            fitness += self.distanceMatrix[solution[i]][solution[i + 1]]
        fitness += self.distanceMatrix[solution[-1]][solution[0]]

        return fitness
    
    def calculatePopulationFitness(self):
        """
        Calculate the fitness of each solution in the population.
        """
        fitness = np.array([])
        for solution in self.population:
            fitness = np.append(fitness, self.calculateFitness(solution))

        self.populationFitness = fitness

    def kTournamentSelection(self, k: int):
        """
        Select k random solutions from the population and return the index of the best one.
        """
        solutions = np.random.randint(self.populationSize, size=k)
        bestSolution = solutions[0]
        for solution in solutions:
            if self.populationFitness[solution] < self.populationFitness[bestSolution]:
                bestSolution = solution

        return bestSolution

    def orderCrossover(self, parent1Index: int, parent2Index: int):
        '''
        Perform order crossover on the given parents. This creates two children, both numpy arrays.
        '''
        # Select two crossover points.
        crossoverPoints = np.random.choice(self.numberOfCities, 2, replace=False)
        crossoverPoints.sort()

        # Get the parents
        parent1 = self.population[parent1Index]
        parent2 = self.population[parent2Index]

        # Create the children.
        child1 = np.empty(self.numberOfCities, dtype=int)
        child2 = np.empty(self.numberOfCities, dtype=int)

        # Copy the part between the crossover points to the children.
        child1[crossoverPoints[0]:crossoverPoints[1]] = parent1[crossoverPoints[0]:crossoverPoints[1]]
        child2[crossoverPoints[0]:crossoverPoints[1]] = parent2[crossoverPoints[0]:crossoverPoints[1]]

        child1Index = crossoverPoints[1]
        child2Index = crossoverPoints[1]
        parent1Index = crossoverPoints[1]
        parent2Index = crossoverPoints[1]

        # Copy the rest of the parent to the children.
        for i in range(self.numberOfCities):
            # If the parent is already in the child, skip it.
            if parent2[parent2Index] in child1:
                parent2Index = (parent2Index + 1) % self.numberOfCities
                continue

            # If the child is already full, skip it.
            if child1Index == self.numberOfCities:
                break

            child1[child1Index] = parent2[parent2Index]
            child1Index = (child1Index + 1) % self.numberOfCities
            parent2Index = (parent2Index + 1) % self.numberOfCities
        
        for i in range(self.numberOfCities):
            # If the parent is already in the child, skip it.
            if parent1[parent1Index] in child2:
                parent1Index = (parent1Index + 1) % self.numberOfCities
                continue

            # If the child is already full, skip it.
            if child2Index == self.numberOfCities:
                break

            child2[child2Index] = parent1[parent1Index]
            child2Index = (child2Index + 1) % self.numberOfCities
            parent1Index = (parent1Index + 1) % self.numberOfCities
        
        return child1, child2

    def eliminationStep(self):
        '''
        Replace the current population with a new population, selected with the kTournamentSelection function.
        '''
        newPopulation = np.empty((0, self.numberOfCities), int)

        # Add the best solution from the previous generation to the new population.
        bestSolution = np.argmin(self.populationFitness)
        newPopulation = np.append(newPopulation, [self.population[bestSolution]], axis=0)
        self.population = np.delete(self.population, bestSolution, axis=0)

        while len(newPopulation) < self.populationSize:
            # Select a solution from the previous generation using kTournamentSelection. If the solution is not already in the new population, add it. TODO check time complexity.
            solution = self.kTournamentSelection(self.kTournamentEliminationSize)
            newPopulation = np.append(newPopulation, [self.population[solution]], axis=0)

            # Remove the solution from the previous generation.
            self.population = np.delete(self.population, solution, axis=0)
                
        self.population = newPopulation

    def normalize(self):
        """
        Shift all cycles in the population such that the first element is 0.
        """
        for i in range(len(self.population)):
            index = np.where(self.population[i] == 0)[0][0]
            self.population[i] = np.roll(self.population[i], -index)